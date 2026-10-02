import type { AppBase } from 'playcanvas';

import type { SogstMeta } from '../parsers/sogst';

import { loadSogst } from './load-sogst';
import { SerialQueue } from './serial-queue';
import { FillHistory, gateOpen, gateQuotients, monolithicProgress, streamProgress } from './sogst-buffer-gate';
import type { GateQuotients } from './sogst-buffer-gate';
import { EntryUnitWriter, idbGetEntryUnit } from './sogst-cache';
import type { EntryManifest } from './sogst-cache';
import type { SogstData } from './sogst-data';
import { SogstDecoder } from './sogst-decoder';
import type { SogstGroup } from './sogst-decoder';
import { SogstGroupAssembler } from './sogst-group-assembler';
import type { AssemblerResult } from './sogst-group-assembler';
import { ZipStreamReader } from './zip';

// Progressive loader for streamed archives.
//
// The encoder writes the ZIP in play order: meta.json, shN_centroids,
// persistent/*, seg_000/*, and so on. It stores every entry uncompressed, so
// this loader can parse each entry from its local file header as the entry
// arrives off the network.
//
// A group decodes when its last entry arrives. The viewer shows the scene
// once the persistent group and the first temporal segment are complete, and
// the later segments continue to decode during playback. data.loadedThrough
// advances once per segment, so the animation driver can hold the playhead
// when the network is too slow.
//
// Two sources feed the same decode driver: the network stream, and a replay
// of the entry units an earlier visit cached in IndexedDB. Neither holds the
// whole archive in memory.
//
// The work is split into single-purpose parts, each usable on its own:
//
//   ZipStreamReader      (zip.ts)                   bytes in, entries out
//   SogstGroupAssembler  (sogst-group-assembler.ts) entries in, decode steps
//                                                   and cache units out
//   gate functions       (sogst-buffer-gate.ts)     measurements in, reveal
//                                                   readiness and progress out
//   SerialQueue          (serial-queue.ts)          decodes, one at a time
//
// The driver below only wires them together and owns the reveal state.

// Cache replay reads the next unit only while fewer than this many group
// decodes are queued. IndexedDB reads far faster than the decoder runs, and
// without the limit a replay would queue the whole archive's compressed
// bytes in memory at once.
const REPLAY_MAX_QUEUED_DECODES = 2;

type SogstStreamCallbacks = {
    /**
     * Download progress in [0, 100]. It measures progress against the reveal
     * point in meta.streams.reveal_bytes, not against the whole file.
     */
    onProgress: (progress: number) => void;
    /**
     * A group finished decoding into the shared arrays. This fires once per
     * group after the reveal.
     *
     * The caller must refresh the GPU data for the given [start, end) splat
     * range, then set data.loadedThrough to the given value. A null range
     * means no new splats, as in the end-of-stream notification.
     *
     * The driver does not advance data.loadedThrough itself. That is
     * deliberate: it keeps the playhead out of any segment whose GPU data the
     * caller has not synced yet.
     */
    onReady: (range: [number, number] | null, loadedThrough: number) => void;
    /**
     * A deferred SH labels payload finished decoding into the f_rest arrays.
     * This fires for sh-deferred archives only.
     *
     * The caller must refresh the GPU SH data for the given [start, end)
     * splat range. This does not change playback gating. The splats render
     * with DC colour only until the payload arrives.
     */
    onShReady?: (range: [number, number]) => void;
};

type SogstStream = {
    /** Resolves with playable data once the reveal set is decoded. */
    reveal: Promise<SogstData>;
    /**
     * Resolves once the whole archive is decoded. It carries the archive bytes
     * for a monolithic archive only, which the caller caches whole. A
     * streamed archive caches itself unit by unit (see sogst-cache.ts), and
     * this resolves with null.
     */
    complete: Promise<ArrayBuffer | null>;
};

// The decode side of a stream. A source calls handleEntry() for each archive
// entry in archive order and setReceived() as bytes arrive, then finish().
type SogstDriver = {
    readonly monolithic: boolean;
    setReceived: (bytes: number, contentLength?: number) => void;
    handleEntry: (name: string, entryData: Uint8Array, offset: number) => void;
    waitForQueuedDecodes: (max: number) => Promise<void>;
    finish: (totalBytes: number) => Promise<void>;
    resolveMonolithic: (decoded: SogstData) => void;
    fail: (err: Error) => void;
};

// Absolute clip time the viewer can play once groups [0, index] are
// decoded: up to the start of the next segment's coverage, the first segment
// not decoded yet.
const loadedThroughAfter = (meta: SogstMeta, groups: SogstGroup[], index: number): number => {
    const next = groups[index + 1];
    return next ? meta.segments.list[next.segIndex].t0 : Infinity;
};

const createSogstDriver = (
    app: AppBase,
    callbacks: SogstStreamCallbacks,
    revealResolve: (data: SogstData) => void,
    revealReject: (err: Error) => void,
    writer: EntryUnitWriter | null
): SogstDriver => {
    const assembler = new SogstGroupAssembler();
    // Decode runs on its own queue, so the network loop never waits for it.
    // Running the download behind the decode wastes bandwidth. It also
    // lowers the measured fill rate, and therefore the buffering gate, well
    // below what the connection supports.
    const decodes = new SerialQueue();
    const fill = new FillHistory();
    let decoder: SogstDecoder | null = null;
    let data: SogstData | null = null;

    // reveal state: the group whose decode completes the reveal set (the
    // persistent group plus the first temporal segment), and whether the
    // scene is decoded but waiting for the buffer gate, or shown
    let revealGroupIdx = 0;
    let revealPending = false;
    let revealed = false;

    // measurements for the gate
    let decodedThrough = 0; // absolute clip time decoded so far
    let downloadStart = 0;
    let received = 0;
    let progressWatermark = -1;

    const quotients = (): GateQuotients => {
        const meta = assembler.meta;
        const geometryBytes = meta?.streams?.geometry_bytes;
        if (!meta || !geometryBytes) {
            return { bytesQ: 1, fillQ: 1 };
        }
        const timeMin = meta.time?.min ?? 0;
        const now = performance.now();
        return gateQuotients({
            received,
            geometryBytes,
            elapsed: (now - downloadStart) / 1000,
            duration: (meta.time?.max ?? 0) - timeMin,
            buffered: decodedThrough - timeMin,
            fill: fill.samples,
            now
        });
    };

    const reportProgress = (progress: number) => {
        if (progress > progressWatermark) {
            progressWatermark = progress;
            callbacks.onProgress(progress);
        }
    };

    const doReveal = () => {
        data!.loadedThrough = decodedThrough;
        revealed = true;
        revealPending = false;
        callbacks.onProgress(100);
        revealResolve(data!);
    };

    const tryReveal = () => {
        if (revealPending && !revealed && gateOpen(quotients())) {
            doReveal();
        }
    };

    const onGroupDecoded = (index: number, group: SogstGroup) => {
        const meta = assembler.meta!;
        if (index >= revealGroupIdx) {
            decodedThrough = loadedThroughAfter(meta, assembler.groups, index);
            if (isFinite(decodedThrough)) {
                fill.push({ t: performance.now(), b: decodedThrough - (meta.time?.min ?? 0) });
            }
        }
        if (!revealed && index === revealGroupIdx) {
            data = decoder!.buildData();
            revealPending = true;
        }
        if (!revealed) {
            // Still buffering: later groups keep decoding into the shared
            // arrays, and the resource created at the reveal reads them.
            tryReveal();
        } else if (data) {
            callbacks.onReady(group.range, decodedThrough);
        }
    };

    // Act on one assembler result: cache its unit, and start what its step
    // needs.
    const apply = ({ step, unit }: AssemblerResult) => {
        if (unit) {
            writer?.add(unit);
        }
        if (!step) {
            return;
        }
        switch (step.kind) {
            case 'meta': {
                if (step.monolithic) {
                    return;
                }
                decoder = new SogstDecoder(app, step.meta);
                const firstSegment = assembler.groups.findIndex((g) => g.segIndex >= 0);
                revealGroupIdx = firstSegment >= 0 ? firstSegment : assembler.groups.length - 1;
                return;
            }
            case 'centroids':
                decoder!.setCentroids(step.bytes);
                return;
            case 'labels':
                // trailing SH pass: the group's geometry is long since
                // decoded and possibly playing DC-only
                decodes.push(async () => {
                    await decoder!.decodeGroupSH(step.group, step.bytes);
                    callbacks.onShReady?.(step.group.range);
                });
                return;
            case 'group':
                decodes.push(async () => {
                    await decoder!.decodeGroup(step.group, step.files);
                    onGroupDecoded(step.index, step.group);
                });
        }
    };

    const setReceived = (bytes: number, contentLength = 0) => {
        if (!downloadStart) {
            downloadStart = performance.now();
        }
        received = bytes;
        // release a pending reveal as soon as the buffer criterion is met;
        // do not wait for the next group to finish decoding
        tryReveal();

        const streams = assembler.meta?.streams;
        if (streams?.reveal_bytes && !revealed) {
            reportProgress(streamProgress(received, streams.reveal_bytes, streams.geometry_bytes ? quotients() : null));
        } else if (assembler.monolithic && contentLength > 0) {
            reportProgress(monolithicProgress(received, contentLength));
        }
    };

    const finish = async (totalBytes: number) => {
        apply(assembler.finish());
        // drain all queued decodes; this also raises any decode error
        await decodes.drain();
        if (revealPending && !revealed) {
            // the download finished, so there is nothing left to buffer
            // against
            doReveal();
        }
        if (!revealed) {
            throw new Error('sogst: stream ended before the reveal set was decoded');
        }
        if (data) {
            callbacks.onReady(null, Infinity);
        }
        decoder?.destroy();
        // The manifest goes last, so a stream that failed above leaves no
        // cache hit behind.
        await writer?.finish(totalBytes);
    };

    const resolveMonolithic = (decoded: SogstData) => {
        revealed = true;
        callbacks.onProgress(100);
        revealResolve(decoded);
    };

    const fail = (err: Error) => {
        decoder?.destroy();
        revealReject(err);
    };

    return {
        get monolithic() {
            return assembler.monolithic;
        },
        setReceived,
        handleEntry: (name, entryData, offset) => apply(assembler.accept(name, entryData, offset)),
        waitForQueuedDecodes: (max) => decodes.waitForAtMost(max),
        finish,
        resolveMonolithic,
        fail
    };
};

// Wire a driver to reveal/complete promises around a source loop.
const runSogstStream = (
    app: AppBase,
    callbacks: SogstStreamCallbacks,
    writer: EntryUnitWriter | null,
    source: (driver: SogstDriver) => Promise<ArrayBuffer | null>
): SogstStream => {
    let revealResolve: (data: SogstData) => void;
    let revealReject: (err: Error) => void;
    const reveal = new Promise<SogstData>((resolve, reject) => {
        revealResolve = resolve;
        revealReject = reject;
    });

    const driver = createSogstDriver(app, callbacks, revealResolve, revealReject, writer);
    const complete = source(driver).catch((err: Error) => {
        driver.fail(err);
        throw err;
    });

    // The reveal consumer handles errors through the complete promise.
    complete.catch(() => {
        /* caller handles it on the returned promise */
    });

    return { reveal, complete };
};

// Stream an archive from the network. With a cache key, each decode step's
// entries are written to IndexedDB as the step completes.
const streamSogst = (
    app: AppBase,
    url: string,
    callbacks: SogstStreamCallbacks,
    cacheKey: string | null = null
): SogstStream => {
    const writer = cacheKey ? new EntryUnitWriter(cacheKey) : null;
    return runSogstStream(app, callbacks, writer, async (driver) => {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to fetch ${url}: ${response.status} ${response.statusText}`);
        }
        const body = response.body?.getReader();
        if (!body) {
            throw new Error('Response body is not readable.');
        }
        const contentLength = parseInt(response.headers.get('content-length') ?? '0', 10);

        const reader = new ZipStreamReader();
        for (;;) {
            const { done, value } = await body.read();
            if (done) {
                break;
            }
            // a monolithic archive decodes from the whole file at the end
            reader.retainAll = driver.monolithic;
            reader.append(value);
            driver.setReceived(reader.received, contentLength);
            for (let entry = reader.next(); entry; entry = reader.next()) {
                driver.handleEntry(entry.name, entry.data, entry.offset);
            }
        }

        if (driver.monolithic) {
            const buffer = reader.archive();
            const decoded = await loadSogst(app, buffer, (p) =>
                callbacks.onProgress(Math.min(100, Math.round(70 + p * 0.3)))
            );
            driver.resolveMonolithic(decoded);
            return buffer;
        }

        await driver.finish(reader.received);
        return null;
    });
};

// Replay an archive an earlier visit cached unit by unit. A missing unit (the
// browser evicted part of the store) fails the load, and the caller falls
// back to the network.
const replaySogst = (
    app: AppBase,
    cacheKey: string,
    manifest: EntryManifest,
    callbacks: SogstStreamCallbacks
): SogstStream => {
    return runSogstStream(app, callbacks, null, async (driver) => {
        let received = 0;
        for (let i = 0; i < manifest.sogstUnits; i++) {
            await driver.waitForQueuedDecodes(REPLAY_MAX_QUEUED_DECODES);
            const entries = await idbGetEntryUnit(cacheKey, i);
            if (!entries) {
                throw new Error(`sogst: cached unit ${i} of ${manifest.sogstUnits} is missing`);
            }
            for (const entry of entries) {
                driver.handleEntry(entry.name, new Uint8Array(entry.bytes), entry.offset);
                received = Math.max(received, entry.offset + entry.bytes.byteLength);
            }
            driver.setReceived(received);
        }
        await driver.finish(manifest.totalBytes);
        return null;
    });
};

export { replaySogst, streamSogst };
