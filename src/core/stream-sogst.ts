import type { AppBase } from 'playcanvas';

import type { SogstMeta } from '../parsers/sogst';

import { loadSogst, parseSogstMeta } from './load-sogst';
import type { SogstData } from './sogst-data';
import { SogstDecoder, enumerateSogstGroups, groupFileList, groupBaseNames } from './sogst-decoder';
import { ZIP_FLAG_DATA_DESCRIPTOR, ZIP_LFH, ZIP_LOCAL_MAGIC } from './zip';

// Progressive loader for streamed archives. The encoder writes the ZIP
// in play order — meta.json, shN_centroids, persistent/*, seg_000/*, ... —
// with every entry stored (uncompressed), so entries can be parsed straight
// off the network stream from their local file headers. Groups decode as
// their last entry arrives; the scene is revealed once the persistent group
// and the first temporal segment are in, while later segments keep decoding
// behind playback. data.loadedThrough advances per segment so the animation
// driver can hold the playhead if the network falls behind.

// Growth buffer for a response with no Content-Length; it doubles from here,
// so this only sets how many reallocations a small archive costs.
const INITIAL_BUFFER_BYTES = 1 << 20;

// -- buffer-ahead gate tuning --------------------------------------------
// These are the margins in the two readiness inequalities below. They are
// tuning, not format constants: chosen against the reference clips on a
// throttled connection, where they were the smallest values that stopped
// the first playback pass from hitching at segment boundaries. Loosening
// them reveals the scene sooner and risks starving the playhead.

// Floor on the clip duration used by the gate, so a still (duration 0)
// cannot make the required-buffer term collapse to zero.
const MIN_GATE_DURATION_S = 0.1;

// Bandwidth measured over the first fraction of a second is dominated by
// connection setup, so the gate refuses to reveal until this much has
// elapsed rather than acting on that estimate.
const BANDWIDTH_WARMUP_S = 0.15;

// Headroom subtracted from the clip duration before asking whether the
// remaining bytes fit in it — the last segment has to land before playback
// reaches it, not exactly as it does.
const DOWNLOAD_HEADROOM_S = 0.3;

// Safety factor on the bandwidth estimate: measured throughput has to beat
// what the clip needs by this much before the byte side of the gate opens.
const DOWNLOAD_MARGIN = 1.3;

// Trailing window for the fill-rate estimate. Long enough to average over
// one segment's decode, short enough to track a connection that degrades.
const FILL_WINDOW_MS = 1500;

// Minimum wall-clock span between the oldest and newest fill sample before
// the fill rate means anything.
const MIN_FILL_SPAN_S = 0.35;

// Floor on the required buffer, so content that decodes faster than it
// plays still buffers a little before revealing.
const MIN_BUFFER_S = 0.3;

// Safety factor on the buffering inequality (buffered >= duration * (1 - f)).
const FILL_MARGIN = 1.25;

// Cap on retained fill samples: FILL_WINDOW_MS of history at the fastest
// segment rate observed, with slack.
const MAX_FILL_SAMPLES = 40;

type SogstStreamCallbacks = {
    /**
     * Download progress in [0, 100], measured against the reveal point
     * (meta.streams.reveal_bytes), not the whole file.
     */
    onProgress: (progress: number) => void;
    /**
     * A group finished decoding into the shared arrays. Fires per group
     * after the reveal; the caller refreshes GPU data for the given
     * [start, end) splat range (null = no new splats, e.g. the
     * end-of-stream notification) and then advances data.loadedThrough
     * to the given value — the driver deliberately does not advance it
     * itself, so the playhead can never enter a segment whose GPU data
     * the caller has not finished syncing.
     */
    onReady: (range: [number, number] | null, loadedThrough: number) => void;
    /**
     * A deferred SH labels payload finished decoding into the f_rest
     * arrays (sh-deferred archives only). The caller refreshes the GPU SH
     * data for the given [start, end) splat range; nothing about playback
     * gating changes — splats render DC-only until this lands.
     */
    onShReady?: (range: [number, number]) => void;
};

type SogstStream = {
    /** Resolves with playable data once the reveal set is decoded. */
    reveal: Promise<SogstData>;
    /** Resolves with the complete archive bytes (for caching). */
    complete: Promise<ArrayBuffer>;
};

const streamSogst = (app: AppBase, url: string, callbacks: SogstStreamCallbacks): SogstStream => {
    let revealResolve: (data: SogstData) => void;
    let revealReject: (err: Error) => void;
    const reveal = new Promise<SogstData>((resolve, reject) => {
        revealResolve = resolve;
        revealReject = reject;
    });

    const complete = (async (): Promise<ArrayBuffer> => {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to fetch ${url}: ${response.status} ${response.statusText}`);
        }
        const reader = response.body?.getReader();
        if (!reader) {
            throw new Error('Response body is not readable.');
        }

        const contentLength = parseInt(response.headers.get('content-length') ?? '0', 10);
        let bytes = new Uint8Array(contentLength > 0 ? contentLength : INITIAL_BUFFER_BYTES);
        let received = 0;
        const ensureCapacity = (needed: number) => {
            if (needed <= bytes.length) {
                return;
            }
            let next = bytes.length;
            while (next < needed) {
                next *= 2;
            }
            const grown = new Uint8Array(next);
            grown.set(bytes, 0);
            bytes = grown;
        };

        // -- incremental stored-entry parser over the accumulated bytes ----
        let parsePos = 0;
        let entriesDone = false;
        const nextEntry = (): { name: string; data: Uint8Array } | null => {
            if (entriesDone || received - parsePos < ZIP_LFH.size) {
                return null;
            }
            const view = new DataView(bytes.buffer, parsePos, ZIP_LFH.size);
            if (view.getUint32(0, true) !== ZIP_LOCAL_MAGIC) {
                // central directory reached — no more entries
                entriesDone = true;
                return null;
            }
            // Walking entries by local-header size only works because the
            // format forbids data descriptors (general-purpose bit 3). A
            // writer that emits them zeroes the local sizes, which would make
            // `total` the header length and march us into the payload as if
            // it were the next entry — garbage names, no error. Fail loudly
            // instead: this is a malformed archive, not a stream underrun.
            const flags = view.getUint16(ZIP_LFH.flags, true);
            if ((flags & ZIP_FLAG_DATA_DESCRIPTOR) !== 0) {
                throw new Error('sogst: archive uses ZIP data descriptors, which the format forbids');
            }
            const compressedSize = view.getUint32(ZIP_LFH.compressedSize, true);
            const nameLength = view.getUint16(ZIP_LFH.nameLength, true);
            const extraLength = view.getUint16(ZIP_LFH.extraLength, true);
            const total = ZIP_LFH.size + nameLength + extraLength + compressedSize;
            if (received - parsePos < total) {
                return null;
            }
            const nameStart = parsePos + ZIP_LFH.size;
            const name = new TextDecoder().decode(bytes.subarray(nameStart, nameStart + nameLength));
            const data = bytes.subarray(nameStart + nameLength + extraLength, parsePos + total);
            parsePos += total;
            return { name, data };
        };

        // -- group-decode driver -------------------------------------------
        let meta: SogstMeta | null = null;
        let monolithic = false;
        let decoder: SogstDecoder | null = null;
        let groups: ReturnType<typeof enumerateSogstGroups> = [];
        let neededNames: string[] = [];
        let groupIdx = 0;
        let pending = new Map<string, Uint8Array>();
        let revealGroupIdx = 0; // last group index needed before reveal
        let data: SogstData | null = null;
        let revealed = false;
        let revealPending = false; // reveal set decoded, awaiting buffer
        let progressWatermark = -1;
        let decodedThrough = 0; // decoded content boundary (absolute)
        let downloadStart = 0;
        // trailing samples of (wall time, buffered content) for the fill-
        // rate estimate — a cumulative mean drags below the sustained rate
        // during the initial decode ramp and over-delays the reveal
        const fillSamples: { t: number; b: number }[] = [];

        // Absolute clip time playable once groups [0, idx] are decoded: up
        // to the start of the next (not yet decoded) segment's coverage.
        const loadedThroughAfter = (idx: number): number => {
            const next = groups[idx + 1];
            return next ? meta.segments.list[next.segIndex].t0 : Infinity;
        };

        // Buffer-ahead readiness. The scene is revealed (and the playhead
        // started) only when the whole pipeline is provably ahead of
        // playback — otherwise the playhead starves at segment boundaries
        // through the first pass, perceived as rhythmic hitching. Two
        // sides, both with margin:
        //
        //  bytesQ: at the measured bandwidth, the remaining geometry bytes
        //          download within the clip duration (network side);
        //  fillQ:  the classic buffering inequality on the measured fill
        //          rate f of decoded content-time (network + decode + sync
        //          combined — this is what protects slow CPUs, where decode
        //          rather than download is the bottleneck):
        //          buffered >= duration * (1 - f) * margin.
        //
        // Both quotients also drive the tail of the progress bar, so the
        // bar reaches 100 exactly when playback can start cleanly.
        const gateInfo = (): { bytesQ: number; fillQ: number } => {
            const gb = meta?.streams?.geometry_bytes;
            if (!gb) {
                return { bytesQ: 1, fillQ: 1 };
            }
            const duration = Math.max(MIN_GATE_DURATION_S, (meta.time?.max ?? 0) - (meta.time?.min ?? 0));
            const elapsed = (performance.now() - downloadStart) / 1000;

            let bytesQ = 1;
            if (received < gb) {
                if (elapsed < BANDWIDTH_WARMUP_S) {
                    bytesQ = 0;
                } else {
                    const bw = received / elapsed;
                    const target = Math.max(1, gb - (bw * (duration - DOWNLOAD_HEADROOM_S)) / DOWNLOAD_MARGIN);
                    bytesQ = Math.min(1, received / target);
                }
            }

            let fillQ = 0;
            if (!isFinite(decodedThrough)) {
                fillQ = 1; // geometry fully decoded
            } else if (fillSamples.length > 1) {
                const buffered = decodedThrough - (meta.time?.min ?? 0);
                const now = performance.now();
                // oldest sample within the trailing window
                let ref = fillSamples[0];
                for (const sample of fillSamples) {
                    if (now - sample.t <= FILL_WINDOW_MS) {
                        break;
                    }
                    ref = sample;
                }
                const span = (now - ref.t) / 1000;
                if (buffered > 0 && span >= MIN_FILL_SPAN_S) {
                    const f = Math.min(1, Math.max(0, buffered - ref.b) / span);
                    const required = Math.max(MIN_BUFFER_S, duration * (1 - f) * FILL_MARGIN);
                    fillQ = Math.min(1, buffered / required);
                }
            }
            return { bytesQ, fillQ };
        };

        const gateReady = (): boolean => {
            const { bytesQ, fillQ } = gateInfo();
            return bytesQ >= 1 && fillQ >= 1;
        };

        const doReveal = () => {
            data!.loadedThrough = decodedThrough;
            revealed = true;
            revealPending = false;
            callbacks.onProgress(100);
            revealResolve(data!);
        };

        // Decode runs on its own promise chain so the network loop never
        // pauses for it — serializing download behind decode both wastes
        // bandwidth and drags the measured fill rate (and therefore the
        // buffering gate) well below what the connection supports.
        let decodeChain: Promise<void> = Promise.resolve();
        const enqueueDecode = (task: () => Promise<void>) => {
            decodeChain = decodeChain.then(task);
            // rejection is delivered where the chain is awaited (stream
            // tail) — this handler only silences the interim unhandled-
            // rejection warning
            decodeChain.catch(() => {
                /* surfaced via the reveal promise */
            });
        };

        const processGroup = async (
            idx: number,
            group: ReturnType<typeof enumerateSogstGroups>[number],
            files: Map<string, Uint8Array>
        ) => {
            await decoder!.decodeGroup(group, files);
            if (idx >= revealGroupIdx) {
                decodedThrough = loadedThroughAfter(idx);
                if (isFinite(decodedThrough)) {
                    fillSamples.push({ t: performance.now(), b: decodedThrough - (meta.time?.min ?? 0) });
                    if (fillSamples.length > MAX_FILL_SAMPLES) {
                        fillSamples.shift();
                    }
                }
            }

            if (!revealed && idx === revealGroupIdx) {
                data = decoder!.buildData();
                revealPending = true;
                if (gateReady()) {
                    doReveal();
                }
            } else if (!revealed && revealPending) {
                // still buffering: later groups keep decoding into the
                // shared arrays (the resource created at reveal picks them
                // up); reveal as soon as the pipeline is provably ahead
                if (gateReady()) {
                    doReveal();
                }
            } else if (revealed && data) {
                callbacks.onReady(group.range, decodedThrough);
            }
        };

        const handleEntry = (name: string, entryData: Uint8Array) => {
            if (name === 'meta.json') {
                meta = parseSogstMeta(entryData.slice());
                if (!meta.streams) {
                    // monolithic archive: keep downloading and decode the
                    // whole buffer once it completes
                    monolithic = true;
                    return;
                }
                decoder = new SogstDecoder(app, meta);
                groups = enumerateSogstGroups(meta);
                // sh-deferred archives ship labels behind all geometry, so
                // geometry groups complete on the base texture set alone
                neededNames = meta.streams.sh_deferred ? [...groupBaseNames(meta)] : groupFileList(meta);
                // reveal set = persistent group plus the first temporal segment
                const firstSeg = groups.findIndex((g) => g.segIndex >= 0);
                revealGroupIdx = firstSeg >= 0 ? firstSeg : groups.length - 1;
                return;
            }
            if (!meta) {
                throw new Error(`sogst: unexpected entry ${name} before meta.json`);
            }
            if (monolithic) {
                return;
            }
            if (name === 'shN_centroids.webp') {
                decoder!.setCentroids(entryData.slice());
                return;
            }
            if (meta.streams.sh_deferred && name.endsWith('/shN_labels.webp')) {
                // trailing SH pass: geometry for this group is long since
                // decoded and (possibly) playing DC-only
                const prefix = name.slice(0, -'/shN_labels.webp'.length);
                const group = groups.find((g) => g.prefix === prefix);
                if (group) {
                    const labels = entryData.slice();
                    enqueueDecode(async () => {
                        await decoder!.decodeGroupSH(group, labels);
                        callbacks.onShReady?.(group.range);
                    });
                }
                return;
            }
            const group = groups[groupIdx];
            if (!group || !name.startsWith(`${group.prefix}/`)) {
                return; // stray entry (or all groups already decoded)
            }
            const bare = name.slice(group.prefix!.length + 1);
            // An entry backing a group member this build doesn't know must be
            // ignored, not treated as the group's data (spec §3.1 — additive
            // groups ship under version 1 and have to degrade like shN/accel).
            // Filtering here is also what keeps the size test below sound: it
            // counts entries, so an unrecognised one would push the count to
            // neededNames.length while a required name was still outstanding,
            // firing the decode a file short — which then throws "missing from
            // archive". Whether that happened depended on the order the
            // encoder wrote the entries in.
            if (!neededNames.includes(bare)) {
                return;
            }
            // copy out of the shared download buffer: it may be grown
            // (reallocated) while the group is still pending
            pending.set(bare, entryData.slice());
            if (pending.size === neededNames.length) {
                const idx = groupIdx;
                const files = pending;
                pending = new Map();
                groupIdx++;
                enqueueDecode(() => processGroup(idx, groups[idx], files));
            }
        };

        try {
            for (;;) {
                const { done, value } = await reader.read();
                if (done) {
                    break;
                }
                if (!downloadStart) {
                    downloadStart = performance.now();
                }
                ensureCapacity(received + value.length);
                bytes.set(value, received);
                received += value.length;

                // release a pending reveal as soon as the buffer criterion
                // is met — don't wait for the next group to finish decoding
                if (revealPending && !revealed && gateReady()) {
                    doReveal();
                }

                if (meta?.streams?.reveal_bytes && !revealed) {
                    // download maps to 0-70; buffer-readiness (the lower of
                    // the bandwidth and fill quotients) maps to 70-99, so
                    // the bar hits 100 exactly at a play-ready reveal
                    let progress;
                    if (meta.streams.geometry_bytes) {
                        const dl = Math.min(1, received / meta.streams.reveal_bytes);
                        const { bytesQ, fillQ } = gateInfo();
                        progress = Math.trunc(70 * dl + 29 * Math.min(bytesQ, fillQ));
                    } else {
                        progress = Math.min(99, Math.trunc((received / meta.streams.reveal_bytes) * 100));
                    }
                    if (progress > progressWatermark) {
                        progressWatermark = progress;
                        callbacks.onProgress(progress);
                    }
                } else if (monolithic && contentLength > 0) {
                    // whole-file download maps to 0-70, decode takes 70-100
                    const progress = Math.min(70, Math.trunc((received / contentLength) * 70));
                    if (progress > progressWatermark) {
                        progressWatermark = progress;
                        callbacks.onProgress(progress);
                    }
                }

                for (;;) {
                    const entry = nextEntry();
                    if (!entry) {
                        break;
                    }
                    handleEntry(entry.name, entry.data);
                }
            }

            const buffer = bytes.byteLength === received ? bytes.buffer : bytes.slice(0, received).buffer;

            if (monolithic) {
                const decoded = await loadSogst(app, buffer, (p) =>
                    callbacks.onProgress(Math.min(100, Math.round(70 + p * 0.3)))
                );
                revealed = true;
                callbacks.onProgress(100);
                revealResolve(decoded);
                return buffer;
            }

            // flush a trailing partial group (shouldn't happen with a well-
            // formed archive, but don't leave decoded splats stranded)
            if (decoder && groupIdx < groups.length && pending.size === neededNames.length) {
                const idx = groupIdx;
                const files = pending;
                pending = new Map();
                groupIdx++;
                enqueueDecode(() => processGroup(idx, groups[idx], files));
            }
            // drain all queued decodes (also surfaces any decode error)
            await decodeChain;
            if (revealPending && !revealed) {
                // download finished — nothing left to buffer against
                doReveal();
            }
            if (!revealed) {
                throw new Error('sogst: stream ended before the reveal set was decoded');
            }
            if (data) {
                callbacks.onReady(null, Infinity);
            }
            decoder?.destroy();

            return buffer;
        } catch (err) {
            decoder?.destroy();
            revealReject(err as Error);
            throw err;
        }
    })();

    // the reveal consumer handles errors via the complete promise
    complete.catch(() => {
        /* caller handles it on the returned promise */
    });

    return { reveal, complete };
};

export { streamSogst };
