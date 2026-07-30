import type { AppBase } from 'playcanvas';

import {
    V3Decoder, enumerateV3Groups, groupFileList, loadOmg4V3, parseV3Meta, Omg4V3Data,
    GROUP_FILE_NAMES
} from './load-omg4-v3';

// Progressive loader for streamed v3 archives. The encoder writes the ZIP
// in play order — meta.json, shN_centroids, persistent/*, seg_000/*, ... —
// with every entry stored (uncompressed), so entries can be parsed straight
// off the network stream from their local file headers. Groups decode as
// their last entry arrives; the scene is revealed once the persistent group
// and the first temporal segment are in, while later segments keep decoding
// behind playback. data.loadedThrough advances per segment so the animation
// driver can hold the playhead if the network falls behind.

const ZIP_LOCAL_MAGIC = 0x04034b50;

type V3StreamCallbacks = {
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

type V3Stream = {
    /** Resolves with playable data once the reveal set is decoded. */
    reveal: Promise<Omg4V3Data>;
    /** Resolves with the complete archive bytes (for caching). */
    complete: Promise<ArrayBuffer>;
};

const streamOmg4V3 = (app: AppBase, url: string, callbacks: V3StreamCallbacks): V3Stream => {
    let revealResolve: (data: Omg4V3Data) => void;
    let revealReject: (err: Error) => void;
    const reveal = new Promise<Omg4V3Data>((resolve, reject) => {
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
        let bytes = new Uint8Array(contentLength > 0 ? contentLength : 1 << 20);
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
        const nextEntry = (): { name: string, data: Uint8Array } | null => {
            if (entriesDone || received - parsePos < 30) {
                return null;
            }
            const view = new DataView(bytes.buffer, parsePos, 30);
            if (view.getUint32(0, true) !== ZIP_LOCAL_MAGIC) {
                // central directory reached — no more entries
                entriesDone = true;
                return null;
            }
            const compressedSize = view.getUint32(18, true);
            const nameLength = view.getUint16(26, true);
            const extraLength = view.getUint16(28, true);
            const total = 30 + nameLength + extraLength + compressedSize;
            if (received - parsePos < total) {
                return null;
            }
            const name = new TextDecoder().decode(bytes.subarray(parsePos + 30, parsePos + 30 + nameLength));
            const data = bytes.subarray(parsePos + 30 + nameLength + extraLength, parsePos + total);
            parsePos += total;
            return { name, data };
        };

        // -- group-decode driver -------------------------------------------
        let meta: any = null;
        let monolithic = false;
        let decoder: V3Decoder | null = null;
        let groups: ReturnType<typeof enumerateV3Groups> = [];
        let neededNames: string[] = [];
        let groupIdx = 0;
        let pending = new Map<string, Uint8Array>();
        let revealGroupIdx = 0;      // last group index needed before reveal
        let data: Omg4V3Data | null = null;
        let revealed = false;
        let progressWatermark = -1;
        let playGate = false;           // playback held at t=0 for buffering
        let gatedLoadedThrough = 0;     // decoded boundary withheld while gated
        let downloadStart = 0;

        // Absolute clip time playable once groups [0, idx] are decoded: up
        // to the start of the next (not yet decoded) segment's coverage.
        const loadedThroughAfter = (idx: number): number => {
            const next = groups[idx + 1];
            return next ? meta.segments.list[next.segIndex].t0 : Infinity;
        };

        // Buffer-ahead criterion: start (or keep) the playhead running only
        // when, at the measured bandwidth, the rest of the geometry will
        // arrive before the playhead needs it. Below-realtime connections
        // otherwise starve the playhead at segment boundaries all through
        // the first pass — perceived as rhythmic hitching on every device,
        // since the stall is network-bound, not compute-bound. Holding at
        // t=0 (scene revealed, camera live) trades that for one predictable
        // buffering wait.
        const gateOpen = (): boolean => {
            const gb = meta?.streams?.geometry_bytes;
            if (!gb || received >= gb) {
                return true;
            }
            const elapsed = (performance.now() - downloadStart) / 1000;
            if (elapsed < 0.15) {
                return false;
            }
            const remaining = (gb - received) / (received / elapsed);
            const duration = Math.max(0.1, (meta.time?.max ?? 0) - (meta.time?.min ?? 0));
            return remaining * 1.3 + 0.3 <= duration;
        };

        const groupComplete = async () => {
            const group = groups[groupIdx];
            const files = pending;
            pending = new Map();
            await decoder!.decodeGroup(group, files);

            if (!revealed && groupIdx === revealGroupIdx) {
                data = decoder!.buildData();
                gatedLoadedThrough = loadedThroughAfter(groupIdx);
                if (gateOpen()) {
                    data.loadedThrough = gatedLoadedThrough;
                } else {
                    playGate = true;
                    data.loadedThrough = data.timeMin;
                }
                revealed = true;
                callbacks.onProgress(100);
                revealResolve(data);
            } else if (revealed && data) {
                gatedLoadedThrough = loadedThroughAfter(groupIdx);
                if (playGate && !gateOpen()) {
                    callbacks.onReady(group.range, data.timeMin);
                } else {
                    playGate = false;
                    callbacks.onReady(group.range, gatedLoadedThrough);
                }
            }
            groupIdx++;
        };

        const handleEntry = async (name: string, entryData: Uint8Array) => {
            if (name === 'meta.json') {
                meta = parseV3Meta(entryData.slice());
                if (!meta.streams) {
                    // monolithic archive: keep downloading and decode the
                    // whole buffer once it completes
                    monolithic = true;
                    return;
                }
                decoder = new V3Decoder(app, meta);
                groups = enumerateV3Groups(meta);
                // sh-deferred archives ship labels behind all geometry, so
                // geometry groups complete on the base texture set alone
                neededNames = meta.streams.sh_deferred ? [...GROUP_FILE_NAMES] : groupFileList(meta);
                // reveal set = persistent group plus the first temporal segment
                const firstSeg = groups.findIndex(g => g.segIndex >= 0);
                revealGroupIdx = firstSeg >= 0 ? firstSeg : groups.length - 1;
                return;
            }
            if (!meta) {
                throw new Error(`omg4 v3: unexpected entry ${name} before meta.json`);
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
                const group = groups.find(g => g.prefix === prefix);
                if (group) {
                    await decoder!.decodeGroupSH(group, entryData.slice());
                    callbacks.onShReady?.(group.range);
                }
                return;
            }
            const group = groups[groupIdx];
            if (!group || !name.startsWith(`${group.prefix}/`)) {
                return;     // stray entry (or all groups already decoded)
            }
            // copy out of the shared download buffer: it may be grown
            // (reallocated) while the group is still pending
            pending.set(name.slice(group.prefix!.length + 1), entryData.slice());
            if (pending.size === neededNames.length) {
                await groupComplete();
            }
        };

        try {
            for (;;) {
                // eslint-disable-next-line no-await-in-loop -- sequential network reads
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

                // release a buffering hold as soon as bandwidth allows —
                // don't wait for the next group to finish decoding
                if (playGate && revealed && data && gateOpen()) {
                    playGate = false;
                    callbacks.onReady(null, gatedLoadedThrough);
                }

                if (meta?.streams?.reveal_bytes && !revealed) {
                    const progress = Math.min(99, Math.trunc((received / meta.streams.reveal_bytes) * 100));
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
                    // eslint-disable-next-line no-await-in-loop -- entries decode in order
                    await handleEntry(entry.name, entry.data);
                }
            }

            const buffer = bytes.byteLength === received ? bytes.buffer : bytes.slice(0, received).buffer;

            if (monolithic) {
                const decoded = await loadOmg4V3(app, buffer,
                    p => callbacks.onProgress(Math.min(100, Math.round(70 + p * 0.3))));
                revealed = true;
                callbacks.onProgress(100);
                revealResolve(decoded);
                return buffer;
            }

            // flush a trailing partial group (shouldn't happen with a well-
            // formed archive, but don't leave decoded splats stranded)
            if (decoder && groupIdx < groups.length && pending.size === neededNames.length) {
                await groupComplete();
            }
            if (!revealed) {
                throw new Error('omg4 v3: stream ended before the reveal set was decoded');
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
    complete.catch(() => {});

    return { reveal, complete };
};

export { streamOmg4V3 };
