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
        let revealPending = false;      // reveal set decoded, awaiting buffer
        let progressWatermark = -1;
        let decodedThrough = 0;         // decoded content boundary (absolute)
        let downloadStart = 0;
        // trailing samples of (wall time, buffered content) for the fill-
        // rate estimate — a cumulative mean drags below the sustained rate
        // during the initial decode ramp and over-delays the reveal
        const fillSamples: { t: number, b: number }[] = [];

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
        const gateInfo = (): { bytesQ: number, fillQ: number } => {
            const gb = meta?.streams?.geometry_bytes;
            if (!gb) {
                return { bytesQ: 1, fillQ: 1 };
            }
            const duration = Math.max(0.1, (meta.time?.max ?? 0) - (meta.time?.min ?? 0));
            const elapsed = (performance.now() - downloadStart) / 1000;

            let bytesQ = 1;
            if (received < gb) {
                if (elapsed < 0.15) {
                    bytesQ = 0;
                } else {
                    const bw = received / elapsed;
                    const target = Math.max(1, gb - (bw * (duration - 0.3)) / 1.3);
                    bytesQ = Math.min(1, received / target);
                }
            }

            let fillQ = 0;
            if (!isFinite(decodedThrough)) {
                fillQ = 1;      // geometry fully decoded
            } else if (fillSamples.length > 1) {
                const buffered = decodedThrough - (meta.time?.min ?? 0);
                const now = performance.now();
                // oldest sample within the trailing window
                let ref = fillSamples[0];
                for (const sample of fillSamples) {
                    if (now - sample.t <= 1500) {
                        break;
                    }
                    ref = sample;
                }
                const span = (now - ref.t) / 1000;
                if (buffered > 0 && span >= 0.35) {
                    const f = Math.min(1, Math.max(0, buffered - ref.b) / span);
                    const required = Math.max(0.3, duration * (1 - f) * 1.25);
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
            decodeChain.catch(() => { });
        };

        const processGroup = async (idx: number, group: ReturnType<typeof enumerateV3Groups>[number], files: Map<string, Uint8Array>) => {
            await decoder!.decodeGroup(group, files);
            if (idx >= revealGroupIdx) {
                decodedThrough = loadedThroughAfter(idx);
                if (isFinite(decodedThrough)) {
                    fillSamples.push({ t: performance.now(), b: decodedThrough - (meta.time?.min ?? 0) });
                    if (fillSamples.length > 40) {
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
                return;     // stray entry (or all groups already decoded)
            }
            // copy out of the shared download buffer: it may be grown
            // (reallocated) while the group is still pending
            pending.set(name.slice(group.prefix!.length + 1), entryData.slice());
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
