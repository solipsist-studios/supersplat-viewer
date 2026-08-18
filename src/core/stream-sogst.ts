import type { AppBase } from 'playcanvas';

import type { SogstMeta } from '../parsers/sogst';

import { loadSogst, parseSogstMeta } from './load-sogst';
import type { SogstData } from './sogst-data';
import { SogstDecoder, enumerateSogstGroups, groupFileList, groupBaseNames } from './sogst-decoder';
import { ZIP_FLAG_DATA_DESCRIPTOR, ZIP_LFH, ZIP_LOCAL_MAGIC } from './zip';

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

// Growth buffer for a response that has no Content-Length. The buffer doubles
// from this size, so the value sets only how many times a small archive
// reallocates.
const INITIAL_BUFFER_BYTES = 1 << 20;

// -- buffer-ahead gate tuning --------------------------------------------
// These constants are the margins in the two readiness tests below. They are
// tuning values, not format constants. We measured them against the
// reference clips on a throttled connection. They were the smallest values
// that stopped the first playback pass from hitching at segment boundaries.
//
// Smaller margins show the scene sooner, but they also risk starving the
// playhead.

// Lowest clip duration the gate will use. Without it, a still image has
// duration 0 and the required-buffer term falls to zero.
const MIN_GATE_DURATION_S = 0.1;

// Connection setup dominates the bandwidth measured in the first fraction of
// a second. The gate therefore waits for this much time before it uses the
// estimate.
const BANDWIDTH_WARMUP_S = 0.15;

// Headroom the gate subtracts from the clip duration before it tests whether
// the remaining bytes fit. The last segment must arrive before playback
// reaches it, not at the same moment.
const DOWNLOAD_HEADROOM_S = 0.3;

// Safety factor on the bandwidth estimate. The measured throughput must
// exceed what the clip needs by this factor before the byte side of the gate
// opens.
const DOWNLOAD_MARGIN = 1.3;

// Trailing window for the fill-rate estimate. It is long enough to average
// one segment's decode, and short enough to follow a connection that gets
// slower.
const FILL_WINDOW_MS = 1500;

// Minimum wall-clock time between the oldest and the newest fill sample. The
// fill rate has no meaning below it.
const MIN_FILL_SPAN_S = 0.35;

// Lowest required buffer. Content that decodes faster than it plays still
// buffers this much before the viewer shows it.
const MIN_BUFFER_S = 0.3;

// Safety factor on the buffering inequality (buffered >= duration * (1 - f)).
const FILL_MARGIN = 1.25;

// Limit on retained fill samples. It holds FILL_WINDOW_MS of history at the
// fastest segment rate we measured, plus some margin.
const MAX_FILL_SAMPLES = 40;

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
            // This parser walks entries by local-header size, which works
            // only because the format forbids data descriptors
            // (general-purpose bit 3).
            //
            // A writer that emits them writes zero local sizes. `total` would
            // then equal the header length, and the next read would start
            // inside the payload and treat it as the next entry. The result
            // is garbage names and no error. Throw instead: this is a
            // malformed archive, not a stream underrun.
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
        // Trailing samples of wall time and buffered content, for the
        // fill-rate estimate. A cumulative mean falls below the sustained
        // rate during the initial decode ramp, and it then delays the reveal
        // too long.
        const fillSamples: { t: number; b: number }[] = [];

        // Absolute clip time the viewer can play once groups [0, idx] are
        // decoded. It runs up to the start of the next segment's coverage,
        // which is the first segment the decoder has not finished.
        const loadedThroughAfter = (idx: number): number => {
            const next = groups[idx + 1];
            return next ? meta.segments.list[next.segIndex].t0 : Infinity;
        };

        // Buffer-ahead readiness. The viewer shows the scene and starts the
        // playhead only when the whole pipeline is ahead of playback. Without
        // this test the playhead starves at segment boundaries through the
        // first pass, which the viewer sees as regular hitching.
        //
        // The test has two sides, and each one carries a margin:
        //
        //  bytesQ: at the measured bandwidth, the remaining geometry bytes
        //          download within the clip duration. This covers the
        //          network.
        //  fillQ:  the standard buffering test on the measured fill rate f of
        //          decoded content-time, which is
        //          buffered >= duration * (1 - f) * margin. The fill rate
        //          covers network, decode and sync together, so this side
        //          protects a slow CPU, where decode and not download is the
        //          bottleneck.
        //
        // The two quotients also drive the last part of the progress bar, so
        // the bar reaches 100 at the moment playback can start cleanly.
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
                fillQ = 1; // The geometry is fully decoded.
            } else if (fillSamples.length > 1) {
                const buffered = decodedThrough - (meta.time?.min ?? 0);
                const now = performance.now();
                // Oldest sample inside the trailing window.
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

        // Decode runs on its own promise chain, so the network loop never
        // waits for it. Running the download behind the decode wastes
        // bandwidth. It also lowers the measured fill rate, and therefore the
        // buffering gate, well below what the connection supports.
        let decodeChain: Promise<void> = Promise.resolve();
        const enqueueDecode = (task: () => Promise<void>) => {
            decodeChain = decodeChain.then(task);
            // The stream tail awaits this chain and receives the rejection
            // there. This handler only silences the unhandled-rejection
            // warning in the meantime.
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
                // Still buffering. Later groups keep decoding into the
                // shared arrays, and the resource created at the reveal
                // reads them. Show the scene as soon as the pipeline is
                // ahead of playback.
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
            // Ignore an entry that backs a group member this build does not
            // know. Do not treat it as the group's data. Spec section 3.1
            // requires this: additive groups ship under version 1, and a
            // player must degrade the way it does for shN and accel.
            //
            // The filter also keeps the size test below correct. That test
            // counts entries. An unrecognised entry would raise the count to
            // neededNames.length while a required name was still missing, and
            // the decode would start one file short. The decode then throws
            // "missing from archive". Whether this happened depended on the
            // order in which the encoder wrote the entries.
            if (!neededNames.includes(bare)) {
                return;
            }
            // Copy out of the shared download buffer. The loop can
            // reallocate that buffer while the group is still pending.
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
                    // The download maps to 0-70. Buffer readiness maps to
                    // 70-99, and it uses the lower of the bandwidth and fill
                    // quotients. The bar therefore reaches 100 at the moment
                    // the scene is ready to play.
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
                    // The whole-file download maps to 0-70. The decode maps
                    // to 70-100.
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

            // Flush a trailing partial group. A well-formed archive does not
            // produce one, but do not leave decoded splats unused.
            if (decoder && groupIdx < groups.length && pending.size === neededNames.length) {
                const idx = groupIdx;
                const files = pending;
                pending = new Map();
                groupIdx++;
                enqueueDecode(() => processGroup(idx, groups[idx], files));
            }
            // Drain all queued decodes. This also raises any decode error.
            await decodeChain;
            if (revealPending && !revealed) {
                // The download finished, so there is nothing left to buffer
                // against.
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

    // The reveal consumer handles errors through the complete promise.
    complete.catch(() => {
        /* caller handles it on the returned promise */
    });

    return { reveal, complete };
};

export { streamSogst };
