import { parseOmg4V2, writeOmg4V2StandardHeader, V2_HEADER_SIZE } from '../parsers/omg4';
import type { Omg4V2Data, Omg4V2Header } from '../parsers/omg4';

// Progressive loader for streamable (tiled) OMG4 v2 files.
//
// The file's tiles are self-contained (all fields for a contiguous group of
// splats), so a single sequential fetch yields renderable splats
// continuously. Bytes are de-tiled on the fly into a full-size STANDARD
// layout buffer: the returned Omg4V2Data's zero-copy views (and therefore
// the engine's texture-update methods) always read coherent data for every
// splat below `readySplats`, and the completed buffer doubles as a regular
// v2 file for the IndexedDB cache.
//
// Splats not yet received are prefilled as invisible (opacity logit -20 →
// alpha ~2e-9, degenerate scale, identity rotation, t_sigma 1 to keep the
// temporal shader math finite), so the resource can be created and rendered
// before the download completes.

type StreamCallbacks = {
    /**
     * Download progress in [0, 100], measured against the first-batch
     * prefetch (the reveal point), not the whole file — the bar reads full
     * when the scene appears while the remainder streams in behind it.
     */
    onProgress: (progress: number) => void;
    /**
     * Splats [0, readySplats) are complete in the buffer. Throttled (time +
     * growth), with a final unthrottled call at completion.
     */
    onReady: (readySplats: number, done: boolean) => void;
};

type Omg4V2Stream = {
    /** Views over the (still-filling) standard-layout buffer. */
    data: Omg4V2Data;
    /** Resolves once enough splats arrived to start rendering. */
    firstBatch: Promise<void>;
    /** Resolves with the completed standard-layout buffer. */
    complete: Promise<ArrayBuffer>;
};

// Start rendering once this share of splats arrived (at least one tile).
const FIRST_BATCH_FRACTION = 0.1;

// GPU re-sync throttle: at least this much time AND splat growth between
// onReady calls (except the final one).
const SYNC_MIN_INTERVAL_MS = 500;
const SYNC_MIN_GROWTH_FRACTION = 0.05;

const streamOmg4V2 = (url: string, header: Omg4V2Header, callbacks: StreamCallbacks): Omg4V2Stream => {
    const { numSplats: N, numFields, tileSize } = header;

    // Full-size standard-layout destination buffer.
    const dest = new ArrayBuffer(V2_HEADER_SIZE + numFields * N * 4);
    writeOmg4V2StandardHeader(dest, header);

    // Prefill unready splats as invisible/inert.
    const fieldView = (i: number) => new Float32Array(dest, V2_HEADER_SIZE + i * N * 4, N);
    fieldView(3).fill(1); // rot_0 (w): identity quaternion
    fieldView(7).fill(-30); // scale_0..2 (log): exp(-30) → sub-pixel
    fieldView(8).fill(-30);
    fieldView(9).fill(-30);
    fieldView(10).fill(-20); // opacity (logit): sigmoid(-20) ≈ 2e-9
    fieldView(18).fill(1); // t_sigma: keep shader math finite

    const data = parseOmg4V2(dest);

    const destBytes = new Uint8Array(dest);
    const tileFloats = (splats: number) => splats * numFields;

    let firstBatchResolve: () => void;
    let firstBatchReject: (err: Error) => void;
    const firstBatch = new Promise<void>((resolve, reject) => {
        firstBatchResolve = resolve;
        firstBatchReject = reject;
    });
    const firstBatchThreshold = Math.min(N, Math.max(tileSize, Math.ceil(N * FIRST_BATCH_FRACTION)));

    // Bytes that must arrive before the first batch resolves: the header plus
    // whole tiles up to the threshold (firstBatch only resolves on tile
    // boundaries). This is the denominator for onProgress, so the bar fills
    // to 100% exactly when the scene becomes renderable.
    const prefetchSplats = Math.min(N, Math.ceil(firstBatchThreshold / tileSize) * tileSize);
    const prefetchBytes = header.headerSize + prefetchSplats * numFields * 4;

    const complete = (async (): Promise<ArrayBuffer> => {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to fetch ${url}: ${response.status} ${response.statusText}`);
        }
        const reader = response.body?.getReader();
        if (!reader) {
            throw new Error('Response body is not readable.');
        }

        // Staging buffer for the largest possible tile.
        const staging = new Uint8Array(tileFloats(tileSize) * 4);

        let received = 0; // total body bytes consumed (incl. header)
        let stagingFill = 0; // bytes accumulated for the current tile
        let readySplats = 0;
        let firstBatchDone = false;
        let lastSyncAt = 0;
        let lastSyncSplats = 0;
        let progressWatermark = 0;

        // Scatter one complete tile from staging into the SoA layout.
        const commitTile = () => {
            const tileStart = readySplats;
            const tileSplats = Math.min(tileSize, N - tileStart);
            const bytesPerField = tileSplats * 4;
            for (let f = 0; f < numFields; f++) {
                destBytes.set(
                    staging.subarray(f * bytesPerField, (f + 1) * bytesPerField),
                    V2_HEADER_SIZE + (f * N + tileStart) * 4
                );
            }
            readySplats += tileSplats;
            stagingFill = 0;
        };

        const notify = (done: boolean) => {
            if (!firstBatchDone && readySplats >= firstBatchThreshold) {
                firstBatchDone = true;
                lastSyncAt = performance.now();
                lastSyncSplats = readySplats;
                callbacks.onReady(readySplats, done);
                firstBatchResolve();
                return;
            }
            if (done) {
                callbacks.onReady(readySplats, true);
                return;
            }
            if (!firstBatchDone) {
                return;
            }
            const now = performance.now();
            if (
                now - lastSyncAt >= SYNC_MIN_INTERVAL_MS &&
                readySplats - lastSyncSplats >= N * SYNC_MIN_GROWTH_FRACTION
            ) {
                lastSyncAt = now;
                lastSyncSplats = readySplats;
                callbacks.onReady(readySplats, false);
            }
        };

        for (;;) {
            // eslint-disable-next-line no-await-in-loop
            const { done, value } = await reader.read();
            if (done) {
                break;
            }

            let offset = 0;

            // Skip the (tiled) file header.
            if (received < header.headerSize) {
                const skip = Math.min(header.headerSize - received, value.length);
                offset += skip;
                received += skip;
            }

            // eslint-disable-next-line no-unmodified-loop-condition -- readySplats advances via commitTile()
            while (offset < value.length && readySplats < N) {
                const tileSplats = Math.min(tileSize, N - readySplats);
                const tileBytes = tileFloats(tileSplats) * 4;
                const take = Math.min(tileBytes - stagingFill, value.length - offset);
                staging.set(value.subarray(offset, offset + take), stagingFill);
                stagingFill += take;
                offset += take;
                received += take;
                if (stagingFill === tileBytes) {
                    commitTile();
                }
            }

            const progress = Math.min(100, Math.trunc((received / prefetchBytes) * 100));
            if (progress > progressWatermark) {
                progressWatermark = progress;
                callbacks.onProgress(progress);
            }
            notify(false);
        }

        if (readySplats < N) {
            throw new Error(`Truncated .omg4 stream: ${readySplats}/${N} splats received`);
        }

        callbacks.onProgress(100);
        notify(true);
        return dest;
    })();

    // A failure before the first batch means nothing is renderable.
    complete.catch((err: Error) => firstBatchReject(err));

    return { data, firstBatch, complete };
};

export { streamOmg4V2 };
export type { Omg4V2Stream };
