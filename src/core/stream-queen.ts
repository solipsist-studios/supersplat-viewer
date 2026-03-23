import { QueenData, computeFrame0MinBytes } from '../parsers/queen';

// Fetch a .queen file at `url` and return a QueenData instance as soon as the base frame (frame 0)
// is available, allowing playback to start immediately.  The remaining residual frames are pumped
// into the QueenData in the background via appendChunk(); availableFrames grows over time.
// The onProgress callback receives integer values in [0, 100].
const streamQueenData = async (url: string, onProgress: (progress: number) => void): Promise<QueenData> => {
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(`Failed to fetch ${url}: ${response.status} ${response.statusText}`);
    }

    const contentLength = parseInt(response.headers.get('content-length') ?? '0', 10);
    const reader = response.body.getReader();
    let bytesReceived = 0;
    let watermark = 0;

    const updateProgress = (size: number) => {
        bytesReceived += size;
        if (contentLength > 0) {
            const p = Math.min(100, Math.trunc(bytesReceived / contentLength * 100));
            if (p > watermark) {
                watermark = p;
                onProgress(watermark);
            }
        }
    };

    // ── Phase 1: accumulate chunks until we have enough bytes for base frame 0 ──────────────────
    // Keep a single growing buffer to avoid repeated concatenation.
    let accumBuf = new Uint8Array(Math.max(65536, contentLength > 0 ? contentLength : 65536));
    let accumLen = 0;

    const pushChunk = (chunk: Uint8Array) => {
        if (accumLen + chunk.length > accumBuf.length) {
            let newSize = accumBuf.length * 2;
            while (newSize < accumLen + chunk.length) newSize *= 2;
            const nb = new Uint8Array(newSize);
            nb.set(accumBuf.subarray(0, accumLen));
            accumBuf = nb;
        }
        accumBuf.set(chunk, accumLen);
        accumLen += chunk.length;
    };

    // Recursive pump avoids the no-await-in-loop lint rule while still handling back-pressure
    // correctly: we only issue the next read() after the previous one resolves.
    let frame0Ready = false;

    const pumpPhase1 = async (): Promise<void> => {
        const { done, value } = await reader.read();
        if (done) return;
        updateProgress(value.length);
        pushChunk(value);

        const needed = computeFrame0MinBytes(accumBuf.buffer, accumLen);
        if (needed > 0 && accumLen >= needed) {
            frame0Ready = true;
            return;
        }
        return pumpPhase1();
    };

    await pumpPhase1();

    if (!frame0Ready) {
        throw new Error('Stream ended before QUEEN base frame was fully received');
    }

    // Build the QueenData from the bytes accumulated so far.
    // slice() gives an exact-size copy so the ArrayBuffer byteLength matches accumLen.
    const queenData = new QueenData(accumBuf.slice(0, accumLen).buffer);

    // ── Phase 2: pump the remaining residual frames into queenData in the background ─────────────
    // This is intentionally "fire and forget" — the animation loop consumes availableFrames
    // and the buffer grows as more HTTP chunks arrive.
    const pumpPhase2 = async (): Promise<void> => {
        const { done, value } = await reader.read();
        if (done) {
            onProgress(100);
            return;
        }
        updateProgress(value.length);
        queenData.appendChunk(value);
        return pumpPhase2();
    };

    pumpPhase2().catch(console.error);

    return queenData;
};

export { streamQueenData };
