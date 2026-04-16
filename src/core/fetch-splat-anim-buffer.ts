// Fetch a 4DGS animation file at the given URL with streaming progress notifications.
// The onProgress callback receives integer values in [0, 100].
// Returns the complete response as an ArrayBuffer.
const fetchSplatAnimBuffer = async (url: string, onProgress: (progress: number) => void): Promise<ArrayBuffer> => {
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(`Failed to fetch ${url}: ${response.status} ${response.statusText}`);
    }

    const contentLength = parseInt(response.headers.get('content-length') ?? '0', 10);
    let watermark = 0;
    let received = 0;

    const reader = response.body?.getReader();
    if (!reader) {
        throw new Error('Response body is not readable.');
    }

    // Use a single destination buffer to avoid holding both chunk arrays and a full concat copy.
    let bytes = new Uint8Array(contentLength > 0 ? contentLength : 1024 * 1024);

    const ensureCapacity = (needed: number) => {
        if (needed <= bytes.length) return;
        let nextSize = bytes.length;
        while (nextSize < needed) {
            nextSize *= 2;
        }
        const grown = new Uint8Array(nextSize);
        grown.set(bytes, 0);
        bytes = grown;
    };

    // Read all chunks while tracking progress, writing directly into the destination buffer.
    const pump = async (): Promise<void> => {
        const { done, value } = await reader.read();
        if (done) return;

        ensureCapacity(received + value.length);
        bytes.set(value, received);
        received += value.length;

        const progress = contentLength > 0 ? Math.min(100, Math.trunc(received / contentLength * 100)) : 0;
        if (progress > watermark) {
            watermark = progress;
            onProgress(watermark);
        }

        return pump();
    };
    await pump();

    if (watermark < 100) {
        onProgress(100);
    }

    if (received === bytes.length) {
        return bytes.buffer;
    }

    return bytes.slice(0, received).buffer;
};

export { fetchSplatAnimBuffer };
