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

    // Collect all chunks while tracking progress, avoiding await-in-loop.
    const chunks: Uint8Array[] = [];
    const reader = response.body.getReader();
    const pump = async (): Promise<void> => {
        const { done, value } = await reader.read();
        if (done) return;
        chunks.push(value);
        received += value.length;
        const progress = contentLength > 0 ? Math.min(100, Math.trunc(received / contentLength * 100)) : 0;
        if (progress > watermark) {
            watermark = progress;
            onProgress(watermark);
        }
        return pump();
    };
    await pump();

    // Concatenate chunks into a single ArrayBuffer.
    const totalBytes = chunks.reduce((s, c) => s + c.length, 0);
    const buffer = new ArrayBuffer(totalBytes);
    const view = new Uint8Array(buffer);
    let offset = 0;
    for (const chunk of chunks) {
        view.set(chunk, offset);
        offset += chunk.length;
    }

    return buffer;
};

export { fetchSplatAnimBuffer };
