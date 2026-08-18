import { idbDeleteByPrefix, idbGetBuffer, idbSetBuffer } from './omg4-cache';

const OMG4_DEBUG_LOG = true;

// Cache key for a full-file payload: URL plus a cheap validator (ETag /
// Last-Modified / size from a HEAD request), so edited files re-download
// while unchanged ones load from IndexedDB. Browsers won't keep responses
// this large (hundreds of MB) in the regular HTTP cache.
const fullFileKeyPrefix = (url: string) => `${new URL(url, location.href).toString()}?__omg4_full=`;

const fullFileCacheKey = async (url: string): Promise<string> => {
    let validator = '';
    try {
        const head = await fetch(url, { method: 'HEAD' });
        if (head.ok) {
            validator =
                head.headers.get('etag') ??
                head.headers.get('last-modified') ??
                head.headers.get('content-length') ??
                '';
        }
    } catch {
        // offline or HEAD unsupported — fall through to the bare key
    }
    return fullFileKeyPrefix(url) + validator;
};

const fetchSplatAnimBufferNetwork = async (
    url: string,
    onProgress: (progress: number) => void
): Promise<ArrayBuffer> => {
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

        const progress = contentLength > 0 ? Math.min(100, Math.trunc((received / contentLength) * 100)) : 0;
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

// Fetch a 4DGS animation file at the given URL with streaming progress notifications.
// The onProgress callback receives integer values in [0, 100].
// Returns the complete response as an ArrayBuffer, served from a durable
// IndexedDB cache when the file is unchanged.
const fetchSplatAnimBuffer = async (url: string, onProgress: (progress: number) => void): Promise<ArrayBuffer> => {
    const cacheKey = await fullFileCacheKey(url);
    const cached = await idbGetBuffer(cacheKey);
    if (cached) {
        if (OMG4_DEBUG_LOG) console.debug('OMG4 full-file cache hit (idb)', cacheKey);
        onProgress(100);
        return cached;
    }

    const buffer = await fetchSplatAnimBufferNetwork(url, onProgress);

    // Store for next time and drop stale copies of this URL (older
    // validators). Deliberately NOT awaited: the scene must never be held
    // hostage to (or lost with) a slow or failing cache write.
    idbSetBuffer(cacheKey, buffer)
        .then(() => idbDeleteByPrefix(fullFileKeyPrefix(url), cacheKey))
        .catch((err) => {
            if (OMG4_DEBUG_LOG) console.debug('OMG4 full-file cache write failed', err);
        });

    return buffer;
};

export { fetchSplatAnimBuffer, fullFileCacheKey, fullFileKeyPrefix };
