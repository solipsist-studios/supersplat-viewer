// Cache key for a full-file payload: URL plus a cheap validator (ETag /
// Last-Modified / size from a HEAD request), so edited files re-download
// while unchanged ones load from IndexedDB. Browsers won't keep responses
// this large (hundreds of MB) in the regular HTTP cache.
//
// The suffix carries the *container* version, not a cache-schema version:
// entries written against a different accepted `meta.version` may hold a
// manifest this build rejects, and a decode-then-throw is a worse failure
// than a re-download. Bump it whenever the accepted `meta.version` changes.
const fullFileKeyPrefix = (url: string) => `${new URL(url, location.href).toString()}?__sogst_v1_full=`;

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

export { fullFileCacheKey, fullFileKeyPrefix };
