// Cache key for a full-file payload. It combines the URL with a cheap
// validator: the ETag, Last-Modified or size from a HEAD request. An edited
// file therefore downloads again, and an unchanged file loads from
// IndexedDB.
//
// This is necessary because a browser does not keep a response of several
// hundred MB in its regular HTTP cache.
//
// The suffix carries the *container* version, not a cache-schema version. An
// entry written against a different accepted `meta.version` can hold a
// manifest this build rejects. A decode that then throws is a worse failure
// than a second download. Raise the suffix whenever the accepted
// `meta.version` changes.
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
