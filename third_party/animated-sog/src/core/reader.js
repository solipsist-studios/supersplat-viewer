/**
 * Byte readers over an .asog source. Both read whole byte ranges front to back
 * (`read(offset, length)` with increasing offsets), so a file is decoded while it downloads
 * and never held whole in memory.
 */

/**
 * @typedef {object} ByteReader
 * @property {(offset: number, length: number) => Promise<Uint8Array>} read - a new array the caller owns
 * @property {() => void} cancel - stops the download
 */

/**
 * @typedef {string | URL | Response | Blob | ReadableStream<Uint8Array> | ArrayBuffer | ArrayBufferView} AsogSource
 */

/**
 * Reads a ReadableStream (a fetch body) front to back, skipping what lies between ranges.
 * @implements {ByteReader}
 */
export class StreamReader {
    /**
     * @param {ReadableStream<Uint8Array>} stream
     * @param {number} total - byte size for progress, 0 if unknown
     * @param {(progress: number) => void} [onProgress]
     */
    constructor(stream, total, onProgress) {
        this._reader = stream.getReader();
        this._chunk = new Uint8Array(0);
        this._at = 0;        // read position in _chunk
        this._received = 0;
        this._pos = 0;       // file position of the next byte
        this._total = total;
        this._onProgress = onProgress;
    }

    async _next() {
        const { done, value } = await this._reader.read();
        if (done) throw new Error("asog: the file ends early");
        this._chunk = value;
        this._at = 0;
        this._received += value.length;
        if (this._total) this._onProgress?.(Math.min(1, this._received / this._total));
    }

    /**
     * @param {number} offset
     * @param {number} length
     * @returns {Promise<Uint8Array>}
     */
    async read(offset, length) {
        if (offset < this._pos) throw new Error("asog: data out of file order");
        while (this._pos < offset) {
            if (this._at === this._chunk.length) await this._next();
            const k = Math.min(offset - this._pos, this._chunk.length - this._at);
            this._at += k;
            this._pos += k;
        }
        const out = new Uint8Array(length);
        for (let o = 0; o < length;) {
            if (this._at === this._chunk.length) await this._next();
            const k = Math.min(length - o, this._chunk.length - this._at);
            out.set(this._chunk.subarray(this._at, this._at + k), o);
            this._at += k;
            this._pos += k;
            o += k;
        }
        return out;
    }

    cancel() {
        this._reader.cancel().catch(() => {});
    }
}

/**
 * Reads a file that is already in memory. Returns copies, so the decoder may free what it reads.
 * @implements {ByteReader}
 */
export class BufferReader {
    /**
     * @param {Uint8Array} bytes
     * @param {(progress: number) => void} [onProgress]
     */
    constructor(bytes, onProgress) {
        this._bytes = bytes;
        this._onProgress = onProgress;
    }

    /**
     * @param {number} offset
     * @param {number} length
     * @returns {Promise<Uint8Array>}
     */
    async read(offset, length) {
        if (offset + length > this._bytes.length) throw new Error("asog: the file ends early");
        this._onProgress?.((offset + length) / this._bytes.length);
        return this._bytes.slice(offset, offset + length);
    }

    cancel() {}
}

/**
 * Opens a reader on a URL, a fetch Response, a Blob, a ReadableStream or bytes in memory.
 * @param {AsogSource} source
 * @param {{ onProgress?: (progress: number) => void, fetchOptions?: RequestInit }} [options]
 * @returns {Promise<ByteReader>}
 */
export async function openReader(source, { onProgress, fetchOptions } = {}) {
    if (typeof source === "string" || source instanceof URL) {
        const resp = await fetch(source, fetchOptions);
        if (!resp.ok) throw new Error(`asog: download failed, HTTP ${resp.status} ${source}`);
        source = resp;
    }
    if (source instanceof ArrayBuffer) return new BufferReader(new Uint8Array(source), onProgress);
    if (ArrayBuffer.isView(source)) {
        return new BufferReader(new Uint8Array(source.buffer, source.byteOffset, source.byteLength), onProgress);
    }
    if (typeof Response !== "undefined" && source instanceof Response) {
        if (!source.body) throw new Error("asog: the response has no body");
        // fetch decodes a content-encoded body, so Content-Length is then not its size
        const total = source.headers.get("content-encoding") ? 0 : Number(source.headers.get("content-length")) || 0;
        return new StreamReader(source.body, total, onProgress);
    }
    if (typeof Blob !== "undefined" && source instanceof Blob) return new StreamReader(source.stream(), source.size, onProgress);
    if (source && typeof (/** @type {any} */ (source).getReader) === "function") {
        return new StreamReader(/** @type {ReadableStream<Uint8Array>} */ (source), 0, onProgress);
    }
    throw new TypeError("asog: unsupported source (expected a URL, Response, Blob, ReadableStream or bytes)");
}
