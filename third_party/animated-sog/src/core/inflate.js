/**
 * zlib (RFC 1950) inflate of a stream whose inflated size is known.
 * Browsers: DecompressionStream("deflate") straight into one preallocated buffer.
 * Node: node:zlib.
 */

const isNode = typeof process !== "undefined" && !!process.versions?.node && typeof window === "undefined";
const NODE_ZLIB = "node:zlib"; // in a variable, so bundlers for the browser leave the import alone
/** @type {Promise<any> | null} */
let nodeZlib = null;

/**
 * Inflates `bytes` to exactly `size` bytes; throws if the stream inflates to another size.
 * `bytes` is not modified or freed.
 * @param {Uint8Array} bytes - zlib-compressed
 * @param {number} size - inflated size
 * @returns {Promise<Uint8Array>} a new array of `size` bytes that owns its whole buffer
 */
export async function inflate(bytes, size) {
    return isNode ? inflateNode(bytes, size) : inflateBrowser(bytes, size);
}

async function inflateBrowser(bytes, size) {
    const out = new Uint8Array(size);
    // the Blob holds its own copy of the bytes
    const reader = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("deflate")).getReader();
    let o = 0;
    for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        if (o + value.length > size) {
            reader.cancel().catch(() => {});
            throw new Error(`asog: a stream inflates to more than its declared ${size} bytes`);
        }
        out.set(value, o);
        o += value.length;
    }
    if (o !== size) throw new Error(`asog: a stream inflates to ${o} bytes, not its declared ${size}`);
    return out;
}

async function inflateNode(bytes, size) {
    nodeZlib ??= import(/* @vite-ignore */ NODE_ZLIB);
    const zlib = await nodeZlib;
    /** @type {Uint8Array} */
    const out = await new Promise((resolve, reject) => {
        zlib.inflate(bytes, { maxOutputLength: Math.max(size, 1) }, (err, result) => (err ? reject(err) : resolve(result)));
    });
    if (out.length !== size) throw new Error(`asog: a stream inflates to ${out.length} bytes, not its declared ${size}`);
    // a small Buffer can be a slice of Node's shared pool: copy it so the caller may free the result
    return out.byteOffset === 0 && out.buffer.byteLength === size ? new Uint8Array(out.buffer) : new Uint8Array(out);
}
