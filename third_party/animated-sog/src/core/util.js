/**
 * Frees typed arrays' memory now rather than at the next GC (a load allocates hundreds of MB for a
 * large clip, none of which is collected before the load ends). Only arrays that own their whole
 * buffer are freed; without ArrayBuffer.prototype.transfer (Safari < 17.4) this does nothing.
 * @param {...(ArrayBufferView | null | undefined)} arrays
 */
export function free(...arrays) {
    for (const a of arrays) {
        const buffer = /** @type {any} */ (a?.buffer);
        if (buffer?.transfer && !buffer.detached && a.byteOffset === 0 && a.byteLength === buffer.byteLength) buffer.transfer(0);
    }
}

/**
 * A pacer for long loops: `await pace()` yields to the event loop (a macrotask, so a frame can
 * render) once `budgetMs` have passed since the last yield, and returns at once otherwise.
 * @param {number} budgetMs
 * @returns {() => Promise<void>}
 */
export function createPacer(budgetMs) {
    let last = performance.now();
    return async () => {
        if (performance.now() - last < budgetMs) return;
        await new Promise((resolve) => setTimeout(resolve, 0));
        last = performance.now();
    };
}
