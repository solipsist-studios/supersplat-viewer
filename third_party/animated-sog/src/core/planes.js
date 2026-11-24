/**
 * Integer coding of a stream (format spec 3.1): byte planes, zigzag, delta.
 */

/**
 * Decodes one channel of an inflated stream into `out[0 .. count)`.
 * Byte `p` of channel `c` of row `i` is at `raw[(p * channels + c) * count + i]`; then zigzag (if set),
 * then the delta sum along the rows (if set). Exact for values below 2^53.
 * @param {import("./container.js").AsogStreamDesc} desc
 * @param {Uint8Array} raw - inflated, `count * channels * planes` bytes
 * @param {number} ch - channel index
 * @param {Float64Array | number[]} out
 */
export function decodeChannel(desc, raw, ch, out) {
    const c = desc.channels, count = desc.count, planes = desc.planes;
    const zigzag = desc.zigzag, delta = desc.delta;
    let prev = 0;
    for (let i = 0; i < count; i++) {
        let v = 0;
        for (let p = planes - 1; p >= 0; p--) v = v * 256 + raw[(p * c + ch) * count + i];
        if (zigzag) v = v % 2 ? -(v + 1) / 2 : v / 2;
        if (delta) v = prev += v;
        out[i] = v;
    }
}

/**
 * Decodes a whole stream, row-major: `x[i * channels + c]`.
 * @param {import("./container.js").AsogStreamDesc} desc
 * @param {Uint8Array} raw
 * @returns {Float64Array}
 */
export function decodeStream(desc, raw) {
    const { count, channels } = desc;
    const out = new Float64Array(count * channels);
    const col = new Float64Array(count);
    for (let ch = 0; ch < channels; ch++) {
        decodeChannel(desc, raw, ch, col);
        for (let i = 0; i < count; i++) out[i * channels + ch] = col[i];
    }
    return out;
}
