/**
 * IEEE float16 conversion for the RGBA16F motion texture.
 */

export const HALF_MAX = 65504;

const f64 = new Float64Array(1);
const u32 = new Uint32Array(f64.buffer); // little-endian: [1] is the high word

/** x rounded to the nearest integer, ties to even */
function roundEven(x) {
    const f = Math.floor(x), d = x - f;
    return d > 0.5 || (d === 0.5 && f % 2 === 1) ? f + 1 : f;
}

/**
 * number -> float16 bits, rounded once (no float32 step) to nearest even. Finite values beyond the
 * float16 range are clamped to +-65504 (a texture sampler would turn infinity into NaN centres).
 * @param {number} value
 * @returns {number}
 */
export function floatToHalf(value) {
    if (Number.isNaN(value)) return 0x7e00;
    const sign = value < 0 || Object.is(value, -0) ? 0x8000 : 0;
    const a = Math.abs(value);
    if (a === Infinity) return sign | 0x7c00;
    if (a >= 65520) return sign | 0x7bff;
    if (a < 2 ** -14) return sign | roundEven(a * 2 ** 24); // subnormal (1024 is the smallest normal's bits)
    // normal: the double's exponent and its top 10 mantissa bits, rounded on the 42 bits below
    f64[0] = a;
    const hi = u32[1], lo = u32[0];
    let h = ((((hi >>> 20) & 0x7ff) - 1023 + 15) << 10) | ((hi >>> 10) & 0x3ff);
    const rest = hi & 0x3ff;
    if (rest > 0x200 || (rest === 0x200 && (lo !== 0 || (h & 1)))) h++; // a carry runs into the exponent, as it should
    return sign | Math.min(h, 0x7bff);
}

/**
 * float16 bits -> float, for every 16-bit pattern. Self-contained (it is also put into the sort worker).
 * @returns {Float32Array} 65536 entries
 */
export function halfTable() {
    const half = new Float32Array(65536);
    for (let h = 0; h < 65536; h++) {
        const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
        half[h] = e === 0 ? s * m * 5.960464477539063e-8
            : e === 31 ? (m ? NaN : s * Infinity)
            : s * Math.pow(2, e - 15) * (1 + m / 1024);
    }
    return half;
}

/** @type {Float32Array | null} */
let table = null;

/**
 * The halfTable(), built once and shared.
 * @returns {Float32Array}
 */
export function getHalfTable() {
    table ??= halfTable();
    return table;
}

/**
 * float16 bits -> float.
 * @param {number} h
 * @returns {number}
 */
export function halfToFloat(h) {
    return getHalfTable()[h];
}
