/**
 * Per-splat access to decoded data (AsogData from decodeAsog), for tools, tests and CPU-side work.
 * Values are read back from the GPU layouts, so they are what a renderer sees (the motion as float16).
 */
import { SH_C0, dequantPosition, dequantQuat } from "./dequant.js";
import { getHalfTable } from "./half.js";
import { clipWeight, createMotion, evaluateSplat, quatMultiply, readMotion } from "./motion.js";

/**
 * @typedef {object} Splat
 * @property {number} index
 * @property {number[]} position - model space (TRiGS: y down)
 * @property {number[]} rotation - unit quaternion x y z w
 * @property {number[]} scale - linear, per axis
 * @property {number[]} color - rgb from the degree-0 SH (0.5 + C0 · c), not clamped
 * @property {number} opacity - 0..1
 * @property {Float32Array | null} sh - degree 1-3 SH, coefficient-major then r g b (coeffs · 3 values)
 * @property {import("./motion.js").SplatMotion | null} motion - needs motion.A, motion.B and centers kept
 */

/**
 * Reads splat i. Needs the SOG arrays (kept by decodeAsog's default `keep`).
 * @param {import("./decode.js").AsogData} data
 * @param {number} i
 * @returns {Splat}
 */
export function readSplat(data, i) {
    const { sog, manifest } = data;
    const t = i * 4;
    const { mins, maxs } = manifest.means;
    const position = [0, 1, 2].map((c) => dequantPosition(sog.means_l[t + c] + (sog.means_u[t + c] << 8), mins[c], maxs[c]));
    const rotation = /** @type {number[]} */ (dequantQuat(sog.quats[t], sog.quats[t + 1], sog.quats[t + 2], sog.quats[t + 3] - 252));
    const scale = [0, 1, 2].map((c) => Math.exp(manifest.scales.codebook[sog.scales[t + c]]));
    const color = [0, 1, 2].map((c) => 0.5 + SH_C0 * manifest.sh0.codebook[sog.sh0[t + c]]);
    const opacity = sog.sh0[t + 3] / 255;
    let sh = null;
    if (sog.sh_labels && sog.sh_centroids) {
        const { coeffs, codebook } = manifest.shN;
        const label = sog.sh_labels[t] | (sog.sh_labels[t + 1] << 8);
        const base = (label >> 6) * sog.shWidth + (label & 63) * coeffs;
        sh = new Float32Array(coeffs * 3);
        for (let j = 0; j < coeffs; j++) {
            for (let c = 0; c < 3; c++) sh[j * 3 + c] = codebook[sog.sh_centroids[(base + j) * 4 + c]];
        }
    }
    return { index: i, position, rotation, scale, color, opacity, sh, motion: readSplatMotion(data, i) };
}

/**
 * Splat i's motion record, or null when the motion arrays or centres were not kept.
 * @param {import("./decode.js").AsogData} data
 * @param {number} i
 * @param {import("./motion.js").SplatMotion} [m] - a record to fill
 * @returns {import("./motion.js").SplatMotion | null}
 */
export function readSplatMotion(data, i, m = createMotion()) {
    const { A, B } = data.motion;
    if (!A || !B || !data.centers) return null;
    return readMotion(A, new Float32Array(A.buffer), B, getHalfTable(), data.centers, i, m);
}

/**
 * Splat i at global time t (seconds, already looped into the timeline).
 * @param {import("./decode.js").AsogData} data
 * @param {number} i
 * @param {number} t
 * @returns {{ center: number[], rotation: number[], weight: number }} rotation: the splat's rotation
 *   at t (x y z w); weight: opacity factor (clip × temporal)
 */
export function evaluateSplatAt(data, i, t) {
    const m = readSplatMotion(data, i);
    if (!m) throw new Error("asog: evaluateSplatAt needs motion.A, motion.B and centers");
    const t4 = i * 4, q = data.sog.quats;
    const base = dequantQuat(q[t4], q[t4 + 1], q[t4 + 2], q[t4 + 3] - 252);
    const cw = clipWeight(data.clips, m.clip, t);
    if (cw <= 0) return { center: [m.x, m.y, m.z], rotation: Array.from(base), weight: 0 };
    const out = new Float64Array(7);
    const weight = cw * evaluateSplat(m, t - data.clips[m.clip * 8], out);
    return { center: [out[0], out[1], out[2]], rotation: /** @type {number[]} */ (quatMultiply(out.subarray(3), base)), weight };
}

/**
 * A robust bounding box of the centres: per axis the `lo` and `hi` quantiles of a sample of splats
 * (outlier splats far from the subject do not count).
 * @param {Float32Array} centers
 * @param {number} [lo]
 * @param {number} [hi]
 * @returns {{ min: number[], max: number[] }}
 */
export function centerBounds(centers, lo = 0.02, hi = 0.98) {
    const n = centers.length / 3;
    const stride = Math.max(1, Math.floor(n / 20000));
    const min = [0, 0, 0], max = [0, 0, 0];
    for (let c = 0; c < 3; c++) {
        const s = [];
        for (let i = 0; i < n; i += stride) s.push(centers[i * 3 + c]);
        s.sort((a, b) => a - b);
        min[c] = s[Math.floor(lo * (s.length - 1))];
        max[c] = s[Math.ceil(hi * (s.length - 1))];
    }
    return { min, max };
}
