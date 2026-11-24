/**
 * Back-to-front depth order of the splats visible at a time, for renderers that sort on the CPU
 * (the three.js player). Self-contained apart from clipWeight() and temporalWeight() (motion.js),
 * because sortWorkerSource() puts its source text into a worker.
 */
import { clipWeight, temporalWeight } from "./motion.js";

/**
 * Orders the splats visible at global time t back to front: `indices` gets their indices, farthest
 * first. Splats with a zero clip weight at t, or a zero temporal weight throughout [t - margin, t + margin], are
 * left out. Counting sort on 16-bit depth keys (ties keep the splat order).
 * @param {number} n - splats
 * @param {Float32Array} centers - centres at t, xyz per splat (evaluateCenters)
 * @param {Uint32Array} A - motionA (layout.js): visibility from the static flag, clip, μ and σ
 * @param {Float32Array} clips - clipTable()
 * @param {number} t - global time (seconds)
 * @param {ArrayLike<number>} depthRow - depth = r0·x + r1·y + r2·z + r3, larger = farther (e.g. minus
 *   the third row of a right-handed model-view matrix)
 * @param {Uint32Array} indices - n entries, filled from 0
 * @param {{ depths?: Float32Array, ids?: Uint32Array, keys?: Uint16Array, counts?: Uint32Array }} [scratch] - buffers
 *   reused between calls
 * @param {number} [margin] - seconds: also keep the splats that become visible within this time of t, so an order
 *   that arrives a few frames late still holds the splats appearing meanwhile (they draw nothing until then)
 * @returns {number} number of visible splats (entries written to indices)
 */
export function sortSplats(n, centers, A, clips, t, depthRow, indices, scratch = {}, margin = 0) {
    const Af = new Float32Array(A.buffer, A.byteOffset, A.length);
    if (!scratch.depths || scratch.depths.length < n) {
        scratch.depths = new Float32Array(n);
        scratch.ids = new Uint32Array(n);
        scratch.keys = new Uint16Array(n);
    }
    if (!scratch.counts) scratch.counts = new Uint32Array(65536);
    const { depths, ids, keys, counts } = scratch;
    const r0 = depthRow[0], r1 = depthRow[1], r2 = depthRow[2], r3 = depthRow[3];
    const numClips = clips.length / 8;
    const clipW = new Float64Array(numClips);
    for (let c = 0; c < numClips; c++) clipW[c] = clipWeight(clips, c, t);

    // the visible splats and their depths
    let count = 0, min = Infinity, max = -Infinity;
    for (let i = 0; i < n; i++) {
        const a = i * 4, aw = A[a + 3], clip = aw >>> 17;
        if (clipW[clip] <= 0) continue;
        if (!(aw & 0x10000) && temporalWeight(Math.max(Math.abs(t - clips[clip * 8] - Af[a]) - margin, 0), Af[a + 1]) <= 0) continue;
        const p = i * 3;
        depths[count] = r0 * centers[p] + r1 * centers[p + 1] + r2 * centers[p + 2] + r3;
        const d = depths[count]; // as stored (float32), so that the keys below stay in 0..65535
        ids[count++] = i;
        if (d < min) min = d;
        if (d > max) max = d;
    }
    if (count === 0) return 0;

    // keys: 65535 = nearest, so ascending keys run far to near
    const scale = max > min ? 65535 / (max - min) : 0;
    counts.fill(0);
    for (let k = 0; k < count; k++) {
        const key = 65535 - Math.floor((depths[k] - min) * scale);
        keys[k] = key;
        counts[key]++;
    }
    for (let b = 0, sum = 0; b < 65536; b++) {
        const c = counts[b];
        counts[b] = sum;
        sum += c;
    }
    for (let k = 0; k < count; k++) indices[counts[keys[k]]++] = ids[k];
    return count;
}
