/**
 * Pure-JS reference of the TRiGS motion (format spec 4.1) and clip timing (section 5).
 *
 * The GPU modifiers (PlayCanvas GLSL / WGSL) are the twins of these functions. Each function here
 * is self-contained apart from calls to the others in this file, because sortWorkerSource() puts
 * their source text into a worker.
 */

/** The temporal opacity cut: weight = max((γ − 0.05) / (1 − 0.05), 0). Literal in temporalWeight(). */
export const TEMPORAL_THRESHOLD = 0.05;

/**
 * @typedef {object} SplatMotion
 * @property {number} x - canonical centre (model space)
 * @property {number} y
 * @property {number} z
 * @property {number} mu - centre time μ (clip-local seconds)
 * @property {number} sigma - temporal width σ (seconds)
 * @property {Float64Array} anchor - a, 3 values
 * @property {Float64Array} omega - O0.xyz O1.xyz O2.xyz (rad/s)
 * @property {Float64Array} g - G0..G4
 * @property {boolean} isStatic
 * @property {number} clip
 */

/**
 * A zeroed motion record to fill (readMotion) and evaluate (evaluateSplat).
 * @returns {SplatMotion}
 */
export function createMotion() {
    return {
        x: 0, y: 0, z: 0, mu: 0, sigma: 1,
        anchor: new Float64Array(3), omega: new Float64Array(9), g: new Float64Array(5),
        isStatic: false, clip: 0,
    };
}

/**
 * Temporal weight of a dynamic splat, `dt` seconds from its centre time.
 * @param {number} dt
 * @param {number} sigma
 * @returns {number} 0..1
 */
export function temporalWeight(dt, sigma) {
    const x = dt / sigma;
    return Math.max((Math.exp(-0.5 * x * x) - 0.05) / (1 - 0.05), 0);
}

/**
 * Weight of a clip at global time t: 0 outside its active range, smoothstep fades outside its core range.
 * @param {ArrayLike<number>} clips - clipTable()
 * @param {number} c - clip index
 * @param {number} t - global time (seconds)
 * @returns {number} 0..1
 */
export function clipWeight(clips, c, t) {
    const o = c * 8;
    if (t < clips[o + 1] || t >= clips[o + 2]) return 0;
    const coreStart = clips[o + 4], coreEnd = clips[o + 5], fadeIn = clips[o + 6], fadeOut = clips[o + 7];
    let x;
    if (fadeIn > 0 && t < coreStart) x = (t - (coreStart - fadeIn)) / fadeIn;
    else if (fadeOut > 0 && t >= coreEnd) x = (coreEnd + fadeOut - t) / fadeOut;
    else return 1;
    x = Math.min(1, Math.max(0, x));
    return x * x * (3 - 2 * x);
}

/**
 * Evaluates one splat at clip-local time t.
 * Writes out[0..2] = centre (model space) and out[3..6] = the rotation q(w) as x y z w, which
 * pre-multiplies the splat's own rotation (rotation(t) = q(w) ⊗ rotation). A static splat, or one
 * that is invisible at t, keeps its canonical centre and an identity rotation.
 * @param {SplatMotion} m
 * @param {number} t - clip-local time (seconds)
 * @param {Float64Array | Float32Array | number[]} out - 7 values
 * @returns {number} temporal weight (opacity factor; 1 for a static splat, 0 = invisible)
 */
export function evaluateSplat(m, t, out) {
    out[0] = m.x; out[1] = m.y; out[2] = m.z;
    out[3] = 0; out[4] = 0; out[5] = 0; out[6] = 1;
    if (m.isStatic) return 1;
    const sigma = m.sigma, dt = t - m.mu;
    const weight = temporalWeight(dt, sigma);
    if (weight <= 0) return 0;

    const tau = Math.min(1, Math.max(0, (t - (m.mu - 2 * sigma)) / (4 * sigma)));
    const u = 1 - tau, b0 = u * u, b1 = 2 * u * tau, b2 = tau * tau;
    const O = m.omega, G = m.g;
    const ox = b0 * O[0] + b1 * O[3] + b2 * O[6];
    const oy = b0 * O[1] + b1 * O[4] + b2 * O[7];
    const oz = b0 * O[2] + b1 * O[5] + b2 * O[8];
    const g = b0 * b0 * G[0] + 2 * b0 * b1 * G[1] + (b1 * b1 + 2 * b0 * b2) * G[2] + 2 * b1 * b2 * G[3] + b2 * b2 * G[4];
    let ax = m.anchor[0], ay = m.anchor[1], az = m.anchor[2];
    const oo = ox * ox + oy * oy + oz * oz;
    if (oo >= 1e-8) {
        const ad = (ax * ox + ay * oy + az * oz) / oo;
        ax -= ad * ox; ay -= ad * oy; az -= ad * oz;
    }
    const gs = g / Math.max(oo, 1e-24);
    const wx = ox * dt, wy = oy * dt, wz = oz * dt;
    const vx = gs * ox * dt, vy = gs * oy * dt, vz = gs * oz * dt;
    const th = Math.sqrt(wx * wx + wy * wy + wz * wz);
    const sh = Math.sin(0.5 * th), ch = Math.cos(0.5 * th);
    const small = th < 1e-6;
    const cA = small ? 1 : 2 * sh * ch / th;
    const cB = small ? 0.5 : 2 * sh * sh / (th * th);
    const cC = small ? 1 / 6 : (th - 2 * sh * ch) / (th * th * th);

    const rx = m.x - ax, ry = m.y - ay, rz = m.z - az;
    const wrx = wy * rz - wz * ry, wry = wz * rx - wx * rz, wrz = wx * ry - wy * rx;
    const wwrx = wy * wrz - wz * wry, wwry = wz * wrx - wx * wrz, wwrz = wx * wry - wy * wrx;
    const wvx = wy * vz - wz * vy, wvy = wz * vx - wx * vz, wvz = wx * vy - wy * vx;
    const wwvx = wy * wvz - wz * wvy, wwvy = wz * wvx - wx * wvz, wwvz = wx * wvy - wy * wvx;
    out[0] = rx + cA * wrx + cB * wwrx + ax + vx + cB * wvx + cC * wwvx;
    out[1] = ry + cA * wry + cB * wwry + ay + vy + cB * wvy + cC * wwvy;
    out[2] = rz + cA * wrz + cB * wwrz + az + vz + cB * wvz + cC * wwvz;
    const s = small ? 0.5 : sh / th; // sin(θ/2) / θ
    out[3] = wx * s; out[4] = wy * s; out[5] = wz * s; out[6] = ch;
    return weight;
}

/**
 * Fills a motion record for splat i from the GPU layouts (layout.js).
 * @param {Uint32Array} A - motionA
 * @param {Float32Array} Af - motionA viewed as floats
 * @param {Uint16Array} B - motionB (float16 bits)
 * @param {Float32Array} half - halfTable()
 * @param {Float32Array} centers - canonical centres, xyz per splat
 * @param {number} i
 * @param {SplatMotion} m
 * @returns {SplatMotion} m
 */
export function readMotion(A, Af, B, half, centers, i, m) {
    const a = i * 4, aw = A[a + 3], p = i * 3, z = i * 16;
    m.x = centers[p]; m.y = centers[p + 1]; m.z = centers[p + 2];
    m.mu = Af[a];
    m.sigma = Af[a + 1];
    m.anchor[0] = half[A[a + 2] & 0xffff];
    m.anchor[1] = half[A[a + 2] >>> 16];
    m.anchor[2] = half[aw & 0xffff];
    for (let k = 0; k < 9; k++) m.omega[k] = half[B[z + k]];
    for (let j = 0; j < 5; j++) m.g[j] = half[B[z + 9 + j]];
    m.isStatic = (aw & 0x10000) !== 0;
    m.clip = aw >>> 17;
    return m;
}

/**
 * Every splat's centre at global time t, for depth sorting: `out` gets the moved centres; splats
 * that are static or invisible at t keep their canonical centre.
 * @param {number} n
 * @param {Float32Array} centers - canonical centres
 * @param {Uint32Array} A - motionA
 * @param {Uint16Array} B - motionB
 * @param {Float32Array} clips - clipTable()
 * @param {Float32Array} half - halfTable()
 * @param {number} t - global time (seconds)
 * @param {Float32Array} out - n * 3
 */
export function evaluateCenters(n, centers, A, B, clips, half, t, out) {
    const Af = new Float32Array(A.buffer, A.byteOffset, A.length);
    const m = createMotion();
    const r = new Float64Array(7);
    out.set(centers.subarray(0, n * 3));
    for (let i = 0; i < n; i++) {
        const a = i * 4, aw = A[a + 3];
        if (aw & 0x10000) continue;
        const clip = aw >>> 17;
        if (clipWeight(clips, clip, t) <= 0) continue;
        const lt = t - clips[clip * 8];
        if (temporalWeight(lt - Af[a], Af[a + 1]) <= 0) continue;
        readMotion(A, Af, B, half, centers, i, m);
        evaluateSplat(m, lt, r);
        out[i * 3] = r[0];
        out[i * 3 + 1] = r[1];
        out[i * 3 + 2] = r[2];
    }
}

/**
 * Hamilton product a ⊗ b of quaternions stored x y z w.
 * @param {ArrayLike<number>} a
 * @param {ArrayLike<number>} b
 * @param {number[] | Float64Array | Float32Array} [out]
 */
export function quatMultiply(a, b, out = [0, 0, 0, 0]) {
    const ax = a[0], ay = a[1], az = a[2], aw = a[3], bx = b[0], by = b[1], bz = b[2], bw = b[3];
    out[0] = aw * bx + ax * bw + ay * bz - az * by;
    out[1] = aw * by - ax * bz + ay * bw + az * bx;
    out[2] = aw * bz + ax * by - ay * bx + az * bw;
    out[3] = aw * bw - ax * bx - ay * by - az * bz;
    return out;
}
