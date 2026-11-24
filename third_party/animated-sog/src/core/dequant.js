/**
 * Dequantisation (format spec 3.3 static attributes, 4.2 motion). Plain scalar functions: the
 * decoder, the per-splat accessor and the tests all use these.
 */

export const SH_C0 = 0.28209479177387814;

/**
 * Position on one axis: the SOG v2 16-bit log mapping.
 * @param {number} q - 0..65535
 * @param {number} min - manifest.means.mins[axis]
 * @param {number} max - manifest.means.maxs[axis]
 * @returns {number}
 */
export function dequantPosition(q, min, max) {
    const v = min + q / 65535 * (max - min);
    return Math.sign(v) * Math.expm1(Math.abs(v));
}

/**
 * Rotation from SOG v2 smallest-three: three 8-bit components in w x y z order with the largest
 * (index `mode`, 0 = w) left out; the largest is positive.
 * @param {number} q0
 * @param {number} q1
 * @param {number} q2
 * @param {number} mode - 0..3 (the `quatmode` stream; the SOG texture stores 252 + mode)
 * @param {number[] | Float32Array | Float64Array} [out]
 * @returns {number[] | Float32Array | Float64Array} out as [x, y, z, w]
 */
export function dequantQuat(q0, q1, q2, mode, out = [0, 0, 0, 0]) {
    const a = (q0 / 255 - 0.5) * Math.SQRT2;
    const b = (q1 / 255 - 0.5) * Math.SQRT2;
    const c = (q2 / 255 - 0.5) * Math.SQRT2;
    const d = Math.sqrt(Math.max(0, 1 - (a * a + b * b + c * c)));
    let w, x, y, z;
    switch (mode) {
        case 0: w = d; x = a; y = b; z = c; break;
        case 1: w = a; x = d; y = b; z = c; break;
        case 2: w = a; x = b; y = d; z = c; break;
        default: w = a; x = b; y = c; z = d; break;
    }
    out[0] = x; out[1] = y; out[2] = z; out[3] = w;
    return out;
}

/**
 * Centre time μ in seconds.
 * @param {number} q
 * @param {import("./container.js").AsogManifest} manifest
 */
export function dequantMu(q, manifest) {
    return q / manifest.mu.scale;
}

/**
 * Temporal width σ in seconds.
 * @param {number} q
 * @param {import("./container.js").AsogManifest} manifest
 */
export function dequantSigma(q, manifest) {
    const ls = manifest.logsigma;
    return Math.exp(ls.min + q / (2 ** ls.bits - 1) * (ls.max - ls.min));
}

/**
 * Visible half-life L = sm · σ, the time the motion steps are scaled to.
 * @param {number} sigma
 * @param {import("./container.js").AsogManifest["motion"]} motion
 */
export function lifeSpan(sigma, motion) {
    return motion.sm * sigma;
}

/**
 * Anchor coordinate.
 * @param {number} q
 * @param {import("./container.js").AsogManifest["motion"]} motion
 */
export function dequantAnchor(q, motion) {
    return q * motion.qa / 1000;
}

/**
 * The Ω quantisation step: qp millimetres of displacement at the splat's distance from its anchor.
 * @param {number} dist - |x − a| (decoded position minus decoded anchor)
 * @param {import("./container.js").AsogManifest["motion"]} motion
 */
export function omegaStep(dist, motion) {
    return motion.qp / 1000 / Math.max(dist, motion.r_min);
}

/**
 * Per-control-point weights of the Ω steps ([1, midO, 1]) and the G steps ([1, midG, midG, midG, 1]).
 * @param {import("./container.js").AsogManifest["motion"]} motion
 */
export function motionWeights(motion) {
    return { wO: [1, motion.midO, 1], wG: [1, motion.midG, motion.midG, motion.midG, 1] };
}

/**
 * Ω control point component O_k (rad/s).
 * @param {number} q
 * @param {number} k - control point 0..2
 * @param {number} step - omegaStep()
 * @param {number} L - lifeSpan()
 * @param {number[]} wO - motionWeights().wO
 */
export function dequantOmega(q, k, step, L, wO) {
    return q * step * wO[k] / L;
}

/**
 * The G quantisation step m_ref · qp / 1000: the smallest |Ω(τ)| over the life (life-scaled), sampled
 * at τ = s / (samples − 1), times the displacement step.
 * @param {ArrayLike<number>} O - O0.xyz O1.xyz O2.xyz (rad/s), as the decoder stores them
 * @param {number} L
 * @param {import("./container.js").AsogManifest["motion"]} motion
 * @returns {number}
 */
export function gStep(O, L, motion) {
    let m = Infinity;
    const samples = motion.mref_samples;
    for (let s = 0; s < samples; s++) {
        const tau = s / (samples - 1), u = 1 - tau, b0 = u * u, b1 = 2 * u * tau, b2 = tau * tau;
        let q = 0;
        for (let c = 0; c < 3; c++) {
            const x = b0 * O[c] + b1 * O[3 + c] + b2 * O[6 + c];
            q += x * x;
        }
        if (q < m) m = q;
    }
    return Math.max(Math.sqrt(m) * L, 1e-9) * motion.qp / 1000;
}

/**
 * Translation quartic coefficient G_j.
 * @param {number} q
 * @param {number} j - 0..4
 * @param {number} step - gStep()
 * @param {number} L
 * @param {number[]} wG - motionWeights().wG
 */
export function dequantG(q, j, step, L, wG) {
    return q * step * wG[j] / (L * L);
}
