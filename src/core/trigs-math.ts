// The TRiGS motion model on the CPU, in double precision: the reference the viewer's shader (trigs-motion.ts) is
// checked against, and what the loader uses to bound the scene. It follows cumuli_core.trigs.slice_at, which is
// written from the TRiGS paper (arXiv 2604.00538, Sec. 3, Eqs. 1 and 4-10):
//
//   gamma(t) = exp(-0.5 (dt / s)^2),  w(t) = max((gamma - 0.05) / 0.95, 0),  alpha(t) = sigmoid(opacity) w
//                                                                           dt = t - t_c,  s = exp(log_duration)
//   tau      = clamp((dt + 2 s) / (4 s), 0, 1)
//   zeta(t)  = base + (1-tau)^2 c0 + 2 (1-tau) tau c1 + tau^2 c2           zeta = (omega, nu)
//   phi = omega dt,  upsilon = nu dt,  R = exp([phi]x),  p = J(phi) upsilon
//   axis = omega / |omega|,  a_perp = a - <a, axis> axis,  p_par = J(phi) <upsilon, axis> axis
//   mean(t) = R (mu - a_perp) + a_perp + p_par,   quat(t) = quat(R) * quat
//
// With |omega|^2 <= 1e-12 there is no axis: a_perp = a and the whole translation is kept.
// Quaternions here are xyzw, as the file stores them.

import type { TrigsArrays } from '../parsers/trigsbin';

const OMEGA_EPS = 1e-12;
const SMALL_ANGLE = 1e-3;
const T_SIGMA_FLOOR = 1e-6;
// The temporal gaussian is cut where it falls under 5% and renormalised: the weight the file format applies (the animated-sog
// spec, and Open4DGS's "smooth_cull_renormalized"), not the paper's plain gamma.
const WEIGHT_FLOOR = 0.05;

type Vec3 = [number, number, number];
type Quat = [number, number, number, number];

type SplatState = { mean: Vec3; quat: Quat; alpha: number };

const dot3 = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

// R = I + A K + B K^2 and J = I + B K + C K^2, with K = [phi]x
const angleTerms = (th2: number) => {
    if (th2 < SMALL_ANGLE * SMALL_ANGLE) {
        return {
            a: 1 - th2 / 6 + (th2 * th2) / 120,
            b: 0.5 - th2 / 24 + (th2 * th2) / 720,
            c: 1 / 6 - th2 / 120 + (th2 * th2) / 5040
        };
    }
    const th = Math.sqrt(th2);
    return { a: Math.sin(th) / th, b: (1 - Math.cos(th)) / th2, c: (th - Math.sin(th)) / (th2 * th) };
};

const skewMul = (k: Vec3, v: Vec3): Vec3 => [
    k[1] * v[2] - k[2] * v[1],
    k[2] * v[0] - k[0] * v[2],
    k[0] * v[1] - k[1] * v[0]
];

// (I + s1 K + s2 K^2) v
const expLike = (phi: Vec3, s1: number, s2: number, v: Vec3): Vec3 => {
    const kv = skewMul(phi, v);
    const kkv = skewMul(phi, kv);
    return [v[0] + s1 * kv[0] + s2 * kkv[0], v[1] + s1 * kv[1] + s2 * kkv[1], v[2] + s1 * kv[2] + s2 * kkv[2]];
};

const quatMul = (a: Quat, b: Quat): Quat => [
    a[3] * b[0] + b[3] * a[0] + (a[1] * b[2] - a[2] * b[1]),
    a[3] * b[1] + b[3] * a[1] + (a[2] * b[0] - a[0] * b[2]),
    a[3] * b[2] + b[3] * a[2] + (a[0] * b[1] - a[1] * b[0]),
    a[3] * b[3] - (a[0] * b[0] + a[1] * b[1] + a[2] * b[2])
];

const rotvecToQuat = (phi: Vec3): Quat => {
    const th2 = dot3(phi, phi);
    let w: number;
    let k: number;
    if (th2 < SMALL_ANGLE * SMALL_ANGLE) {
        w = 1 - th2 / 8 + (th2 * th2) / 384;
        k = 0.5 - th2 / 48 + (th2 * th2) / 3840;
    } else {
        const th = Math.sqrt(th2);
        w = Math.cos(0.5 * th);
        k = Math.sin(0.5 * th) / th;
    }
    return [k * phi[0], k * phi[1], k * phi[2], w];
};

const row = (a: Float32Array, i: number, n: number): number[] => Array.from(a.subarray(i * n, i * n + n));

// Splat `i` of `arrays` (a clip's float32 tensors, or the loader's expanded ones) at time `t` seconds, in the clip's own time.
const evalSplat = (arrays: TrigsArrays, i: number, t: number): SplatState => {
    const sigma = Math.max(Math.exp(arrays.log_durations[i]), T_SIGMA_FLOOR);
    const dt = t - arrays.times[i];
    const tau = Math.min(1, Math.max(0, (dt + 2 * sigma) / (4 * sigma)));
    const b = 1 - tau;
    const w0 = b * b;
    const w1 = 2 * b * tau;
    const w2 = tau * tau;
    const twist = (name: string): Vec3 => {
        const base = row(arrays[`zeta_base_${name}`], i, 3);
        const c0 = row(arrays[`zeta_c0_${name}`], i, 3);
        const c1 = row(arrays[`zeta_c1_${name}`], i, 3);
        const c2 = row(arrays[`zeta_c2_${name}`], i, 3);
        return [0, 1, 2].map((k) => base[k] + w0 * c0[k] + w1 * c1[k] + w2 * c2[k]) as Vec3;
    };
    const omega = twist('r');
    const nu = twist('t');
    const phi = omega.map((v) => v * dt) as Vec3;
    const ups = nu.map((v) => v * dt) as Vec3;
    const { a, b: bb, c } = angleTerms(dot3(phi, phi));

    const mu = row(arrays.means, i, 3) as Vec3;
    const anchor = row(arrays.anchors, i, 3) as Vec3;
    const w2n = dot3(omega, omega);
    const moving = w2n > OMEGA_EPS;
    const inv = moving ? 1 / Math.sqrt(w2n) : 0;
    const axis = omega.map((v) => v * inv) as Vec3;
    const ad = dot3(anchor, axis);
    const aPerp = anchor.map((v, k) => v - ad * axis[k]) as Vec3;
    const ud = dot3(ups, axis);
    const upsUsed = (moving ? axis.map((v) => v * ud) : ups) as Vec3;

    const centred = mu.map((v, k) => v - aPerp[k]) as Vec3;
    const rotated = expLike(phi, a, bb, centred);
    const shift = expLike(phi, bb, c, upsUsed);
    const mean = rotated.map((v, k) => v + aPerp[k] + shift[k]) as Vec3;

    const q = row(arrays.quats, i, 4) as Quat;
    const quat = quatMul(rotvecToQuat(phi), q);
    const logit = arrays.opacities[i];
    const weight = Math.max((Math.exp(-0.5 * (dt / sigma) ** 2) - WEIGHT_FLOOR) / (1 - WEIGHT_FLOOR), 0);
    const alpha = (1 / (1 + Math.exp(-logit))) * weight;
    return { mean, quat, alpha };
};

// The mean of every splat at time `t`, written into `out` (3 floats per splat). The same maths as evalSplat without
// the allocation, for the CPU sorter's centers on WebGL, where the engine sorts by what the resource says rather than by
// the work buffer. Only the means are needed there: a depth order does not need the rotation or the opacity.
const evalCenters = (arrays: TrigsArrays, count: number, t: number, out: Float32Array) => {
    const { means, anchors, times, log_durations: logDur } = arrays;
    const zbr = arrays.zeta_base_r;
    const zbt = arrays.zeta_base_t;
    const c0r = arrays.zeta_c0_r;
    const c0t = arrays.zeta_c0_t;
    const c1r = arrays.zeta_c1_r;
    const c1t = arrays.zeta_c1_t;
    const c2r = arrays.zeta_c2_r;
    const c2t = arrays.zeta_c2_t;
    for (let i = 0; i < count; i++) {
        const o = i * 3;
        const sigma = Math.max(Math.exp(logDur[i]), T_SIGMA_FLOOR);
        const dt = t - times[i];
        const tau = Math.min(1, Math.max(0, (dt + 2 * sigma) / (4 * sigma)));
        const b = 1 - tau;
        const w0 = b * b;
        const w1 = 2 * b * tau;
        const w2 = tau * tau;
        const ox = zbr[o] + w0 * c0r[o] + w1 * c1r[o] + w2 * c2r[o];
        const oy = zbr[o + 1] + w0 * c0r[o + 1] + w1 * c1r[o + 1] + w2 * c2r[o + 1];
        const oz = zbr[o + 2] + w0 * c0r[o + 2] + w1 * c1r[o + 2] + w2 * c2r[o + 2];
        const px = ox * dt;
        const py = oy * dt;
        const pz = oz * dt;
        const ux = (zbt[o] + w0 * c0t[o] + w1 * c1t[o] + w2 * c2t[o]) * dt;
        const uy = (zbt[o + 1] + w0 * c0t[o + 1] + w1 * c1t[o + 1] + w2 * c2t[o + 1]) * dt;
        const uz = (zbt[o + 2] + w0 * c0t[o + 2] + w1 * c1t[o + 2] + w2 * c2t[o + 2]) * dt;
        const { a, b: bb, c } = angleTerms(px * px + py * py + pz * pz);

        const w2n = ox * ox + oy * oy + oz * oz;
        const moving = w2n > OMEGA_EPS;
        const inv = moving ? 1 / Math.sqrt(w2n) : 0;
        const ax = ox * inv;
        const ay = oy * inv;
        const az = oz * inv;
        const ad = anchors[o] * ax + anchors[o + 1] * ay + anchors[o + 2] * az;
        const apx = anchors[o] - ad * ax;
        const apy = anchors[o + 1] - ad * ay;
        const apz = anchors[o + 2] - ad * az;
        const ud = ux * ax + uy * ay + uz * az;
        const sx = moving ? ud * ax : ux;
        const sy = moving ? ud * ay : uy;
        const sz = moving ? ud * az : uz;

        // (I + a K + bb K^2)(mu - a_perp) + a_perp + (I + bb K + c K^2) s
        const cx = means[o] - apx;
        const cy = means[o + 1] - apy;
        const cz = means[o + 2] - apz;
        let kx = py * cz - pz * cy;
        let ky = pz * cx - px * cz;
        let kz = px * cy - py * cx;
        let kkx = py * kz - pz * ky;
        let kky = pz * kx - px * kz;
        let kkz = px * ky - py * kx;
        const rx = cx + a * kx + bb * kkx;
        const ry = cy + a * ky + bb * kky;
        const rz = cz + a * kz + bb * kkz;
        kx = py * sz - pz * sy;
        ky = pz * sx - px * sz;
        kz = px * sy - py * sx;
        kkx = py * kz - pz * ky;
        kky = pz * kx - px * kz;
        kkz = px * ky - py * kx;
        out[o] = rx + apx + sx + bb * kx + c * kkx;
        out[o + 1] = ry + apy + sy + bb * ky + c * kky;
        out[o + 2] = rz + apz + sz + bb * kz + c * kkz;
    }
};

export { OMEGA_EPS, SMALL_ANGLE, evalCenters, evalSplat, quatMul, rotvecToQuat };
export type { Quat, SplatState, Vec3 };
