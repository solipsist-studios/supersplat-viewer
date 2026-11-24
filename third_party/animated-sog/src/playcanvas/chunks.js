/**
 * The gsplat work-buffer modifier that moves the splats (GLSL for WebGL 2, WGSL for WebGPU; keep the
 * two in sync). GPU twin of core/motion.js: clipWeight(), temporalWeight(), evaluateSplat().
 * Reads the motion from motionA / motionB and the clip table (core/layout.js).
 */
import { MOTION_B_TEXELS, MOTION_TEX_WIDTH } from "../core/layout.js";
import { TEMPORAL_THRESHOLD } from "../core/motion.js";

const W = MOTION_TEX_WIDTH;
const BT = MOTION_B_TEXELS;
const TH = TEMPORAL_THRESHOLD.toFixed(4);

export const ASOG_CHUNK_GLSL = /* glsl */ `
uniform highp usampler2D uAsogA;
uniform highp sampler2D uAsogB;
uniform highp sampler2D uAsogClips;
uniform float uAsogTime;
uniform mat4 uAsogModelInv;
vec4 gAsogRelQuat = vec4(0.0, 0.0, 0.0, 1.0);
float gAsogWeight = 1.0;

ivec2 asogUv(int texel) {
    return ivec2(texel % ${W}, texel / ${W});
}

vec4 asogQuatMul(vec4 a, vec4 b) {
    return vec4(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z);
}

float asogSmooth(float x) {
    x = clamp(x, 0.0, 1.0);
    return x * x * (3.0 - 2.0 * x);
}

void modifySplatCenter(inout vec3 center) {
    gAsogRelQuat = vec4(0.0, 0.0, 0.0, 1.0);
    int i = int(splat.index);
    uvec4 a = texelFetch(uAsogA, asogUv(i), 0);
    int clip = int(a.w >> 17u);
    vec4 c0 = texelFetch(uAsogClips, ivec2(0, clip), 0);
    vec4 c1 = texelFetch(uAsogClips, ivec2(1, clip), 0);
    float t = uAsogTime;
    gAsogWeight = (t < c0.y || t >= c0.z) ? 0.0
        : (c1.z > 0.0 && t < c1.x) ? asogSmooth((t - (c1.x - c1.z)) / c1.z)
        : (c1.w > 0.0 && t >= c1.y) ? asogSmooth((c1.y + c1.w - t) / c1.w)
        : 1.0;
    if ((a.w & 0x10000u) != 0u || gAsogWeight <= 0.0) return;

    float lt = t - c0.x;
    float mu = uintBitsToFloat(a.x);
    float sigma = uintBitsToFloat(a.y);
    float dt = lt - mu;
    float g = exp(-0.5 * (dt / sigma) * (dt / sigma));
    gAsogWeight *= max((g - ${TH}) / (1.0 - ${TH}), 0.0);
    if (gAsogWeight <= 0.0) return;

    float tau = clamp((lt - (mu - 2.0 * sigma)) / (4.0 * sigma), 0.0, 1.0);
    float u = 1.0 - tau;
    float b0 = u * u, b1 = 2.0 * u * tau, b2 = tau * tau;
    int zt = i * ${BT};
    vec4 t0 = texelFetch(uAsogB, asogUv(zt), 0);
    vec4 t1 = texelFetch(uAsogB, asogUv(zt + 1), 0);
    vec4 t2 = texelFetch(uAsogB, asogUv(zt + 2), 0);
    vec4 t3 = texelFetch(uAsogB, asogUv(zt + 3), 0);
    vec3 omega = b0 * t0.xyz + b1 * vec3(t0.w, t1.xy) + b2 * vec3(t1.zw, t2.x);
    float gq = b0 * b0 * t2.y + 2.0 * b0 * b1 * t2.z + (b1 * b1 + 2.0 * b0 * b2) * t2.w + 2.0 * b1 * b2 * t3.x + b2 * b2 * t3.y;
    vec3 anchor = vec3(unpackHalf2x16(a.z), unpackHalf2x16(a.w).x);
    float oo = dot(omega, omega);
    if (oo >= 1e-8) anchor -= (dot(anchor, omega) / oo) * omega;
    vec3 nu = (gq / max(oo, 1e-24)) * omega;
    vec3 w = omega * dt;
    vec3 v = nu * dt;
    float th = length(w);
    float sh = sin(0.5 * th);
    float ch = cos(0.5 * th);
    bool small = th < 1e-6;
    float A = small ? 1.0 : 2.0 * sh * ch / th;
    float B = small ? 0.5 : 2.0 * sh * sh / (th * th);
    float C = small ? 1.0 / 6.0 : (th - 2.0 * sh * ch) / (th * th * th);

    vec3 rel = (uAsogModelInv * vec4(center, 1.0)).xyz - anchor;
    vec3 wr = cross(w, rel);
    vec3 wv = cross(w, v);
    vec3 moved = rel + A * wr + B * cross(w, wr) + anchor + v + B * wv + C * cross(w, wv);
    center = (matrix_model * vec4(moved, 1.0)).xyz;
    gAsogRelQuat = vec4(w * (small ? 0.5 : sh / th), ch);
}

void modifySplatRotationScale(vec3 originalCenter, vec3 modifiedCenter, inout vec4 rotation, inout vec3 scale) {
    if (gAsogWeight <= 0.0) {
        scale = vec3(0.0);
        return;
    }
    // the motion is in model space: world = qModel * rel * conj(qModel) * worldRotation
    vec4 rel = asogQuatMul(asogQuatMul(model_rotation, gAsogRelQuat), vec4(-model_rotation.xyz, model_rotation.w));
    rotation = normalize(asogQuatMul(rel, rotation));
    // work-buffer formats rebuild w with a square root and need w >= 0
    if (rotation.w < 0.0) rotation = -rotation;
}

void modifySplatColor(vec3 center, inout vec4 color) {
    color.a *= gAsogWeight;
}
`;

export const ASOG_CHUNK_WGSL = /* wgsl */ `
var uAsogA: texture_2d<u32>;
var uAsogB: texture_2d<uff>;
var uAsogClips: texture_2d<uff>;
uniform uAsogTime: f32;
uniform uAsogModelInv: mat4x4f;
var<private> gAsogRelQuat: vec4f = vec4f(0.0, 0.0, 0.0, 1.0);
var<private> gAsogWeight: f32 = 1.0;

fn asogUv(texel: i32) -> vec2i {
    return vec2i(texel % ${W}, texel / ${W});
}

fn asogQuatMul(a: vec4f, b: vec4f) -> vec4f {
    return vec4f(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z);
}

fn asogSmooth(x0: f32) -> f32 {
    let x = clamp(x0, 0.0, 1.0);
    return x * x * (3.0 - 2.0 * x);
}

fn modifySplatCenter(center: ptr<function, vec3f>) {
    gAsogRelQuat = vec4f(0.0, 0.0, 0.0, 1.0);
    let i = i32(splat.index);
    let a = textureLoad(uAsogA, asogUv(i), 0);
    let clip = i32(a.w >> 17u);
    let c0 = textureLoad(uAsogClips, vec2i(0, clip), 0);
    let c1 = textureLoad(uAsogClips, vec2i(1, clip), 0);
    let t = uniform.uAsogTime;
    var weight = 1.0;
    if (t < c0.y || t >= c0.z) {
        weight = 0.0;
    } else if (c1.z > 0.0 && t < c1.x) {
        weight = asogSmooth((t - (c1.x - c1.z)) / c1.z);
    } else if (c1.w > 0.0 && t >= c1.y) {
        weight = asogSmooth((c1.y + c1.w - t) / c1.w);
    }
    gAsogWeight = weight;
    if ((a.w & 0x10000u) != 0u || weight <= 0.0) { return; }

    let lt = t - c0.x;
    let mu = bitcast<f32>(a.x);
    let sigma = bitcast<f32>(a.y);
    let dt = lt - mu;
    let g = exp(-0.5 * (dt / sigma) * (dt / sigma));
    gAsogWeight = weight * max((g - ${TH}) / (1.0 - ${TH}), 0.0);
    if (gAsogWeight <= 0.0) { return; }

    let tau = clamp((lt - (mu - 2.0 * sigma)) / (4.0 * sigma), 0.0, 1.0);
    let u = 1.0 - tau;
    let b0 = u * u;
    let b1 = 2.0 * u * tau;
    let b2 = tau * tau;
    let zt = i * ${BT};
    let t0 = textureLoad(uAsogB, asogUv(zt), 0);
    let t1 = textureLoad(uAsogB, asogUv(zt + 1), 0);
    let t2 = textureLoad(uAsogB, asogUv(zt + 2), 0);
    let t3 = textureLoad(uAsogB, asogUv(zt + 3), 0);
    let omega = b0 * t0.xyz + b1 * vec3f(t0.w, t1.xy) + b2 * vec3f(t1.zw, t2.x);
    let gq = b0 * b0 * t2.y + 2.0 * b0 * b1 * t2.z + (b1 * b1 + 2.0 * b0 * b2) * t2.w + 2.0 * b1 * b2 * t3.x + b2 * b2 * t3.y;
    var anchor = vec3f(unpack2x16float(a.z), unpack2x16float(a.w).x);
    let oo = dot(omega, omega);
    if (oo >= 1e-8) {
        anchor = anchor - (dot(anchor, omega) / oo) * omega;
    }
    let nu = (gq / max(oo, 1e-24)) * omega;
    let w = omega * dt;
    let v = nu * dt;
    let th = length(w);
    let sh = sin(0.5 * th);
    let ch = cos(0.5 * th);
    let small = th < 1e-6;
    let A = select(2.0 * sh * ch / th, 1.0, small);
    let B = select(2.0 * sh * sh / (th * th), 0.5, small);
    let C = select((th - 2.0 * sh * ch) / (th * th * th), 1.0 / 6.0, small);

    let rel = (uniform.uAsogModelInv * vec4f(*center, 1.0)).xyz - anchor;
    let wr = cross(w, rel);
    let wv = cross(w, v);
    let moved = rel + A * wr + B * cross(w, wr) + anchor + v + B * wv + C * cross(w, wv);
    *center = (uniform.matrix_model * vec4f(moved, 1.0)).xyz;
    gAsogRelQuat = vec4f(w * select(sh / th, 0.5, small), ch);
}

fn modifySplatRotationScale(originalCenter: vec3f, modifiedCenter: vec3f, rotation: ptr<function, vec4f>, scale: ptr<function, vec3f>) {
    if (gAsogWeight <= 0.0) {
        *scale = vec3f(0.0);
        return;
    }
    let qm = uniform.model_rotation;
    let rel = asogQuatMul(asogQuatMul(qm, gAsogRelQuat), vec4f(-qm.xyz, qm.w));
    *rotation = normalize(asogQuatMul(rel, *rotation));
    if ((*rotation).w < 0.0) { *rotation = -*rotation; }
}

fn modifySplatColor(center: vec3f, color: ptr<function, vec4f>) {
    *color = vec4f((*color).xyz, (*color).w * gAsogWeight);
}
`;
