import { PIXELFORMAT_RGBA32F, Vec3 } from 'playcanvas';
import type { Entity, GSplatResource } from 'playcanvas';

import type { TrigsArrays } from '../parsers/trigsbin';

// GPU evaluation of the TRiGS motion model ("trigs.se3_bezier_v1") on the engine's unified gsplat pipeline.
//
// The same mechanism as the .sogst motion (sogst-motion.ts): per-splat textures are added to the resource's streams,
// the engine binds them to the work-buffer material, and the component's work-buffer modifier evaluates the model at
// the time uniform `trigsTime`. The maths is core/trigs-math.ts (the CPU reference) and cumuli_core.trigs, which are
// written from the TRiGS paper (arXiv 2604.00538, Eqs. 1 and 4-10). Unlike the .sogst model a splat also turns, so the
// modifier changes the splat's rotation as well as its centre and its alpha.
//
// Eight RGBA32F textures hold the 32 floats of a splat that the shader needs (the colour, scale and opacity are the
// resource's own):
//
//   trigsA  zeta_base_r.xyz | t_center        trigsE  zeta_c1_r.xyz | anchor.z
//   trigsB  zeta_base_t.xyz | sigma           trigsF  zeta_c1_t.xyz | mean.x
//   trigsC  zeta_c0_r.xyz   | anchor.x        trigsG  zeta_c2_r.xyz | mean.y
//   trigsD  zeta_c0_t.xyz   | anchor.y        trigsH  zeta_c2_t.xyz | mean.z
//
// t_center and sigma are seconds (sigma = exp(log_duration)); the mean is the canonical position the resource also holds.
//
// The work-buffer pass works on world-space centers, so the model-space displacement is scaled by the entity's scale and
// rotated by its rotation, and the rotation's turn is carried the same way (as in the .sogst chunk). The twist is small
// next to a pi turn in a frame, and uses the series forms of its sinc terms below an angle of 0.01 rad, which float32
// needs where float64 does not.

const glslModifyChunk = /* glsl */ `
uniform highp sampler2D trigsA;
uniform highp sampler2D trigsB;
uniform highp sampler2D trigsC;
uniform highp sampler2D trigsD;
uniform highp sampler2D trigsE;
uniform highp sampler2D trigsF;
uniform highp sampler2D trigsG;
uniform highp sampler2D trigsH;
uniform float trigsTime;
uniform vec4 trigsModelRotation;   // entity world rotation (x, y, z, w)
uniform vec3 trigsModelScale;      // entity world scale

vec3 trigsQuatRotate(vec4 q, vec3 v) {
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}
vec4 trigsQuatMul(vec4 a, vec4 b) {
    return vec4(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz), a.w * b.w - dot(a.xyz, b.xyz));
}
mat3 trigsSkew(vec3 v) {
    return mat3(0.0, v.z, -v.y, -v.z, 0.0, v.x, v.y, -v.x, 0.0);
}

// the model-space displacement of the splat's center, the turn of its rotation (xyzw) and its temporal factor at trigsTime
void trigsEval(out vec3 disp, out vec4 turn, out float gamma) {
    vec4 A = texelFetch(trigsA, splat.uv, 0);
    vec4 B = texelFetch(trigsB, splat.uv, 0);
    vec4 C = texelFetch(trigsC, splat.uv, 0);
    vec4 D = texelFetch(trigsD, splat.uv, 0);
    vec4 E = texelFetch(trigsE, splat.uv, 0);
    vec4 F = texelFetch(trigsF, splat.uv, 0);
    vec4 G = texelFetch(trigsG, splat.uv, 0);
    vec4 H = texelFetch(trigsH, splat.uv, 0);
    float sigma = max(B.w, 1e-6);
    float dt = trigsTime - A.w;
    gamma = exp(-0.5 * (dt / sigma) * (dt / sigma));
    float tau = clamp((dt + 2.0 * sigma) / (4.0 * sigma), 0.0, 1.0);
    float b = 1.0 - tau;
    float w0 = b * b;
    float w1 = 2.0 * b * tau;
    float w2 = tau * tau;
    vec3 omega = A.xyz + w0 * C.xyz + w1 * E.xyz + w2 * G.xyz;
    vec3 nu = B.xyz + w0 * D.xyz + w1 * F.xyz + w2 * H.xyz;
    vec3 anchor = vec3(C.w, D.w, E.w);
    vec3 mu = vec3(F.w, G.w, H.w);
    vec3 phi = omega * dt;
    vec3 ups = nu * dt;
    float th2 = dot(phi, phi);
    float ca, cb, cc, qw, qk;
    if (th2 < 1e-4) {
        ca = 1.0 - th2 / 6.0 + th2 * th2 / 120.0;
        cb = 0.5 - th2 / 24.0 + th2 * th2 / 720.0;
        cc = 1.0 / 6.0 - th2 / 120.0 + th2 * th2 / 5040.0;
        qw = 1.0 - th2 / 8.0 + th2 * th2 / 384.0;
        qk = 0.5 - th2 / 48.0 + th2 * th2 / 3840.0;
    } else {
        float th = sqrt(th2);
        ca = sin(th) / th;
        cb = (1.0 - cos(th)) / th2;
        cc = (th - sin(th)) / (th2 * th);
        qw = cos(0.5 * th);
        qk = sin(0.5 * th) / th;
    }
    mat3 K = trigsSkew(phi);
    mat3 K2 = K * K;
    mat3 R = mat3(1.0) + ca * K + cb * K2;
    mat3 J = mat3(1.0) + cb * K + cc * K2;
    float w2n = dot(omega, omega);
    bool moving = w2n > 1e-12;
    vec3 axis = moving ? omega * inversesqrt(w2n) : vec3(0.0);
    vec3 aPerp = anchor - dot(anchor, axis) * axis;
    vec3 upsUsed = moving ? dot(ups, axis) * axis : ups;
    disp = R * (mu - aPerp) + aPerp + J * upsUsed - mu;
    turn = vec4(qk * phi, qw);
}

void modifySplatCenter(inout vec3 center) {
    vec3 disp; vec4 turn; float gamma;
    trigsEval(disp, turn, gamma);
    center += trigsQuatRotate(trigsModelRotation, disp * trigsModelScale);
}
void modifySplatRotationScale(vec3 originalCenter, vec3 modifiedCenter, inout vec4 rotation, inout vec3 scale) {
    vec3 disp; vec4 turn; float gamma;
    trigsEval(disp, turn, gamma);
    // the turn is about a model-space axis; the splat's rotation here is already in the world
    vec4 worldTurn = vec4(trigsQuatRotate(trigsModelRotation, turn.xyz), turn.w);
    rotation = trigsQuatMul(worldTurn, rotation);
}
void modifySplatColor(vec3 center, inout vec4 color) {
    vec3 disp; vec4 turn; float gamma;
    trigsEval(disp, turn, gamma);
    color.a *= gamma;
}
`;

const wgslModifyChunk = /* wgsl */ `
var trigsA: texture_2d<uff>;
var trigsB: texture_2d<uff>;
var trigsC: texture_2d<uff>;
var trigsD: texture_2d<uff>;
var trigsE: texture_2d<uff>;
var trigsF: texture_2d<uff>;
var trigsG: texture_2d<uff>;
var trigsH: texture_2d<uff>;
uniform trigsTime: f32;
uniform trigsModelRotation: vec4f;
uniform trigsModelScale: vec3f;

fn trigsQuatRotate(q: vec4f, v: vec3f) -> vec3f {
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}
fn trigsQuatMul(a: vec4f, b: vec4f) -> vec4f {
    return vec4f(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz), a.w * b.w - dot(a.xyz, b.xyz));
}
fn trigsSkew(v: vec3f) -> mat3x3f {
    return mat3x3f(vec3f(0.0, v.z, -v.y), vec3f(-v.z, 0.0, v.x), vec3f(v.y, -v.x, 0.0));
}

struct TrigsState {
    disp: vec3f,
    turn: vec4f,
    gamma: f32,
}

fn trigsEval() -> TrigsState {
    let A = textureLoad(trigsA, splat.uv, 0);
    let B = textureLoad(trigsB, splat.uv, 0);
    let C = textureLoad(trigsC, splat.uv, 0);
    let D = textureLoad(trigsD, splat.uv, 0);
    let E = textureLoad(trigsE, splat.uv, 0);
    let F = textureLoad(trigsF, splat.uv, 0);
    let G = textureLoad(trigsG, splat.uv, 0);
    let H = textureLoad(trigsH, splat.uv, 0);
    let sigma = max(B.w, 1e-6);
    let dt = uniform.trigsTime - A.w;
    let gamma = exp(-0.5 * (dt / sigma) * (dt / sigma));
    let tau = clamp((dt + 2.0 * sigma) / (4.0 * sigma), 0.0, 1.0);
    let b = 1.0 - tau;
    let w0 = b * b;
    let w1 = 2.0 * b * tau;
    let w2 = tau * tau;
    let omega = A.xyz + w0 * C.xyz + w1 * E.xyz + w2 * G.xyz;
    let nu = B.xyz + w0 * D.xyz + w1 * F.xyz + w2 * H.xyz;
    let anchor = vec3f(C.w, D.w, E.w);
    let mu = vec3f(F.w, G.w, H.w);
    let phi = omega * dt;
    let ups = nu * dt;
    let th2 = dot(phi, phi);
    var ca: f32; var cb: f32; var cc: f32; var qw: f32; var qk: f32;
    if (th2 < 1e-4) {
        ca = 1.0 - th2 / 6.0 + th2 * th2 / 120.0;
        cb = 0.5 - th2 / 24.0 + th2 * th2 / 720.0;
        cc = 1.0 / 6.0 - th2 / 120.0 + th2 * th2 / 5040.0;
        qw = 1.0 - th2 / 8.0 + th2 * th2 / 384.0;
        qk = 0.5 - th2 / 48.0 + th2 * th2 / 3840.0;
    } else {
        let th = sqrt(th2);
        ca = sin(th) / th;
        cb = (1.0 - cos(th)) / th2;
        cc = (th - sin(th)) / (th2 * th);
        qw = cos(0.5 * th);
        qk = sin(0.5 * th) / th;
    }
    let K = trigsSkew(phi);
    let K2 = K * K;
    let eye = mat3x3f(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0));
    let R = eye + ca * K + cb * K2;
    let J = eye + cb * K + cc * K2;
    let w2n = dot(omega, omega);
    let moving = w2n > 1e-12;
    var axis = vec3f(0.0);
    if (moving) { axis = omega * inverseSqrt(w2n); }
    let aPerp = anchor - dot(anchor, axis) * axis;
    var upsUsed = ups;
    if (moving) { upsUsed = dot(ups, axis) * axis; }
    var s: TrigsState;
    s.disp = R * (mu - aPerp) + aPerp + J * upsUsed - mu;
    s.turn = vec4f(qk * phi, qw);
    s.gamma = gamma;
    return s;
}

fn modifySplatCenter(center: ptr<function, vec3f>) {
    let s = trigsEval();
    *center += trigsQuatRotate(uniform.trigsModelRotation, s.disp * uniform.trigsModelScale);
}
fn modifySplatRotationScale(originalCenter: vec3f, modifiedCenter: vec3f, rotation: ptr<function, vec4f>, scale: ptr<function, vec3f>) {
    let s = trigsEval();
    let worldTurn = vec4f(trigsQuatRotate(uniform.trigsModelRotation, s.turn.xyz), s.turn.w);
    *rotation = trigsQuatMul(worldTurn, *rotation);
}
fn modifySplatColor(center: vec3f, color: ptr<function, vec4f>) {
    let s = trigsEval();
    (*color).a = (*color).a * s.gamma;
}
`;

const TEXTURE_NAMES = ['trigsA', 'trigsB', 'trigsC', 'trigsD', 'trigsE', 'trigsF', 'trigsG', 'trigsH'] as const;

// Create and fill the eight per-splat textures from splat-ordered arrays (the loader's concatenated clip and static
// splats), and register them with the resource's stream collection so they are bound to the work-buffer material. Must
// run before the entity's gsplat component receives the resource. `sigma` is in seconds, the same as `times`.
const createTrigsTextures = (resource: GSplatResource, count: number, a: TrigsArrays) => {
    const streams = resource.streams;
    const dims = streams.textureDimensions;
    const texels = dims.x * dims.y;
    const data = TEXTURE_NAMES.map(() => new Float32Array(texels * 4));
    // texels past the last splat stay at sigma 1 so the shader math is finite; their opacity is the empty logit
    for (let i = count; i < texels; i++) {
        data[1][i * 4 + 3] = 1;
    }
    const put = (tex: number, i: number, xyz: ArrayLike<number> | null, offset: number, w: number) => {
        const d = data[tex];
        if (xyz) {
            d[i * 4] = xyz[offset];
            d[i * 4 + 1] = xyz[offset + 1];
            d[i * 4 + 2] = xyz[offset + 2];
        }
        d[i * 4 + 3] = w;
    };
    for (let i = 0; i < count; i++) {
        const o = i * 3;
        put(0, i, a.zeta_base_r, o, a.times[i]);
        put(1, i, a.zeta_base_t, o, Math.exp(a.log_durations[i]));
        put(2, i, a.zeta_c0_r, o, a.anchors[o]);
        put(3, i, a.zeta_c0_t, o, a.anchors[o + 1]);
        put(4, i, a.zeta_c1_r, o, a.anchors[o + 2]);
        put(5, i, a.zeta_c1_t, o, a.means[o]);
        put(6, i, a.zeta_c2_r, o, a.means[o + 1]);
        put(7, i, a.zeta_c2_t, o, a.means[o + 2]);
    }
    TEXTURE_NAMES.forEach((name, k) => {
        streams.textures.set(name, streams.createTexture(name, PIXELFORMAT_RGBA32F, dims, data[k]));
    });
};

// Install the TRiGS modifier on the gsplat component.
const bindTrigsModifier = (entity: Entity) => {
    const component = entity.gsplat;
    if (!component) {
        throw new Error('trigsbin: entity has no gsplat component');
    }
    component.setWorkBufferModifier({ glsl: glslModifyChunk, wgsl: wgslModifyChunk });
};

const scratchScale = new Vec3();

// Update the time and transform uniforms. Setting a parameter marks the placement render-dirty, so only call when a
// value actually changed.
const setTrigsParams = (entity: Entity, time: number) => {
    const component = entity.gsplat;
    if (!component) {
        return;
    }
    const q = entity.getRotation();
    const s = entity.getWorldTransform().getScale(scratchScale);
    component.setParameter('trigsTime', time);
    component.setParameter('trigsModelRotation', [q.x, q.y, q.z, q.w]);
    component.setParameter('trigsModelScale', [s.x, s.y, s.z]);
};

export { bindTrigsModifier, createTrigsTextures, glslModifyChunk, setTrigsParams, wgslModifyChunk };
