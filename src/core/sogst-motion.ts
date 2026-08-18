import { PIXELFORMAT_R32F, PIXELFORMAT_RGBA32F } from 'playcanvas';
import type { Entity, GSplatResource } from 'playcanvas';

import { uploadTextureRows } from './gsplat-range-sync';
import type { SogstData } from './sogst-data';

// GPU evaluation of the .sogst temporal model on the engine's unified gsplat
// pipeline.
//
// This module attaches two extra per-splat textures to the GSplatResource
// streams, and the engine binds them to the work-buffer material. The
// component's work-buffer modifier hook then evaluates the model at the time
// uniform `sogstTime`:
//
//   center(t) = center + R_model * velocity * (t - t_center)
//   alpha(t)  = alpha * exp(-0.5 * ((t - t_center) / t_sigma)^2)
//
// The two textures hold:
//
//   splatMotion   (RGBA32F): xyz = velocity (units/sec), w = t_center (sec)
//   splatTemporal (R32F)   : r = t_sigma (sec)
//
// The work-buffer pass works on world-space centers. The shader therefore
// rotates the model-space velocity by the entity's world rotation, which
// arrives in the `sogstModelRotation` quaternion uniform. The shader ignores
// scale, because the viewer never scales the splat entity.
//
// The modifier runs in the work-buffer pass, so depth sorting uses the
// motion-displaced centers without any further work.
//
// Segmented content also culls splats outside the active temporal window. The
// file orders splats as [persistent | segment 0 | ...], and each frame the
// animation driver pushes sogstCullRanges = (persistentEnd, dynamicStart,
// dynamicEnd, 0).
//
// A splat outside [0, persistentEnd) ∪ [dynamicStart, dynamicEnd) has a
// temporal opacity near 0 at the current time. The shader collapses it to
// zero scale and alpha, which saves its motion math, its blending and its
// fill.

const glslModifyChunk = /* glsl */ `
uniform highp sampler2D splatMotion;
uniform highp sampler2D splatTemporal;
#ifdef SOGST_ACCEL
uniform highp sampler2D splatAccel;
#endif // SOGST_ACCEL
uniform float sogstTime;
uniform vec4 sogstModelRotation;   // entity world rotation (x, y, z, w)
uniform vec4 sogstCamRot;          // camera world rotation (x, y, z, w)
uniform vec3 sogstCamPos;          // camera world position
#ifdef SOGST_SEG_CULL
uniform vec4 sogstCullRanges;      // x = persistent end, y/z = dynamic [start, end)

bool sogstCulled() {
    float idx = float(splat.uv.y * textureSize(splatMotion, 0).x + splat.uv.x);
    return idx >= sogstCullRanges.x && (idx < sogstCullRanges.y || idx >= sogstCullRanges.z);
}
#endif // SOGST_SEG_CULL

vec3 sogstQuatRotate(vec4 q, vec3 v) {
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}
mat3 sogstQuatToMat(vec4 q) {
    float x = q.x, y = q.y, z = q.z, w = q.w;
    return mat3(
        1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y),
        2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x),
        2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y));
}
vec4 sogstMatToQuat(mat3 m) {
    float r00 = m[0].x, r01 = m[1].x, r02 = m[2].x;
    float r10 = m[0].y, r11 = m[1].y, r12 = m[2].y;
    float r20 = m[0].z, r21 = m[1].z, r22 = m[2].z;
    float tr = r00 + r11 + r22;
    if (tr > 0.0) {
        float s = sqrt(tr + 1.0) * 2.0;
        return vec4((r21 - r12) / s, (r02 - r20) / s, (r10 - r01) / s, 0.25 * s);
    } else if (r00 > r11 && r00 > r22) {
        float s = sqrt(1.0 + r00 - r11 - r22) * 2.0;
        return vec4(0.25 * s, (r01 + r10) / s, (r02 + r20) / s, (r21 - r12) / s);
    } else if (r11 > r22) {
        float s = sqrt(1.0 + r11 - r00 - r22) * 2.0;
        return vec4((r01 + r10) / s, 0.25 * s, (r12 + r21) / s, (r02 - r20) / s);
    }
    float s = sqrt(1.0 + r22 - r00 - r11) * 2.0;
    return vec4((r02 + r20) / s, (r12 + r21) / s, 0.25 * s, (r10 - r01) / s);
}
void modifySplatCenter(inout vec3 center) {
#ifdef SOGST_SEG_CULL
    if (sogstCulled()) {
        return;
    }
#endif // SOGST_SEG_CULL
    vec4 m = texelFetch(splatMotion, splat.uv, 0);
    float sogstDt = sogstTime - m.w;
    vec3 sogstDisp = m.xyz * sogstDt;
#ifdef SOGST_ACCEL
    sogstDisp += texelFetch(splatAccel, splat.uv, 0).xyz * (sogstDt * sogstDt);
#endif // SOGST_ACCEL
    center += sogstQuatRotate(sogstModelRotation, sogstDisp);
}
void modifySplatRotationScale(vec3 originalCenter, vec3 modifiedCenter, inout vec4 rotation, inout vec3 scale) {
#ifdef SOGST_SEG_CULL
    if (sogstCulled()) {
        scale = vec3(0.0);
        return;
    }
#endif // SOGST_SEG_CULL
#ifdef SOGST_COV_COMP
    // Reproduce the screen-space footprint the OMG4 reference rasterizer
    // produced during training, which its FoV-sentinel bug caused. This takes
    // two steps. First inflate by (KX, KY) along the camera's right and up
    // axes. Then replace the view-dependent perspective tilt with the
    // reference rasterizer's degenerate constant one.
    mat3 Rc = sogstQuatToMat(sogstCamRot);
    vec3 vcam = transpose(Rc) * (modifiedCenter - sogstCamPos);
    float invz = 1.0 / min(vcam.z, -1e-4);
    mat3 X = mat3(
        SOGST_KX, 0.0, 0.0,
        0.0, SOGST_KY, 0.0,
        SOGST_TILT * SOGST_KX + vcam.x * invz, SOGST_TILT * SOGST_KY + vcam.y * invz, 1.0);
    mat3 M = Rc * X * transpose(Rc);
    mat3 Rs = sogstQuatToMat(rotation);
    mat3 A = M * mat3(Rs[0] * scale.x, Rs[1] * scale.y, Rs[2] * scale.z);
    mat3 S = A * transpose(A);
    // cyclic Jacobi eigensolve (symmetric 3x3)
    mat3 V = mat3(1.0);
    for (int sweep = 0; sweep < 4; sweep++) {
        for (int k = 0; k < 3; k++) {
            int p = (k == 2) ? 1 : 0;
            int q = (k == 0) ? 1 : 2;
            float spq = S[q][p];
            if (abs(spq) > 1e-12) {
                float tau = (S[q][q] - S[p][p]) / (2.0 * spq);
                float t = (tau == 0.0) ? 1.0 : sign(tau) / (abs(tau) + sqrt(1.0 + tau * tau));
                float c = inversesqrt(1.0 + t * t);
                float s = t * c;
                for (int i = 0; i < 3; i++) {
                    float sp = S[p][i], sq = S[q][i];
                    S[p][i] = c * sp - s * sq;
                    S[q][i] = s * sp + c * sq;
                }
                for (int i = 0; i < 3; i++) {
                    float sp = S[i][p], sq = S[i][q];
                    S[i][p] = c * sp - s * sq;
                    S[i][q] = s * sp + c * sq;
                }
                for (int i = 0; i < 3; i++) {
                    float vp_ = V[p][i], vq = V[q][i];
                    V[p][i] = c * vp_ - s * vq;
                    V[q][i] = s * vp_ + c * vq;
                }
            }
        }
    }
    scale = sqrt(max(vec3(S[0][0], S[1][1], S[2][2]), vec3(1e-12)));
    if (dot(cross(V[0], V[1]), V[2]) < 0.0) {
        V[2] = -V[2];
    }
    rotation = sogstMatToQuat(V);
#endif // SOGST_COV_COMP
}
void modifySplatColor(vec3 center, inout vec4 color) {
#ifdef SOGST_SEG_CULL
    if (sogstCulled()) {
        color.a = 0.0;
        return;
    }
#endif // SOGST_SEG_CULL
    float tCenter = texelFetch(splatMotion, splat.uv, 0).w;
    float tSigma = texelFetch(splatTemporal, splat.uv, 0).r;
    float dt = (sogstTime - tCenter) / max(tSigma, 1e-6);
    color.a *= exp(-0.5 * dt * dt);
}
`;

const wgslModifyChunk = /* wgsl */ `
var splatMotion: texture_2d<f32>;
var splatTemporal: texture_2d<f32>;
#ifdef SOGST_ACCEL
var splatAccel: texture_2d<f32>;
#endif // SOGST_ACCEL
uniform sogstTime: f32;
uniform sogstModelRotation: vec4f;
uniform sogstCamRot: vec4f;
uniform sogstCamPos: vec3f;
#ifdef SOGST_SEG_CULL
uniform sogstCullRanges: vec4f;

fn sogstCulled() -> bool {
    let dims = textureDimensions(splatMotion, 0);
    let idx = f32(splat.uv.y * i32(dims.x) + splat.uv.x);
    return idx >= uniform.sogstCullRanges.x && (idx < uniform.sogstCullRanges.y || idx >= uniform.sogstCullRanges.z);
}
#endif // SOGST_SEG_CULL

fn sogstQuatRotate(q: vec4f, v: vec3f) -> vec3f {
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}
fn sogstQuatToMat(q: vec4f) -> mat3x3f {
    let x = q.x; let y = q.y; let z = q.z; let w = q.w;
    return mat3x3f(
        vec3f(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)),
        vec3f(2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x)),
        vec3f(2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)));
}
fn sogstMatToQuat(m: mat3x3f) -> vec4f {
    let r00 = m[0].x; let r01 = m[1].x; let r02 = m[2].x;
    let r10 = m[0].y; let r11 = m[1].y; let r12 = m[2].y;
    let r20 = m[0].z; let r21 = m[1].z; let r22 = m[2].z;
    let tr = r00 + r11 + r22;
    if (tr > 0.0) {
        let s = sqrt(tr + 1.0) * 2.0;
        return vec4f((r21 - r12) / s, (r02 - r20) / s, (r10 - r01) / s, 0.25 * s);
    } else if (r00 > r11 && r00 > r22) {
        let s = sqrt(1.0 + r00 - r11 - r22) * 2.0;
        return vec4f(0.25 * s, (r01 + r10) / s, (r02 + r20) / s, (r21 - r12) / s);
    } else if (r11 > r22) {
        let s = sqrt(1.0 + r11 - r00 - r22) * 2.0;
        return vec4f((r01 + r10) / s, 0.25 * s, (r12 + r21) / s, (r02 - r20) / s);
    }
    let s = sqrt(1.0 + r22 - r00 - r11) * 2.0;
    return vec4f((r02 + r20) / s, (r12 + r21) / s, 0.25 * s, (r10 - r01) / s);
}
fn modifySplatCenter(center: ptr<function, vec3f>) {
#ifdef SOGST_SEG_CULL
    if (sogstCulled()) {
        return;
    }
#endif // SOGST_SEG_CULL
    let m = textureLoad(splatMotion, splat.uv, 0);
    let sogstDt = uniform.sogstTime - m.w;
    var sogstDisp = m.xyz * sogstDt;
#ifdef SOGST_ACCEL
    sogstDisp += textureLoad(splatAccel, splat.uv, 0).xyz * (sogstDt * sogstDt);
#endif // SOGST_ACCEL
    *center += sogstQuatRotate(uniform.sogstModelRotation, sogstDisp);
}
fn modifySplatRotationScale(originalCenter: vec3f, modifiedCenter: vec3f, rotation: ptr<function, vec4f>, scale: ptr<function, vec3f>) {
#ifdef SOGST_SEG_CULL
    if (sogstCulled()) {
        *scale = vec3f(0.0);
        return;
    }
#endif // SOGST_SEG_CULL
#ifdef SOGST_COV_COMP
    let Rc = sogstQuatToMat(uniform.sogstCamRot);
    let vcam = transpose(Rc) * (modifiedCenter - uniform.sogstCamPos);
    let invz = 1.0 / min(vcam.z, -1e-4);
    let X = mat3x3f(
        vec3f(SOGST_KX, 0.0, 0.0),
        vec3f(0.0, SOGST_KY, 0.0),
        vec3f(SOGST_TILT * SOGST_KX + vcam.x * invz, SOGST_TILT * SOGST_KY + vcam.y * invz, 1.0));
    let M = Rc * X * transpose(Rc);
    let Rs = sogstQuatToMat(*rotation);
    let A = M * mat3x3f(Rs[0] * (*scale).x, Rs[1] * (*scale).y, Rs[2] * (*scale).z);
    var S = A * transpose(A);
    var V = mat3x3f(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0));
    for (var sweep = 0; sweep < 4; sweep++) {
        for (var k = 0; k < 3; k++) {
            let p = select(0, 1, k == 2);
            let q = select(2, 1, k == 0);
            let spq = S[q][p];
            if (abs(spq) > 1e-12) {
                let tau = (S[q][q] - S[p][p]) / (2.0 * spq);
                var t = 1.0;
                if (tau != 0.0) { t = sign(tau) / (abs(tau) + sqrt(1.0 + tau * tau)); }
                let c = inverseSqrt(1.0 + t * t);
                let s = t * c;
                for (var i = 0; i < 3; i++) {
                    let sp = S[p][i]; let sq = S[q][i];
                    S[p][i] = c * sp - s * sq;
                    S[q][i] = s * sp + c * sq;
                }
                for (var i = 0; i < 3; i++) {
                    let sp = S[i][p]; let sq = S[i][q];
                    S[i][p] = c * sp - s * sq;
                    S[i][q] = s * sp + c * sq;
                }
                for (var i = 0; i < 3; i++) {
                    let vp_ = V[p][i]; let vq = V[q][i];
                    V[p][i] = c * vp_ - s * vq;
                    V[q][i] = s * vp_ + c * vq;
                }
            }
        }
    }
    *scale = sqrt(max(vec3f(S[0][0], S[1][1], S[2][2]), vec3f(1e-12)));
    if (dot(cross(V[0], V[1]), V[2]) < 0.0) {
        V[2] = -V[2];
    }
    *rotation = sogstMatToQuat(V);
#endif // SOGST_COV_COMP
}
fn modifySplatColor(center: vec3f, color: ptr<function, vec4f>) {
#ifdef SOGST_SEG_CULL
    if (sogstCulled()) {
        (*color).a = 0.0;
        return;
    }
#endif // SOGST_SEG_CULL
    let tCenter = textureLoad(splatMotion, splat.uv, 0).w;
    let tSigma = textureLoad(splatTemporal, splat.uv, 0).x;
    let dt = (uniform.sogstTime - tCenter) / max(tSigma, 1e-6);
    (*color).a = (*color).a * exp(-0.5 * dt * dt);
}
`;

// Create the per-splat motion/temporal textures and register them with the
// resource's stream collection so they are bound to the work-buffer material.
// Must be called before the entity's gsplat component receives the resource.
const attachSogstMotion = (resource: GSplatResource, data: SogstData) => {
    const streams = resource.streams;
    const dims = streams.textureDimensions;
    const w = dims.x;
    const h = dims.y;
    const N = data.numSplats;

    const motion = new Float32Array(w * h * 4);
    const temporal = new Float32Array(w * h);
    // Unused texels keep t_sigma = 1 so the shader math stays finite (those
    // splats are never drawn: numSplats bounds the draw count).
    temporal.fill(1);

    for (let i = 0; i < N; i++) {
        motion[i * 4 + 0] = data.velocityX[i];
        motion[i * 4 + 1] = data.velocityY[i];
        motion[i * 4 + 2] = data.velocityZ[i];
        motion[i * 4 + 3] = data.tCenter[i];
        temporal[i] = data.tSigma[i];
    }

    const motionTex = streams.createTexture('splatMotion', PIXELFORMAT_RGBA32F, dims, motion);
    const temporalTex = streams.createTexture('splatTemporal', PIXELFORMAT_R32F, dims, temporal);
    streams.textures.set('splatMotion', motionTex);
    streams.textures.set('splatTemporal', temporalTex);

    if (data.accelX && data.accelY && data.accelZ) {
        const accel = new Float32Array(w * h * 4);
        for (let i = 0; i < N; i++) {
            accel[i * 4 + 0] = data.accelX[i];
            accel[i * 4 + 1] = data.accelY[i];
            accel[i * 4 + 2] = data.accelZ[i];
        }
        streams.textures.set('splatAccel', streams.createTexture('splatAccel', PIXELFORMAT_RGBA32F, dims, accel));
    }
};

// Upload the motion/temporal texture rows covering [a, b).
const uploadSogstMotionRows = (resource: GSplatResource, a: number, b: number) => {
    const streams = resource.streams;
    uploadTextureRows(streams.textures.get('splatMotion'), 4, a, b);
    uploadTextureRows(streams.textures.get('splatTemporal'), 1, a, b);
    const accelTex = streams.textures.get('splatAccel');
    if (accelTex) {
        uploadTextureRows(accelTex, 4, a, b);
    }
};

// Rewrite splats [a, b) of the motion and temporal textures from the
// textures' persistent CPU copies, then upload the covering rows only. The
// segment streamer uses this. A full O(numSplats) rewrite for every 0.1s
// segment would stall a weak device.
const syncSogstMotionRange = (resource: GSplatResource, data: SogstData, a: number, b: number, upload = true) => {
    const streams = resource.streams;
    const motionTex = streams.textures.get('splatMotion');
    const temporalTex = streams.textures.get('splatTemporal');
    const motion = motionTex?._levels?.[0] as Float32Array | undefined;
    const temporal = temporalTex?._levels?.[0] as Float32Array | undefined;
    if (!motion || !temporal) {
        return;
    }
    const n = Math.min(b, data.numSplats);
    for (let i = a; i < n; i++) {
        motion[i * 4 + 0] = data.velocityX[i];
        motion[i * 4 + 1] = data.velocityY[i];
        motion[i * 4 + 2] = data.velocityZ[i];
        motion[i * 4 + 3] = data.tCenter[i];
        temporal[i] = data.tSigma[i];
    }
    const accelTex = streams.textures.get('splatAccel');
    const accel = accelTex?._levels?.[0] as Float32Array | undefined;
    if (accel && data.accelX && data.accelY && data.accelZ) {
        for (let i = a; i < n; i++) {
            accel[i * 4 + 0] = data.accelX[i];
            accel[i * 4 + 1] = data.accelY[i];
            accel[i * 4 + 2] = data.accelZ[i];
        }
    }
    if (upload) {
        uploadSogstMotionRows(resource, a, n);
    }
};

// Keep or strip a `#ifdef TAG ... #endif // TAG` template block. Blocks are
// tag-terminated so multiple independent blocks resolve safely.
const resolveBlock = (src: string, tag: string, keep: boolean) => {
    const re = new RegExp(`#ifdef ${tag}\\n([\\s\\S]*?)#endif // ${tag}\\n`, 'g');
    return src.replace(re, keep ? '$1' : '');
};

// The perspective tilt the OMG4 reference rasterizer applied during
// training. It is a constant, not the geometrically correct view-dependent
// term. The reference rasterizer's projection was degenerate, so the training
// fitted every splat against this one shear.
//
// The value is empirical, not derived. It is the shear that best reproduces
// the trained footprints. We measured it at 15.8 dB PSNR against the
// reference render at the test view. A different value returns the stringy
// artifacts at zoomed, off-training viewpoints that this one removes.
const COV_COMP_TILT = 0.7096;

// Resolve the chunk template.
//
// This keeps the SOGST_COV_COMP block only when a cov2d scale is present. It
// inlines the KX, KY and tilt literals there, so the compute path needs no
// extra uniforms.
//
// It keeps the SOGST_SEG_CULL block only for segmented content. A file
// without a segment table therefore compiles the same shader as before.
const buildModifyChunk = (src: string, cov2dScale: [number, number] | null, segmented: boolean, accel = false) => {
    let out = resolveBlock(src, 'SOGST_SEG_CULL', segmented);
    out = resolveBlock(out, 'SOGST_COV_COMP', !!cov2dScale);
    out = resolveBlock(out, 'SOGST_ACCEL', accel);
    if (cov2dScale) {
        out = out
            .replace(/SOGST_KX/g, cov2dScale[0].toFixed(6))
            .replace(/SOGST_KY/g, cov2dScale[1].toFixed(6))
            .replace(/SOGST_TILT/g, COV_COMP_TILT.toFixed(6));
    }
    return out;
};

// Install the temporal-evaluation modifier on the gsplat component.
const bindSogstModifier = (
    entity: Entity,
    cov2dScale: [number, number] | null = null,
    segmented = false,
    accel = false
) => {
    const component = entity.gsplat;
    if (!component) {
        throw new Error('sogst: entity has no gsplat component');
    }
    component.setWorkBufferModifier({
        glsl: buildModifyChunk(glslModifyChunk, cov2dScale, segmented, accel),
        wgsl: buildModifyChunk(wgslModifyChunk, cov2dScale, segmented, accel)
    });
};

// Update the time / rotation uniforms. Setting a parameter marks the
// placement render-dirty, so only call when a value actually changed.
// `cullRanges` = (persistentEnd, dynamicStart, dynamicEnd) splat-index
// bounds for segmented content (see the chunk comment above).
const setSogstParams = (
    entity: Entity,
    time: number,
    camera?: Entity,
    cullRanges?: [number, number, number] | null
) => {
    const component = entity.gsplat;
    if (!component) {
        return;
    }
    const q = entity.getRotation();
    component.setParameter('sogstTime', time);
    component.setParameter('sogstModelRotation', [q.x, q.y, q.z, q.w]);
    const c = camera?.getRotation();
    component.setParameter('sogstCamRot', c ? [c.x, c.y, c.z, c.w] : [0, 0, 0, 1]);
    const p = camera?.getPosition();
    component.setParameter('sogstCamPos', p ? [p.x, p.y, p.z] : [0, 0, 0]);
    if (cullRanges) {
        component.setParameter('sogstCullRanges', [cullRanges[0], cullRanges[1], cullRanges[2], 0]);
    }
};

export { attachSogstMotion, syncSogstMotionRange, uploadSogstMotionRows, bindSogstModifier, setSogstParams };
