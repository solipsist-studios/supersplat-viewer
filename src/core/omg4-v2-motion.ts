import {
    PIXELFORMAT_R32F,
    PIXELFORMAT_RGBA32F,
    type Entity,
    type GSplatResource
} from 'playcanvas';

import type { Omg4V2Data } from '../parsers/omg4';

// GPU evaluation of the .omg4 v2 temporal model on the engine's unified
// gsplat pipeline. Two extra per-splat textures are attached to the
// GSplatResource streams (auto-bound to the work-buffer material), and the
// component's work-buffer modifier hook evaluates, at time uniform
// `omg4Time`:
//
//   center(t) = center + R_model * velocity * (t - t_center)
//   alpha(t)  = alpha * exp(-0.5 * ((t - t_center) / t_sigma)^2)
//
// The work-buffer pass operates on world-space centers, so the model-space
// velocity is rotated by the entity's world rotation, passed in as the
// `omg4ModelRotation` quaternion uniform (the viewer never scales the splat
// entity, so scale is ignored). Because the modifier runs in the work-buffer
// pass, depth sorting uses the motion-displaced centers automatically.
//
// splatMotion   (RGBA32F): xyz = velocity (units/sec), w = t_center (sec)
// splatTemporal (R32F)   : r = t_sigma (sec)

const glslModifyChunk = /* glsl */ `
uniform highp sampler2D splatMotion;
uniform highp sampler2D splatTemporal;
uniform float omg4Time;
uniform vec4 omg4ModelRotation;   // entity world rotation (x, y, z, w)
uniform vec4 omg4CamRot;          // camera world rotation (x, y, z, w)

vec3 omg4QuatRotate(vec4 q, vec3 v) {
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}
mat3 omg4QuatToMat(vec4 q) {
    float x = q.x, y = q.y, z = q.z, w = q.w;
    return mat3(
        1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y),
        2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x),
        2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y));
}
vec4 omg4MatToQuat(mat3 m) {
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
    vec4 m = texelFetch(splatMotion, splat.uv, 0);
    center += omg4QuatRotate(omg4ModelRotation, m.xyz) * (omg4Time - m.w);
}
void modifySplatRotationScale(vec3 originalCenter, vec3 modifiedCenter, inout vec4 rotation, inout vec3 scale) {
#ifdef OMG4_COV_COMP
    // Inflate each splat's covariance by (KX, KY) along the current camera's
    // right/up axes, matching the screen-space footprint the OMG4 reference
    // rasterizer produced during training (FoV-sentinel bug).
    mat3 Rc = omg4QuatToMat(omg4CamRot);
    mat3 M = Rc * mat3(OMG4_KX, 0.0, 0.0, 0.0, OMG4_KY, 0.0, 0.0, 0.0, 1.0) * transpose(Rc);
    mat3 Rs = omg4QuatToMat(rotation);
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
    rotation = omg4MatToQuat(V);
#endif
}
void modifySplatColor(vec3 center, inout vec4 color) {
    float tCenter = texelFetch(splatMotion, splat.uv, 0).w;
    float tSigma = texelFetch(splatTemporal, splat.uv, 0).r;
    float dt = (omg4Time - tCenter) / max(tSigma, 1e-6);
    color.a *= exp(-0.5 * dt * dt);
}
`;

const wgslModifyChunk = /* wgsl */ `
var splatMotion: texture_2d<f32>;
var splatTemporal: texture_2d<f32>;
uniform omg4Time: f32;
uniform omg4ModelRotation: vec4f;
uniform omg4CamRot: vec4f;

fn omg4QuatRotate(q: vec4f, v: vec3f) -> vec3f {
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}
fn omg4QuatToMat(q: vec4f) -> mat3x3f {
    let x = q.x; let y = q.y; let z = q.z; let w = q.w;
    return mat3x3f(
        vec3f(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)),
        vec3f(2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x)),
        vec3f(2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)));
}
fn omg4MatToQuat(m: mat3x3f) -> vec4f {
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
    let m = textureLoad(splatMotion, splat.uv, 0);
    *center += omg4QuatRotate(uniform.omg4ModelRotation, m.xyz) * (uniform.omg4Time - m.w);
}
fn modifySplatRotationScale(originalCenter: vec3f, modifiedCenter: vec3f, rotation: ptr<function, vec4f>, scale: ptr<function, vec3f>) {
#ifdef OMG4_COV_COMP
    let Rc = omg4QuatToMat(uniform.omg4CamRot);
    let M = Rc * mat3x3f(vec3f(OMG4_KX, 0.0, 0.0), vec3f(0.0, OMG4_KY, 0.0), vec3f(0.0, 0.0, 1.0)) * transpose(Rc);
    let Rs = omg4QuatToMat(*rotation);
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
    *rotation = omg4MatToQuat(V);
#endif
}
fn modifySplatColor(center: vec3f, color: ptr<function, vec4f>) {
    let tCenter = textureLoad(splatMotion, splat.uv, 0).w;
    let tSigma = textureLoad(splatTemporal, splat.uv, 0).x;
    let dt = (uniform.omg4Time - tCenter) / max(tSigma, 1e-6);
    (*color).a = (*color).a * exp(-0.5 * dt * dt);
}
`;

// Create the per-splat motion/temporal textures and register them with the
// resource's stream collection so they are bound to the work-buffer material.
// Must be called before the entity's gsplat component receives the resource.
const attachOmg4V2Motion = (resource: GSplatResource, data: Omg4V2Data) => {
    const streams = (resource as any).streams;
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
};

// Resolve the OMG4_COV_COMP template: with a cov2d scale the block is kept
// and the KX/KY literals are inlined (no extra uniforms needed on the
// compute path); without it the block is stripped.
const buildModifyChunk = (src: string, cov2dScale: [number, number] | null) => {
    if (cov2dScale) {
        return src
            .replace('#ifdef OMG4_COV_COMP\n', '')
            .replace('#endif\n', '')
            .replace(/OMG4_KX/g, cov2dScale[0].toFixed(6))
            .replace(/OMG4_KY/g, cov2dScale[1].toFixed(6));
    }
    return src.replace(/#ifdef OMG4_COV_COMP[\s\S]*?#endif\n/, '');
};

// Install the temporal-evaluation modifier on the gsplat component.
const bindOmg4V2Modifier = (entity: Entity, cov2dScale: [number, number] | null = null) => {
    const component = entity.gsplat as any;
    if (!component) {
        throw new Error('omg4 v2: entity has no gsplat component');
    }
    component.setWorkBufferModifier({
        glsl: buildModifyChunk(glslModifyChunk, cov2dScale),
        wgsl: buildModifyChunk(wgslModifyChunk, cov2dScale)
    });
};

// Update the time / rotation uniforms. Setting a parameter marks the
// placement render-dirty, so only call when a value actually changed.
const setOmg4V2Params = (entity: Entity, time: number, camera?: Entity) => {
    const component = entity.gsplat as any;
    if (!component) {
        return;
    }
    const q = entity.getRotation();
    component.setParameter('omg4Time', time);
    component.setParameter('omg4ModelRotation', [q.x, q.y, q.z, q.w]);
    const c = camera?.getRotation();
    component.setParameter('omg4CamRot', c ? [c.x, c.y, c.z, c.w] : [0, 0, 0, 1]);
};

export { attachOmg4V2Motion, bindOmg4V2Modifier, setOmg4V2Params };
