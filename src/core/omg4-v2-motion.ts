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

vec3 omg4QuatRotate(vec4 q, vec3 v) {
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}
void modifySplatCenter(inout vec3 center) {
    vec4 m = texelFetch(splatMotion, splat.uv, 0);
    center += omg4QuatRotate(omg4ModelRotation, m.xyz) * (omg4Time - m.w);
}
void modifySplatRotationScale(vec3 originalCenter, vec3 modifiedCenter, inout vec4 rotation, inout vec3 scale) {
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

fn omg4QuatRotate(q: vec4f, v: vec3f) -> vec3f {
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}
fn modifySplatCenter(center: ptr<function, vec3f>) {
    let m = textureLoad(splatMotion, splat.uv, 0);
    *center += omg4QuatRotate(uniform.omg4ModelRotation, m.xyz) * (uniform.omg4Time - m.w);
}
fn modifySplatRotationScale(originalCenter: vec3f, modifiedCenter: vec3f, rotation: ptr<function, vec4f>, scale: ptr<function, vec3f>) {
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

// Install the temporal-evaluation modifier on the gsplat component.
const bindOmg4V2Modifier = (entity: Entity) => {
    const component = entity.gsplat as any;
    if (!component) {
        throw new Error('omg4 v2: entity has no gsplat component');
    }
    component.setWorkBufferModifier({ glsl: glslModifyChunk, wgsl: wgslModifyChunk });
};

// Update the time / model-rotation uniforms. Setting a parameter marks the
// placement render-dirty, so only call when a value actually changed.
const setOmg4V2Params = (entity: Entity, time: number) => {
    const component = entity.gsplat as any;
    if (!component) {
        return;
    }
    const q = entity.getRotation();
    component.setParameter('omg4Time', time);
    component.setParameter('omg4ModelRotation', [q.x, q.y, q.z, q.w]);
};

export { attachOmg4V2Motion, bindOmg4V2Modifier, setOmg4V2Params };
