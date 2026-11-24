/**
 * GLSL (WebGL 2) of the three.js player. Two passes:
 *
 * 1. The work pass draws one fragment a splat into two RGBA32UI targets (splat i at texel
 *    (i mod width, i / width), the SOG layout): it decodes the splat from the SOG v2 textures, moves it
 *    with the motion modifier shared with the PlayCanvas player (ASOG_CHUNK_GLSL, the GPU twin of
 *    core/motion.js) and evaluates its colour with SH for the camera position.
 *      target 0: centre xyz (float bits), colour rgba (unorm8, clamped)
 *      target 1: rotation xyzw (4 halves), scale xyz (3 halves)
 *    It runs when the time or the camera position changed.
 * 2. The splat pass draws an instanced quad a splat in the order the CPU sort gives (splatIndex),
 *    projecting the 3D covariance (EWA, as the PlayCanvas gsplat shader does) every frame.
 *
 * SOG decoding and the projection follow PlayCanvas 2.22 (gsplat chunks) so that both players draw
 * the same image.
 */
import { ASOG_CHUNK_GLSL } from "../playcanvas/chunks.js";

/** Full-screen triangle for the work pass. */
export const WORK_VS = /* glsl */ `
in vec3 position;
void main(void) {
    gl_Position = vec4(position.xy, 0.0, 1.0);
}
`;

export const WORK_FS = /* glsl */ `
precision highp float;
precision highp int;
precision highp usampler2D;
precision highp sampler2D;

uniform sampler2D means_l;
uniform sampler2D means_u;
uniform sampler2D quats;
uniform sampler2D scales;
uniform sampler2D sh0;
uniform sampler2D sogCodebook;   // 256 x 1: r = scales, g = sh0, b = shN codebook
uniform vec3 means_mins;
uniform vec3 means_maxs;
uniform uint uNumSplats;
uniform vec3 uCameraPosition;    // model space

#if SH_BANDS > 0
uniform sampler2D sh_labels;
uniform sampler2D sh_centroids;
#endif

// the PlayCanvas modifier interface the motion chunk is written against: the motion is evaluated in
// model space here, so the model transform is the identity
struct Splat { uint index; ivec2 uv; };
Splat splat;
const mat4 matrix_model = mat4(1.0);
const vec4 model_rotation = vec4(0.0, 0.0, 0.0, 1.0);

${ASOG_CHUNK_GLSL}

layout(location = 0) out highp uvec4 out0;
layout(location = 1) out highp uvec4 out1;

const float SH_C0 = 0.28209479177387814;
const float norm = sqrt(2.0);

float lutScales(int b) { return texelFetch(sogCodebook, ivec2(b, 0), 0).r; }
float lutSh0(int b) { return texelFetch(sogCodebook, ivec2(b, 0), 0).g; }
float lutShN(int b) { return texelFetch(sogCodebook, ivec2(b, 0), 0).b; }

vec3 getCenter() {
    vec3 l = texelFetch(means_l, splat.uv, 0).xyz;
    vec3 u = texelFetch(means_u, splat.uv, 0).xyz;
    vec3 n = (l + u * 256.0) / 257.0;
    vec3 v = mix(means_mins, means_maxs, n);
    return sign(v) * (exp(abs(v)) - 1.0);
}

// x y z w
vec4 getRotation() {
    vec4 q = texelFetch(quats, splat.uv, 0);
    uint qmode = uint(q.w * 255.0 + 0.5) - 252u;
    vec3 abc = (q.xyz - 0.5) * norm;
    float d = sqrt(max(0.0, 1.0 - dot(abc, abc)));
    vec4 wxyz = (qmode == 0u) ? vec4(d, abc) :
        ((qmode == 1u) ? vec4(abc.x, d, abc.yz) :
        ((qmode == 2u) ? vec4(abc.xy, d, abc.z) : vec4(abc, d)));
    return wxyz.yzwx;
}

vec3 getScale() {
    ivec3 i = ivec3(texelFetch(scales, splat.uv, 0).xyz * 255.0 + 0.5);
    return exp(vec3(lutScales(i.x), lutScales(i.y), lutScales(i.z)));
}

vec4 getColor() {
    vec4 c = texelFetch(sh0, splat.uv, 0);
    ivec3 i = ivec3(c.xyz * 255.0 + 0.5);
    return vec4(vec3(0.5) + vec3(lutSh0(i.x), lutSh0(i.y), lutSh0(i.z)) * SH_C0, c.w);
}

#if SH_BANDS == 1
    #define SH_COEFFS 3
#elif SH_BANDS == 2
    #define SH_COEFFS 8
#elif SH_BANDS == 3
    #define SH_COEFFS 15
#endif

#if SH_BANDS > 0
void readSHData(out vec3 sh[SH_COEFFS]) {
    vec4 labels = texelFetch(sh_labels, splat.uv, 0);
    int n = int(labels.x * 255.0 + 0.5) + int(labels.y * 255.0 + 0.5) * 256;
    int u = (n % 64) * SH_COEFFS;
    int v = n / 64;
    for (int i = 0; i < SH_COEFFS; i++) {
        ivec3 idx = ivec3(texelFetch(sh_centroids, ivec2(u + i, v), 0).xyz * 255.0 + 0.5);
        sh[i] = vec3(lutShN(idx.x), lutShN(idx.y), lutShN(idx.z));
    }
}

const float SH_C1 = 0.4886025119029199;
#if SH_BANDS > 1
const float SH_C2_0 = 1.0925484305920792;
const float SH_C2_1 = -1.0925484305920792;
const float SH_C2_2 = 0.31539156525252005;
const float SH_C2_3 = -1.0925484305920792;
const float SH_C2_4 = 0.5462742152960396;
#endif
#if SH_BANDS > 2
const float SH_C3_0 = -0.5900435899266435;
const float SH_C3_1 = 2.890611442640554;
const float SH_C3_2 = -0.4570457994644658;
const float SH_C3_3 = 0.3731763325901154;
const float SH_C3_4 = -0.4570457994644658;
const float SH_C3_5 = 1.445305721320277;
const float SH_C3_6 = -0.5900435899266435;
#endif

vec3 evalSH(in vec3 sh[SH_COEFFS], in vec3 dir) {
    float x = dir.x, y = dir.y, z = dir.z;
    vec3 result = SH_C1 * (-sh[0] * y + sh[1] * z - sh[2] * x);
#if SH_BANDS > 1
    float xx = x * x, yy = y * y, zz = z * z, xy = x * y, yz = y * z, xz = x * z;
    result +=
        sh[3] * (SH_C2_0 * xy) +
        sh[4] * (SH_C2_1 * yz) +
        sh[5] * (SH_C2_2 * (2.0 * zz - xx - yy)) +
        sh[6] * (SH_C2_3 * xz) +
        sh[7] * (SH_C2_4 * (xx - yy));
#endif
#if SH_BANDS > 2
    result +=
        sh[8]  * (SH_C3_0 * y * (3.0 * xx - yy)) +
        sh[9]  * (SH_C3_1 * xy * z) +
        sh[10] * (SH_C3_2 * y * (4.0 * zz - xx - yy)) +
        sh[11] * (SH_C3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy)) +
        sh[12] * (SH_C3_4 * x * (4.0 * zz - xx - yy)) +
        sh[13] * (SH_C3_5 * z * (xx - yy)) +
        sh[14] * (SH_C3_6 * x * (xx - 3.0 * yy));
#endif
    return result;
}
#endif

// packUnorm4x8 is GLSL ES 3.1
uint packColor(vec4 c) {
    uvec4 b = uvec4(clamp(c, 0.0, 1.0) * 255.0 + 0.5);
    return b.x | (b.y << 8) | (b.z << 16) | (b.w << 24);
}

void main(void) {
    ivec2 uv = ivec2(gl_FragCoord.xy);
    uint index = uint(uv.y * textureSize(means_l, 0).x + uv.x);
    if (index >= uNumSplats) {
        out0 = uvec4(0u);
        out1 = uvec4(0u);
        return;
    }
    splat.index = index;
    splat.uv = uv;

    vec3 original = getCenter();
    vec3 center = original;
    modifySplatCenter(center);
    vec4 rotation = getRotation();
    vec3 scale = getScale();
    modifySplatRotationScale(original, center, rotation, scale);
    vec4 color = getColor();
#if SH_BANDS > 0
    vec3 sh[SH_COEFFS];
    readSHData(sh);
    color.rgb += evalSH(sh, normalize(center - uCameraPosition));
#endif
    modifySplatColor(center, color);

    out0 = uvec4(floatBitsToUint(center), packColor(color));
    out1 = uvec4(packHalf2x16(rotation.xy), packHalf2x16(rotation.zw), packHalf2x16(scale.xy), packHalf2x16(vec2(scale.z, clamp(color.a, 0.0, 1.0))));   // opacity as a half: 8 bits drop faint splats
}
`;

export const SPLAT_VS = /* glsl */ `
precision highp float;
precision highp int;
precision highp usampler2D;

in vec3 position;        // quad corner, xy in -1..1
in uint splatIndex;      // instanced: the sorted splat order
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform usampler2D uWork0;
uniform usampler2D uWork1;
uniform vec4 uViewport;  // width, height, 1 / width, 1 / height (pixels)
uniform float uMinPixelSize;
uniform float uAlphaClip;

out mediump vec2 vUv;
out mediump vec4 vColor;

const vec4 discardVec = vec4(0.0, 0.0, 2.0, 1.0);

// R: w x y z
mat3 quatToMat3(vec4 R) {
    vec4 R2 = R + R;
    float X = R2.x * R.w;
    vec4 Y = R2.y * R;
    vec4 Z = R2.z * R;
    float W = R2.w * R.w;
    return mat3(
        1.0 - Z.z - W, Y.z + X, Y.w - Z.x,
        Y.z - X, 1.0 - Y.y - W, Z.w + Y.x,
        Y.w + Z.x, Z.w - Y.x, 1.0 - Y.y - Z.z);
}

vec4 unpackColor(uint c) {
    return vec4((uvec4(c, c >> 8, c >> 16, c >> 24) & 0xffu)) / 255.0;
}

void main(void) {
    int width = textureSize(uWork0, 0).x;
    ivec2 uv = ivec2(int(splatIndex) % width, int(splatIndex) / width);
    uvec4 w0 = texelFetch(uWork0, uv, 0);
    uvec4 w1 = texelFetch(uWork1, uv, 0);
    vec4 color = vec4(unpackColor(w0.w).rgb, unpackHalf2x16(w1.w).y);
    if (color.a <= uAlphaClip) {
        gl_Position = discardVec;
        return;
    }
    vec3 center = uintBitsToFloat(w0.xyz);
    vec4 view = modelViewMatrix * vec4(center, 1.0);
    if (view.z > 0.0) {
        gl_Position = discardVec;
        return;
    }
    vec4 proj = projectionMatrix * view;
    proj.z = clamp(proj.z, -abs(proj.w), abs(proj.w));

    vec4 rotation = vec4(unpackHalf2x16(w1.x), unpackHalf2x16(w1.y));
    vec3 scale = vec3(unpackHalf2x16(w1.z), unpackHalf2x16(w1.w).x);
    mat3 rot = quatToMat3(rotation.wxyz);
    mat3 M = transpose(mat3(scale.x * rot[0], scale.y * rot[1], scale.z * rot[2]));
    mat3 Vrk = mat3(
        dot(M[0], M[0]), dot(M[0], M[1]), dot(M[0], M[2]),
        dot(M[0], M[1]), dot(M[1], M[1]), dot(M[1], M[2]),
        dot(M[0], M[2]), dot(M[1], M[2]), dot(M[2], M[2]));

    float focal = uViewport.x * projectionMatrix[0][0];
    vec3 v = projectionMatrix[3][3] == 1.0 ? vec3(0.0, 0.0, 1.0) : view.xyz;
    float J1 = focal / v.z;
    vec2 J2 = -J1 / v.z * v.xy;
    mat3 J = mat3(J1, 0.0, J2.x, 0.0, J1, J2.y, 0.0, 0.0, 0.0);
    mat3 W = transpose(mat3(modelViewMatrix));
    mat3 T = W * J;
    mat3 cov = transpose(T) * Vrk * T;

    float diagonal1 = cov[0][0] + 0.3;
    float offDiagonal = cov[0][1];
    float diagonal2 = cov[1][1] + 0.3;
    float mid = 0.5 * (diagonal1 + diagonal2);
    float radius = length(vec2((diagonal1 - diagonal2) / 2.0, offDiagonal));
    float lambda1 = mid + radius;
    float lambda2 = max(mid - radius, 0.1);
    float vmin = min(1024.0, min(uViewport.x, uViewport.y));
    float l1 = 2.0 * min(sqrt(2.0 * lambda1), vmin);
    float l2 = 2.0 * min(sqrt(2.0 * lambda2), vmin);
    if (max(l1, l2) < uMinPixelSize) {
        gl_Position = discardVec;
        return;
    }
    vec2 c = proj.ww * uViewport.zw;
    if (any(greaterThan(abs(proj.xy) - vec2(max(l1, l2)) * c, proj.ww))) {
        gl_Position = discardVec;
        return;
    }
    vec2 diagonalVector = normalize(vec2(offDiagonal, lambda1 - diagonal1));
    vec2 v1 = l1 * diagonalVector;
    vec2 v2 = l2 * vec2(diagonalVector.y, -diagonalVector.x);

    // shrink the quad to where the Gaussian falls under the alpha clip
    float clip = min(1.0, sqrt(max(0.0, log(color.a / uAlphaClip))) * 0.5);
    vec2 corner = position.xy * clip;
    gl_Position = proj + vec4((corner.x * v1 + corner.y * v2) * c, 0.0, 0.0);
    vUv = corner;
    vColor = color;
}
`;

export const SPLAT_FS = /* glsl */ `
precision highp float;

in mediump vec2 vUv;
in mediump vec4 vColor;
uniform float uAlphaClip;
out vec4 fragColor;

const float EXP4 = exp(-4.0);
const float INV_EXP4 = 1.0 / (1.0 - EXP4);

void main(void) {
    mediump float A = dot(vUv, vUv);
    if (A > 1.0) discard;
    mediump float alpha = (exp(A * -4.0) - EXP4) * INV_EXP4 * vColor.a;
    if (alpha < uAlphaClip) discard;
    fragColor = vec4(vColor.rgb * alpha, alpha);   // premultiplied
}
`;
