import {
    drawQuadWithShader,
    PIXELFORMAT_R32F,
    PIXELFORMAT_R32U,
    PIXELFORMAT_RGBA16F,
    PIXELFORMAT_RGBA32F,
    PIXELFORMAT_RGBA32U,
    PIXELFORMAT_RGBA8,
    RenderTarget,
    SEMANTIC_POSITION,
    ShaderUtils,
    Texture,
    Vec4
} from 'playcanvas';
import type { GraphicsDevice, Shader } from 'playcanvas';

import type { SogstMeta } from '../parsers/sogst';

import { writeTexelBlock } from './gsplat-range-sync';
import { SH_TEXTURES, shCoefficients, shTextureCount, shWords, WORDS_PER_TEXEL } from './sh-textures';
import type { SogstGroup, SplatPlacement } from './sogst-decoder';
import type { PackMeta, RawTexels } from './sogst-pack-worker';
import type { SogstTarget } from './sogst-target';

// GPU decode of a .sogst group into the existing stream layout.
//
// The CPU path (sogst-pack-worker.ts) decodes WebP, unpacks the SOG and
// temporal attributes, and packs them into the stream textures' texel
// layouts. Here the pack workers only decode the WebP payloads (in parallel,
// one file per job) and compute the centers; the main thread writes the
// texels into reused staging textures, and three fragment passes, limited to
// the destination rows, write the same layouts into the stream textures:
//
//   geometry  splatColor (RGBA16F), transformA (RGBA32U), transformB (RGBA16F)
//   temporal  splatMotion (RGBA32F), splatTemporal (R32F), splatAccel
//   sh        the SH textures (sh-textures.ts)
//
// The CPU sorter (WebGL) still needs the centers on the CPU; the worker that
// decodes the means pair computes them, so nothing is read back from the
// GPU. (WebGPU sorts on the GPU and ignores them.)
//
// Each pass exists in GLSL (WebGL2) and WGSL (WebGPU). Two WebGPU rules
// shape the WGSL side:
//
// - A pass's outputs together may write at most the device's
//   maxColorAttachmentBytesPerSample per pixel: 32 bytes unless the adapter
//   allows more. The temporal pass with accel writes 36 and the SH3 pass 64,
//   so on WebGPU a pass whose outputs exceed the limit is split into
//   several, each writing the outputs that fit (passGroups).
// - Staging textures are rewritten through the queue, which runs at once,
//   while the passes are only recorded until the device submits. The
//   decoder therefore submits after each group's passes, before another
//   group can rewrite the staging textures.
//
// An earlier version decoded with createImageBitmap on the main thread and
// read centers back with readPixels: on Vision Pro those cost ~1.2 s and
// ~0.6 s per million splats of main-thread time, and playback fell to 19 fps.
//
// Not byte-identical to the CPU path: the GPU rounds float to half with its
// own rounding mode, and does the maths in float32.

// codebook rows in the codebook texture
const CB_SCALES = 0;
const CB_SH0 = 1;
const CB_SHN = 2;
const CB_TCENTER = 3;
const CB_TSIGMA = 4;
const CB_ROWS = 5;

const COMMON_GLSL = /* glsl */ `
uniform highp int uDest;
uniform highp int uCount;
uniform highp int uSrcStart;
uniform highp int uSrcWidth;
uniform highp int uDstWidth;
uniform highp sampler2D codebooks;

// The payloads are RGBA8 WebP images, so every channel arrives as a byte
// normalised to [0, 1].
const float BYTE_MAX = 255.0;

// the byte a channel stores
int byteOf(float v) {
    return int(v * BYTE_MAX + 0.5);
}

// source index of this fragment's destination splat, or discard: the pass
// writes the run [uDest, uDest + uCount) from source [uSrcStart, ...)
int srcIndex() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    int s = p.x + p.y * uDstWidth - uDest;
    if (s < 0 || s >= uCount) {
        discard;
    }
    return s + uSrcStart;
}

// Source textures are reused staging textures, often larger than the
// group, so the group's own width comes in as a uniform.
ivec2 srcUV(int s) {
    return ivec2(s % uSrcWidth, s / uSrcWidth);
}

// codebook entry for a byte index
float cb(int row, float v) {
    return texelFetch(codebooks, ivec2(byteOf(v), row), 0).r;
}

vec3 unlog(vec3 v) {
    return sign(v) * (exp(abs(v)) - 1.0);
}

// A split-16 attribute stores a 16-bit fraction of [mins, maxs] (log space)
// as its low byte in one texture and its high byte in another.
const float U16_MAX = 65535.0;

vec3 split16(highp sampler2D lo, highp sampler2D hi, ivec2 uv, vec3 mins, vec3 maxs) {
    vec3 lo8 = floor(texelFetch(lo, uv, 0).xyz * BYTE_MAX + 0.5);
    vec3 hi8 = floor(texelFetch(hi, uv, 0).xyz * BYTE_MAX + 0.5);
    return unlog(mix(mins, maxs, (lo8 + hi8 * 256.0) / U16_MAX));
}
`;

const GEOMETRY_GLSL = /* glsl */ `
${COMMON_GLSL}
uniform highp sampler2D means_l;
uniform highp sampler2D means_u;
uniform highp sampler2D quats;
uniform highp sampler2D scales;
uniform highp sampler2D sh0;
uniform vec3 means_mins;
uniform vec3 means_maxs;

// Smallest-three rotation: quats.webp stores three components, each mapped
// from [-1/sqrt(2), 1/sqrt(2)] to a byte, and drops the largest, which unit
// length recovers. Alpha holds QUAT_MODE_BASE + the dropped component's
// index (0 = w, 1 = x, 2 = y, 3 = z).
const float QUAT_RANGE = 1.41421356237; // sqrt(2)
const int QUAT_MODE_BASE = 252;

const float SH_C0 = 0.28209479177387814;

void main(void) {
    int s = srcIndex();
    ivec2 uv = srcUV(s);
    vec3 pos = split16(means_l, means_u, uv, means_mins, means_maxs);

    vec4 q = texelFetch(quats, uv, 0);
    float a = (q.r - 0.5) * QUAT_RANGE;
    float b = (q.g - 0.5) * QUAT_RANGE;
    float c = (q.b - 0.5) * QUAT_RANGE;
    float d = sqrt(max(0.0, 1.0 - (a * a + b * b + c * c)));
    int mode = byteOf(q.a) - QUAT_MODE_BASE;
    vec4 r = mode == 0 ? vec4(a, b, c, d) : mode == 1 ? vec4(d, b, c, a) : mode == 2 ? vec4(b, d, c, a) : vec4(b, c, d, a);
    float len = length(r);
    r = len == 0.0 ? vec4(0.0, 0.0, 0.0, 1.0) : r / len;
    if (r.w < 0.0) {
        r = -r;
    }

    vec4 sc = texelFetch(scales, uv, 0);
    vec4 col = texelFetch(sh0, uv, 0);

    pcFragColor0 = vec4(0.5 + vec3(cb(${CB_SH0}, col.r), cb(${CB_SH0}, col.g), cb(${CB_SH0}, col.b)) * SH_C0, col.a);
    pcFragColor1 = uvec4(floatBitsToUint(pos), packHalf2x16(r.xy));
    pcFragColor2 = vec4(exp(vec3(cb(${CB_SCALES}, sc.r), cb(${CB_SCALES}, sc.g), cb(${CB_SCALES}, sc.b))), r.z);
}
`;

const temporalGlsl = (accel: boolean) => /* glsl */ `
${COMMON_GLSL}
uniform highp sampler2D motion_l;
uniform highp sampler2D motion_u;
uniform highp sampler2D trbf;
uniform vec3 motion_mins;
uniform vec3 motion_maxs;
${accel ? 'uniform highp sampler2D accel_l;\nuniform highp sampler2D accel_u;\nuniform vec3 accel_mins;\nuniform vec3 accel_maxs;' : ''}

void main(void) {
    int s = srcIndex();
    ivec2 uv = srcUV(s);
    vec4 t = texelFetch(trbf, uv, 0);
    pcFragColor0 = vec4(split16(motion_l, motion_u, uv, motion_mins, motion_maxs), cb(${CB_TCENTER}, t.r));
    pcFragColor1 = vec4(cb(${CB_TSIGMA}, t.g), 0.0, 0.0, 0.0);
    ${accel ? 'pcFragColor2 = vec4(split16(accel_l, accel_u, uv, accel_mins, accel_maxs), 0.0);' : ''}
}
`;

// One statement per SH texture the degree uses, assigning texture t its
// texel of words from the shader's array w (0u past the last word).
const shOutputs = (bands: number, statement: (t: number, lanes: string) => string) => {
    const words = shWords(bands);
    return Array.from({ length: shTextureCount(bands) }, (_, t) => {
        const lanes = Array.from({ length: WORDS_PER_TEXEL }, (_, l) => t * WORDS_PER_TEXEL + l).map((w) =>
            w < words ? `w[${w}]` : '0u'
        );
        return statement(t, lanes.join(', '));
    }).join('\n    ');
};

const shGlsl = (bands: number) => {
    const coeffs = shCoefficients(bands);
    const words = shWords(bands);
    return /* glsl */ `
${COMMON_GLSL}
uniform highp sampler2D sh_labels;
uniform highp sampler2D sh_centroids;

// shN_labels.webp stores each splat's 16-bit centroid label in its R (low)
// and G (high) bytes. shN_centroids.webp holds this many centroids per row,
// each a run of one texel per coefficient.
const int CENTROIDS_PER_ROW = 64;

// The stream layout packs each coefficient's RGB, divided by the splat's
// largest magnitude, into one 11:10:11 word (gsplat-range-sync.ts).
const float SH_R_MAX = 2047.0;
const float SH_G_MAX = 1023.0;
const float SH_B_MAX = 2047.0;
const uint SH_R_SHIFT = 21u;
const uint SH_G_SHIFT = 11u;

// v in [-mx, mx] to an integer in [0, levels]
uint quantize(float v, float mx, float levels) {
    return uint(clamp(floor((v / mx * 0.5 + 0.5) * levels + 0.5), 0.0, levels));
}

uint word(float r, float g, float b, float mx) {
    return (quantize(r, mx, SH_R_MAX) << SH_R_SHIFT) | (quantize(g, mx, SH_G_MAX) << SH_G_SHIFT) | quantize(b, mx, SH_B_MAX);
}

void main(void) {
    int s = srcIndex();
    ivec2 uv = srcUV(s);
    vec4 l = texelFetch(sh_labels, uv, 0);
    int label = byteOf(l.r) + byteOf(l.g) * 256;
    int u = (label % CENTROIDS_PER_ROW) * ${coeffs};
    int v = label / CENTROIDS_PER_ROW;
    float c[${coeffs * 3}];
    for (int k = 0; k < ${coeffs}; k++) {
        vec4 t = texelFetch(sh_centroids, ivec2(u + k, v), 0);
        c[k * 3] = cb(${CB_SHN}, t.r);
        c[k * 3 + 1] = cb(${CB_SHN}, t.g);
        c[k * 3 + 2] = cb(${CB_SHN}, t.b);
    }
    float mx = c[0];
    for (int j = 1; j < ${coeffs * 3}; j++) {
        mx = max(mx, abs(c[j]));
    }
    // the scale word, then one word per coefficient (sh-textures.ts)
    uint w[${words}];
    w[0] = mx == 0.0 ? 0u : floatBitsToUint(mx);
    for (int k = 0; k < ${coeffs}; k++) {
        w[k + 1] = mx == 0.0 ? 0u : word(c[k * 3], c[k * 3 + 1], c[k * 3 + 2], mx);
    }
    ${shOutputs(bands, (t, lanes) => `pcFragColor${t} = uvec4(${lanes});`)}
}
`;
};

const COMMON_WGSL = /* wgsl */ `
uniform uDest: i32;
uniform uCount: i32;
uniform uSrcStart: i32;
uniform uSrcWidth: i32;
uniform uDstWidth: i32;
var codebooks: texture_2d<uff>;

// constants as in COMMON_GLSL
const BYTE_MAX: f32 = 255.0;
const U16_MAX: f32 = 65535.0;

fn byteOf(v: f32) -> i32 {
    return i32(v * BYTE_MAX + 0.5);
}

// Source textures are reused staging textures, often larger than the
// group, so the group's own width comes in as a uniform.
fn srcUV(s: i32) -> vec2i {
    return vec2i(s % uniform.uSrcWidth, s / uniform.uSrcWidth);
}

fn cb(row: i32, v: f32) -> f32 {
    return textureLoad(codebooks, vec2i(byteOf(v), row), 0).r;
}

fn unlog(v: vec3f) -> vec3f {
    return sign(v) * (exp(abs(v)) - 1.0);
}

fn split16(lo: vec3f, hi: vec3f, mins: vec3f, maxs: vec3f) -> vec3f {
    let lo8 = floor(lo * BYTE_MAX + 0.5);
    let hi8 = floor(hi * BYTE_MAX + 0.5);
    return unlog(mix(mins, maxs, (lo8 + hi8 * 256.0) / U16_MAX));
}
`;

// A WGSL pass: module-scope declarations, and a body that computes the
// pass's outputs as o0, o1, ... for the source texel `uv`. The entry point
// writes the outputs listed in `outs`, in order, to color, color1, ...
const wgslPass = (decls: string, body: string, outs: number[]) => /* wgsl */ `
${COMMON_WGSL}
${decls}

@fragment
fn fragmentMain(input: FragmentInput) -> FragmentOutput {
    var output: FragmentOutput;
    // the pass writes the run [uDest, uDest + uCount) from source
    // [uSrcStart, ...)
    let p = vec2i(input.position.xy);
    let local = p.x + p.y * uniform.uDstWidth - uniform.uDest;
    if (local < 0 || local >= uniform.uCount) {
        discard;
        return output;
    }
    let uv = srcUV(local + uniform.uSrcStart);
${body}
${outs.map((o, i) => `    output.color${i > 0 ? i : ''} = o${o};`).join('\n')}
    return output;
}
`;

const GEOMETRY_WGSL_DECLS = /* wgsl */ `
var means_l: texture_2d<f32>;
var means_u: texture_2d<f32>;
var quats: texture_2d<f32>;
var scales: texture_2d<f32>;
var sh0: texture_2d<f32>;
uniform means_mins: vec3f;
uniform means_maxs: vec3f;

// as in GEOMETRY_GLSL
const QUAT_RANGE: f32 = 1.41421356237; // sqrt(2)
const QUAT_MODE_BASE: i32 = 252;
const SH_C0: f32 = 0.28209479177387814;
`;

const GEOMETRY_WGSL_BODY = /* wgsl */ `
    let pos = split16(
        textureLoad(means_l, uv, 0).xyz,
        textureLoad(means_u, uv, 0).xyz,
        uniform.means_mins,
        uniform.means_maxs
    );

    let q = textureLoad(quats, uv, 0);
    let a = (q.r - 0.5) * QUAT_RANGE;
    let b = (q.g - 0.5) * QUAT_RANGE;
    let c = (q.b - 0.5) * QUAT_RANGE;
    let d = sqrt(max(0.0, 1.0 - (a * a + b * b + c * c)));
    let mode = byteOf(q.a) - QUAT_MODE_BASE;
    var r = vec4f(b, c, d, a);
    if (mode == 0) {
        r = vec4f(a, b, c, d);
    } else if (mode == 1) {
        r = vec4f(d, b, c, a);
    } else if (mode == 2) {
        r = vec4f(b, d, c, a);
    }
    let len = length(r);
    if (len == 0.0) {
        r = vec4f(0.0, 0.0, 0.0, 1.0);
    } else {
        r = r / len;
    }
    if (r.w < 0.0) {
        r = -r;
    }

    let sc = textureLoad(scales, uv, 0);
    let col = textureLoad(sh0, uv, 0);

    let o0 = vec4f(0.5 + vec3f(cb(${CB_SH0}, col.r), cb(${CB_SH0}, col.g), cb(${CB_SH0}, col.b)) * SH_C0, col.a);
    let o1 = vec4u(bitcast<vec3u>(pos), pack2x16float(r.xy));
    let o2 = vec4f(exp(vec3f(cb(${CB_SCALES}, sc.r), cb(${CB_SCALES}, sc.g), cb(${CB_SCALES}, sc.b))), r.z);
`;

const temporalWgslDecls = (accel: boolean) => /* wgsl */ `
var motion_l: texture_2d<f32>;
var motion_u: texture_2d<f32>;
var trbf: texture_2d<f32>;
uniform motion_mins: vec3f;
uniform motion_maxs: vec3f;
${accel ? 'var accel_l: texture_2d<f32>;\nvar accel_u: texture_2d<f32>;\nuniform accel_mins: vec3f;\nuniform accel_maxs: vec3f;' : ''}
`;

const temporalWgslBody = (accel: boolean) => /* wgsl */ `
    let t = textureLoad(trbf, uv, 0);
    let motion = split16(
        textureLoad(motion_l, uv, 0).xyz,
        textureLoad(motion_u, uv, 0).xyz,
        uniform.motion_mins,
        uniform.motion_maxs
    );
    let o0 = vec4f(motion, cb(${CB_TCENTER}, t.r));
    let o1 = vec4f(cb(${CB_TSIGMA}, t.g), 0.0, 0.0, 0.0);
    ${
        accel
            ? `let o2 = vec4f(split16(
        textureLoad(accel_l, uv, 0).xyz,
        textureLoad(accel_u, uv, 0).xyz,
        uniform.accel_mins,
        uniform.accel_maxs
    ), 0.0);`
            : ''
    }
`;

const SH_WGSL_DECLS = /* wgsl */ `
var sh_labels: texture_2d<f32>;
var sh_centroids: texture_2d<f32>;

// as in shGlsl
const CENTROIDS_PER_ROW: i32 = 64;
const SH_R_MAX: f32 = 2047.0;
const SH_G_MAX: f32 = 1023.0;
const SH_B_MAX: f32 = 2047.0;
const SH_R_SHIFT: u32 = 21u;
const SH_G_SHIFT: u32 = 11u;

fn quantize(v: f32, mx: f32, levels: f32) -> u32 {
    return u32(clamp(floor((v / mx * 0.5 + 0.5) * levels + 0.5), 0.0, levels));
}

fn word(r: f32, g: f32, b: f32, mx: f32) -> u32 {
    return (quantize(r, mx, SH_R_MAX) << SH_R_SHIFT) | (quantize(g, mx, SH_G_MAX) << SH_G_SHIFT) | quantize(b, mx, SH_B_MAX);
}
`;

const shWgslBody = (bands: number) => {
    const coeffs = shCoefficients(bands);
    return /* wgsl */ `
    let l = textureLoad(sh_labels, uv, 0);
    let label = byteOf(l.r) + byteOf(l.g) * 256;
    let u = (label % CENTROIDS_PER_ROW) * ${coeffs};
    let v = label / CENTROIDS_PER_ROW;
    var c: array<f32, ${coeffs * 3}>;
    for (var k = 0; k < ${coeffs}; k++) {
        let t = textureLoad(sh_centroids, vec2i(u + k, v), 0);
        c[k * 3] = cb(${CB_SHN}, t.r);
        c[k * 3 + 1] = cb(${CB_SHN}, t.g);
        c[k * 3 + 2] = cb(${CB_SHN}, t.b);
    }
    var mx = c[0];
    for (var j = 1; j < ${coeffs * 3}; j++) {
        mx = max(mx, abs(c[j]));
    }
    // the scale word, then one word per coefficient (sh-textures.ts)
    var w: array<u32, ${shWords(bands)}>;
    if (mx != 0.0) {
        w[0] = bitcast<u32>(mx);
        for (var k = 0; k < ${coeffs}; k++) {
            w[k + 1] = word(c[k * 3], c[k * 3 + 1], c[k * 3 + 2], mx);
        }
    }
    ${shOutputs(bands, (t, lanes) => `let o${t} = vec4u(${lanes});`)}
`;
};

// Bytes per pixel each stream format adds to a pass's color attachments
// (WebGPU's render target pixel byte cost).
const ATTACHMENT_BYTES: Record<number, number> = {
    [PIXELFORMAT_RGBA16F]: 8,
    [PIXELFORMAT_RGBA32F]: 16,
    [PIXELFORMAT_R32F]: 4,
    [PIXELFORMAT_RGBA32U]: 16,
    [PIXELFORMAT_R32U]: 4
};

// Split a pass's outputs (by attachment size) into consecutive groups that
// each fit `limit` bytes per pixel. One group when they all fit.
const passGroups = (bytes: number[], limit: number): number[][] => {
    const groups: number[][] = [];
    let used = 0;
    bytes.forEach((size, i) => {
        if (groups.length === 0 || used + size > limit) {
            groups.push([]);
            used = 0;
        }
        groups[groups.length - 1].push(i);
        used += size;
    });
    return groups;
};

// One decode pass: the stream textures it writes, their fragment output
// types, and its source in each shading language.
type DecodePass = {
    key: string;
    names: string[];
    types: string[];
    glsl: string;
    wgsl: (outs: number[]) => string;
};

type DecodeTexels = (
    files: Map<string, Uint8Array>,
    count: number,
    priority: number
) => Promise<{ texels: Record<string, RawTexels>; centers: Float32Array | null }>;

class SogstGpuDecoder {
    private device: GraphicsDevice;
    private meta: SogstMeta;
    private packMeta: PackMeta;
    private target: SogstTarget;
    private codebooks: Texture;
    private centroids: Texture | null = null;
    private centroidsBytes: Uint8Array | null = null;
    private decodeTexels: DecodeTexels;

    // staging textures for source payloads, by bare name; reused while big
    // enough, replaced when a larger group arrives
    private staging = new Map<string, Texture>();

    private shaders = new Map<string, Shader>();
    private targets = new Map<string, RenderTarget>();

    constructor(
        device: GraphicsDevice,
        meta: SogstMeta,
        packMeta: PackMeta,
        target: SogstTarget,
        decodeTexels: DecodeTexels
    ) {
        this.device = device;
        this.decodeTexels = decodeTexels;
        this.meta = meta;
        this.packMeta = packMeta;
        this.target = target;
        const data = new Float32Array(256 * CB_ROWS);
        const put = (row: number, values: number[] | undefined) => values?.forEach((v, i) => (data[row * 256 + i] = v));
        put(CB_SCALES, packMeta.scales.codebook);
        put(CB_SH0, packMeta.sh0.codebook);
        put(CB_SHN, packMeta.shN?.codebook);
        put(CB_TCENTER, packMeta.trbf.center);
        put(CB_TSIGMA, packMeta.trbf.sigma);
        this.codebooks = Texture.createDataTexture2D(device, 'sogst-codebooks', 256, CB_ROWS, PIXELFORMAT_R32F, [data]);
    }

    static supported(device: GraphicsDevice) {
        return device.isWebGL2 || device.isWebGPU;
    }

    // The most bytes per pixel a pass may write across its outputs: WebGPU's
    // limit, or no limit on WebGL2, where every pass already fits.
    private get attachmentBytes(): number {
        if (!this.device.isWebGPU) {
            return Infinity;
        }
        const limits = (this.device as unknown as { wgpu?: { limits?: { maxColorAttachmentBytesPerSample?: number } } })
            .wgpu?.limits;
        return limits?.maxColorAttachmentBytesPerSample ?? 32;
    }

    // Draw a pass over texel rows [rowA, rowB) of the stream textures, split
    // into as many draws as the attachment limit needs.
    private drawPass(pass: DecodePass, rowA: number, rowB: number) {
        const textures = this.target.resource.streams.textures;
        const bytes = pass.names.map((name) => ATTACHMENT_BYTES[textures.get(name)!.format] ?? 16);
        const { width, height } = textures.get(pass.names[0])!;
        // WebGPU flips the rect vertically for a render target, so mirror it
        // first: the draw then covers the same texel rows as on WebGL
        const y = this.device.isWebGPU ? height - rowB : rowA;
        const rect = new Vec4(0, y, width, rowB - rowA);
        for (const outs of passGroups(bytes, this.attachmentBytes)) {
            const key = `${pass.key}:${outs.join('')}`;
            const names = outs.map((i) => pass.names[i]);
            const shader = this.shader(
                key,
                pass.glsl,
                pass.wgsl(outs),
                outs.map((i) => pass.types[i])
            );
            drawQuadWithShader(this.device, this.renderTarget(key, names), shader, rect);
        }
    }

    setCentroids(bytes: Uint8Array) {
        this.centroidsBytes = bytes;
    }

    private shader(key: string, glsl: string, wgsl: string, outputs: string[]): Shader {
        let shader = this.shaders.get(key);
        if (!shader) {
            shader = ShaderUtils.createShader(this.device, {
                uniqueName: `SogstGpuDecode-${key}`,
                attributes: { vertex_position: SEMANTIC_POSITION },
                vertexChunk: 'fullscreenQuadVS',
                fragmentGLSL: glsl,
                fragmentWGSL: wgsl,
                fragmentOutputTypes: outputs
            });
            this.shaders.set(key, shader);
        }
        return shader;
    }

    private renderTarget(key: string, names: string[]): RenderTarget {
        let rt = this.targets.get(key);
        if (!rt) {
            const textures = this.target.resource.streams.textures;
            rt = new RenderTarget({
                colorBuffers: names.map((name) => textures.get(name)!),
                depth: false
            });
            this.targets.set(key, rt);
        }
        return rt;
    }

    // Write decoded texels into the staging texture for `name`.
    private stage(name: string, texels: RawTexels): Texture {
        const width = texels.width;
        const height = texels.data.length / (width * 4);
        let texture = this.staging.get(name);
        if (texture && texture.width >= width && texture.height >= height) {
            writeTexelBlock(texture, width, height, texels.data);
            return texture;
        }
        texture?.destroy();
        const w = Math.max(width, texture?.width ?? 0);
        const h = Math.max(height, texture?.height ?? 0);
        const data = new Uint8Array(w * h * 4);
        for (let row = 0; row < height; row++) {
            data.set(texels.data.subarray(row * width * 4, (row + 1) * width * 4), row * w * 4);
        }
        texture = Texture.createDataTexture2D(this.device, `sogst-staging-${name}`, w, h, PIXELFORMAT_RGBA8, [data]);
        this.staging.set(name, texture);
        return texture;
    }

    private setUniforms(values: Record<string, number | number[] | Texture>) {
        const scope = this.device.scope;
        for (const [name, value] of Object.entries(values)) {
            scope.resolve(name).setValue(value);
        }
    }

    async decodeGroup(group: SogstGroup, files: Map<string, Uint8Array>, placement: SplatPlacement, priority = 0) {
        const m = group.range[1] - group.range[0];
        if (m <= 0 || this.target.destroyed) {
            return;
        }
        const sh = !!this.meta.shN && files.has('shN_labels.webp');
        if (sh && !this.centroids && this.centroidsBytes) {
            const decoded = await this.decodeTexels(
                new Map([['shN_centroids.webp', this.centroidsBytes]]),
                0,
                priority
            );
            const c = decoded.texels['shN_centroids.webp'];
            this.centroids = Texture.createDataTexture2D(
                this.device,
                'sogst-centroids',
                c.width,
                c.data.length / (c.width * 4),
                PIXELFORMAT_RGBA8,
                [c.data]
            );
        }
        const { texels, centers: groupCenters } = await this.decodeTexels(files, m, priority);
        if (this.target.destroyed) {
            return;
        }
        const tex: Record<string, Texture> = {};
        let srcWidth = 1;
        for (const [name, data] of Object.entries(texels)) {
            tex[name.replace('.webp', '')] = this.stage(name, data);
            srcWidth = data.width;
        }

        const width = this.target.resource.streams.textureDimensions.x;
        const accel = !!this.packMeta.accel;
        const bands = this.meta.shN?.bands ?? 0;
        for (const run of placement) {
            const dest = run.dest;
            const rowA = Math.floor(dest / width);
            const rowB = Math.ceil((dest + run.count) / width);
            const common = {
                uDest: dest,
                uCount: run.count,
                uSrcStart: run.src,
                uSrcWidth: srcWidth,
                uDstWidth: width,
                codebooks: this.codebooks
            };
            if (tex.means_l) {
                this.setUniforms({
                    ...common,
                    ...tex,
                    means_mins: this.packMeta.means.mins,
                    means_maxs: this.packMeta.means.maxs
                });
                this.drawPass(
                    {
                        key: 'geometry',
                        names: ['splatColor', 'transformA', 'transformB'],
                        types: ['vec4', 'uvec4', 'vec4'],
                        glsl: GEOMETRY_GLSL,
                        wgsl: (outs) => wgslPass(GEOMETRY_WGSL_DECLS, GEOMETRY_WGSL_BODY, outs)
                    },
                    rowA,
                    rowB
                );
                this.setUniforms({
                    ...common,
                    ...tex,
                    motion_mins: this.packMeta.motion.mins,
                    motion_maxs: this.packMeta.motion.maxs,
                    ...(accel ? { accel_mins: this.packMeta.accel!.mins, accel_maxs: this.packMeta.accel!.maxs } : {})
                });
                this.drawPass(
                    {
                        key: `temporal${accel ? '-accel' : ''}`,
                        names: accel
                            ? ['splatMotion', 'splatTemporal', 'splatAccel']
                            : ['splatMotion', 'splatTemporal'],
                        types: accel ? ['vec4', 'vec4', 'vec4'] : ['vec4', 'vec4'],
                        glsl: temporalGlsl(accel),
                        wgsl: (outs) => wgslPass(temporalWgslDecls(accel), temporalWgslBody(accel), outs)
                    },
                    rowA,
                    rowB
                );
            }
            if (sh && this.centroids) {
                const names = SH_TEXTURES.slice(0, shTextureCount(bands));
                this.setUniforms({ ...common, sh_labels: tex.shN_labels, sh_centroids: this.centroids });
                this.drawPass(
                    {
                        key: `sh${bands}`,
                        names,
                        types: names.map(() => 'uvec4'),
                        glsl: shGlsl(bands),
                        wgsl: (outs) => wgslPass(SH_WGSL_DECLS, shWgslBody(bands), outs)
                    },
                    rowA,
                    rowB
                );
            }
            // centers for the CPU sorter, from the worker
            const centers = this.target.resource.centers as Float32Array | undefined;
            if (groupCenters && centers) {
                centers.set(groupCenters.subarray(run.src * 3, (run.src + run.count) * 3), dest * 3);
            }
        }
        // WebGPU: run the recorded passes now, before another group can
        // rewrite the staging textures they read
        (this.device as unknown as { submit?: () => void }).submit?.();
    }

    destroy() {
        this.targets.forEach((rt) => rt.destroy());
        this.shaders.clear();
        this.codebooks.destroy();
        this.centroids?.destroy();
        this.staging.forEach((texture) => texture.destroy());
        this.staging.clear();
    }
}

export { SogstGpuDecoder };
