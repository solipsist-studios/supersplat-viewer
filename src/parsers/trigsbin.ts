// .trigsbin: the Open4DGS runtime file (TRiGS motion model "trigs.se3_bezier_v1").
// This file describes what a .trigsbin file *is* and reads one. core/load-trigsbin.ts puts it on the GPU.
// ─────────────────────────────────────────────────────────────────────────────
// Layout (Open4DGS-Core, deliver/make-trigs-timeline.py, runtime format 1, MIT):
//
//   [64-byte header][UTF-8 JSON manifest][zero padding][aligned raw tensor payload]
//
//   header = "<8sIIQQQQQQ" (little-endian): magic "TRIGSBIN", version, header_bytes, manifest_offset,
//            manifest_length, payload_offset, payload_length, alignment, reserved
//
// Each tensor is described in the manifest (clips[i].dynamic.tensors, and the top-level static group) with a
// byteOffset relative to payload_offset, a dtype (float32 or float16 here) and a shape. Quaternions are xyzw,
// scales are natural logs, opacity is a logit, and a clip's times are seconds relative to its own training
// start ("clip-local", the packager's default).
//
// This reader is written against that layout and the TRiGS paper (arXiv 2604.00538); it is the viewer's twin of
// cumuli_core.trigsbin.
// ─────────────────────────────────────────────────────────────────────────────

const TRIGSBIN_MAGIC = 'TRIGSBIN';
const TRIGSBIN_HEADER_BYTES = 64;
const TRIGSBIN_MOTION_MODEL = 'trigs.se3_bezier_v1';

const isTrigsbinFilename = (filename: string): boolean => filename.toLowerCase().endsWith('.trigsbin');

type TrigsArrays = Record<string, Float32Array>;

type TrigsClip = {
    id: string;
    // global frame ranges, [start, end)
    core: [number, number];
    active: [number, number];
    fadeInFrames: number;
    fadeOutFrames: number;
    // seconds added to this clip's tensor times to give global time
    timeOriginS: number;
    arrays: TrigsArrays;
};

// The parts of the manifest this reader uses (the file carries more).
type TensorSpec = { dtype: string; shape: number[]; byteOffset: number };
type ClipManifest = {
    id: string;
    global: {
        coreFrameStart: number;
        coreFrameEndExclusive: number;
        activeFrameStart: number;
        activeFrameEndExclusive: number;
    };
    render?: { fadeInFrames?: number; fadeOutFrames?: number };
    training: { globalFrameStart: number };
    source?: { temporalSelection?: { timesRebasedBySeconds?: number } };
    dynamic?: { tensors?: Record<string, TensorSpec> };
};
type Manifest = {
    format: string;
    semantics?: { dynamicMotionModel?: string; dynamicTimesDomain?: string };
    timeline: { frameRate: number; totalFrames: number };
    clips?: ClipManifest[];
    static?: { tensors?: Record<string, TensorSpec> };
};

type TrigsFile = {
    manifest: Manifest;
    fps: number;
    totalFrames: number;
    // the time of the last frame, in seconds
    duration: number;
    clips: TrigsClip[];
    // splats that never move, shared by every clip (may be empty)
    static: TrigsArrays;
};

// IEEE half to single, for the float16 tensors (Float16Array is not in every browser yet).
const f16ToF32 = (src: Uint16Array): Float32Array => {
    const out = new Float32Array(src.length);
    for (let i = 0; i < src.length; i++) {
        const h = src[i];
        const sign = h & 0x8000 ? -1 : 1;
        const exp = (h >> 10) & 0x1f;
        const frac = h & 0x3ff;
        if (exp === 0) {
            out[i] = sign * frac * 2 ** -24;
        } else if (exp === 31) {
            out[i] = frac ? NaN : sign * Infinity;
        } else {
            out[i] = sign * (1 + frac / 1024) * 2 ** (exp - 15);
        }
    }
    return out;
};

const readTensor = (buffer: ArrayBuffer, payloadOffset: number, spec: TensorSpec): Float32Array => {
    const count = (spec.shape as number[]).reduce((a, b) => a * b, 1);
    const offset = payloadOffset + spec.byteOffset;
    if (spec.dtype === 'float32') {
        // a copy: the payload offsets are aligned in the file, and a copy keeps the result independent of the buffer
        return new Float32Array(buffer.slice(offset, offset + count * 4));
    }
    if (spec.dtype === 'float16') {
        return f16ToF32(new Uint16Array(buffer.slice(offset, offset + count * 2)));
    }
    throw new Error(`trigsbin: unsupported tensor dtype ${spec.dtype}`);
};

const readGroup = (
    buffer: ArrayBuffer,
    payloadOffset: number,
    tensors: Record<string, TensorSpec> | undefined
): TrigsArrays => {
    const out: TrigsArrays = {};
    for (const [name, spec] of Object.entries(tensors ?? {})) {
        if (spec.dtype === 'float32' || spec.dtype === 'float16') {
            out[name] = readTensor(buffer, payloadOffset, spec);
        }
    }
    return out;
};

const parseTrigsbin = (buffer: ArrayBuffer): TrigsFile => {
    if (buffer.byteLength < TRIGSBIN_HEADER_BYTES) {
        throw new Error('trigsbin: the file is shorter than a header');
    }
    const view = new DataView(buffer);
    const magic = new TextDecoder().decode(new Uint8Array(buffer, 0, 8));
    if (magic !== TRIGSBIN_MAGIC) {
        throw new Error(`trigsbin: not a .trigsbin file (magic ${JSON.stringify(magic)})`);
    }
    const version = view.getUint32(8, true);
    if (version !== 1) {
        throw new Error(`trigsbin: unsupported version ${version}`);
    }
    const manifestOffset = Number(view.getBigUint64(16, true));
    const manifestLength = Number(view.getBigUint64(24, true));
    const payloadOffset = Number(view.getBigUint64(32, true));
    const manifest: Manifest = JSON.parse(
        new TextDecoder().decode(new Uint8Array(buffer, manifestOffset, manifestLength))
    );
    if (manifest.format !== 'trigs.runtime') {
        throw new Error(`trigsbin: the manifest says format ${JSON.stringify(manifest.format)}`);
    }
    const model = manifest.semantics?.dynamicMotionModel;
    if (model !== TRIGSBIN_MOTION_MODEL) {
        throw new Error(`trigsbin: unsupported motion model ${JSON.stringify(model)}`);
    }
    const fps = Number(manifest.timeline.frameRate);
    const totalFrames = Number(manifest.timeline.totalFrames);
    const domain = manifest.semantics?.dynamicTimesDomain ?? 'clip-local';
    const clips: TrigsClip[] = (manifest.clips ?? []).map((raw) => {
        const rebased = raw.source?.temporalSelection?.timesRebasedBySeconds;
        const origin = rebased ?? (domain === 'global' ? 0 : raw.training.globalFrameStart / fps);
        return {
            id: raw.id,
            core: [raw.global.coreFrameStart, raw.global.coreFrameEndExclusive],
            active: [raw.global.activeFrameStart, raw.global.activeFrameEndExclusive],
            fadeInFrames: raw.render?.fadeInFrames ?? 0,
            fadeOutFrames: raw.render?.fadeOutFrames ?? 0,
            timeOriginS: Number(origin),
            arrays: readGroup(buffer, payloadOffset, raw.dynamic?.tensors)
        } as TrigsClip;
    });
    const staticArrays = readGroup(buffer, payloadOffset, manifest.static?.tensors);
    return {
        manifest,
        fps,
        totalFrames,
        duration: Math.max(0, (totalFrames - 1) / fps),
        clips,
        static: staticArrays
    };
};

export { TRIGSBIN_MOTION_MODEL, f16ToF32, isTrigsbinFilename, parseTrigsbin };
export type { TrigsArrays, TrigsClip, TrigsFile };
