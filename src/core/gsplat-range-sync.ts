import { FloatPacking, Quat } from 'playcanvas';
import type { GSplatResource, Texture } from 'playcanvas';

type TypedArray = Uint8Array | Uint16Array | Uint32Array | Float32Array;

// Ranged variants of the engine's GSplatResource GPU-data updates:
// updateColorData, updateTransformData and updateSHData.
//
// The engine methods repack every splat and re-upload whole textures. That is
// acceptable for a single refresh. The segment streamer, however, refreshes
// once per decoded ~0.1s segment, and O(numSplats) work per segment freezes a
// weak device for the whole first playback pass.
//
// These variants pack a run of decoded splats into the textures' persistent
// CPU copies, and upload only the covering rows. The source is a chunk of
// decoded attributes, not a whole-clip GSplatData, and the destination index
// is independent of the source index, so a decoder can pack straight from a
// small scratch buffer into any slot of the resource. The packing math stays
// byte-identical to the engine's.

// Zeroth-order spherical-harmonic basis function, 1 / (2 * sqrt(pi)).
//
// The format stores colour as the DC coefficient. To recover the [0, 1] value
// the packing expects, compute SH_C0 * f_dc + 0.5. The engine's own gsplat
// code and every 3DGS reference implementation use this same constant.
const SH_C0 = 0.28209479177387814;

// Three spherical-harmonic coefficients share one uint32 in the engine's
// 11:10:11 layout. R takes 11 bits at 31..21, G takes 10 bits at 20..11, and
// B takes 11 bits at 10..0.
//
// We do not choose these widths. The shader unpacks the word with
// `unpack111011` in the engine's gsplat chunks. A mismatch here corrupts
// colour silently. It does not raise an error.
const SH_R_BITS = 11;
const SH_G_BITS = 10;
const SH_B_BITS = 11;

// Left shift that moves each field to its bit position. B occupies the low
// bits and needs no shift.
const SH_R_SHIFT = SH_G_BITS + SH_B_BITS;
const SH_G_SHIFT = SH_B_BITS;

// Largest value each field can hold, which is also the quantisation scale.
const SH_R_MAX = (1 << SH_R_BITS) - 1;
const SH_G_MAX = (1 << SH_G_BITS) - 1;
const SH_B_MAX = (1 << SH_B_BITS) - 1;

// Upload the texture rows that cover splat texel indices [a, b). The caller
// has already updated the CPU level copy in place. A device with no partial
// write path, such as WebGPU, therefore uploads that whole copy instead.
const uploadTextureRows = (texture: Texture, elemsPerTexel: number, a: number, b: number) => {
    // Texture._levels has a union type that also covers image sources. A
    // typed array creates every gsplat stream texture, so narrow it to
    // that.
    const level = texture?._levels?.[0] as TypedArray | undefined;
    if (!level) {
        return;
    }
    const w = texture.width;
    if (texture.impl?.write) {
        const rowA = Math.floor(a / w);
        const rowB = Math.min(texture.height, Math.ceil(b / w));
        const rows = level.subarray(rowA * w * elemsPerTexel, rowB * w * elemsPerTexel);
        (texture.write(0, rowA, w, rowB - rowA, rows) as Promise<unknown>)?.catch(() => {
            /* upload races teardown; nothing to recover */
        });
    } else {
        texture.upload();
    }
};

// Decoded per-splat attributes, keyed by GSplatData property name (x, f_dc_0,
// opacity, scale_0, rot_0, f_rest_*, and so on). The values use the same
// conventions as a GSplatData that is not "activated": opacity is a logit and
// scales are logs. The source can be shorter than the resource: a packer
// reads n splats from source index `s` and writes them at resource index `d`.
type SplatSource = Record<string, Float32Array>;

// The CPU copy of a stream texture. Every gsplat stream texture is created
// from a typed array, so a missing copy means the resource was not built the
// way these packers assume, and writing nothing would fail silently.
const levelOf = <T extends TypedArray>(resource: GSplatResource, name: string): T => {
    const level = resource.streams.getTexture(name)?._levels?.[0] as T | undefined;
    if (!level) {
        throw new Error(`sogst: gsplat stream ${name} has no CPU copy`);
    }
    return level;
};

const packColor = (resource: GSplatResource, src: SplatSource, s: number, d: number, n: number) => {
    const level = levelOf<Uint16Array>(resource, 'splatColor');
    const float2Half = FloatPacking.float2Half;
    const cr = src.f_dc_0;
    const cg = src.f_dc_1;
    const cb = src.f_dc_2;
    const ca = src.opacity;
    for (let j = 0; j < n; ++j) {
        const i = s + j;
        const o = (d + j) * 4;
        level[o + 0] = float2Half(cr[i] * SH_C0 + 0.5);
        level[o + 1] = float2Half(cg[i] * SH_C0 + 0.5);
        level[o + 2] = float2Half(cb[i] * SH_C0 + 0.5);
        level[o + 3] = float2Half(1 / (1 + Math.exp(-ca[i])));
    }
};

// The same values the engine's GSplatData iterator produces: rotation as
// (rot_1, rot_2, rot_3, rot_0) and scale as exp(scale_*).
const packTransform = (resource: GSplatResource, src: SplatSource, s: number, d: number, n: number) => {
    const dataA = levelOf<Uint32Array>(resource, 'transformA');
    const dataB = levelOf<Uint16Array>(resource, 'transformB');
    const float2Half = FloatPacking.float2Half;
    const dataAFloat32 = new Float32Array(dataA.buffer);
    const r = new Quat();
    for (let j = 0; j < n; j++) {
        const i = s + j;
        const o = (d + j) * 4;
        r.set(src.rot_1[i], src.rot_2[i], src.rot_3[i], src.rot_0[i]);
        r.normalize();
        if (r.w < 0) {
            r.mulScalar(-1);
        }
        dataAFloat32[o + 0] = src.x[i];
        dataAFloat32[o + 1] = src.y[i];
        dataAFloat32[o + 2] = src.z[i];
        dataA[o + 3] = float2Half(r.x) | (float2Half(r.y) << 16);
        dataB[o + 0] = float2Half(Math.exp(src.scale_0[i]));
        dataB[o + 1] = float2Half(Math.exp(src.scale_1[i]));
        dataB[o + 2] = float2Half(Math.exp(src.scale_2[i]));
        dataB[o + 3] = float2Half(r.z);
    }
};

// The resource's centers copy, which the CPU sorter (WebGL) reads.
const packCenters = (resource: GSplatResource, src: SplatSource, s: number, d: number, n: number) => {
    const centers = resource.centers as Float32Array | undefined;
    if (!centers) {
        return;
    }
    for (let j = 0; j < n; j++) {
        const i = s + j;
        const o = (d + j) * 3;
        centers[o + 0] = src.x[i];
        centers[o + 1] = src.y[i];
        centers[o + 2] = src.z[i];
    }
};

const uploadSHRows = (resource: GSplatResource, a: number, b: number) => {
    const shBands = resource.shBands;
    if (shBands <= 0) {
        return;
    }
    uploadTextureRows(resource.streams.getTexture('splatSH_1to3'), 4, a, b);
    if (shBands > 1) {
        uploadTextureRows(resource.streams.getTexture('splatSH_4to7'), 4, a, b);
        uploadTextureRows(resource.streams.getTexture('splatSH_8to11'), shBands > 2 ? 4 : 1, a, b);
        if (shBands > 2) {
            uploadTextureRows(resource.streams.getTexture('splatSH_12to15'), 4, a, b);
        }
    }
};

const packSH = (resource: GSplatResource, src: SplatSource, s: number, d: number, n: number) => {
    const shBands = resource.shBands;
    if (shBands <= 0) {
        return;
    }
    const sh1to3Data = levelOf<Uint32Array>(resource, 'splatSH_1to3');
    const sh4to7Data = shBands > 1 ? levelOf<Uint32Array>(resource, 'splatSH_4to7') : undefined;
    const sh8to11Data = shBands > 1 ? levelOf<Uint32Array>(resource, 'splatSH_8to11') : undefined;
    const sh12to15Data = shBands > 2 ? levelOf<Uint32Array>(resource, 'splatSH_12to15') : undefined;
    const numCoeffs = ({ 1: 3, 2: 8, 3: 15 } as Record<number, number>)[shBands];
    const rest: Float32Array[] = [];
    for (let i = 0; i < numCoeffs * 3; ++i) {
        rest.push(src[`f_rest_${i}`]);
    }
    const float32 = new Float32Array(1);
    const uint32 = new Uint32Array(float32.buffer);
    const c = new Array(numCoeffs * 3).fill(0);
    for (let jj = 0; jj < n; ++jj) {
        const i = s + jj;
        const t = d + jj;
        for (let j = 0; j < numCoeffs; ++j) {
            c[j * 3] = rest[j][i];
            c[j * 3 + 1] = rest[j + numCoeffs][i];
            c[j * 3 + 2] = rest[j + numCoeffs * 2][i];
        }
        let max = c[0];
        for (let j = 1; j < numCoeffs * 3; ++j) {
            max = Math.max(max, Math.abs(c[j]));
        }
        if (max === 0) {
            // The engine skips such a splat and leaves the texel at its
            // initial zero. A destination slot can hold an earlier splat's
            // SH here, so clear it instead. A zero scale word decodes every
            // coefficient to zero.
            sh1to3Data.fill(0, t * 4, t * 4 + 4);
            sh4to7Data?.fill(0, t * 4, t * 4 + 4);
            if (shBands > 2) {
                sh8to11Data?.fill(0, t * 4, t * 4 + 4);
                sh12to15Data?.fill(0, t * 4, t * 4 + 4);
            } else {
                sh8to11Data?.fill(0, t, t + 1);
            }
            continue;
        }
        for (let j = 0; j < numCoeffs; ++j) {
            c[j * 3 + 0] = Math.max(
                0,
                Math.min(SH_R_MAX, Math.floor(((c[j * 3 + 0] / max) * 0.5 + 0.5) * SH_R_MAX + 0.5))
            );
            c[j * 3 + 1] = Math.max(
                0,
                Math.min(SH_G_MAX, Math.floor(((c[j * 3 + 1] / max) * 0.5 + 0.5) * SH_G_MAX + 0.5))
            );
            c[j * 3 + 2] = Math.max(
                0,
                Math.min(SH_B_MAX, Math.floor(((c[j * 3 + 2] / max) * 0.5 + 0.5) * SH_B_MAX + 0.5))
            );
        }
        float32[0] = max;
        sh1to3Data[t * 4 + 0] = uint32[0];
        sh1to3Data[t * 4 + 1] = (c[0] << SH_R_SHIFT) | (c[1] << SH_G_SHIFT) | c[2];
        sh1to3Data[t * 4 + 2] = (c[3] << SH_R_SHIFT) | (c[4] << SH_G_SHIFT) | c[5];
        sh1to3Data[t * 4 + 3] = (c[6] << SH_R_SHIFT) | (c[7] << SH_G_SHIFT) | c[8];
        if (shBands > 1 && sh4to7Data && sh8to11Data) {
            sh4to7Data[t * 4 + 0] = (c[9] << SH_R_SHIFT) | (c[10] << SH_G_SHIFT) | c[11];
            sh4to7Data[t * 4 + 1] = (c[12] << SH_R_SHIFT) | (c[13] << SH_G_SHIFT) | c[14];
            sh4to7Data[t * 4 + 2] = (c[15] << SH_R_SHIFT) | (c[16] << SH_G_SHIFT) | c[17];
            sh4to7Data[t * 4 + 3] = (c[18] << SH_R_SHIFT) | (c[19] << SH_G_SHIFT) | c[20];
            if (shBands > 2 && sh12to15Data) {
                sh8to11Data[t * 4 + 0] = (c[21] << SH_R_SHIFT) | (c[22] << SH_G_SHIFT) | c[23];
                sh8to11Data[t * 4 + 1] = (c[24] << SH_R_SHIFT) | (c[25] << SH_G_SHIFT) | c[26];
                sh8to11Data[t * 4 + 2] = (c[27] << SH_R_SHIFT) | (c[28] << SH_G_SHIFT) | c[29];
                sh8to11Data[t * 4 + 3] = (c[30] << SH_R_SHIFT) | (c[31] << SH_G_SHIFT) | c[32];
                sh12to15Data[t * 4 + 0] = (c[33] << SH_R_SHIFT) | (c[34] << SH_G_SHIFT) | c[35];
                sh12to15Data[t * 4 + 1] = (c[36] << SH_R_SHIFT) | (c[37] << SH_G_SHIFT) | c[38];
                sh12to15Data[t * 4 + 2] = (c[39] << SH_R_SHIFT) | (c[40] << SH_G_SHIFT) | c[41];
                sh12to15Data[t * 4 + 3] = (c[42] << SH_R_SHIFT) | (c[43] << SH_G_SHIFT) | c[44];
            } else {
                sh8to11Data[t] = (c[21] << SH_R_SHIFT) | (c[22] << SH_G_SHIFT) | c[23];
            }
        }
    }
};

// Pack n decoded splats from source index s into the resource at index d:
// colour, transform, centers and, with `withSH`, SH. This writes
// the CPU copies only; uploadGsplatRows sends the covering rows afterwards,
// so a caller that packs many small chunks pays the upload once.
const packGsplatRange = (
    resource: GSplatResource,
    src: SplatSource,
    s: number,
    d: number,
    n: number,
    withSH: boolean
) => {
    if (n <= 0) {
        return;
    }
    packColor(resource, src, s, d, n);
    packTransform(resource, src, s, d, n);
    packCenters(resource, src, s, d, n);
    if (withSH && resource.shBands > 0) {
        packSH(resource, src, s, d, n);
    }
};

// SH-only pack, for deferred SH labels that arrive after a range's geometry
// is already live.
const packGsplatSHRange = (resource: GSplatResource, src: SplatSource, s: number, d: number, n: number) => {
    if (n <= 0 || resource.shBands <= 0) {
        return;
    }
    packSH(resource, src, s, d, n);
};

// Upload the texture rows covering [a, b) for the streams repacked above.
const uploadGsplatRows = (resource: GSplatResource, a: number, b: number, shOnly = false) => {
    if (b <= a) {
        return;
    }
    if (!shOnly) {
        uploadTextureRows(resource.streams.getTexture('splatColor'), 4, a, b);
        uploadTextureRows(resource.streams.getTexture('transformA'), 4, a, b);
        uploadTextureRows(resource.streams.getTexture('transformB'), 4, a, b);
    }
    uploadSHRows(resource, a, b);
};

export { packGsplatRange, packGsplatSHRange, uploadGsplatRows, uploadTextureRows };
export type { SplatSource };
