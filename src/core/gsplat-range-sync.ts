import { FloatPacking, Quat, Vec3 } from 'playcanvas';
import type { GSplatData, GSplatResource, Texture } from 'playcanvas';

type TypedArray = Uint8Array | Uint16Array | Uint32Array | Float32Array;

// Ranged variants of the engine's GSplatResource GPU-data updates:
// updateColorData, updateTransformData and updateSHData.
//
// The engine methods repack every splat and re-upload whole textures. That is
// acceptable for a single refresh. The segment streamer, however, refreshes
// once per decoded ~0.1s segment, and O(numSplats) work per segment freezes a
// weak device for the whole first playback pass.
//
// These variants repack splats [a, b) only, into the textures' persistent CPU
// copies, and they upload the covering rows only. The packing math stays
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

const updateColorRange = (resource: GSplatResource, gsplatData: GSplatData, a: number, b: number, upload: boolean) => {
    const texture = resource.streams.getTexture('splatColor');
    const level = texture?._levels?.[0] as Uint16Array | undefined;
    if (!level) {
        resource.updateColorData(gsplatData);
        return;
    }
    const float2Half = FloatPacking.float2Half;
    const cr = gsplatData.getProp('f_dc_0') as Float32Array;
    const cg = gsplatData.getProp('f_dc_1') as Float32Array;
    const cb = gsplatData.getProp('f_dc_2') as Float32Array;
    const ca = gsplatData.getProp('opacity') as Float32Array;
    for (let i = a; i < b; ++i) {
        level[i * 4 + 0] = float2Half(cr[i] * SH_C0 + 0.5);
        level[i * 4 + 1] = float2Half(cg[i] * SH_C0 + 0.5);
        level[i * 4 + 2] = float2Half(cb[i] * SH_C0 + 0.5);
        level[i * 4 + 3] = float2Half(1 / (1 + Math.exp(-ca[i])));
    }
    if (upload) {
        uploadTextureRows(texture, 4, a, b);
    }
};

const updateTransformRange = (
    resource: GSplatResource,
    gsplatData: GSplatData,
    a: number,
    b: number,
    upload: boolean
) => {
    const transformA = resource.streams.getTexture('transformA');
    const transformB = resource.streams.getTexture('transformB');
    const dataA = transformA?._levels?.[0] as Uint32Array | undefined;
    const dataB = transformB?._levels?.[0] as Uint16Array | undefined;
    if (!dataA || !dataB) {
        resource.updateTransformData(gsplatData);
        return;
    }
    const float2Half = FloatPacking.float2Half;
    const dataAFloat32 = new Float32Array(dataA.buffer);
    const p = new Vec3();
    const r = new Quat();
    const s = new Vec3();
    const iter = gsplatData.createIter(p, r, s);
    for (let i = a; i < b; i++) {
        iter.read(i);
        r.normalize();
        if (r.w < 0) {
            r.mulScalar(-1);
        }
        dataAFloat32[i * 4 + 0] = p.x;
        dataAFloat32[i * 4 + 1] = p.y;
        dataAFloat32[i * 4 + 2] = p.z;
        dataA[i * 4 + 3] = float2Half(r.x) | (float2Half(r.y) << 16);
        dataB[i * 4 + 0] = float2Half(s.x);
        dataB[i * 4 + 1] = float2Half(s.y);
        dataB[i * 4 + 2] = float2Half(s.z);
        dataB[i * 4 + 3] = float2Half(r.z);
    }
    if (upload) {
        uploadTextureRows(transformA, 4, a, b);
        uploadTextureRows(transformB, 4, a, b);
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

const updateSHRange = (resource: GSplatResource, gsplatData: GSplatData, a: number, b: number, upload: boolean) => {
    const shBands = resource.shBands;
    const sh1to3Texture = resource.streams.getTexture('splatSH_1to3');
    const sh4to7Texture = resource.streams.getTexture('splatSH_4to7');
    const sh8to11Texture = resource.streams.getTexture('splatSH_8to11');
    const sh12to15Texture = resource.streams.getTexture('splatSH_12to15');
    const sh1to3Data = sh1to3Texture?._levels?.[0] as Uint32Array | undefined;
    if (!sh1to3Data) {
        resource.updateSHData(gsplatData);
        return;
    }
    const sh4to7Data = sh4to7Texture?._levels?.[0] as Uint32Array | undefined;
    const sh8to11Data = sh8to11Texture?._levels?.[0] as Uint32Array | undefined;
    const sh12to15Data = sh12to15Texture?._levels?.[0] as Uint32Array | undefined;
    const numCoeffs = ({ 1: 3, 2: 8, 3: 15 } as Record<number, number>)[shBands];
    const src: Float32Array[] = [];
    for (let i = 0; i < numCoeffs * 3; ++i) {
        src.push(gsplatData.getProp(`f_rest_${i}`) as Float32Array);
    }
    const float32 = new Float32Array(1);
    const uint32 = new Uint32Array(float32.buffer);
    const c = new Array(numCoeffs * 3).fill(0);
    for (let i = a; i < b; ++i) {
        for (let j = 0; j < numCoeffs; ++j) {
            c[j * 3] = src[j][i];
            c[j * 3 + 1] = src[j + numCoeffs][i];
            c[j * 3 + 2] = src[j + numCoeffs * 2][i];
        }
        let max = c[0];
        for (let j = 1; j < numCoeffs * 3; ++j) {
            max = Math.max(max, Math.abs(c[j]));
        }
        if (max === 0) {
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
        sh1to3Data[i * 4 + 0] = uint32[0];
        sh1to3Data[i * 4 + 1] = (c[0] << SH_R_SHIFT) | (c[1] << SH_G_SHIFT) | c[2];
        sh1to3Data[i * 4 + 2] = (c[3] << SH_R_SHIFT) | (c[4] << SH_G_SHIFT) | c[5];
        sh1to3Data[i * 4 + 3] = (c[6] << SH_R_SHIFT) | (c[7] << SH_G_SHIFT) | c[8];
        if (shBands > 1 && sh4to7Data && sh8to11Data) {
            sh4to7Data[i * 4 + 0] = (c[9] << SH_R_SHIFT) | (c[10] << SH_G_SHIFT) | c[11];
            sh4to7Data[i * 4 + 1] = (c[12] << SH_R_SHIFT) | (c[13] << SH_G_SHIFT) | c[14];
            sh4to7Data[i * 4 + 2] = (c[15] << SH_R_SHIFT) | (c[16] << SH_G_SHIFT) | c[17];
            sh4to7Data[i * 4 + 3] = (c[18] << SH_R_SHIFT) | (c[19] << SH_G_SHIFT) | c[20];
            if (shBands > 2 && sh12to15Data) {
                sh8to11Data[i * 4 + 0] = (c[21] << SH_R_SHIFT) | (c[22] << SH_G_SHIFT) | c[23];
                sh8to11Data[i * 4 + 1] = (c[24] << SH_R_SHIFT) | (c[25] << SH_G_SHIFT) | c[26];
                sh8to11Data[i * 4 + 2] = (c[27] << SH_R_SHIFT) | (c[28] << SH_G_SHIFT) | c[29];
                sh8to11Data[i * 4 + 3] = (c[30] << SH_R_SHIFT) | (c[31] << SH_G_SHIFT) | c[32];
                sh12to15Data[i * 4 + 0] = (c[33] << SH_R_SHIFT) | (c[34] << SH_G_SHIFT) | c[35];
                sh12to15Data[i * 4 + 1] = (c[36] << SH_R_SHIFT) | (c[37] << SH_G_SHIFT) | c[38];
                sh12to15Data[i * 4 + 2] = (c[39] << SH_R_SHIFT) | (c[40] << SH_G_SHIFT) | c[41];
                sh12to15Data[i * 4 + 3] = (c[42] << SH_R_SHIFT) | (c[43] << SH_G_SHIFT) | c[44];
            } else {
                sh8to11Data[i] = (c[21] << SH_R_SHIFT) | (c[22] << SH_G_SHIFT) | c[23];
            }
        }
    }
    if (upload) {
        uploadSHRows(resource, a, b);
    }
};

// Repack GPU splat data for splats [a, b). With upload=false this writes the
// CPU level copies only.
//
// A caller that cuts a large range into many small repack chunks should pass
// false, then make one uploadGsplatRows call over the whole range. It then
// pays the upload cost once instead of once per chunk.
const updateGsplatRangeData = (
    resource: GSplatResource,
    gsplatData: GSplatData,
    a: number,
    b: number,
    upload = true
) => {
    if (b <= a) {
        return;
    }
    updateColorRange(resource, gsplatData, a, b, upload);
    updateTransformRange(resource, gsplatData, a, b, upload);
    if (resource.shBands > 0) {
        updateSHRange(resource, gsplatData, a, b, upload);
    }
};

// SH-only repack for splats [a, b) — used when deferred SH coefficients
// arrive after a range's geometry is already live.
const updateGsplatSHRange = (resource: GSplatResource, gsplatData: GSplatData, a: number, b: number, upload = true) => {
    if (b <= a || resource.shBands <= 0) {
        return;
    }
    updateSHRange(resource, gsplatData, a, b, upload);
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

export { updateGsplatRangeData, updateGsplatSHRange, uploadGsplatRows, uploadTextureRows };
