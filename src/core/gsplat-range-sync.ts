import { FloatPacking, Quat, Vec3, type GSplatData } from 'playcanvas';

// Ranged variants of the engine's GSplatResource GPU-data updates
// (updateColorData / updateTransformData / updateSHData). The engine
// methods repack every splat and re-upload whole textures — fine for a
// one-off refresh, but the v3 segment streamer refreshes once per decoded
// ~0.1s segment, and O(numSplats) work per segment freezes weak devices
// for the whole first playback pass. These variants repack only splats
// [a, b) into the textures' persistent CPU copies and upload just the
// covering rows. Packing math is kept byte-identical to the engine's.

const SH_C0 = 0.28209479177387814;

// Upload the texture rows covering splat texel indices [a, b). The CPU
// level copy is already updated in place, so devices without a partial
// write path (WebGPU) fall back to a full upload of that copy.
const uploadTextureRows = (texture: any, elemsPerTexel: number, a: number, b: number) => {
    const w = texture.width as number;
    const level = texture._levels?.[0];
    if (!level) {
        return;
    }
    if (texture.impl?.write) {
        const rowA = Math.floor(a / w);
        const rowB = Math.min(texture.height as number, Math.ceil(b / w));
        const rows = level.subarray(rowA * w * elemsPerTexel, rowB * w * elemsPerTexel);
        (texture.write(0, rowA, w, rowB - rowA, rows) as Promise<unknown>)?.catch(() => { });
    } else {
        texture.upload();
    }
};

const updateColorRange = (resource: any, gsplatData: GSplatData, a: number, b: number) => {
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
    uploadTextureRows(texture, 4, a, b);
};

const updateTransformRange = (resource: any, gsplatData: GSplatData, a: number, b: number) => {
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
    const iter = (gsplatData as any).createIter(p, r, s);
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
    uploadTextureRows(transformA, 4, a, b);
    uploadTextureRows(transformB, 4, a, b);
};

const updateSHRange = (resource: any, gsplatData: GSplatData, a: number, b: number) => {
    const shBands = resource.shBands as number;
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
    const t11 = (1 << 11) - 1;
    const t10 = (1 << 10) - 1;
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
            c[j * 3 + 0] = Math.max(0, Math.min(t11, Math.floor(((c[j * 3 + 0] / max) * 0.5 + 0.5) * t11 + 0.5)));
            c[j * 3 + 1] = Math.max(0, Math.min(t10, Math.floor(((c[j * 3 + 1] / max) * 0.5 + 0.5) * t10 + 0.5)));
            c[j * 3 + 2] = Math.max(0, Math.min(t11, Math.floor(((c[j * 3 + 2] / max) * 0.5 + 0.5) * t11 + 0.5)));
        }
        float32[0] = max;
        sh1to3Data[i * 4 + 0] = uint32[0];
        sh1to3Data[i * 4 + 1] = (c[0] << 21) | (c[1] << 11) | c[2];
        sh1to3Data[i * 4 + 2] = (c[3] << 21) | (c[4] << 11) | c[5];
        sh1to3Data[i * 4 + 3] = (c[6] << 21) | (c[7] << 11) | c[8];
        if (shBands > 1 && sh4to7Data && sh8to11Data) {
            sh4to7Data[i * 4 + 0] = (c[9] << 21) | (c[10] << 11) | c[11];
            sh4to7Data[i * 4 + 1] = (c[12] << 21) | (c[13] << 11) | c[14];
            sh4to7Data[i * 4 + 2] = (c[15] << 21) | (c[16] << 11) | c[17];
            sh4to7Data[i * 4 + 3] = (c[18] << 21) | (c[19] << 11) | c[20];
            if (shBands > 2 && sh12to15Data) {
                sh8to11Data[i * 4 + 0] = (c[21] << 21) | (c[22] << 11) | c[23];
                sh8to11Data[i * 4 + 1] = (c[24] << 21) | (c[25] << 11) | c[26];
                sh8to11Data[i * 4 + 2] = (c[27] << 21) | (c[28] << 11) | c[29];
                sh8to11Data[i * 4 + 3] = (c[30] << 21) | (c[31] << 11) | c[32];
                sh12to15Data[i * 4 + 0] = (c[33] << 21) | (c[34] << 11) | c[35];
                sh12to15Data[i * 4 + 1] = (c[36] << 21) | (c[37] << 11) | c[38];
                sh12to15Data[i * 4 + 2] = (c[39] << 21) | (c[40] << 11) | c[41];
                sh12to15Data[i * 4 + 3] = (c[42] << 21) | (c[43] << 11) | c[44];
            } else {
                sh8to11Data[i] = (c[21] << 21) | (c[22] << 11) | c[23];
            }
        }
    }
    uploadTextureRows(sh1to3Texture, 4, a, b);
    if (shBands > 1) {
        uploadTextureRows(sh4to7Texture, 4, a, b);
        uploadTextureRows(sh8to11Texture, shBands > 2 ? 4 : 1, a, b);
        if (shBands > 2) {
            uploadTextureRows(sh12to15Texture, 4, a, b);
        }
    }
};

// Refresh GPU splat data for splats [a, b) only.
const updateGsplatRangeData = (resource: any, gsplatData: GSplatData, a: number, b: number) => {
    if (b <= a) {
        return;
    }
    updateColorRange(resource, gsplatData, a, b);
    updateTransformRange(resource, gsplatData, a, b);
    if (resource.shBands > 0) {
        updateSHRange(resource, gsplatData, a, b);
    }
};

export { updateGsplatRangeData, uploadTextureRows };
