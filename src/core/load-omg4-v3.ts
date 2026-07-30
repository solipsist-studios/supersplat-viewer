import {
    GSplatData,
    GSplatSogData,
    PIXELFORMAT_RGBA8,
    Texture,
    type AppBase
} from 'playcanvas';

import type { Omg4Segments } from '../parsers/omg4';

// .omg4 version 3: SOG-compressed temporal splats.
//
// The file is a ZIP archive (identified by the leading "PK\x03\x04" magic
// instead of the OMG4 magic word) holding a meta.json plus lossless-webp
// attribute textures. Static attributes follow the PlayCanvas SOG v2
// conventions exactly — means_l/u (16-bit split, log-transformed), quats
// (smallest-three), scales/sh0 (256-entry codebook indices, opacity in
// sh0's alpha), optional VQ'd higher-order SH (shN_centroids/shN_labels) —
// so the engine's own GSplatSogData decoder reconstructs them unmodified.
// Two additional textures carry the temporal model:
//
//   motion_l/motion_u : per-axis 16-bit split of velocity, same
//                       sign(x)*ln(1+|x|) transform + mins/maxs as means
//   trbf              : R = index into trbf.center codebook (t_center, s),
//                       G = index into trbf.sigma codebook (t_sigma, s)
//
// The decoded result is structurally identical to Omg4V2Data, so the whole
// v2 playback path (motion textures, work-buffer modifier, animation
// driver) is reused as-is.

const ZIP_LOCAL_MAGIC = 0x04034b50;      // "PK\x03\x04"
const ZIP_EOCD_MAGIC = 0x06054b50;
const ZIP_CDR_MAGIC = 0x02014b50;

interface ZipEntry {
    filename: string;
    deflated: boolean;
    data: Uint8Array;
}

// Minimal ZIP reader (central-directory walk; stored + deflate entries).
// Our encoder always writes stored entries, deflate is handled for
// robustness against re-zipped files.
const parseZipEntries = (buffer: ArrayBuffer): ZipEntry[] => {
    const view = new DataView(buffer);
    const u16 = (o: number) => view.getUint16(o, true);
    const u32 = (o: number) => view.getUint32(o, true);

    // EOCD: scan back over a possible trailing comment (max 64KB + 22)
    let eocd = -1;
    const scanEnd = Math.max(0, buffer.byteLength - 22 - 65535);
    for (let o = buffer.byteLength - 22; o >= scanEnd; o--) {
        if (u32(o) === ZIP_EOCD_MAGIC) {
            eocd = o;
            break;
        }
    }
    if (eocd < 0) {
        throw new Error('omg4 v3: invalid zip (no end-of-central-directory)');
    }

    const numFiles = u16(eocd + 8);
    let offset = u32(eocd + 16);
    const entries: ZipEntry[] = [];
    for (let i = 0; i < numFiles; i++) {
        if (u32(offset) !== ZIP_CDR_MAGIC) {
            throw new Error('omg4 v3: invalid zip (bad central-directory record)');
        }
        const compression = u16(offset + 10);
        const compressedSize = u32(offset + 20);
        const filenameLength = u16(offset + 28);
        const extraLength = u16(offset + 30);
        const commentLength = u16(offset + 32);
        const lfhOffset = u32(offset + 42);
        const filename = new TextDecoder().decode(new Uint8Array(buffer, offset + 46, filenameLength));

        if (u32(lfhOffset) !== ZIP_LOCAL_MAGIC) {
            throw new Error('omg4 v3: invalid zip (bad local file header)');
        }
        const dataOffset = lfhOffset + 30 + u16(lfhOffset + 26) + u16(lfhOffset + 28);

        if (compression !== 0 && compression !== 8) {
            throw new Error(`omg4 v3: unsupported zip compression method ${compression}`);
        }
        entries.push({
            filename,
            deflated: compression === 8,
            data: new Uint8Array(buffer, dataOffset, compressedSize)
        });
        offset += 46 + filenameLength + extraLength + commentLength;
    }
    return entries;
};

const inflateRaw = async (compressed: Uint8Array): Promise<Uint8Array> => {
    const stream = new Blob([compressed as unknown as ArrayBuffer]).stream()
    .pipeThrough(new DecompressionStream('deflate-raw'));
    return new Uint8Array(await new Response(stream).arrayBuffer());
};

// Decode webp bytes into an engine texture. Decoding must not premultiply:
// several textures carry codebook indices next to a data-bearing alpha
// channel (sh0 stores opacity there), and premultiplication would corrupt
// the indices of low-alpha splats.
const decodeTexture = async (app: AppBase, bytes: Uint8Array, name: string): Promise<Texture> => {
    const bitmap = await createImageBitmap(new Blob([bytes as unknown as ArrayBuffer], { type: 'image/webp' }), {
        premultiplyAlpha: 'none',
        colorSpaceConversion: 'none'
    });
    const texture = new Texture(app.graphicsDevice, {
        name: `omg4v3-${name}`,
        width: bitmap.width,
        height: bitmap.height,
        format: PIXELFORMAT_RGBA8,
        mipmaps: false
    });
    // The engine's setSource typing predates ImageBitmap, but the runtime
    // upload path handles it (same as the engine's own image parser).
    texture.setSource(bitmap as unknown as HTMLCanvasElement);
    return texture;
};

// Read a texture's texels back as raw RGBA bytes.
const readTexels = async (texture: Texture): Promise<Uint8Array> => {
    const texels = await texture.read(0, 0, texture.width, texture.height, { mipLevel: 0, face: 0, immediate: true });
    return texels as Uint8Array;
};

// Structural twin of Omg4V2Data (parsers/omg4.ts): the v2 setup path —
// GSplatResource creation, attachOmg4V2Motion, Omg4V2SplatAnimation — is
// typed against that shape and works on this unchanged.
class Omg4V3Data {
    // Temporal segment table for per-segment culling (see parsers/omg4.ts).
    segments?: Omg4Segments;

    readonly numSplats: number;

    readonly timeMin: number;

    readonly timeMax: number;

    readonly fps: number;

    cov2dScale: [number, number] | null = null;

    readonly gsplatData: GSplatData;

    readonly velocityX: Float32Array;

    readonly velocityY: Float32Array;

    readonly velocityZ: Float32Array;

    readonly tCenter: Float32Array;

    readonly tSigma: Float32Array;

    constructor(meta: any, gsplatData: GSplatData,
        velocity: [Float32Array, Float32Array, Float32Array],
        tCenter: Float32Array, tSigma: Float32Array) {
        this.numSplats = meta.count;
        this.timeMin = meta.time?.min ?? 0;
        this.timeMax = meta.time?.max ?? 0;
        this.fps = meta.time?.fps ?? 30;
        this.cov2dScale = meta.cov2d_scale ? [meta.cov2d_scale[0], meta.cov2d_scale[1]] : null;
        if (meta.segments?.list?.length && meta.segments.persistent) {
            this.segments = meta.segments as Omg4Segments;
        }
        this.gsplatData = gsplatData;
        [this.velocityX, this.velocityY, this.velocityZ] = velocity;
        this.tCenter = tCenter;
        this.tSigma = tSigma;
    }

    get duration(): number {
        return Math.max(0, this.timeMax - this.timeMin);
    }
}

// True if the buffer starts with the ZIP local-file magic (v3 container).
const isOmg4V3 = (buffer: ArrayBuffer): boolean => {
    return buffer.byteLength >= 4 && new DataView(buffer).getUint32(0, true) === ZIP_LOCAL_MAGIC;
};

const loadOmg4V3 = async (app: AppBase, buffer: ArrayBuffer): Promise<Omg4V3Data> => {
    const entries = parseZipEntries(buffer);
    const files = new Map<string, Uint8Array>();
    for (const entry of entries) {
        // eslint-disable-next-line no-await-in-loop -- deflate entries are a rare fallback; our encoder stores
        files.set(entry.filename, entry.deflated ? await inflateRaw(entry.data) : entry.data);
    }

    const metaBytes = files.get('meta.json');
    if (!metaBytes) {
        throw new Error('omg4 v3: meta.json not found in archive');
    }
    const meta = JSON.parse(new TextDecoder().decode(metaBytes));
    if (meta.version !== 3) {
        throw new Error(`omg4 v3: expected meta version 3, got ${meta.version}`);
    }

    const texOf = (filename: string): Promise<Texture> => {
        const bytes = files.get(filename);
        if (!bytes) {
            throw new Error(`omg4 v3: ${filename} not found in archive`);
        }
        return decodeTexture(app, bytes, filename);
    };

    // Static attributes: hand the SOG textures to the engine's own decoder.
    // GSplatSogData's codebook/iterator paths key off meta.version === 2 —
    // v3's static conventions are exactly SOG v2, so present it as such.
    const sog = new GSplatSogData() as any;
    sog.meta = { ...meta, version: 2 };
    sog.numSplats = meta.count;
    sog.means_l = await texOf(meta.means.files[0]);
    sog.means_u = await texOf(meta.means.files[1]);
    sog.quats = await texOf(meta.quats.files[0]);
    sog.scales = await texOf(meta.scales.files[0]);
    sog.sh0 = await texOf(meta.sh0.files[0]);
    if (meta.shN) {
        sog.sh_centroids = await texOf(meta.shN.files[0]);
        sog.sh_labels = await texOf(meta.shN.files[1]);
        sog.shBands = meta.shN.bands;
    } else {
        sog.shBands = 0;
    }

    // Temporal attributes (decoded here; not part of the engine's SOG model).
    const readEntry = async (filename: string) => {
        const texture = await texOf(filename);
        const texels = await readTexels(texture);
        texture.destroy();
        return texels;
    };
    const motionL = await readEntry(meta.motion.files[0]);
    const motionU = await readEntry(meta.motion.files[1]);
    const trbf = await readEntry(meta.trbf.files[0]);

    const gsplatData: GSplatData = await sog.decompress();
    sog.destroy();

    const n = meta.count as number;
    const velocity: [Float32Array, Float32Array, Float32Array] =
        [new Float32Array(n), new Float32Array(n), new Float32Array(n)];
    const vMins = meta.motion.mins as number[];
    const vMaxs = meta.motion.maxs as number[];
    for (let c = 0; c < 3; c++) {
        const mn = vMins[c];
        const span = vMaxs[c] - vMins[c];
        const out = velocity[c];
        for (let i = 0; i < n; i++) {
            const t = mn + span * ((motionU[i * 4 + c] << 8) + motionL[i * 4 + c]) / 65535;
            out[i] = Math.sign(t) * (Math.exp(Math.abs(t)) - 1);
        }
    }

    const centerCodebook = meta.trbf.center.codebook as number[];
    const sigmaCodebook = meta.trbf.sigma.codebook as number[];
    const tCenter = new Float32Array(n);
    const tSigma = new Float32Array(n);
    for (let i = 0; i < n; i++) {
        tCenter[i] = centerCodebook[trbf[i * 4]];
        tSigma[i] = sigmaCodebook[trbf[i * 4 + 1]];
    }

    return new Omg4V3Data(meta, gsplatData, velocity, tCenter, tSigma);
};

export { isOmg4V3, loadOmg4V3, Omg4V3Data };
