import {
    GSplatData,
    GSplatSogData,
    PIXELFORMAT_RGBA8,
    Quat,
    Texture,
    Vec3,
    Vec4,
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

    // Highest absolute clip time that is fully decoded — streaming loads
    // advance this per segment and the animation driver holds the playhead
    // at it. Infinity once (or when) everything is loaded.
    loadedThrough = Infinity;

    readonly meta: any;

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
        this.meta = meta;
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

// Yield to the event loop so progress UI can repaint mid-decode. setTimeout
// rather than requestAnimationFrame: rAF never fires in hidden tabs.
const yieldToUi = () => new Promise((resolve) => {
    setTimeout(resolve, 0);
});

const SH_C0 = 0.28209479177387814;

// Canonical per-group texture names. Monolithic archives use these bare;
// streamed archives prefix them with the group directory ("persistent/",
// "seg_000/", ...). shN_centroids is global either way.
const GROUP_FILE_NAMES = [
    'means_l.webp', 'means_u.webp', 'quats.webp', 'scales.webp', 'sh0.webp',
    'motion_l.webp', 'motion_u.webp', 'trbf.webp'
];

interface V3Group {
    prefix: string | null;          // null => monolithic (bare names)
    range: [number, number];
    segIndex: number;               // index into meta.segments.list; -1 otherwise
}

// Decode groups in play order: [whole file] for monolithic archives, or
// [persistent, seg_000, seg_001, ...] (empty groups omitted) for streamed.
const enumerateV3Groups = (meta: any): V3Group[] => {
    if (!meta.streams) {
        return [{ prefix: null, range: [0, meta.count], segIndex: -1 }];
    }
    const groups: V3Group[] = [];
    if (meta.streams.persistent) {
        groups.push({ prefix: meta.streams.persistent, range: meta.segments.persistent, segIndex: -1 });
    }
    (meta.streams.segments as (string | null)[]).forEach((prefix, i) => {
        if (prefix) {
            groups.push({ prefix, range: meta.segments.list[i].range, segIndex: i });
        }
    });
    return groups;
};

const groupFileList = (meta: any): string[] => {
    return meta.shN ? [...GROUP_FILE_NAMES, 'shN_labels.webp'] : GROUP_FILE_NAMES;
};

// Incremental decoder: allocates the full-length attribute arrays up front
// (prefilled invisible) and fills index ranges as group payloads become
// available — the same arrays back the GSplatData, so a streaming caller
// can create the GPU resource after the first groups and refresh it as
// later groups land. Also used for complete buffers (all groups at once).
class V3Decoder {
    private app: AppBase;

    readonly meta: any;

    readonly n: number;

    private members: string[];

    readonly arrays: Record<string, Float32Array>;

    readonly velocity: [Float32Array, Float32Array, Float32Array];

    readonly tCenter: Float32Array;

    readonly tSigma: Float32Array;

    private centroidsBytes: Uint8Array | null = null;

    private centroidsTexture: Texture | null = null;

    constructor(app: AppBase, meta: any) {
        this.app = app;
        this.meta = meta;
        this.n = meta.count;
        this.members = [
            'x', 'y', 'z',
            'f_dc_0', 'f_dc_1', 'f_dc_2',
            'opacity',
            'scale_0', 'scale_1', 'scale_2',
            'rot_0', 'rot_1', 'rot_2', 'rot_3'
        ];
        if (meta.shN) {
            for (let i = 0; i < 45; i++) {
                this.members.push(`f_rest_${i}`);
            }
        }
        this.arrays = {};
        this.members.forEach((name) => {
            this.arrays[name] = new Float32Array(this.n);
        });
        // not-yet-loaded splats must be invisible: alpha ~0 via a deeply
        // negative opacity logit (same trick as the v2 streaming prefill)
        this.arrays.opacity.fill(-40);
        this.velocity = [new Float32Array(this.n), new Float32Array(this.n), new Float32Array(this.n)];
        this.tCenter = new Float32Array(this.n);
        this.tSigma = new Float32Array(this.n);
        this.tSigma.fill(1);
    }

    setCentroids(bytes: Uint8Array) {
        this.centroidsBytes = bytes;
    }

    // Decode one group's texture payloads (keyed by bare canonical name)
    // into [range[0], range[1]) of the full arrays.
    async decodeGroup(group: V3Group, files: Map<string, Uint8Array>,
        onProgress?: (frac: number) => void) {
        const [a, b] = group.range;
        const m = b - a;
        if (m <= 0) {
            return;
        }
        const texFor = (name: string): Promise<Texture> => {
            const bytes = files.get(name);
            if (!bytes) {
                throw new Error(`omg4 v3: ${group.prefix ?? ''}/${name} missing from archive`);
            }
            return decodeTexture(this.app, bytes, `${group.prefix ?? 'mono'}-${name}`);
        };

        // Static attributes decode through the engine's own SOG iterator; a
        // per-group shim presents the group textures with the global
        // codebooks/mins (GSplatSogData keys off meta.version === 2, and
        // v3's static conventions are exactly SOG v2).
        const sog = new GSplatSogData() as any;
        sog.meta = { ...this.meta, version: 2, count: m };
        sog.numSplats = m;
        sog.means_l = await texFor('means_l.webp');
        sog.means_u = await texFor('means_u.webp');
        sog.quats = await texFor('quats.webp');
        sog.scales = await texFor('scales.webp');
        sog.sh0 = await texFor('sh0.webp');
        if (this.meta.shN) {
            if (!this.centroidsTexture) {
                if (!this.centroidsBytes) {
                    throw new Error('omg4 v3: shN_centroids payload not provided before group decode');
                }
                this.centroidsTexture = await decodeTexture(this.app, this.centroidsBytes, 'shN_centroids');
                (this.centroidsTexture as any)._levels[0] = await readTexels(this.centroidsTexture);
            }
            sog.sh_centroids = this.centroidsTexture;
            sog.sh_labels = await texFor('shN_labels.webp');
            sog.shBands = this.meta.shN.bands;
        } else {
            sog.shBands = 0;
        }
        sog._patchCodebooks?.();

        const groupTextures = [sog.means_l, sog.means_u, sog.quats, sog.scales, sog.sh0];
        if (sog.shBands > 0) {
            groupTextures.push(sog.sh_labels);
        }
        for (const texture of groupTextures) {
            // eslint-disable-next-line no-await-in-loop -- sequential GPU readbacks
            texture._levels[0] = await readTexels(texture);
        }

        const p = new Vec3();
        const r = new Quat();
        const s = new Vec3();
        const c = new Vec4();
        const sh = sog.shBands > 0 ? new Float32Array(45) : null;
        const iter = sog.createIter(p, r, s, c, sh);
        const arrays = this.arrays;

        const CHUNK = 131072;
        for (let start = 0; start < m; start += CHUNK) {
            const end = Math.min(m, start + CHUNK);
            for (let i = start; i < end; i++) {
                iter.read(i);
                const o = a + i;
                arrays.x[o] = p.x;
                arrays.y[o] = p.y;
                arrays.z[o] = p.z;
                arrays.rot_0[o] = r.w;
                arrays.rot_1[o] = r.x;
                arrays.rot_2[o] = r.y;
                arrays.rot_3[o] = r.z;
                arrays.scale_0[o] = s.x;
                arrays.scale_1[o] = s.y;
                arrays.scale_2[o] = s.z;
                arrays.f_dc_0[o] = (c.x - 0.5) / SH_C0;
                arrays.f_dc_1[o] = (c.y - 0.5) / SH_C0;
                arrays.f_dc_2[o] = (c.z - 0.5) / SH_C0;
                arrays.opacity[o] = c.w <= 0 ? -40 : (c.w >= 1 ? 40 : -Math.log(1 / c.w - 1));
                if (sh) {
                    for (let j = 0; j < 45; j++) {
                        arrays[`f_rest_${j}`][o] = sh[j];
                    }
                }
            }
            onProgress?.(end / m);
            // eslint-disable-next-line no-await-in-loop -- deliberate UI yield
            await yieldToUi();
        }

        // the centroids texture is shared across groups — detach it so the
        // shim's destroy() only releases the group-local textures
        sog.sh_centroids = null;
        sog.destroy();

        // Temporal attributes (not part of the engine's SOG model).
        const readEntry = async (name: string) => {
            const texture = await texFor(name);
            const texels = await readTexels(texture);
            texture.destroy();
            return texels;
        };
        const motionL = await readEntry('motion_l.webp');
        const motionU = await readEntry('motion_u.webp');
        const trbf = await readEntry('trbf.webp');

        const vMins = this.meta.motion.mins as number[];
        const vMaxs = this.meta.motion.maxs as number[];
        for (let ch = 0; ch < 3; ch++) {
            const mn = vMins[ch];
            const span = vMaxs[ch] - vMins[ch];
            const out = this.velocity[ch];
            for (let i = 0; i < m; i++) {
                const t = mn + span * ((motionU[i * 4 + ch] << 8) + motionL[i * 4 + ch]) / 65535;
                out[a + i] = Math.sign(t) * (Math.exp(Math.abs(t)) - 1);
            }
        }
        const centerCodebook = this.meta.trbf.center.codebook as number[];
        const sigmaCodebook = this.meta.trbf.sigma.codebook as number[];
        for (let i = 0; i < m; i++) {
            this.tCenter[a + i] = centerCodebook[trbf[i * 4]];
            this.tSigma[a + i] = sigmaCodebook[trbf[i * 4 + 1]];
        }
    }

    buildData(): Omg4V3Data {
        const gsplatData = new GSplatData([{
            name: 'vertex',
            count: this.n,
            properties: this.members.map(name => ({
                name,
                type: 'float' as const,
                byteSize: 4,
                storage: this.arrays[name]
            }))
        }]);
        return new Omg4V3Data(this.meta, gsplatData, this.velocity, this.tCenter, this.tSigma);
    }

    destroy() {
        try {
            this.centroidsTexture?.destroy();
        } catch {
            // the graphics device may already be torn down (viewer was
            // destroyed while a stream was still decoding) — nothing to free
        }
        this.centroidsTexture = null;
    }
}

const parseV3Meta = (bytes: Uint8Array | undefined): any => {
    if (!bytes) {
        throw new Error('omg4 v3: meta.json not found in archive');
    }
    const meta = JSON.parse(new TextDecoder().decode(bytes));
    if (meta.version !== 3) {
        throw new Error(`omg4 v3: expected meta version 3, got ${meta.version}`);
    }
    return meta;
};

// Decode a complete v3 archive (either layout) into playable data. Used for
// non-streamed archives and for cache hits on streamed ones.
const loadOmg4V3 = async (app: AppBase, buffer: ArrayBuffer,
    onProgress?: (progress: number) => void): Promise<Omg4V3Data> => {
    const report = (value: number) => onProgress?.(Math.min(100, Math.round(value)));

    const entries = parseZipEntries(buffer);
    const files = new Map<string, Uint8Array>();
    for (const entry of entries) {
        // eslint-disable-next-line no-await-in-loop -- deflate entries are a rare fallback; our encoder stores
        files.set(entry.filename, entry.deflated ? await inflateRaw(entry.data) : entry.data);
    }

    const meta = parseV3Meta(files.get('meta.json'));
    const decoder = new V3Decoder(app, meta);
    if (meta.shN) {
        const centroidsName = meta.streams ? 'shN_centroids.webp' : meta.shN.files[0];
        decoder.setCentroids(files.get(centroidsName)!);
    }

    const groups = enumerateV3Groups(meta);
    const names = groupFileList(meta);
    let done = 0;
    for (const group of groups) {
        const groupBytes = new Map<string, Uint8Array>();
        for (const name of names) {
            const stored = files.get(group.prefix ? `${group.prefix}/${name}` : name);
            if (stored) {
                groupBytes.set(name, stored);
            }
        }
        const m = group.range[1] - group.range[0];
        const base = done;
        // eslint-disable-next-line no-await-in-loop -- groups decode sequentially
        await decoder.decodeGroup(group, groupBytes, frac => report(((base + frac * m) / meta.count) * 100));
        done += m;
    }

    const data = decoder.buildData();
    decoder.destroy();
    report(100);
    return data;
};

// Set exact scene bounds from the archive's global means range — during a
// streaming load the attribute arrays are only partially filled, so bounds
// computed from them would understate the scene. The mins/maxs live in the
// SOG log-transformed space; invert with sign(v) * (e^|v| - 1).
const setAabbFromV3Meta = (meta: any, aabb: any) => {
    const mins = meta.means.mins as number[];
    const maxs = meta.means.maxs as number[];
    const map = (v: number) => Math.sign(v) * (Math.exp(Math.abs(v)) - 1);
    aabb.center.set(
        (map(mins[0]) + map(maxs[0])) * 0.5,
        (map(mins[1]) + map(maxs[1])) * 0.5,
        (map(mins[2]) + map(maxs[2])) * 0.5
    );
    aabb.halfExtents.set(
        (map(maxs[0]) - map(mins[0])) * 0.5,
        (map(maxs[1]) - map(mins[1])) * 0.5,
        (map(maxs[2]) - map(mins[2])) * 0.5
    );
};

export {
    isOmg4V3, loadOmg4V3, Omg4V3Data,
    V3Decoder, enumerateV3Groups, groupFileList, parseV3Meta, setAabbFromV3Meta
};
