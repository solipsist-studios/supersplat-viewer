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

import type { SogstSegments } from '../parsers/sogst';

// .sogst version 3: SOG-compressed temporal splats.
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
// The decoded result is structurally identical to SogstData, so the whole
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
        throw new Error('sogst v3: invalid zip (no end-of-central-directory)');
    }

    const numFiles = u16(eocd + 8);
    let offset = u32(eocd + 16);
    const entries: ZipEntry[] = [];
    for (let i = 0; i < numFiles; i++) {
        if (u32(offset) !== ZIP_CDR_MAGIC) {
            throw new Error('sogst v3: invalid zip (bad central-directory record)');
        }
        const compression = u16(offset + 10);
        const compressedSize = u32(offset + 20);
        const filenameLength = u16(offset + 28);
        const extraLength = u16(offset + 30);
        const commentLength = u16(offset + 32);
        const lfhOffset = u32(offset + 42);
        const filename = new TextDecoder().decode(new Uint8Array(buffer, offset + 46, filenameLength));

        if (u32(lfhOffset) !== ZIP_LOCAL_MAGIC) {
            throw new Error('sogst v3: invalid zip (bad local file header)');
        }
        const dataOffset = lfhOffset + 30 + u16(lfhOffset + 26) + u16(lfhOffset + 28);

        if (compression !== 0 && compression !== 8) {
            throw new Error(`sogst v3: unsupported zip compression method ${compression}`);
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
        name: `sogstv3-${name}`,
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

// The SOG iterator only reads texel arrays (_levels[0]) plus dimensions,
// so decode feeds it these plain holders instead of real GPU textures.
interface TexelImage {
    width: number;
    height: number;
    _levels: [Uint8Array];
    destroy: () => void;
}

// Decodes webp payloads to raw RGBA texels in a worker with its own
// OffscreenCanvas WebGL context. The main-thread alternative (upload +
// texture.read on the app's context) forces every readback to sync behind
// queued rendering work — profiled at ~40% of the main thread during
// streaming playback, and the single biggest source of first-pass stutter
// on weak devices. A 2D canvas cannot be used instead: getImageData
// premultiplies, corrupting codebook indices next to data-bearing alpha
// (sh0 stores opacity there). In the worker, readPixels blocks only the
// worker.
const TEXEL_WORKER_SRC = `
let canvas = null, gl = null, tex = null, fbo = null;
self.onmessage = async (e) => {
    const { id, bytes } = e.data;
    try {
        const bitmap = await createImageBitmap(new Blob([bytes], { type: 'image/webp' }), {
            premultiplyAlpha: 'none',
            colorSpaceConversion: 'none'
        });
        const w = bitmap.width, h = bitmap.height;
        if (!gl) {
            canvas = new OffscreenCanvas(1, 1);
            gl = canvas.getContext('webgl2', { antialias: false, depth: false });
            if (!gl) throw new Error('no webgl2 in worker');
            tex = gl.createTexture();
            fbo = gl.createFramebuffer();
        }
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
        gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.NONE);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, bitmap);
        bitmap.close();
        gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
        const data = new Uint8Array(w * h * 4);
        gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, data);
        self.postMessage({ id, width: w, height: h, data }, [data.buffer]);
    } catch (err) {
        self.postMessage({ id, error: String(err) });
    }
};
`;

class WebpTexelWorker {
    private worker: Worker | null = null;

    private pending = new Map<number, { resolve:(t: TexelImage) => void, reject: (e: Error) => void }>();

    private nextId = 0;

    decode(bytes: Uint8Array): Promise<TexelImage> {
        if (!this.worker) {
            const blob = new Blob([TEXEL_WORKER_SRC], { type: 'text/javascript' });
            this.worker = new Worker(URL.createObjectURL(blob));
            this.worker.onmessage = (e: MessageEvent) => {
                const { id, width, height, data, error } = e.data;
                const entry = this.pending.get(id);
                if (!entry) {
                    return;
                }
                this.pending.delete(id);
                if (error) {
                    entry.reject(new Error(error));
                } else {
                    entry.resolve({
                        width, height, _levels: [data], destroy: () => { }
                    });
                }
            };
        }
        const id = this.nextId++;
        // exact-size copy so the underlying buffer can transfer without
        // detaching (or wholesale-cloning) the caller's archive buffer
        const copy = bytes.slice();
        return new Promise<TexelImage>((resolve, reject) => {
            this.pending.set(id, { resolve, reject });
            this.worker!.postMessage({ id, bytes: copy }, [copy.buffer]);
        });
    }

    destroy() {
        this.worker?.terminate();
        this.worker = null;
        const failure = new Error('texel worker destroyed');
        this.pending.forEach(entry => entry.reject(failure));
        this.pending.clear();
    }
}

// Structural twin of SogstData (parsers/sogst.ts): the v2 setup path —
// GSplatResource creation, attachSogstMotion, SogstSplatAnimation — is
// typed against that shape and works on this unchanged.
class SogstV3Data {
    // Temporal segment table for per-segment culling (see parsers/sogst.ts).
    segments?: SogstSegments;

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

    // Degree-2 motion: quadratic coefficient arrays (units/sec^2), or null
    // on degree-1 content. Same contract as SogstData.
    accelX: Float32Array | null = null;

    accelY: Float32Array | null = null;

    accelZ: Float32Array | null = null;

    constructor(meta: any, gsplatData: GSplatData,
        velocity: [Float32Array, Float32Array, Float32Array],
        tCenter: Float32Array, tSigma: Float32Array,
        accel: [Float32Array, Float32Array, Float32Array] | null = null) {
        this.meta = meta;
        this.numSplats = meta.count;
        this.timeMin = meta.time?.min ?? 0;
        this.timeMax = meta.time?.max ?? 0;
        this.fps = meta.time?.fps ?? 30;
        this.cov2dScale = meta.cov2d_scale ? [meta.cov2d_scale[0], meta.cov2d_scale[1]] : null;
        if (meta.segments?.list?.length && meta.segments.persistent) {
            this.segments = meta.segments as SogstSegments;
        }
        this.gsplatData = gsplatData;
        [this.velocityX, this.velocityY, this.velocityZ] = velocity;
        this.tCenter = tCenter;
        this.tSigma = tSigma;
        if (accel) {
            [this.accelX, this.accelY, this.accelZ] = accel;
        }
    }

    get duration(): number {
        return Math.max(0, this.timeMax - this.timeMin);
    }
}

// True if the buffer starts with the ZIP local-file magic (v3 container).
const isSogstV3 = (buffer: ArrayBuffer): boolean => {
    return buffer.byteLength >= 4 && new DataView(buffer).getUint32(0, true) === ZIP_LOCAL_MAGIC;
};

// Yield to the event loop so progress UI can repaint mid-decode. setTimeout
// rather than requestAnimationFrame: rAF never fires in hidden tabs.
const yieldToUi = () => new Promise((resolve) => {
    setTimeout(resolve, 0);
});

const SH_C0 = 0.28209479177387814;

// Main-thread budget per decode slice. Decode runs behind live playback on
// streaming loads; slices above ~a half frame read as visible stutter.
const DECODE_SLICE_MS = 6;

// Canonical per-group texture names. Monolithic archives use these bare;
// streamed archives prefix them with the group directory ("persistent/",
// "seg_000/", ...). shN_centroids is global either way.
const GROUP_FILE_NAMES = [
    'means_l.webp', 'means_u.webp', 'quats.webp', 'scales.webp', 'sh0.webp',
    'motion_l.webp', 'motion_u.webp', 'trbf.webp'
];

// Per-group texture names for this archive: degree-2 (accel) content adds
// the accel split pair to every group.
const groupBaseNames = (meta: any): string[] => {
    return meta.accel ? [...GROUP_FILE_NAMES, 'accel_l.webp', 'accel_u.webp'] : GROUP_FILE_NAMES;
};

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
    const base = groupBaseNames(meta);
    return meta.shN ? [...base, 'shN_labels.webp'] : base;
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

    readonly accel: [Float32Array, Float32Array, Float32Array] | null;

    readonly tCenter: Float32Array;

    readonly tSigma: Float32Array;

    private centroidsBytes: Uint8Array | null = null;

    private centroidsTexture: TexelImage | Texture | null = null;

    private texelWorker = new WebpTexelWorker();

    private texelWorkerBroken = false;

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
        this.accel = meta.accel ?
            [new Float32Array(this.n), new Float32Array(this.n), new Float32Array(this.n)] : null;
        this.tCenter = new Float32Array(this.n);
        this.tSigma = new Float32Array(this.n);
        this.tSigma.fill(1);
    }

    setCentroids(bytes: Uint8Array) {
        this.centroidsBytes = bytes;
    }

    // webp -> raw texels, off the main thread; falls back to the app
    // context's upload + readback path if the worker cannot run.
    private async decodeTexels(bytes: Uint8Array, name: string): Promise<TexelImage | Texture> {
        if (!this.texelWorkerBroken) {
            try {
                return await this.texelWorker.decode(bytes);
            } catch (err) {
                console.warn('sogst v3: worker texel decode unavailable, using GPU readback:', err);
                this.texelWorkerBroken = true;
            }
        }
        const texture = await decodeTexture(this.app, bytes, name);
        (texture as any)._levels[0] = await readTexels(texture);
        return texture;
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
        const texFor = (name: string): Promise<TexelImage | Texture> => {
            const bytes = files.get(name);
            if (!bytes) {
                throw new Error(`sogst v3: ${group.prefix ?? ''}/${name} missing from archive`);
            }
            return this.decodeTexels(bytes, `${group.prefix ?? 'mono'}-${name}`);
        };

        // SH decodes with the group only when its labels are present —
        // sh-deferred archives ship labels behind all geometry, and those
        // groups take a later decodeGroupSH pass instead.
        const useSH = !!this.meta.shN && files.has('shN_labels.webp');
        if (useSH && !this.centroidsTexture) {
            if (!this.centroidsBytes) {
                throw new Error('sogst v3: shN_centroids payload not provided before group decode');
            }
            // decoded once, shared by every group
            this.centroidsTexture = await this.decodeTexels(this.centroidsBytes, 'shN_centroids');
        }

        // all payloads decode concurrently in the worker
        const names = [...groupBaseNames(this.meta)];
        if (useSH) {
            names.push('shN_labels.webp');
        }
        const textures = await Promise.all(names.map(name => texFor(name)));
        const tex = new Map<string, TexelImage | Texture>();
        names.forEach((name, i) => {
            tex.set(name, textures[i]);
        });

        // Static attributes decode through the engine's own SOG iterator; a
        // per-group shim presents the group textures with the global
        // codebooks/mins (GSplatSogData keys off meta.version === 2, and
        // v3's static conventions are exactly SOG v2).
        const sog = new GSplatSogData() as any;
        sog.meta = { ...this.meta, version: 2, count: m };
        sog.numSplats = m;
        sog.means_l = tex.get('means_l.webp');
        sog.means_u = tex.get('means_u.webp');
        sog.quats = tex.get('quats.webp');
        sog.scales = tex.get('scales.webp');
        sog.sh0 = tex.get('sh0.webp');
        if (useSH) {
            sog.sh_centroids = this.centroidsTexture;
            sog.sh_labels = tex.get('shN_labels.webp');
            sog.shBands = this.meta.shN.bands;
        } else {
            sog.shBands = 0;
        }
        sog._patchCodebooks?.();

        const p = new Vec3();
        const r = new Quat();
        const s = new Vec3();
        const c = new Vec4();
        const sh = sog.shBands > 0 ? new Float32Array(45) : null;
        const iter = sog.createIter(p, r, s, c, sh);
        const arrays = this.arrays;
        const restArrays = sh ? Array.from({ length: 45 }, (_, j) => arrays[`f_rest_${j}`]) : null;

        // Time-budgeted slices rather than a fixed chunk size: decode runs
        // behind live playback on streaming loads, so no single slice may
        // hold the main thread past a few milliseconds — segment sizes vary
        // wildly (tens of thousands of splats on dense scenes) and a
        // count-based chunk either yields too rarely (jank) or too often.
        for (let i = 0; i < m;) {
            const sliceStart = performance.now();
            while (i < m) {
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
                // The spec permits an encoder to lose the RGB of any texel
                // whose alpha is zero (libwebp may rewrite fully-transparent
                // blocks when the `exact` flag is unavailable), so nothing may
                // *depend* on the colour read here. Storing it unconditionally
                // is safe only because the splat stays invisible: opacity
                // saturates to -40 below, and the temporal factor in
                // sogst-motion.ts multiplies alpha by exp(-0.5*dt^2) <= 1 and
                // so can never raise it. Do not add a path that scales alpha
                // up without guarding on c.w > 0 here.
                arrays.f_dc_0[o] = (c.x - 0.5) / SH_C0;
                arrays.f_dc_1[o] = (c.y - 0.5) / SH_C0;
                arrays.f_dc_2[o] = (c.z - 0.5) / SH_C0;
                arrays.opacity[o] = c.w <= 0 ? -40 : (c.w >= 1 ? 40 : -Math.log(1 / c.w - 1));
                if (sh && restArrays) {
                    for (let j = 0; j < 45; j++) {
                        restArrays[j][o] = sh[j];
                    }
                }
                i++;
                if ((i & 63) === 0 && performance.now() - sliceStart >= DECODE_SLICE_MS) {
                    break;
                }
            }
            onProgress?.(i / m);
            // eslint-disable-next-line no-await-in-loop -- deliberate UI yield
            await yieldToUi();
        }

        // the centroids texture is shared across groups — detach it so the
        // shim's destroy() only releases the group-local textures
        sog.sh_centroids = null;
        sog.destroy();

        // Temporal attributes (not part of the engine's SOG model) — texels
        // were read back in the batch above; only the textures remain to free.
        const motionL = (tex.get('motion_l.webp') as any)._levels[0] as Uint8Array;
        const motionU = (tex.get('motion_u.webp') as any)._levels[0] as Uint8Array;
        const trbf = (tex.get('trbf.webp') as any)._levels[0] as Uint8Array;
        tex.get('motion_l.webp')!.destroy();
        tex.get('motion_u.webp')!.destroy();
        tex.get('trbf.webp')!.destroy();

        let accelL: Uint8Array | null = null;
        let accelU: Uint8Array | null = null;
        const aMins = this.meta.accel?.mins as number[] | undefined;
        const aMaxs = this.meta.accel?.maxs as number[] | undefined;
        if (this.accel) {
            accelL = (tex.get('accel_l.webp') as any)._levels[0] as Uint8Array;
            accelU = (tex.get('accel_u.webp') as any)._levels[0] as Uint8Array;
            tex.get('accel_l.webp')!.destroy();
            tex.get('accel_u.webp')!.destroy();
        }

        const vMins = this.meta.motion.mins as number[];
        const vMaxs = this.meta.motion.maxs as number[];
        const centerCodebook = this.meta.trbf.center.codebook as number[];
        const sigmaCodebook = this.meta.trbf.sigma.codebook as number[];
        for (let i = 0; i < m;) {
            const sliceStart = performance.now();
            while (i < m) {
                const o = a + i;
                for (let ch = 0; ch < 3; ch++) {
                    const t = vMins[ch] + (vMaxs[ch] - vMins[ch]) * ((motionU[i * 4 + ch] << 8) + motionL[i * 4 + ch]) / 65535;
                    this.velocity[ch][o] = Math.sign(t) * (Math.exp(Math.abs(t)) - 1);
                }
                if (this.accel && accelL && accelU && aMins && aMaxs) {
                    for (let ch = 0; ch < 3; ch++) {
                        const t = aMins[ch] + (aMaxs[ch] - aMins[ch]) * ((accelU[i * 4 + ch] << 8) + accelL[i * 4 + ch]) / 65535;
                        this.accel[ch][o] = Math.sign(t) * (Math.exp(Math.abs(t)) - 1);
                    }
                }
                this.tCenter[o] = centerCodebook[trbf[i * 4]];
                this.tSigma[o] = sigmaCodebook[trbf[i * 4 + 1]];
                i++;
                if ((i & 255) === 0 && performance.now() - sliceStart >= DECODE_SLICE_MS) {
                    break;
                }
            }
            // eslint-disable-next-line no-await-in-loop -- deliberate UI yield
            await yieldToUi();
        }
    }

    // Decode a deferred SH labels payload for one group into the f_rest
    // arrays (sh-deferred archives ship all labels behind the geometry so
    // the scene can reveal DC-only and layer view dependence in later).
    async decodeGroupSH(group: V3Group, labelsBytes: Uint8Array) {
        const [a, b] = group.range;
        const m = b - a;
        if (m <= 0 || !this.meta.shN) {
            return;
        }
        if (!this.centroidsTexture) {
            if (!this.centroidsBytes) {
                throw new Error('sogst v3: shN_centroids payload must precede deferred labels');
            }
            this.centroidsTexture = await this.decodeTexels(this.centroidsBytes, 'shN_centroids');
        }
        const labels = await this.decodeTexels(labelsBytes, `${group.prefix ?? 'mono'}-shN_labels`);

        // SH-only iterator: null attribute targets skip every texture but
        // sh_labels/sh_centroids
        const sog = new GSplatSogData() as any;
        sog.meta = { ...this.meta, version: 2, count: m };
        sog.numSplats = m;
        sog.sh_labels = labels;
        sog.sh_centroids = this.centroidsTexture;
        sog.shBands = this.meta.shN.bands;
        sog._patchCodebooks?.();

        const sh = new Float32Array(45);
        const iter = sog.createIter(null, null, null, null, sh);
        const arrays = this.arrays;
        const restArrays = Array.from({ length: 45 }, (_, j) => arrays[`f_rest_${j}`]);
        for (let i = 0; i < m;) {
            const sliceStart = performance.now();
            while (i < m) {
                iter.read(i);
                const o = a + i;
                for (let j = 0; j < 45; j++) {
                    restArrays[j][o] = sh[j];
                }
                i++;
                if ((i & 63) === 0 && performance.now() - sliceStart >= DECODE_SLICE_MS) {
                    break;
                }
            }
            // eslint-disable-next-line no-await-in-loop -- deliberate UI yield
            await yieldToUi();
        }

        sog.sh_centroids = null;
        sog.destroy();
    }

    buildData(): SogstV3Data {
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
        return new SogstV3Data(this.meta, gsplatData, this.velocity, this.tCenter, this.tSigma, this.accel);
    }

    destroy() {
        this.texelWorker.destroy();
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
        throw new Error('sogst v3: meta.json not found in archive');
    }
    const meta = JSON.parse(new TextDecoder().decode(bytes));
    if (meta.version !== 3) {
        throw new Error(`sogst v3: expected meta version 3, got ${meta.version}`);
    }
    // `format` was added when the format was renamed to .sogst; v3 archives
    // baked before that carry no such key, so absence means .sogst too.
    if (meta.format !== undefined && meta.format !== 'sogst') {
        throw new Error(`sogst v3: unsupported meta format '${meta.format}'`);
    }
    return meta;
};

// Decode a complete v3 archive (either layout) into playable data. Used for
// non-streamed archives and for cache hits on streamed ones.
const loadSogstV3 = async (app: AppBase, buffer: ArrayBuffer,
    onProgress?: (progress: number) => void): Promise<SogstV3Data> => {
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
    isSogstV3, loadSogstV3, SogstV3Data,
    V3Decoder, enumerateV3Groups, groupFileList, groupBaseNames, parseV3Meta, setAabbFromV3Meta,
    GROUP_FILE_NAMES
};
