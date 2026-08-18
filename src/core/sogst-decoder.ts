import { GSplatData, GSplatSogData, Quat, Vec3, Vec4 } from 'playcanvas';
import type { AppBase, Texture } from 'playcanvas';

import type { SogstMeta } from '../parsers/sogst';

import { SogstData } from './sogst-data';
import { decodeTexture, readTexels, texelsOf, WebpTexelWorker } from './sogst-texels';
import type { TexelImage } from './sogst-texels';

// GSplatSogData._patchCodebooks is private to the engine. It is the one member
// here with no public equivalent, so the dependency is spelled out rather than
// hidden behind a blanket `as any` — if the engine drops it, this is the line
// to look at.
type SogCodebookPatch = { _patchCodebooks?: () => void };

// SOG's own container version. .sogst carries meta.version 1, but its static
// attributes follow the SOG v2 conventions byte for byte, so the per-group
// shim below advertises 2 — GSplatSogData selects its decode path from it.
const SOG_META_VERSION = 2;

// Zeroth-order SH basis function, and the offset SOG applies before storing
// DC colour so it lands in [0, 1]. Inverting both recovers the f_dc_* value.
const SH_C0 = 0.28209479177387814;
const SH_DC_OFFSET = 0.5;

// Higher-order SH coefficients per splat: bands 1-3 are 3 + 5 + 7 = 15
// coefficients, one set per colour channel.
const SH_REST_COEFFS = 45;

// Opacity is stored as a logit. Splats that are not yet decoded, and splats
// whose stored alpha saturates, are pinned to +/- this magnitude: sigmoid
// is flat to float precision well before it, so a larger value buys nothing
// and NaNs on the inverse.
const OPACITY_LOGIT_LIMIT = 40;

// Quantisation of the 16-bit split-plane textures, and the RGBA texel
// stride every attribute texture is read at.
const U16_MAX = 65535;
const RGBA_STRIDE = 4;

// trbf.webp channel assignment (see parsers/sogst.ts): R indexes the
// t_center codebook, G the t_sigma codebook.
const TRBF_CENTER_CHANNEL = 0;
const TRBF_SIGMA_CHANNEL = 1;

// Main-thread budget per decode slice. Decode runs behind live playback on
// streaming loads; slices above ~a half frame read as visible stutter.
const DECODE_SLICE_MS = 6;

// Reading the clock costs more than a loop iteration, so the elapsed-time
// test runs on a power-of-two stride (`i & (STRIDE - 1)`). The temporal
// loop's body is the cheaper of the two, so it checks less often.
const STATIC_CLOCK_STRIDE = 64;
const TEMPORAL_CLOCK_STRIDE = 256;

// Yield to the event loop so progress UI can repaint mid-decode. setTimeout
// rather than requestAnimationFrame: rAF never fires in hidden tabs.
const yieldToUi = () =>
    new Promise((resolve) => {
        setTimeout(resolve, 0);
    });

// Invert one channel of a 16-bit split-plane pair: the `_u` texture holds
// the high byte and `_l` the low, together quantising [min, max] across the
// full u16 range. Values are stored log-transformed (sign(x)*ln(1+|x|)), so
// the exponential undoes that.
const decodeSplit16 = (
    lo: Uint8Array,
    hi: Uint8Array,
    texel: number,
    channel: number,
    min: number,
    max: number
): number => {
    const o = texel * RGBA_STRIDE + channel;
    const t = min + ((max - min) * ((hi[o] << 8) + lo[o])) / U16_MAX;
    return Math.sign(t) * (Math.exp(Math.abs(t)) - 1);
};

// Canonical per-group texture names. Monolithic archives use these bare;
// streamed archives prefix them with the group directory ("persistent/",
// "seg_000/", ...). shN_centroids is global either way.
const GROUP_FILE_NAMES = [
    'means_l.webp',
    'means_u.webp',
    'quats.webp',
    'scales.webp',
    'sh0.webp',
    'motion_l.webp',
    'motion_u.webp',
    'trbf.webp'
];

// Per-group texture names for this archive: degree-2 (accel) content adds
// the accel split pair to every group.
const groupBaseNames = (meta: SogstMeta): string[] => {
    return meta.accel ? [...GROUP_FILE_NAMES, 'accel_l.webp', 'accel_u.webp'] : GROUP_FILE_NAMES;
};

type SogstGroup = {
    prefix: string | null; // null => monolithic (bare names)
    range: [number, number];
    segIndex: number; // index into meta.segments.list; -1 otherwise
};

// Decode groups in play order: [whole file] for monolithic archives, or
// [persistent, seg_000, seg_001, ...] (empty groups omitted) for streamed.
const enumerateSogstGroups = (meta: SogstMeta): SogstGroup[] => {
    if (!meta.streams) {
        return [{ prefix: null, range: [0, meta.count], segIndex: -1 }];
    }
    const groups: SogstGroup[] = [];
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

const groupFileList = (meta: SogstMeta): string[] => {
    const base = groupBaseNames(meta);
    return meta.shN ? [...base, 'shN_labels.webp'] : base;
};

// Incremental decoder: allocates the full-length attribute arrays up front
// (prefilled invisible) and fills index ranges as group payloads become
// available — the same arrays back the GSplatData, so a streaming caller
// can create the GPU resource after the first groups and refresh it as
// later groups land. Also used for complete buffers (all groups at once).
class SogstDecoder {
    private app: AppBase;

    readonly meta: SogstMeta;

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

    constructor(app: AppBase, meta: SogstMeta) {
        this.app = app;
        this.meta = meta;
        this.n = meta.count;
        this.members = [
            'x',
            'y',
            'z',
            'f_dc_0',
            'f_dc_1',
            'f_dc_2',
            'opacity',
            'scale_0',
            'scale_1',
            'scale_2',
            'rot_0',
            'rot_1',
            'rot_2',
            'rot_3'
        ];
        if (meta.shN) {
            for (let i = 0; i < SH_REST_COEFFS; i++) {
                this.members.push(`f_rest_${i}`);
            }
        }
        this.arrays = {};
        this.members.forEach((name) => {
            this.arrays[name] = new Float32Array(this.n);
        });
        // not-yet-loaded splats must be invisible: sigmoid of a deeply
        // negative logit is zero to float precision
        this.arrays.opacity.fill(-OPACITY_LOGIT_LIMIT);
        this.velocity = [new Float32Array(this.n), new Float32Array(this.n), new Float32Array(this.n)];
        this.accel = meta.accel ? [new Float32Array(this.n), new Float32Array(this.n), new Float32Array(this.n)] : null;
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
                console.warn('sogst: worker texel decode unavailable, using GPU readback:', err);
                this.texelWorkerBroken = true;
            }
        }
        const texture = await decodeTexture(this.app, bytes, name);
        texture._levels[0] = await readTexels(texture);
        return texture;
    }

    // Decode one group's texture payloads (keyed by bare canonical name)
    // into [range[0], range[1]) of the full arrays.
    async decodeGroup(group: SogstGroup, files: Map<string, Uint8Array>, onProgress?: (frac: number) => void) {
        const [a, b] = group.range;
        const m = b - a;
        if (m <= 0) {
            return;
        }
        const texFor = (name: string): Promise<TexelImage | Texture> => {
            const bytes = files.get(name);
            if (!bytes) {
                throw new Error(`sogst: ${group.prefix ?? ''}/${name} missing from archive`);
            }
            return this.decodeTexels(bytes, `${group.prefix ?? 'mono'}-${name}`);
        };

        // SH decodes with the group only when its labels are present —
        // sh-deferred archives ship labels behind all geometry, and those
        // groups take a later decodeGroupSH pass instead.
        const useSH = !!this.meta.shN && files.has('shN_labels.webp');
        if (useSH && !this.centroidsTexture) {
            if (!this.centroidsBytes) {
                throw new Error('sogst: shN_centroids payload not provided before group decode');
            }
            // decoded once, shared by every group
            this.centroidsTexture = await this.decodeTexels(this.centroidsBytes, 'shN_centroids');
        }

        // all payloads decode concurrently in the worker
        const names = [...groupBaseNames(this.meta)];
        if (useSH) {
            names.push('shN_labels.webp');
        }
        const textures = await Promise.all(names.map((name) => texFor(name)));
        const tex = new Map<string, TexelImage | Texture>();
        names.forEach((name, i) => {
            tex.set(name, textures[i]);
        });

        // Static attributes decode through the engine's own SOG iterator; a
        // per-group shim presents the group textures with the global
        // codebooks/mins. GSplatSogData selects its decode path from
        // meta.version, so the shim advertises the SOG container version
        // rather than .sogst's own.
        const sog = new GSplatSogData();
        sog.meta = { ...this.meta, version: SOG_META_VERSION, count: m };
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
        (sog as unknown as SogCodebookPatch)._patchCodebooks?.();

        const p = new Vec3();
        const r = new Quat();
        const s = new Vec3();
        const c = new Vec4();
        const sh = sog.shBands > 0 ? new Float32Array(SH_REST_COEFFS) : null;
        const iter = sog.createIter(p, r, s, c, sh);
        const arrays = this.arrays;
        const restArrays = sh ? Array.from({ length: SH_REST_COEFFS }, (_, j) => arrays[`f_rest_${j}`]) : null;

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
                // saturates to -OPACITY_LOGIT_LIMIT below, and the temporal factor in
                // sogst-motion.ts multiplies alpha by exp(-0.5*dt^2) <= 1 and
                // so can never raise it. Do not add a path that scales alpha
                // up without guarding on c.w > 0 here.
                arrays.f_dc_0[o] = (c.x - SH_DC_OFFSET) / SH_C0;
                arrays.f_dc_1[o] = (c.y - SH_DC_OFFSET) / SH_C0;
                arrays.f_dc_2[o] = (c.z - SH_DC_OFFSET) / SH_C0;
                arrays.opacity[o] =
                    c.w <= 0 ? -OPACITY_LOGIT_LIMIT : c.w >= 1 ? OPACITY_LOGIT_LIMIT : -Math.log(1 / c.w - 1);
                if (sh && restArrays) {
                    for (let j = 0; j < SH_REST_COEFFS; j++) {
                        restArrays[j][o] = sh[j];
                    }
                }
                i++;
                if ((i & (STATIC_CLOCK_STRIDE - 1)) === 0 && performance.now() - sliceStart >= DECODE_SLICE_MS) {
                    break;
                }
            }
            onProgress?.(i / m);

            await yieldToUi();
        }

        // the centroids texture is shared across groups — detach it so the
        // shim's destroy() only releases the group-local textures
        sog.sh_centroids = null;
        sog.destroy();

        // Temporal attributes (not part of the engine's SOG model) — texels
        // were read back in the batch above; only the textures remain to free.
        const motionL = texelsOf(tex.get('motion_l.webp'));
        const motionU = texelsOf(tex.get('motion_u.webp'));
        const trbf = texelsOf(tex.get('trbf.webp'));
        tex.get('motion_l.webp')!.destroy();
        tex.get('motion_u.webp')!.destroy();
        tex.get('trbf.webp')!.destroy();

        let accelL: Uint8Array | null = null;
        let accelU: Uint8Array | null = null;
        const aMins = this.meta.accel?.mins as number[] | undefined;
        const aMaxs = this.meta.accel?.maxs as number[] | undefined;
        if (this.accel) {
            accelL = texelsOf(tex.get('accel_l.webp'));
            accelU = texelsOf(tex.get('accel_u.webp'));
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
                    this.velocity[ch][o] = decodeSplit16(motionL, motionU, i, ch, vMins[ch], vMaxs[ch]);
                }
                if (this.accel && accelL && accelU && aMins && aMaxs) {
                    for (let ch = 0; ch < 3; ch++) {
                        this.accel[ch][o] = decodeSplit16(accelL, accelU, i, ch, aMins[ch], aMaxs[ch]);
                    }
                }
                this.tCenter[o] = centerCodebook[trbf[i * RGBA_STRIDE + TRBF_CENTER_CHANNEL]];
                this.tSigma[o] = sigmaCodebook[trbf[i * RGBA_STRIDE + TRBF_SIGMA_CHANNEL]];
                i++;
                if ((i & (TEMPORAL_CLOCK_STRIDE - 1)) === 0 && performance.now() - sliceStart >= DECODE_SLICE_MS) {
                    break;
                }
            }

            await yieldToUi();
        }
    }

    // Decode a deferred SH labels payload for one group into the f_rest
    // arrays (sh-deferred archives ship all labels behind the geometry so
    // the scene can reveal DC-only and layer view dependence in later).
    async decodeGroupSH(group: SogstGroup, labelsBytes: Uint8Array) {
        const [a, b] = group.range;
        const m = b - a;
        if (m <= 0 || !this.meta.shN) {
            return;
        }
        if (!this.centroidsTexture) {
            if (!this.centroidsBytes) {
                throw new Error('sogst: shN_centroids payload must precede deferred labels');
            }
            this.centroidsTexture = await this.decodeTexels(this.centroidsBytes, 'shN_centroids');
        }
        const labels = await this.decodeTexels(labelsBytes, `${group.prefix ?? 'mono'}-shN_labels`);

        // SH-only iterator: null attribute targets skip every texture but
        // sh_labels/sh_centroids
        const sog = new GSplatSogData();
        sog.meta = { ...this.meta, version: SOG_META_VERSION, count: m };
        sog.numSplats = m;
        sog.sh_labels = labels;
        sog.sh_centroids = this.centroidsTexture;
        sog.shBands = this.meta.shN.bands;
        (sog as unknown as SogCodebookPatch)._patchCodebooks?.();

        const sh = new Float32Array(SH_REST_COEFFS);
        const iter = sog.createIter(null, null, null, null, sh);
        const arrays = this.arrays;
        const restArrays = Array.from({ length: SH_REST_COEFFS }, (_, j) => arrays[`f_rest_${j}`]);
        for (let i = 0; i < m;) {
            const sliceStart = performance.now();
            while (i < m) {
                iter.read(i);
                const o = a + i;
                for (let j = 0; j < SH_REST_COEFFS; j++) {
                    restArrays[j][o] = sh[j];
                }
                i++;
                if ((i & (STATIC_CLOCK_STRIDE - 1)) === 0 && performance.now() - sliceStart >= DECODE_SLICE_MS) {
                    break;
                }
            }

            await yieldToUi();
        }

        sog.sh_centroids = null;
        sog.destroy();
    }

    buildData(): SogstData {
        const gsplatData = new GSplatData([
            {
                name: 'vertex',
                count: this.n,
                properties: this.members.map((name) => ({
                    name,
                    type: 'float' as const,
                    byteSize: 4,
                    storage: this.arrays[name]
                }))
            }
        ]);
        return new SogstData(this.meta, gsplatData, this.velocity, this.tCenter, this.tSigma, this.accel);
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

export {
    DECODE_SLICE_MS,
    enumerateSogstGroups,
    groupBaseNames,
    groupFileList,
    GROUP_FILE_NAMES,
    SogstDecoder,
    yieldToUi
};
export type { SogstGroup };
