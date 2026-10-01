import { GSplatSogData, Quat, Vec3, Vec4 } from 'playcanvas';
import type { AppBase, Texture } from 'playcanvas';

import type { SogstMeta } from '../parsers/sogst';

import { packGsplatRange, packGsplatSHRange } from './gsplat-range-sync';
import type { SplatSource } from './gsplat-range-sync';
import { SogstData } from './sogst-data';
import { packSogstMotion } from './sogst-motion';
import type { SogstMotionSource } from './sogst-motion';
import { SH_COEFFS_PER_CHANNEL, shRestCount } from './sogst-target';
import type { SogstTarget } from './sogst-target';
import { decodeTexture, readTexels, texelsOf, WebpTexelWorker } from './sogst-texels';
import type { TexelImage } from './sogst-texels';

// GSplatSogData._patchCodebooks is private to the engine. It is the one
// member this file uses that has no public equivalent. The cast names the
// member instead of hiding it behind a blanket `as any`. If the engine drops
// the member, look at this line first.
type SogCodebookPatch = { _patchCodebooks?: () => void };

// SOG's own container version. A .sogst archive carries meta.version 1, but
// its static attributes follow the SOG v2 conventions byte for byte.
// GSplatSogData selects its decode path from meta.version, so the per-group
// shim below advertises 2.
const SOG_META_VERSION = 2;

// Zeroth-order SH basis function. SH_DC_OFFSET is the offset SOG adds before
// it stores DC colour, which moves the value into [0, 1]. Inverting both
// recovers the f_dc_* value.
const SH_C0 = 0.28209479177387814;
const SH_DC_OFFSET = 0.5;

// The engine's SOG iterator writes higher-order SH into a 45-float scratch
// buffer with a fixed stride of 15 coefficients per colour channel, whatever
// the band count. Bands 1 to 3 hold 3 + 5 + 7 = 15 coefficients per channel.
const SH_ITER_COEFFS = 45;
const SH_ITER_CHANNEL_STRIDE = 15;

// The decoder stores opacity as a logit. It pins two kinds of splat to plus
// or minus this magnitude: splats it has not decoded yet, and splats whose
// stored alpha saturates. Sigmoid is flat to float precision well before
// this magnitude, so a larger value adds no range. A larger value also makes
// the inverse return NaN.
const OPACITY_LOGIT_LIMIT = 40;

// Quantisation range of the 16-bit split-plane textures. RGBA_STRIDE is the
// texel stride the decoder reads every attribute texture at.
const U16_MAX = 65535;
const RGBA_STRIDE = 4;

// trbf.webp channel assignment (see parsers/sogst.ts). R indexes the
// t_center codebook and G indexes the t_sigma codebook.
const TRBF_CENTER_CHANNEL = 0;
const TRBF_SIGMA_CHANNEL = 1;

// Main-thread budget per decode slice. On streaming loads the decode runs
// behind live playback. A slice longer than about half a frame causes
// visible stutter.
const DECODE_SLICE_MS = 6;

// Reading the clock costs more than one loop iteration, so the elapsed-time
// test runs on a power-of-two stride (`i & (STRIDE - 1)`). The temporal loop
// has the cheaper body of the two, so it tests less often.
const STATIC_CLOCK_STRIDE = 64;
const TEMPORAL_CLOCK_STRIDE = 256;

// Yield to the event loop so that the progress UI can repaint during a
// decode. This uses setTimeout and not requestAnimationFrame, because rAF
// never fires in a hidden tab.
const yieldToUi = () =>
    new Promise((resolve) => {
        setTimeout(resolve, 0);
    });

// Invert one channel of a 16-bit split-plane pair. The `_u` texture holds
// the high byte and `_l` holds the low byte. Together they quantise
// [min, max] across the full u16 range. The encoder stores the values
// log-transformed as sign(x)*ln(1+|x|), and the exponential inverts that.
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

// Canonical per-group texture names. A monolithic archive uses these names
// unchanged. A streamed archive prefixes each one with its group directory
// ("persistent/", "seg_000/", and so on). shN_centroids is global in both
// layouts.
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

// Per-group texture names for this archive. Degree-2 content adds the accel
// split pair to every group.
const groupBaseNames = (meta: SogstMeta): string[] => {
    return meta.accel ? [...GROUP_FILE_NAMES, 'accel_l.webp', 'accel_u.webp'] : GROUP_FILE_NAMES;
};

type SogstGroup = {
    prefix: string | null; // null => monolithic (bare names)
    range: [number, number];
    segIndex: number; // index into meta.segments.list; -1 otherwise
};

// Decode groups in play order. A monolithic archive has one group covering
// the whole file. A streamed archive has [persistent, seg_000, seg_001, ...]
// and omits empty groups.
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

// Splats per decode chunk. The decoder never holds more than one chunk of
// decoded attributes: it decodes a chunk into scratch arrays, packs it into
// the target, and reuses the arrays. The scratch costs about 64 floats per
// splat (about 8 MB at this size); the value only trades that against loop
// overhead.
const DECODE_CHUNK_SPLATS = 32768;

// Splats per pack call. Packing is several times the per-splat cost of
// decoding (half-float conversion, SH quantisation), so it checks the clock
// after every call of this many splats.
const PACK_STEP_SPLATS = 256;

// Incremental decoder. It decodes one group's payloads at a time and packs
// the result straight into a SogstTarget at a destination index, a chunk at
// a time. It holds no whole-clip arrays.
//
// A streaming caller decodes groups as they arrive; a caller with a complete
// buffer uses the same class and decodes all groups in turn.
class SogstDecoder {
    private app: AppBase;

    readonly meta: SogstMeta;

    readonly target: SogstTarget;

    private scratch: SplatSource;

    private scratchMotion: SogstMotionSource;

    private restCount: number;

    private centroidsBytes: Uint8Array | null = null;

    private centroidsTexture: TexelImage | Texture | null = null;

    private texelWorker = new WebpTexelWorker();

    private texelWorkerBroken = false;

    constructor(app: AppBase, meta: SogstMeta, target: SogstTarget) {
        this.app = app;
        this.meta = meta;
        this.target = target;
        this.restCount = meta.shN ? shRestCount(meta.shN.bands) : 0;
        const names = [
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
        for (let i = 0; i < this.restCount; i++) {
            names.push(`f_rest_${i}`);
        }
        const chunk = () => new Float32Array(DECODE_CHUNK_SPLATS);
        this.scratch = {};
        names.forEach((name) => {
            this.scratch[name] = chunk();
        });
        this.scratchMotion = {
            velocity: [chunk(), chunk(), chunk()],
            accel: meta.accel ? [chunk(), chunk(), chunk()] : null,
            tCenter: chunk(),
            tSigma: chunk()
        };
    }

    // The compact f_rest_* scratch arrays, in index order.
    private restArrays(): Float32Array[] {
        return Array.from({ length: this.restCount }, (_, j) => this.scratch[`f_rest_${j}`]);
    }

    setCentroids(bytes: Uint8Array) {
        this.centroidsBytes = bytes;
    }

    // Decode webp to raw texels off the main thread. If the worker cannot
    // run, this uses the app context's upload and readback path instead.
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

    // Pack scratch splats [0, n) into the target at [dest, dest + n), in
    // time-budgeted slices. `withSH` packs the f_rest scratch too; a group
    // decoded without its SH labels must not, or it would pack stale values.
    private async packChunk(dest: number, n: number, withSH: boolean) {
        const resource = this.target.resource;
        for (let j = 0; j < n;) {
            const sliceStart = performance.now();
            while (j < n) {
                if (this.target.destroyed) {
                    return;
                }
                const k = Math.min(n - j, PACK_STEP_SPLATS);
                packGsplatRange(resource, this.scratch, j, dest + j, k, withSH);
                packSogstMotion(resource, this.scratchMotion, j, dest + j, k);
                j += k;
                if (performance.now() - sliceStart >= DECODE_SLICE_MS) {
                    break;
                }
            }

            await yieldToUi();
        }
    }

    // Decode one group's texture payloads and pack them into the target at
    // [dest, dest + group size), then upload those rows. `dest` defaults to
    // the group's file range, the fully resident layout. The payload map is
    // keyed by bare canonical name.
    async decodeGroup(
        group: SogstGroup,
        files: Map<string, Uint8Array>,
        onProgress?: (frac: number) => void,
        dest = group.range[0]
    ) {
        const m = group.range[1] - group.range[0];
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

        // SH decodes with the group only when its labels are present. An
        // sh-deferred archive writes all labels after the geometry, so those
        // groups use a later decodeGroupSH pass instead.
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

        // Static attributes decode through the engine's own SOG iterator. A
        // per-group shim gives the iterator the group textures together with
        // the global codebooks and mins. GSplatSogData selects its decode
        // path from meta.version, so the shim advertises the SOG container
        // version and not the .sogst one.
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

        // Temporal attributes, which the engine's SOG model does not cover.
        // The batch above already read their texels back, so only the
        // textures remain to free.
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
        if (this.meta.accel) {
            accelL = texelsOf(tex.get('accel_l.webp'));
            accelU = texelsOf(tex.get('accel_u.webp'));
            tex.get('accel_l.webp')!.destroy();
            tex.get('accel_u.webp')!.destroy();
        }

        const vMins = this.meta.motion.mins as number[];
        const vMaxs = this.meta.motion.maxs as number[];
        const centerCodebook = this.meta.trbf.center.codebook as number[];
        const sigmaCodebook = this.meta.trbf.sigma.codebook as number[];

        const p = new Vec3();
        const r = new Quat();
        const s = new Vec3();
        const c = new Vec4();
        const sh = sog.shBands > 0 ? new Float32Array(SH_ITER_COEFFS) : null;
        const iter = sog.createIter(p, r, s, c, sh);
        const arrays = this.scratch;
        const restArrays = sh ? this.restArrays() : null;
        const coeffs = SH_COEFFS_PER_CHANNEL[sog.shBands] ?? 0;
        const { velocity, accel, tCenter, tSigma } = this.scratchMotion;

        // Each loop below runs in time-budgeted slices, not a fixed count. On
        // streaming loads the decode runs behind live playback, so no slice
        // may hold the main thread for more than a few milliseconds.
        //
        // A count-based slice cannot meet that bound. Segment sizes vary
        // widely, up to hundreds of thousands of splats on a dense scene. A
        // fixed count therefore yields either too rarely, which causes
        // stutter, or too often, which wastes time.
        for (let c0 = 0; c0 < m; c0 += DECODE_CHUNK_SPLATS) {
            const n = Math.min(DECODE_CHUNK_SPLATS, m - c0);

            for (let j = 0; j < n;) {
                const sliceStart = performance.now();
                while (j < n) {
                    iter.read(c0 + j);
                    arrays.x[j] = p.x;
                    arrays.y[j] = p.y;
                    arrays.z[j] = p.z;
                    arrays.rot_0[j] = r.w;
                    arrays.rot_1[j] = r.x;
                    arrays.rot_2[j] = r.y;
                    arrays.rot_3[j] = r.z;
                    arrays.scale_0[j] = s.x;
                    arrays.scale_1[j] = s.y;
                    arrays.scale_2[j] = s.z;
                    // The spec lets an encoder lose the RGB of any texel
                    // whose alpha is zero. libwebp can rewrite a
                    // fully-transparent block when the `exact` flag is not
                    // available. Nothing may therefore *depend* on the colour
                    // this line reads.
                    //
                    // Storing the colour unconditionally is safe only because
                    // the splat stays invisible. Opacity saturates to
                    // -OPACITY_LOGIT_LIMIT below, and the temporal factor in
                    // sogst-motion.ts multiplies alpha by exp(-0.5*dt^2),
                    // which is never above 1. Do not add a path that scales
                    // alpha up unless it first tests c.w > 0 here.
                    arrays.f_dc_0[j] = (c.x - SH_DC_OFFSET) / SH_C0;
                    arrays.f_dc_1[j] = (c.y - SH_DC_OFFSET) / SH_C0;
                    arrays.f_dc_2[j] = (c.z - SH_DC_OFFSET) / SH_C0;
                    arrays.opacity[j] =
                        c.w <= 0 ? -OPACITY_LOGIT_LIMIT : c.w >= 1 ? OPACITY_LOGIT_LIMIT : -Math.log(1 / c.w - 1);
                    if (sh && restArrays) {
                        for (let ch = 0; ch < 3; ch++) {
                            for (let k = 0; k < coeffs; k++) {
                                restArrays[ch * coeffs + k][j] = sh[ch * SH_ITER_CHANNEL_STRIDE + k];
                            }
                        }
                    }
                    j++;
                    if ((j & (STATIC_CLOCK_STRIDE - 1)) === 0 && performance.now() - sliceStart >= DECODE_SLICE_MS) {
                        break;
                    }
                }

                await yieldToUi();
            }

            for (let j = 0; j < n;) {
                const sliceStart = performance.now();
                while (j < n) {
                    const i = c0 + j;
                    for (let ch = 0; ch < 3; ch++) {
                        velocity[ch][j] = decodeSplit16(motionL, motionU, i, ch, vMins[ch], vMaxs[ch]);
                    }
                    if (accel && accelL && accelU && aMins && aMaxs) {
                        for (let ch = 0; ch < 3; ch++) {
                            accel[ch][j] = decodeSplit16(accelL, accelU, i, ch, aMins[ch], aMaxs[ch]);
                        }
                    }
                    tCenter[j] = centerCodebook[trbf[i * RGBA_STRIDE + TRBF_CENTER_CHANNEL]];
                    tSigma[j] = sigmaCodebook[trbf[i * RGBA_STRIDE + TRBF_SIGMA_CHANNEL]];
                    j++;
                    if ((j & (TEMPORAL_CLOCK_STRIDE - 1)) === 0 && performance.now() - sliceStart >= DECODE_SLICE_MS) {
                        break;
                    }
                }

                await yieldToUi();
            }

            await this.packChunk(dest + c0, n, !!sh);
            if (this.target.destroyed) {
                break;
            }
            onProgress?.((c0 + n) / m);
        }

        // The groups share the centroids texture. Detach it so that the
        // shim's destroy() releases the group-local textures only.
        sog.sh_centroids = null;
        sog.destroy();

        await this.target.upload(dest, dest + m);
    }

    // Decode a deferred SH labels payload for one group and pack it into the
    // target at [dest, dest + group size). An sh-deferred archive writes all
    // labels after the geometry, so the scene can appear with DC colour only
    // and add view dependence later.
    async decodeGroupSH(group: SogstGroup, labelsBytes: Uint8Array, dest = group.range[0]) {
        const m = group.range[1] - group.range[0];
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

        // SH-only iterator. Null attribute targets make it skip every
        // texture except sh_labels and sh_centroids.
        const sog = new GSplatSogData();
        sog.meta = { ...this.meta, version: SOG_META_VERSION, count: m };
        sog.numSplats = m;
        sog.sh_labels = labels;
        sog.sh_centroids = this.centroidsTexture;
        sog.shBands = this.meta.shN.bands;
        (sog as unknown as SogCodebookPatch)._patchCodebooks?.();

        const sh = new Float32Array(SH_ITER_COEFFS);
        const iter = sog.createIter(null, null, null, null, sh);
        const restArrays = this.restArrays();
        const coeffs = SH_COEFFS_PER_CHANNEL[sog.shBands] ?? 0;
        const resource = this.target.resource;
        for (let c0 = 0; c0 < m; c0 += DECODE_CHUNK_SPLATS) {
            const n = Math.min(DECODE_CHUNK_SPLATS, m - c0);
            for (let j = 0; j < n;) {
                const sliceStart = performance.now();
                while (j < n) {
                    iter.read(c0 + j);
                    for (let ch = 0; ch < 3; ch++) {
                        for (let k = 0; k < coeffs; k++) {
                            restArrays[ch * coeffs + k][j] = sh[ch * SH_ITER_CHANNEL_STRIDE + k];
                        }
                    }
                    j++;
                    if ((j & (STATIC_CLOCK_STRIDE - 1)) === 0 && performance.now() - sliceStart >= DECODE_SLICE_MS) {
                        break;
                    }
                }

                await yieldToUi();
            }

            for (let j = 0; j < n;) {
                const sliceStart = performance.now();
                while (j < n) {
                    if (this.target.destroyed) {
                        return;
                    }
                    const k = Math.min(n - j, PACK_STEP_SPLATS);
                    packGsplatSHRange(resource, this.scratch, j, dest + c0 + j, k);
                    j += k;
                    if (performance.now() - sliceStart >= DECODE_SLICE_MS) {
                        break;
                    }
                }

                await yieldToUi();
            }
        }

        sog.sh_centroids = null;
        sog.destroy();

        await this.target.upload(dest, dest + m, true);
    }

    buildData(): SogstData {
        return new SogstData(this.meta, this.target.resource);
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
