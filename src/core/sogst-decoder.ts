import type { AppBase } from 'playcanvas';

import type { SogstMeta } from '../parsers/sogst';

import { SogstData } from './sogst-data';
import { SogstGpuDecoder } from './sogst-gpu-decode';
import { SogstPackPool } from './sogst-pack-pool';
import type { PackedRows, PackMeta, RawTexels } from './sogst-pack-worker';
import type { SogstTarget } from './sogst-target';
import { decodeTexture, readTexels } from './sogst-texels';

// Group decoding for .sogst. Unpacking the SOG and temporal attributes and
// packing them into the GPU stream layouts runs in a pool of workers
// (sogst-pack-worker.ts); the main thread only splits a group into jobs,
// copies each job's rows into the target's texture CPU copies at the
// destination index, and uploads them.

const LABELS_NAME = 'shN_labels.webp';

// Least splats per worker job. Every job decodes all of its group's WebP
// payloads, so splitting a group costs one more decode of them; below this
// size a split does not pay for that.
const MIN_JOB_SPLATS = 131072;

// Fields of the SOG manifest the pack workers need that SogstMeta does not
// declare. They follow SOG v2: 256-entry codebooks.
type SogCodebooks = {
    scales: { codebook: (number | null)[] };
    sh0: { codebook: (number | null)[] };
    shN?: { codebook: (number | null)[]; bands: number };
};

// The engine's GSplatSogData._patchCodebooks: an encoder may write the first
// entry as null, to be extrapolated from its neighbours.
const patchCodebook = (codebook: (number | null)[]): number[] => {
    const out = [...codebook];
    if (out[0] === null) {
        out[0] = (out[1] as number) + ((out[1] as number) - (out[255] as number)) / 255;
    }
    return out as number[];
};

const packMetaOf = (meta: SogstMeta): PackMeta => {
    const sog = meta as unknown as SogCodebooks;
    return {
        means: { mins: meta.means.mins, maxs: meta.means.maxs },
        scales: { codebook: patchCodebook(sog.scales.codebook) },
        sh0: { codebook: patchCodebook(sog.sh0.codebook) },
        shN: meta.shN && sog.shN ? { codebook: patchCodebook(sog.shN.codebook), bands: meta.shN.bands } : null,
        motion: { mins: meta.motion.mins, maxs: meta.motion.maxs },
        accel: meta.accel ? { mins: meta.accel.mins, maxs: meta.accel.maxs } : null,
        trbf: { center: meta.trbf.center.codebook, sigma: meta.trbf.sigma.codebook }
    };
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

// Where a group's splats go in the target: runs mapping group-local source
// splats [src, src + count) onto target splats [dest, dest + count). A fully
// resident group is one run at its file range; a windowed segment has one run
// per stretch of consecutive arena pages.
type SplatPlacement = { src: number; dest: number; count: number }[];

const placementOf = (dest: number | SplatPlacement, count: number): SplatPlacement =>
    typeof dest === 'number' ? [{ src: 0, dest, count }] : dest;

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

// Decodes groups into a SogstTarget at a destination index. A streaming
// caller decodes groups as they arrive; a caller with a complete buffer
// decodes all groups in turn; the segment window decodes evicted segments
// again into whatever slot is free. Decodes may overlap: the pool runs their
// jobs in priority order.
class SogstDecoder {
    private app: AppBase;
    readonly meta: SogstMeta;
    readonly target: SogstTarget;
    private pool = new SogstPackPool();
    private packMeta: PackMeta;
    private centroids: Uint8Array | null = null;

    // Set once a worker reports no WebGL2: the main thread then decodes the
    // WebP payloads to texels and the workers only unpack and pack.
    private rawTexels = false;

    // GPU decode (sogst-gpu-decode.ts), when requested and supported.
    private gpu: SogstGpuDecoder | null = null;

    constructor(app: AppBase, meta: SogstMeta, target: SogstTarget, options: { gpu?: boolean } = {}) {
        this.app = app;
        this.meta = meta;
        this.target = target;
        this.packMeta = packMetaOf(meta);
        this.pool.init(this.packMeta, null);
        if (options.gpu && SogstGpuDecoder.supported(app.graphicsDevice)) {
            this.gpu = new SogstGpuDecoder(app.graphicsDevice, meta, this.packMeta, target, (files, count, priority) =>
                this.decodeTexels(files, count, priority)
            );
        }
    }

    // Decodes that can usefully run at once: one per pack worker.
    get concurrency(): number {
        // GPU passes run one at a time; two in flight overlap one install's
        // WebP decode with another's passes
        return this.gpu ? 2 : this.pool.size;
    }

    setCentroids(bytes: Uint8Array) {
        this.centroids = bytes;
        this.pool.init(this.packMeta, bytes);
        this.gpu?.setCentroids(bytes);
    }

    // Main-thread WebP decode, for the no-WebGL2-in-workers fallback.
    private async toTexels(bytes: Uint8Array, name: string): Promise<RawTexels> {
        const texture = await decodeTexture(this.app, bytes, name);
        const data = await readTexels(texture);
        const width = texture.width;
        texture.destroy();
        return { width, data };
    }

    private async texelFiles(files: Record<string, Uint8Array | RawTexels>) {
        const out: Record<string, RawTexels> = {};
        for (const [name, payload] of Object.entries(files)) {
            out[name] = payload instanceof Uint8Array ? await this.toTexels(payload, name) : payload;
        }
        return out;
    }

    private async pack(request: Parameters<SogstPackPool['run']>[0], priority: number): Promise<PackedRows> {
        if (this.rawTexels) {
            return this.pool.run({ ...request, files: await this.texelFiles(request.files) }, priority);
        }
        try {
            return await this.pool.run(request, priority);
        } catch (err) {
            if (!/webgl2/i.test(String(err))) {
                throw err;
            }
            if (!this.rawTexels) {
                console.warn('sogst: pack workers have no WebGL2; decoding WebP on the main thread instead');
                this.rawTexels = true;
                if (this.centroids) {
                    this.pool.init(this.packMeta, await this.toTexels(this.centroids, 'shN_centroids'));
                }
            }
            return this.pool.run({ ...request, files: await this.texelFiles(request.files) }, priority);
        }
    }

    // Decode WebP payloads to raw texels in the pack workers, for the GPU
    // path: one job per file, so a group's files decode in parallel, except
    // the means pair, which shares a job that also computes the group's
    // `count` centers (the CPU sorter's input) from it.
    async decodeTexels(
        files: Map<string, Uint8Array>,
        count: number,
        priority: number
    ): Promise<{ texels: Record<string, RawTexels>; centers: Float32Array | null }> {
        const means = ['means_l.webp', 'means_u.webp'];
        const jobs: string[][] = [];
        if (means.every((name) => files.has(name))) {
            jobs.push(means);
        }
        for (const name of files.keys()) {
            if (!jobs[0]?.includes(name)) {
                jobs.push([name]);
            }
        }
        const results = await Promise.all(
            jobs.map((names) =>
                this.pack(
                    {
                        files: Object.fromEntries(names.map((name) => [name, files.get(name)!])),
                        start: 0,
                        count: names === means ? count : 0,
                        geometry: false,
                        sh: false,
                        mode: 'texels'
                    },
                    priority
                )
            )
        );
        const texels: Record<string, RawTexels> = {};
        let centers: Float32Array | null = null;
        for (const result of results) {
            Object.assign(texels, result.texels);
            centers = result.centers ?? centers;
        }
        return { texels, centers };
    }

    // Split a group into jobs, and write and upload each job's rows into the
    // placement runs they overlap as the job finishes.
    private async run(
        group: SogstGroup,
        files: Map<string, Uint8Array>,
        placement: SplatPlacement,
        geometry: boolean,
        sh: boolean,
        priority: number,
        onProgress?: (frac: number) => void
    ) {
        const m = group.range[1] - group.range[0];
        if (m <= 0) {
            return;
        }
        const jobs = Math.min(this.pool.size, Math.max(1, Math.ceil(m / MIN_JOB_SPLATS)));
        const per = Math.ceil(m / jobs);
        const record = Object.fromEntries(files);
        let done = 0;
        await Promise.all(
            Array.from({ length: jobs }, async (_, k) => {
                const start = k * per;
                const count = Math.min(per, m - start);
                if (count <= 0) {
                    return;
                }
                const rows = await this.pack({ files: record, start, count, geometry, sh }, priority);
                if (this.target.destroyed) {
                    return;
                }
                const pieces: [number, number][] = [];
                for (const run of placement) {
                    const a = Math.max(start, run.src);
                    const b = Math.min(start + count, run.src + run.count);
                    if (b > a) {
                        const d = run.dest + (a - run.src);
                        this.target.writeRows(d, rows, a - start, b - a);
                        pieces.push([d, d + b - a]);
                    }
                }
                for (const [a, b] of pieces) {
                    await this.target.upload(a, b, !geometry);
                }
                done += count;
                onProgress?.(done / m);
            })
        );
    }

    // Decode one group's payloads into the target and upload them. `dest` is
    // the first target splat, or a placement of runs; it defaults to the
    // group's file range, the fully resident layout. The payload map is keyed
    // by bare canonical name; with shN_labels.webp in it, SH decodes with the
    // group. Lower `priority` runs first.
    decodeGroup(
        group: SogstGroup,
        files: Map<string, Uint8Array>,
        onProgress?: (frac: number) => void,
        dest: number | SplatPlacement = group.range[0],
        priority = 0
    ): Promise<void> {
        const placement = placementOf(dest, group.range[1] - group.range[0]);
        for (const name of groupBaseNames(this.meta)) {
            if (!files.has(name)) {
                return Promise.reject(new Error(`sogst: ${group.prefix ?? ''}/${name} missing from archive`));
            }
        }
        const sh = !!this.meta.shN && files.has(LABELS_NAME);
        if (sh && !this.centroids) {
            return Promise.reject(new Error('sogst: shN_centroids payload not provided before group decode'));
        }
        if (this.gpu) {
            return this.gpu.decodeGroup(group, files, placement, priority).then(() => onProgress?.(1));
        }
        return this.run(group, files, placement, true, sh, priority, onProgress);
    }

    // Decode a deferred SH labels payload for one group into the target at
    // `dest` (see decodeGroup). An sh-deferred archive writes all labels
    // after the geometry, so the scene can appear with DC colour only and
    // add view dependence later.
    decodeGroupSH(
        group: SogstGroup,
        labels: Uint8Array,
        dest: number | SplatPlacement = group.range[0],
        priority = 0
    ): Promise<void> {
        const placement = placementOf(dest, group.range[1] - group.range[0]);
        if (!this.meta.shN) {
            return Promise.resolve();
        }
        if (!this.centroids) {
            return Promise.reject(new Error('sogst: shN_centroids payload must precede deferred labels'));
        }
        if (this.gpu) {
            return this.gpu.decodeGroup(group, new Map([[LABELS_NAME, labels]]), placement, priority);
        }
        return this.run(group, new Map([[LABELS_NAME, labels]]), placement, false, true, priority);
    }

    buildData(): SogstData {
        return new SogstData(this.meta, this.target.resource);
    }

    destroy() {
        this.pool.destroy();
        this.gpu?.destroy();
    }
}

export { enumerateSogstGroups, groupBaseNames, groupFileList, GROUP_FILE_NAMES, SogstDecoder };
export type { SogstGroup, SplatPlacement };
