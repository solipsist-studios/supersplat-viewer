import { GSplatData } from 'playcanvas';

import type { Omg4FrameData } from '../parsers/omg4';

const MAGIC = 0x34474D4F;
const HEADER_SIZE = 28;
const FLOATS_PER_SPLAT = 14;
const MAX_CACHED_FRAMES = 32;
const FRAMES_PER_CHUNK = 8;
const OMG4_CACHE_NAME = 'supersplat-omg4-v1';
const OMG4_IDB_NAME = 'supersplat-omg4-chunks';
const OMG4_IDB_STORE = 'ranges';
const OMG4_DEBUG_LOG = true;

type Omg4Header = {
    version: number;
    numSplats: number;
    numFrames: number;
    fps: number;
    timeDurationMin: number;
    timeDurationMax: number;
};

type WorkArrays = {
    x: Float32Array;
    y: Float32Array;
    z: Float32Array;
    rot0: Float32Array;
    rot1: Float32Array;
    rot2: Float32Array;
    rot3: Float32Array;
    scale0: Float32Array;
    scale1: Float32Array;
    scale2: Float32Array;
    opacity: Float32Array;
    fdc0: Float32Array;
    fdc1: Float32Array;
    fdc2: Float32Array;
};

const rangeCacheKey = (url: string, start: number, end: number) => `${new URL(url, location.href).toString()}?__omg4_range=${start}-${end}`;

let omg4DbPromise: Promise<IDBDatabase | null> | null = null;

const openOmg4Db = (): Promise<IDBDatabase | null> => {
    if (typeof indexedDB === 'undefined') {
        return Promise.resolve(null);
    }

    if (omg4DbPromise) {
        return omg4DbPromise;
    }

    omg4DbPromise = new Promise((resolve) => {
        const request = indexedDB.open(OMG4_IDB_NAME, 1);

        request.onupgradeneeded = () => {
            const db = request.result;
            if (!db.objectStoreNames.contains(OMG4_IDB_STORE)) {
                db.createObjectStore(OMG4_IDB_STORE);
            }
        };

        request.onsuccess = () => resolve(request.result);
        request.onerror = () => resolve(null);
    });

    return omg4DbPromise;
};

const idbGetRange = async (key: string): Promise<ArrayBuffer | null> => {
    const db = await openOmg4Db();
    if (!db) {
        return null;
    }

    return new Promise((resolve) => {
        const tx = db.transaction(OMG4_IDB_STORE, 'readonly');
        const store = tx.objectStore(OMG4_IDB_STORE);
        const request = store.get(key);

        request.onsuccess = () => {
            const value = request.result;
            resolve(value instanceof ArrayBuffer ? value : null);
        };
        request.onerror = () => resolve(null);
    });
};

const idbSetRange = async (key: string, buffer: ArrayBuffer): Promise<void> => {
    const db = await openOmg4Db();
    if (!db) {
        return;
    }

    await new Promise<void>((resolve) => {
        const tx = db.transaction(OMG4_IDB_STORE, 'readwrite');
        const store = tx.objectStore(OMG4_IDB_STORE);
        store.put(buffer, key);
        tx.oncomplete = () => resolve();
        tx.onerror = () => resolve();
        tx.onabort = () => resolve();
    });
};

const fetchRangeNetwork = async (url: string, start: number, end: number): Promise<ArrayBuffer> => {
    const response = await fetch(url, {
        headers: {
            Range: `bytes=${start}-${end}`
        }
    });

    if (response.status !== 206) {
        throw new Error(`OMG4 streaming requires byte-range support. ${url} responded with ${response.status}.`);
    }

    return response.arrayBuffer();
};

const fetchRange = async (url: string, start: number, end: number): Promise<ArrayBuffer> => {
    const key = rangeCacheKey(url, start, end);

    // First try IndexedDB for durable local chunk cache.
    const idbCached = await idbGetRange(key);
    if (idbCached) {
        if (OMG4_DEBUG_LOG) console.debug('OMG4 range cache hit (idb)', key);
        return idbCached;
    }

    // Keep a persistent local cache of fetched byte ranges so playback loops
    // do not repeatedly hammer the network for the same OMG4 data.
    const hasCacheApi = typeof caches !== 'undefined';
    if (!hasCacheApi) {
        const networkBuffer = await fetchRangeNetwork(url, start, end);
        if (OMG4_DEBUG_LOG) console.debug('OMG4 range fetch (network/no-cache-api)', key);
        await idbSetRange(key, networkBuffer);
        return networkBuffer;
    }

    const cache = await caches.open(OMG4_CACHE_NAME);
    const request = new Request(key);

    const cached = await cache.match(request);
    if (cached) {
        if (OMG4_DEBUG_LOG) console.debug('OMG4 range cache hit (cache storage)', key);
        const buffer = await cached.arrayBuffer();
        await idbSetRange(key, buffer);
        return buffer;
    }

    const buffer = await fetchRangeNetwork(url, start, end);
    if (OMG4_DEBUG_LOG) console.debug('OMG4 range fetch (network)', key);
    await cache.put(request, new Response(buffer, {
        headers: {
            'Content-Type': 'application/octet-stream'
        }
    }));
    await idbSetRange(key, buffer);

    return buffer;
};

class StreamedOmg4Data implements Omg4FrameData {
    readonly header: Omg4Header;

    readonly gsplatData: GSplatData;

    private url: string;

    private frameByteSize: number;

    private work: WorkArrays;

    private frameCache: Map<number, Uint8Array>;

    private frameOrder: number[];

    private prefetchTargetFrame: number;

    private nextFrameToFetch: number;

    private prefetchPromise: Promise<void> | null;

    private maxSequentialReadyFrame: number;

    private constructor(url: string, header: Omg4Header, frame0: Uint8Array) {
        this.url = url;
        this.header = header;
        this.frameByteSize = 4 + header.numSplats * FLOATS_PER_SPLAT * 4;
        this.frameCache = new Map([[0, frame0]]);
        this.frameOrder = [0];
        this.prefetchTargetFrame = 0;
        this.nextFrameToFetch = 1;
        this.prefetchPromise = null;
        this.maxSequentialReadyFrame = 0;

        const count = header.numSplats;
        this.work = {
            x: new Float32Array(count),
            y: new Float32Array(count),
            z: new Float32Array(count),
            rot0: new Float32Array(count),
            rot1: new Float32Array(count),
            rot2: new Float32Array(count),
            rot3: new Float32Array(count),
            scale0: new Float32Array(count),
            scale1: new Float32Array(count),
            scale2: new Float32Array(count),
            opacity: new Float32Array(count),
            fdc0: new Float32Array(count),
            fdc1: new Float32Array(count),
            fdc2: new Float32Array(count)
        };

        const prop = (name: string, storage: Float32Array) => ({
            type: 'float' as const,
            name,
            storage,
            byteSize: 4
        });

        this.gsplatData = new GSplatData([{
            name: 'vertex',
            count,
            properties: [
                prop('x', this.work.x),
                prop('y', this.work.y),
                prop('z', this.work.z),
                prop('rot_0', this.work.rot0),
                prop('rot_1', this.work.rot1),
                prop('rot_2', this.work.rot2),
                prop('rot_3', this.work.rot3),
                prop('scale_0', this.work.scale0),
                prop('scale_1', this.work.scale1),
                prop('scale_2', this.work.scale2),
                prop('opacity', this.work.opacity),
                prop('f_dc_0', this.work.fdc0),
                prop('f_dc_1', this.work.fdc1),
                prop('f_dc_2', this.work.fdc2)
            ]
        }]);

        this.copyFrame(frame0);
    }

    static async create(url: string, onProgress: (progress: number) => void): Promise<StreamedOmg4Data> {
        const headerBuffer = await fetchRange(url, 0, HEADER_SIZE - 1);
        onProgress(5);

        const headerView = new DataView(headerBuffer);
        const magic = headerView.getUint32(0, true);
        if (magic !== MAGIC) {
            throw new Error(`Invalid .omg4 file: expected magic 0x${MAGIC.toString(16)}, got 0x${magic.toString(16)}`);
        }

        const header: Omg4Header = {
            version: headerView.getUint32(4, true),
            numSplats: headerView.getUint32(8, true),
            numFrames: headerView.getUint32(12, true),
            fps: headerView.getFloat32(16, true),
            timeDurationMin: headerView.getFloat32(20, true),
            timeDurationMax: headerView.getFloat32(24, true)
        };

        const frameByteSize = 4 + header.numSplats * FLOATS_PER_SPLAT * 4;
        const frame0 = new Uint8Array(await fetchRange(url, HEADER_SIZE, HEADER_SIZE + frameByteSize - 1));
        onProgress(100);

        return new StreamedOmg4Data(url, header, frame0);
    }

    get numFrames(): number {
        return this.header.numFrames;
    }

    get duration(): number {
        const timestampDuration = this.header.timeDurationMax - this.header.timeDurationMin;
        if (this.header.numFrames <= 1) {
            return 0;
        }
        return timestampDuration > 0 ? timestampDuration : (this.header.fps > 0 ? (this.header.numFrames - 1) / this.header.fps : 0);
    }

    getFrameIndex(time: number): number {
        if (this.header.numFrames <= 1) {
            return 0;
        }

        const duration = this.duration;
        if (duration <= 0) {
            return this.header.fps > 0 ? Math.max(0, Math.min(this.header.numFrames - 1, Math.round(time * this.header.fps))) : 0;
        }

        const clampedTime = Math.max(0, Math.min(duration, time));
        const desiredFrame = Math.max(0, Math.min(this.header.numFrames - 1, Math.round(clampedTime / duration * (this.header.numFrames - 1))));
        return Math.min(desiredFrame, this.maxSequentialReadyFrame);
    }

    async loadFrame(frameIndex: number): Promise<void> {
        const frame = await this.getFrameBuffer(frameIndex);
        this.copyFrame(frame);
    }

    prefetchFrame(frameIndex: number): void {
        if (frameIndex < 0 || frameIndex >= this.header.numFrames || this.frameCache.has(frameIndex)) {
            return;
        }
        this.prefetchTargetFrame = Math.max(this.prefetchTargetFrame, frameIndex);
        this.ensurePrefetchRunning();
    }

    private async getFrameBuffer(frameIndex: number): Promise<Uint8Array> {
        const cached = this.frameCache.get(frameIndex);
        if (cached) {
            this.touchFrame(frameIndex);
            return cached;
        }

        if (frameIndex < this.nextFrameToFetch) {
            await this.fetchSpecificChunkForFrame(frameIndex);
        } else {
            this.prefetchTargetFrame = Math.max(this.prefetchTargetFrame, frameIndex);
            await this.ensurePrefetchRunning();
        }

        if (!this.frameCache.has(frameIndex)) {
            // If the requested frame still isn't present, prefetch may have stopped at an older target.
            await this.fetchSpecificChunkForFrame(frameIndex);
        }

        const fetched = this.frameCache.get(frameIndex);
        if (!fetched) {
            throw new Error(`Failed to load OMG4 frame ${frameIndex}`);
        }

        this.touchFrame(frameIndex);
        return fetched;
    }

    private async ensurePrefetchRunning(): Promise<void> {
        if (this.prefetchPromise) {
            return this.prefetchPromise;
        }

        this.prefetchPromise = this.runPrefetch().finally(() => {
            this.prefetchPromise = null;
        });

        return this.prefetchPromise;
    }

    private async runPrefetch(): Promise<void> {
        while (this.nextFrameToFetch <= this.prefetchTargetFrame && this.nextFrameToFetch < this.header.numFrames) {
            const startFrame = this.nextFrameToFetch;
            const endFrame = Math.min(this.header.numFrames - 1, startFrame + FRAMES_PER_CHUNK - 1);
            await this.fetchChunkRange(startFrame, endFrame);

            this.nextFrameToFetch = endFrame + 1;
            this.maxSequentialReadyFrame = Math.max(this.maxSequentialReadyFrame, endFrame);
            this.pruneCache();
        }

        if (this.nextFrameToFetch <= this.prefetchTargetFrame && this.nextFrameToFetch < this.header.numFrames) {
            // Target moved while awaiting I/O; continue pumping in order.
            return this.runPrefetch();
        }
    }

    private async fetchSpecificChunkForFrame(frameIndex: number): Promise<void> {
        const chunkStartFrame = Math.floor(frameIndex / FRAMES_PER_CHUNK) * FRAMES_PER_CHUNK;
        const chunkEndFrame = Math.min(this.header.numFrames - 1, chunkStartFrame + FRAMES_PER_CHUNK - 1);
        await this.fetchChunkRange(chunkStartFrame, chunkEndFrame);
        this.pruneCache();
    }

    private async fetchChunkRange(startFrame: number, endFrame: number): Promise<void> {
        const start = HEADER_SIZE + startFrame * this.frameByteSize;
        const end = HEADER_SIZE + (endFrame + 1) * this.frameByteSize - 1;
        const chunkBuffer = await fetchRange(this.url, start, end);
        const chunkBytes = new Uint8Array(chunkBuffer);

        for (let frame = startFrame; frame <= endFrame; frame++) {
            if (!this.frameCache.has(frame)) {
                const byteOffset = (frame - startFrame) * this.frameByteSize;
                const frameBytes = new Uint8Array(chunkBytes.buffer, chunkBytes.byteOffset + byteOffset, this.frameByteSize);
                this.frameCache.set(frame, frameBytes);
            }
            this.touchFrame(frame);
        }
    }

    private touchFrame(frameIndex: number): void {
        this.frameOrder = this.frameOrder.filter(index => index !== frameIndex);
        this.frameOrder.push(frameIndex);
    }

    private pruneCache(): void {
        while (this.frameOrder.length > MAX_CACHED_FRAMES) {
            const evict = this.frameOrder.shift();
            if (evict === undefined) {
                return;
            }
            if (evict === 0) {
                // Keep frame 0 resident so loop restarts never hard-fail.
                this.frameOrder.push(evict);
                continue;
            }
            this.frameCache.delete(evict);
        }
    }

    private copyFrame(frame: Uint8Array): void {
        const count = this.header.numSplats;
        const floats = new Float32Array(frame.buffer, frame.byteOffset + 4, count * FLOATS_PER_SPLAT);
        const { x, y, z, rot0, rot1, rot2, rot3, scale0, scale1, scale2, opacity, fdc0, fdc1, fdc2 } = this.work;

        for (let i = 0; i < count; i++) {
            const base = i * FLOATS_PER_SPLAT;
            x[i] = floats[base];
            y[i] = floats[base + 1];
            z[i] = floats[base + 2];
            rot0[i] = floats[base + 3];
            rot1[i] = floats[base + 4];
            rot2[i] = floats[base + 5];
            rot3[i] = floats[base + 6];
            scale0[i] = floats[base + 7];
            scale1[i] = floats[base + 8];
            scale2[i] = floats[base + 9];
            opacity[i] = floats[base + 10];
            fdc0[i] = floats[base + 11];
            fdc1[i] = floats[base + 12];
            fdc2[i] = floats[base + 13];
        }
    }
}

const streamOmg4Data = (url: string, onProgress: (progress: number) => void) => StreamedOmg4Data.create(url, onProgress);

export { StreamedOmg4Data, streamOmg4Data };