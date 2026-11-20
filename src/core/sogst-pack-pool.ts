import { sogstPackWorker } from './sogst-pack-worker';
import type { PackedRows, PackInitMessage, PackJobMessage, PackMeta, RawTexels } from './sogst-pack-worker';

// A pool of pack workers (sogst-pack-worker.ts). Jobs run in priority order,
// lowest number first, and in submission order within a priority.

type PackRequest = Omit<PackJobMessage, 'type' | 'id'>;

type QueuedJob = {
    request: PackRequest;
    priority: number;
    seq: number;
    resolve: (rows: PackedRows) => void;
    reject: (err: Error) => void;
};

type PoolWorker = {
    worker: Worker;
    job: QueuedJob | null;
};

// Workers in the pool: one per core but one, which leaves the main thread
// its own core, and at most this many. The jobs are memory-bound enough that
// more workers stop paying off.
const MAX_PACK_WORKERS = 4;

// A reported core count at or below this is treated as masked. Brave
// randomises navigator.hardwareConcurrency as anti-fingerprinting and
// reported 2 on an 8+ core Mac, which sized the pool to one worker and
// queued every decode behind it.
const MASKED_CORE_COUNT = 2;

const poolSize = () => {
    const reported = navigator.hardwareConcurrency || 0;
    if (reported <= MASKED_CORE_COUNT) {
        return MAX_PACK_WORKERS;
    }
    return Math.max(1, Math.min(MAX_PACK_WORKERS, reported - 1));
};

class SogstPackPool {
    readonly size: number;
    private workers: PoolWorker[] = [];
    private queue: QueuedJob[] = [];
    private seq = 0;
    private url: string;
    private destroyed = false;

    constructor() {
        this.size = poolSize();
        this.url = URL.createObjectURL(new Blob([`(${sogstPackWorker.toString()})()`], { type: 'text/javascript' }));
        for (let i = 0; i < this.size; i++) {
            const entry: PoolWorker = { worker: new Worker(this.url), job: null };
            entry.worker.onmessage = (e: MessageEvent) => this.finished(entry, e.data);
            entry.worker.onerror = (e: ErrorEvent) => this.finished(entry, { error: e.message || 'pack worker error' });
            this.workers.push(entry);
        }
    }

    // Send the clip's codebooks, and the SH centroids once they are known, to
    // every worker. A worker handles messages in order, so jobs posted after
    // this see it.
    init(meta: PackMeta, centroids: Uint8Array | RawTexels | null) {
        const message: PackInitMessage = { type: 'init', meta, centroids };
        this.workers.forEach(({ worker }) => worker.postMessage(message));
    }

    run(request: PackRequest, priority: number): Promise<PackedRows> {
        if (this.destroyed) {
            return Promise.reject(new Error('pack pool destroyed'));
        }
        return new Promise<PackedRows>((resolve, reject) => {
            this.queue.push({ request, priority, seq: this.seq++, resolve, reject });
            this.dispatch();
        });
    }

    private dispatch() {
        for (const entry of this.workers) {
            if (entry.job || !this.queue.length) {
                continue;
            }
            let best = 0;
            for (let i = 1; i < this.queue.length; i++) {
                const a = this.queue[i];
                const b = this.queue[best];
                if (a.priority < b.priority || (a.priority === b.priority && a.seq < b.seq)) {
                    best = i;
                }
            }
            const job = this.queue.splice(best, 1)[0];
            entry.job = job;
            const message: PackJobMessage = { type: 'pack', id: job.seq, ...job.request };
            entry.worker.postMessage(message);
        }
    }

    private finished(entry: PoolWorker, data: { rows?: PackedRows; error?: string }) {
        const job = entry.job;
        entry.job = null;
        if (job) {
            if (data.error !== undefined) {
                job.reject(new Error(data.error));
            } else {
                job.resolve(data.rows!);
            }
        }
        this.dispatch();
    }

    destroy() {
        this.destroyed = true;
        const failure = new Error('pack pool destroyed');
        this.queue.forEach((job) => job.reject(failure));
        this.queue = [];
        this.workers.forEach((entry) => {
            entry.job?.reject(failure);
            entry.worker.terminate();
        });
        this.workers = [];
        URL.revokeObjectURL(this.url);
    }
}

export { SogstPackPool };
export type { PackRequest };
