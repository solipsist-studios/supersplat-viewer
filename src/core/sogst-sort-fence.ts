import type { AppBase } from 'playcanvas';

// Tells a segment window when the CPU sorter (WebGL) has sorted with the
// centers of the segments installed before a centersVersion bump.
//
// The CPU sorter orders splats from the resource's centers, which it copies
// to its worker only on a centersVersion bump. A segment installed into
// pages that held another segment keeps that segment's sort keys until the
// next bump is sorted: drawn in the meantime, its splats blend in the wrong
// order, and they visibly pop into place when the sort lands. So the window
// holds a new segment back until a fence reports it sorted.
//
// The sorter worker handles messages in order, so a sort posted after the
// new centers uses them, while the jobs already in flight at the bump do
// not. After a bump the fence therefore waits for (jobs in flight + 1)
// `gsplat:sorted` events: the stale jobs, then one sorted with the new
// centers. The engine reaches the next bump's feed before it posts another
// sort, so no stale job can be posted between the bump and the feed.

// The engine members the fence reads to count sort jobs in flight. All are
// private; this names exactly the path (GSplatDirector -> per-camera data ->
// per-layer data -> GSplatManager -> GSplatUnifiedSorter).
type SorterChain = {
    renderer: {
        gsplatDirector?: {
            camerasMap: Map<
                unknown,
                {
                    layersMap: Map<unknown, { gsplatManager: { cpuSorter: { jobsInFlight: number } | null } | null }>;
                }
            >;
        } | null;
    };
};

// Sorted events to wait for when the in-flight count cannot be read: the
// usual case of at most one stale job, plus the fresh one.
const FALLBACK_EVENTS = 2;

const jobsInFlight = (app: AppBase): number | null => {
    const director = (app as unknown as SorterChain).renderer.gsplatDirector;
    if (!director?.camerasMap) {
        return null;
    }
    let jobs = 0;
    for (const camera of director.camerasMap.values()) {
        for (const layer of camera.layersMap?.values() ?? []) {
            jobs += layer.gsplatManager?.cpuSorter?.jobsInFlight ?? 0;
        }
    }
    return jobs;
};

class SogstSortFence {
    private app: AppBase;

    private onSorted: (seq: number) => void;

    // fences in bump order: the install sequence each covers, and the sorted
    // events it still waits for
    private pending: { seq: number; remaining: number }[] = [];

    private warned = false;

    private handler = () => this.sorted();

    constructor(app: AppBase, onSorted: (seq: number) => void) {
        this.app = app;
        this.onSorted = onSorted;
        app.scene.on('gsplat:sorted', this.handler);
    }

    // Call right after bumping centersVersion: `seq` is the window's install
    // sequence the bumped centers include.
    fence(seq: number) {
        let jobs = jobsInFlight(this.app);
        if (jobs === null) {
            if (!this.warned) {
                this.warned = true;
                console.warn('sogst: cannot read the CPU sorter state; waiting a fixed number of sorts instead');
            }
            jobs = FALLBACK_EVENTS - 1;
        }
        this.pending.push({ seq, remaining: jobs + 1 });
    }

    private sorted() {
        let done = -1;
        for (const fence of this.pending) {
            fence.remaining--;
        }
        while (this.pending.length && this.pending[0].remaining <= 0) {
            done = this.pending.shift()!.seq;
        }
        if (done >= 0) {
            this.onSorted(done);
        }
    }

    destroy() {
        this.app.scene.off('gsplat:sorted', this.handler);
        this.pending = [];
    }
}

export { SogstSortFence };
