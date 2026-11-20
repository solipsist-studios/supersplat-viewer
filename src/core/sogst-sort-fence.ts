import type { AppBase, GSplatResource } from 'playcanvas';

import type { SplatPlacement } from './sogst-decoder';
import type { SogstSorterController } from './sogst-sorter';

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

// The same contract with the patch-aware sorter (sogst-sorter.ts): an
// install's centers go to the sort worker as patches the moment it
// completes, and the install is reported sorted once a sort that includes
// those patches lands. No centersVersion bump, no 54 MB copy.
//
// Sorts are requested sparingly. Each completed sort makes the engine upload
// a whole new order buffer on the main thread (18 MB at 4.5M splats), and on
// a Mac or Vision Pro a sort took ~100 ms in the worker; requested for every
// install, at ~6 a second, every one of them landed as a 100–160 ms frame.
// Most installs are prefetches the playhead will not reach for a while, and
// one sort covers every patch sent before it, so a sort is requested only
// when the caller says one is needed soon, or when a patch has waited
// MAX_SORT_DEFER_MS. Camera moves sort anyway, and those sorts count too.

// Longest a patch waits for a sort that nothing urgent asked for.
const MAX_SORT_DEFER_MS = 500;
class SogstPatchFence {
    private app: AppBase;
    private controller: SogstSorterController;
    private onSorted: (installSeq: number) => void;

    // fences in order: the patch sequence a sort must include, the install
    // sequence that sort covers, and when the fence was set
    private pending: { patchSeq: number; installSeq: number; at: number }[] = [];

    // whether a pending install is needed soon (the window's view)
    private urgent: () => boolean;

    private kick = () => {
        if (!this.pending.length) {
            return;
        }
        if (this.urgent() || performance.now() - this.pending[0].at >= MAX_SORT_DEFER_MS) {
            this.controller.requestSort();
        }
    };

    constructor(
        app: AppBase,
        controller: SogstSorterController,
        onSorted: (installSeq: number) => void,
        urgent: () => boolean = () => false
    ) {
        this.app = app;
        this.controller = controller;
        this.onSorted = onSorted;
        this.urgent = urgent;
        controller.onSorted((seq) => this.sorted(seq));
        app.on('update', this.kick);
    }

    // Patch an install's placement into the sorter and fence it.
    install(resource: GSplatResource, placement: SplatPlacement, installSeq: number) {
        let last: number | null = null;
        for (const run of placement) {
            last = this.controller.patch(resource, run.dest, run.count) ?? last;
        }
        this.fence(last ?? this.controller.mark(), installSeq);
    }

    // Fence installs up to `installSeq` on the next sort after now: for a
    // centersVersion bump, or a patch the sorter could not take yet.
    fenceAll(installSeq: number) {
        this.fence(this.controller.mark(), installSeq);
    }

    private fence(patchSeq: number, installSeq: number) {
        this.pending.push({ patchSeq, installSeq, at: performance.now() });
    }

    private sorted(seq: number) {
        let done = -1;
        while (this.pending.length && this.pending[0].patchSeq <= seq) {
            done = this.pending.shift()!.installSeq;
        }
        if (done >= 0) {
            this.onSorted(done);
        }
    }

    destroy() {
        this.app.off('update', this.kick);
        this.pending = [];
    }
}

export { SogstPatchFence, SogstSortFence };
