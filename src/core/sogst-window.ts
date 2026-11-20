import type { SogstMeta } from '../parsers/sogst';
import type { LoopMode } from '../types';

import { enumerateSogstGroups, groupBaseNames } from './sogst-decoder';
import type { SogstDecoder, SogstGroup, SplatPlacement } from './sogst-decoder';
import { createSogstPageMask } from './sogst-motion';
import type { SogstSegmentSource } from './sogst-segment-source';
import type { SogstTarget } from './sogst-target';

// Segment window: plays a segmented .sogst clip whose splats do not all fit
// the device's splat budget.
//
// The resource holds the persistent group at [0, persistentEnd), always.
// The rest of its capacity is an arena of fixed-size pages (PAGE_SPLATS). A
// segment takes any free pages, preferring one contiguous run; when too few
// are free, segments are evicted until enough are. An evicted segment is
// fetched again later from the segment source (IndexedDB, then HTTP Range)
// and decoded into whatever pages are free.
//
// Pages, not contiguous slots, because an active segment is never evicted:
// with contiguous slots the free space could fragment into holes each too
// small for the next segment, and playback would wait forever (it did, at
// a tight budget). Pages trade that for under one page of waste per
// segment.
//
// Which segment to install and which to evict both follow the segments'
// *next use*: the wall-clock time until the playhead next enters the
// segment's [t0, t1] span. Playback is deterministic given the time,
// direction, speed and loop mode, so the next use is exact, and evicting the
// resident segment used farthest in the future is optimal (Belady's rule).
// The one rule gives each loop mode the right behaviour:
//
//   none      played segments are never used again and go first
//   repeat    played segments are used again only after the wrap, so they
//             go first, and the start of the clip refills before the wrap
//   pingpong  segments near a turnaround are used again soon after, so they
//             stay
//
// A segment active at the playhead (next use 0) is never evicted. The
// animation driver holds the playhead while isReady() is false, and culls
// every page outside activePages() (see setSogstCullPages), so an evicted
// or half-written slot never draws.

type WindowPlayback = {
    // absolute clip time
    time: number;

    direction: 1 | -1;
    speed: number;
    mode: LoopMode;
};

type Slot = {
    // arena pages, in segment order: page k holds the segment's splats
    // [k * PAGE_SPLATS, (k + 1) * PAGE_SPLATS)
    pages: number[];
    // whether the slot has its SH (sh-deferred labels can arrive later)
    sh: boolean;
    // install sequence number, for the sort fence (see markSorted)
    seq: number;
};

type InstallPlan = {
    seg: number;
    size: number;
    pages: number[];
    victims: number[];
};

const LABELS_NAME = 'shN_labels.webp';

// Splats per arena page. Small enough that a segment wastes little of its
// last page (segments run from about 5k to 440k splats), large enough that
// the page mask stays a few hundred texels and a segment needs few runs.
const PAGE_SPLATS = 16384;

const pagesFor = (splats: number) => Math.ceil(splats / PAGE_SPLATS);

// Segment payloads the stream offered (first pass) that the window keeps in
// memory for a pending install, at most. Beyond that it drops the farthest
// and reads them back from the cache when needed.
const MAX_OFFERED_SEGMENTS = 4;

// How far ahead of its first use a resident segment's sort is requested.
// Covers a CPU sort on the slower devices measured (~100–200 ms on a Mac or
// Vision Pro) plus the frame that applies it.
const SORT_LEAD_S = 0.4;

// Wait before retrying a segment whose install failed. A failure is usually
// transient: its cache unit not committed yet, or a network error.
const RETRY_AFTER_MS = 2000;

// Speed below which next-use times are computed as if at this speed. A
// paused or near-zero transport would otherwise make every future segment
// infinitely far away and the window would stop prefetching.
const MIN_PLANNING_SPEED = 0.05;

// Splats in each segment group, from the manifest.
const segmentSize = (meta: SogstMeta, seg: number) => {
    const [a, b] = meta.segments.list[seg].range;
    return b - a;
};

// Capacity for a clip under a splat budget. A clip that fits, has no
// segment table, or was given no budget plays fully resident, which is the
// layout before windowing. Otherwise the capacity is the budget, raised if
// needed to the least that can play: the persistent group, the pages of the
// largest set of segments active at one instant, and the pages to install
// the largest segment beside them.
const planSogstCapacity = (meta: SogstMeta, budget?: number): { capacity: number; windowed: boolean } => {
    const segments = meta.segments;
    const resident = { capacity: meta.count, windowed: false };
    if (!budget || !(budget > 0) || !meta.streams || !segments?.list?.length || !segments.persistent) {
        return resident;
    }
    if (meta.count <= budget) {
        return resident;
    }
    const groups = enumerateSogstGroups(meta).filter((g) => g.segIndex >= 0);
    const span = (g: SogstGroup) => segments.list[g.segIndex];
    // the active set only grows at some segment's t0
    let peakPages = 0;
    for (const g of groups) {
        const t = span(g).t0;
        let sum = 0;
        for (const h of groups) {
            if (span(h).t0 <= t && t <= span(h).t1) {
                sum += pagesFor(segmentSize(meta, h.segIndex));
            }
        }
        peakPages = Math.max(peakPages, sum);
    }
    const largestPages = Math.max(0, ...groups.map((g) => pagesFor(segmentSize(meta, g.segIndex))));
    const minimum = segments.persistent[1] + (peakPages + largestPages) * PAGE_SPLATS;
    if (minimum >= meta.count) {
        return resident;
    }
    if (budget < minimum) {
        console.warn(
            `sogst: splat budget ${budget} is below the ${minimum} this clip needs to play windowed; using ${minimum}`
        );
        return { capacity: minimum, windowed: true };
    }
    return { capacity: budget, windowed: true };
};

class SogstWindow {
    readonly persistentEnd: number;
    readonly capacity: number;
    private meta: SogstMeta;
    private decoder: SogstDecoder;
    private target: SogstTarget;
    private source: SogstSegmentSource;
    private groups = new Map<number, SogstGroup>();
    private baseNames: string[];
    private resident = new Map<number, Slot>();
    readonly pageCount: number;

    // free arena pages, ascending
    private freePages: number[];

    private offered = new Map<number, Map<string, Uint8Array>>();
    private pendingLabels = new Map<number, Uint8Array>();

    // segments whose install failed (fetch or decode), and when to retry
    private retryAt = new Map<number, number>();

    // segments with an install in flight
    private installing = new Set<number>();

    // installs and SH upgrades in flight; at most one per pack worker
    private inflight = 0;

    private playback: WindowPlayback;
    private running = false;
    private destroyed = false;
    private listeners: (() => void)[] = [];
    private installListeners: ((placement: SplatPlacement, installSeq: number) => void)[] = [];

    // Sequence number of the latest completed install.
    installSeq = 0;

    // Installs up to this sequence number are sorted and may draw. Infinity
    // unless a sort fence is in use (useSortFence).
    private sortedSeq = Infinity;

    constructor(
        meta: SogstMeta,
        decoder: SogstDecoder,
        target: SogstTarget,
        source: SogstSegmentSource,
        persistentEnd: number
    ) {
        this.meta = meta;
        this.decoder = decoder;
        this.target = target;
        this.source = source;
        this.persistentEnd = persistentEnd;
        this.capacity = target.capacity;
        this.pageCount = Math.floor((target.capacity - persistentEnd) / PAGE_SPLATS);
        this.freePages = Array.from({ length: this.pageCount }, (_, i) => i);
        createSogstPageMask(target.resource, this.pageCount);
        this.baseNames = groupBaseNames(meta);
        for (const group of enumerateSogstGroups(meta)) {
            if (group.segIndex >= 0) {
                this.groups.set(group.segIndex, group);
            }
        }
        this.playback = { time: meta.time?.min ?? 0, direction: 1, speed: 1, mode: 'repeat' };
    }

    // Called on every change to what the window can show: a segment
    // installed, evicted, given its SH, or reported sorted.
    onChange(listener: () => void) {
        this.listeners.push(listener);
    }

    // Called when a segment's install completes, with where it went and its
    // install sequence, since it changes the centers the CPU sorter needs.
    onInstalled(listener: (placement: SplatPlacement, installSeq: number) => void) {
        this.installListeners.push(listener);
    }

    // Hold every later install back until markSorted() reports it. For the
    // CPU sorter, which only sees new centers after a centersVersion bump
    // and a sort (see sogst-sort-fence.ts). Installs before this call count
    // as sorted: the sorter takes the whole centers buffer when the resource
    // first draws.
    useSortFence() {
        this.sortedSeq = this.installSeq;
    }

    // The CPU sorter has sorted with the centers of installs up to `seq`.
    markSorted(seq: number) {
        if (seq > this.sortedSeq) {
            this.sortedSeq = seq;
            this.changed();
        }
    }

    private visible(slot: Slot) {
        return slot.seq <= this.sortedSeq;
    }

    // True when a resident segment still waiting for its sort is needed
    // within SORT_LEAD_S (or is needed now, holding the playhead), so the
    // sort fence should ask for a sort instead of letting patches batch.
    needsSortSoon(): boolean {
        for (const [seg, slot] of this.resident) {
            if (!this.visible(slot) && this.nextUse(seg) < SORT_LEAD_S) {
                return true;
            }
        }
        return false;
    }

    private changed() {
        this.listeners.forEach((listener) => listener());
    }

    // The animation driver reports the transport every frame.
    setPlayback(playback: WindowPlayback) {
        this.playback = playback;
        this.wake();
    }

    // First-pass payloads of a segment, straight off the stream.
    offer(seg: number, files: Map<string, Uint8Array>) {
        if (this.resident.has(seg) || !this.groups.has(seg)) {
            return;
        }
        this.offered.set(seg, files);
        if (this.offered.size > MAX_OFFERED_SEGMENTS) {
            let farthest = -1;
            let farthestUse = -1;
            for (const s of this.offered.keys()) {
                const use = this.nextUse(s);
                if (use > farthestUse) {
                    farthest = s;
                    farthestUse = use;
                }
            }
            this.offered.delete(farthest);
        }
        this.wake();
    }

    // Deferred SH labels of a segment, straight off the stream. Kept only
    // for a segment that is resident without its SH or being installed;
    // otherwise a later install reads them from the cache.
    offerLabels(seg: number, labels: Uint8Array) {
        const slot = this.resident.get(seg);
        if ((slot && !slot.sh) || this.installing.has(seg)) {
            this.pendingLabels.set(seg, labels);
            this.wake();
        }
    }

    private covers(seg: number, time: number) {
        const span = this.meta.segments.list[seg];
        return span.t0 <= time && time <= span.t1;
    }

    // True when every segment that covers `time` is resident and sorted.
    isReady(time: number): boolean {
        for (const seg of this.groups.keys()) {
            if (this.covers(seg, time)) {
                const slot = this.resident.get(seg);
                if (!slot || !this.visible(slot)) {
                    return false;
                }
            }
        }
        return true;
    }

    // Pages of the resident segments that cover `time`, for the cull test.
    activePages(time: number): number[] {
        const pages: number[] = [];
        for (const [seg, slot] of this.resident) {
            if (this.covers(seg, time) && this.visible(slot)) {
                pages.push(...slot.pages);
            }
        }
        return pages.sort((a, b) => a - b);
    }

    // Where a segment's splats go: contiguous source ranges mapped onto runs
    // of consecutive pages.
    private placement(pages: number[], size: number): SplatPlacement {
        const runs: SplatPlacement = [];
        for (let k = 0; k < pages.length; k++) {
            const src = k * PAGE_SPLATS;
            const count = Math.min(PAGE_SPLATS, size - src);
            const dest = this.persistentEnd + pages[k] * PAGE_SPLATS;
            const last = runs[runs.length - 1];
            if (last && last.src + last.count === src && last.dest + last.count === dest) {
                last.count += count;
            } else {
                runs.push({ src, dest, count });
            }
        }
        return runs;
    }

    // Wall-clock seconds until the playhead next enters the segment's span:
    // 0 when it is inside now, Infinity when it never will (loop mode none,
    // after the segment).
    private nextUse(seg: number): number {
        const { time, direction, mode } = this.playback;
        const speed = Math.max(MIN_PLANNING_SPEED, Math.abs(this.playback.speed));
        const t0Clip = this.meta.time?.min ?? 0;
        const duration = Math.max(0, (this.meta.time?.max ?? 0) - t0Clip);
        const p = time - t0Clip;
        const span = this.meta.segments.list[seg];
        const a = span.t0 - t0Clip;
        const b = span.t1 - t0Clip;
        if (a <= p && p <= b) {
            return 0;
        }
        const ahead = a > p;
        switch (mode) {
            case 'none':
                return ahead ? (a - p) / speed : Infinity;
            case 'repeat':
                return ahead ? (a - p) / speed : (duration - p + Math.max(0, a)) / speed;
            case 'pingpong':
                if (direction > 0) {
                    return ahead ? (a - p) / speed : (2 * duration - p - Math.min(b, duration)) / speed;
                }
                return ahead ? (p + Math.max(0, a)) / speed : (p - b) / speed;
        }
        return Infinity;
    }

    // Start installs (and SH upgrades) until every pack worker has one or
    // nothing more is worth installing. Planning, eviction and slot
    // assignment all happen synchronously here, so overlapping installs
    // never plan against each other's slots.
    private wake() {
        if (this.running || this.destroyed || this.target.destroyed) {
            return;
        }
        this.running = true;
        try {
            while (this.inflight < this.decoder.concurrency) {
                const upgrade = this.nextUpgrade();
                if (upgrade) {
                    this.launch(this.upgradeSH(upgrade.seg, upgrade.labels));
                    continue;
                }
                const plan = this.choose();
                if (!plan) {
                    break;
                }
                this.launch(this.install(plan));
            }
        } finally {
            this.running = false;
        }
    }

    private launch(task: Promise<void>) {
        this.inflight++;
        task.catch((err: Error) => {
            console.warn('sogst: segment window task failed:', err);
        }).finally(() => {
            this.inflight--;
            this.wake();
        });
    }

    // Pack-pool priority for a segment's jobs: its next use, in ms, so the
    // pool runs the most urgent install's jobs first.
    private priorityOf(seg: number) {
        return Math.min(Math.round(this.nextUse(seg) * 1000), Number.MAX_SAFE_INTEGER);
    }

    // A resident segment whose SH labels arrived after its install.
    private nextUpgrade(): { seg: number; labels: Uint8Array } | null {
        for (const [seg, labels] of this.pendingLabels) {
            const slot = this.resident.get(seg);
            this.pendingLabels.delete(seg);
            if (slot && !slot.sh) {
                return { seg, labels };
            }
        }
        return null;
    }

    private async upgradeSH(seg: number, labels: Uint8Array) {
        const slot = this.resident.get(seg)!;
        // the slot is marked now, so a second copy of the labels is ignored
        slot.sh = true;
        await this.decoder.decodeGroupSH(
            this.groups.get(seg)!,
            labels,
            this.placement(slot.pages, segmentSize(this.meta, seg)),
            this.priorityOf(seg)
        );
        if (this.resident.get(seg) === slot) {
            this.changed();
        }
    }

    // `count` free pages from `free` (ascending): the first contiguous run
    // that fits, or else the lowest free pages.
    private static pick(free: number[], count: number): number[] {
        for (let i = 0; i + count <= free.length; i++) {
            if (free[i + count - 1] - free[i] === count - 1) {
                return free.slice(i, i + count);
            }
        }
        return free.slice(0, count);
    }

    // The next install: the nearest-used segment that is not resident, can be
    // fetched, and fits once segments used later than it are evicted.
    private choose(): InstallPlan | null {
        const candidates: { seg: number; use: number; size: number }[] = [];
        for (const [seg, group] of this.groups) {
            if (
                this.resident.has(seg) ||
                this.installing.has(seg) ||
                (this.retryAt.get(seg) ?? 0) > performance.now()
            ) {
                continue;
            }
            const use = this.nextUse(seg);
            if (!isFinite(use)) {
                continue;
            }
            if (!this.offered.has(seg) && !this.source.has(group.prefix!, this.baseNames)) {
                continue;
            }
            const size = segmentSize(this.meta, seg);
            if (pagesFor(size) > this.pageCount) {
                continue;
            }
            candidates.push({ seg, use, size });
        }
        candidates.sort((x, y) => x.use - y.use);

        const nearest = candidates[0];
        if (!nearest) {
            return null;
        }
        const needed = pagesFor(nearest.size);
        if (this.freePages.length >= needed) {
            const pages = SogstWindow.pick(this.freePages, needed);
            return { seg: nearest.seg, size: nearest.size, pages, victims: [] };
        }
        // evict from the farthest next use down, never one used as soon as
        // the candidate (which includes every active segment)
        const victims = [...this.resident.keys()]
            .map((seg) => ({ seg, use: this.nextUse(seg) }))
            .filter((v) => v.use > nearest.use)
            .sort((x, y) => y.use - x.use);
        let free = this.freePages;
        const chosen: number[] = [];
        for (const victim of victims) {
            free = [...free, ...this.resident.get(victim.seg)!.pages].sort((a, b) => a - b);
            chosen.push(victim.seg);
            if (free.length >= needed) {
                return { seg: nearest.seg, size: nearest.size, pages: SogstWindow.pick(free, needed), victims: chosen };
            }
        }
        return null;
    }

    private evict(seg: number) {
        const slot = this.resident.get(seg);
        if (!slot) {
            return;
        }
        this.resident.delete(seg);
        this.release(slot.pages);
    }

    private release(pages: number[]) {
        this.freePages = [...this.freePages, ...pages].sort((a, b) => a - b);
    }

    private take(pages: number[]) {
        const taken = new Set(pages);
        this.freePages = this.freePages.filter((p) => !taken.has(p));
    }

    private async install(plan: InstallPlan) {
        plan.victims.forEach((seg) => this.evict(seg));
        if (plan.victims.length) {
            // cull the evicted slots before anything overwrites them
            this.changed();
        }
        this.take(plan.pages);
        const placement = this.placement(plan.pages, plan.size);
        this.installing.add(plan.seg);
        const group = this.groups.get(plan.seg)!;
        try {
            let files = this.offered.get(plan.seg) ?? null;
            this.offered.delete(plan.seg);
            if (!files) {
                files = await this.source.get(group.prefix!, this.baseNames, this.meta.shN ? [LABELS_NAME] : []);
            }
            if (!files || this.destroyed) {
                this.release(plan.pages);
                return;
            }
            const labels = this.pendingLabels.get(plan.seg);
            if (labels && !files.has(LABELS_NAME)) {
                files.set(LABELS_NAME, labels);
            }
            this.pendingLabels.delete(plan.seg);

            await this.decoder.decodeGroup(group, files, undefined, placement, this.priorityOf(plan.seg));
            const sh = !!this.meta.shN && files.has(LABELS_NAME);
            if (this.meta.shN && !sh) {
                for (const run of placement) {
                    await this.target.clearSH(run.dest, run.dest + run.count);
                }
            }
            this.resident.set(plan.seg, { pages: plan.pages, sh, seq: ++this.installSeq });
            this.changed();
            const installSeq = this.installSeq;
            this.installListeners.forEach((listener) => listener(placement, installSeq));
        } catch (err) {
            console.warn(`sogst: could not install segment ${plan.seg}:`, err);
            this.retryAt.set(plan.seg, performance.now() + RETRY_AFTER_MS);
            this.release(plan.pages);
        } finally {
            this.installing.delete(plan.seg);
        }
    }

    destroy() {
        this.destroyed = true;
        this.offered.clear();
        this.pendingLabels.clear();
        this.decoder.destroy();
    }
}

export { PAGE_SPLATS, planSogstCapacity, SogstWindow };
export type { WindowPlayback };
