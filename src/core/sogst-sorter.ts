import type { AppBase, GSplatResource } from 'playcanvas';
import { GSplatSortBinWeights } from 'playcanvas/build/playcanvas/src/scene/gsplat-unified/gsplat-sort-bin-weights.js';
import { UnifiedSortWorker } from 'playcanvas/build/playcanvas/src/scene/gsplat-unified/gsplat-unified-sort-worker.js';

// A CPU splat sorter (WebGL) that takes centers in pieces.
//
// The engine's CPU sorter copies a resource's whole centers buffer to its
// worker whenever the resource's centersVersion changes: 54 MB for a 4.5M
// splat resource. A .sogst clip changes a few hundred thousand centers at a
// time (a decoded group, or a segment the window installed), so this sorter
// adds `patchCenters`, which sends just those, and stamps each sort result
// with the last patch it included. A caller can then wait for exactly the
// sort that includes its splats.
//
// It is not a fork. The worker is the engine's own UnifiedSortWorker, built
// from its source text with one message case added; the sorter class is the
// engine's own, taken from the live instance and subclassed to use that
// worker. It is installed through the unified manager's createSorter(), the
// one place the engine constructs a CPU sorter.
//
// Everything it relies on is engine-private: the manager's createSorter,
// cpuSorter and onSorted, the director's camera and layer maps that reach
// the manager, the sorter's worker and onSorted, and the worker's source
// text. installSogstSorter() checks each before it changes anything, and
// returns null (leaving the engine sorter in place) when one is missing.
// Re-check after every engine bump, as with patchChunk() in viewer.ts.

// The engine members this module reaches, by name.
type EngineSorter = {
    worker: Worker | null;
    centersSet: Set<number>;
    onSorted(message: unknown): void;
    on(name: string, callback: (...args: unknown[]) => void): void;
    destroy(): void;
};

type EngineSorterCtor = new (scene: unknown) => EngineSorter;

type EngineManager = {
    scene: unknown;
    sortNeeded: boolean;
    cpuSorter: EngineSorter | null;
    createSorter(): EngineSorter;
    onSorted(count: unknown, version: unknown, orderData: unknown): void;
};

type EngineDirector = {
    camerasMap: Map<unknown, { layersMap: Map<unknown, { gsplatManager: EngineManager | null }> }>;
};

// Frames to wait for the unified manager to create its CPU sorter.
const MAX_WAIT_FRAMES = 240;

// The engine worker's source with a `patchCenters` case, and each sort
// result stamped with the last patch applied before it. Null when the
// source no longer has the shape the injection expects.
const workerSource = (): string | null => {
    let src = UnifiedSortWorker.toString();
    const centersMap = /case\s*"addCenters"\s*:\s*\{\s*([\w$]+)\.set\(/.exec(src)?.[1];
    const message = /switch\s*\(\s*([\w$]+)\.command\s*\)/.exec(src)?.[1];
    const removeCase = /case\s*"removeCenters"\s*:/.exec(src)?.[0];
    const response = /version:\s*[\w$]+\.version\s*,/.exec(src)?.[0];
    if (!centersMap || !message || !removeCase || !response) {
        return null;
    }
    src = src.replace(
        removeCase,
        `case "patchCenters": {
            const target = ${centersMap}.get(${message}.id);
            if (target) {
                target.set(new Float32Array(${message}.centers), ${message}.offset * 3);
            }
            globalThis.__sogstPatchSeq = Math.max(globalThis.__sogstPatchSeq ?? 0, ${message}.seq);
            break;
        }
        ${removeCase}`
    );
    src = src.replace(response, `sogstPatchSeq: globalThis.__sogstPatchSeq ?? 0, ${response}`);
    return `const GSplatSortBinWeights = ${GSplatSortBinWeights.toString()};\n(${src})()`;
};

const managersOf = (app: AppBase): EngineManager[] => {
    const director = (app as unknown as { renderer: { gsplatDirector?: EngineDirector | null } }).renderer
        .gsplatDirector;
    const managers: EngineManager[] = [];
    director?.camerasMap?.forEach((camera) => {
        camera.layersMap?.forEach((layer) => {
            if (layer.gsplatManager) {
                managers.push(layer.gsplatManager);
            }
        });
    });
    return managers;
};

class SogstSorterController {
    private app: AppBase;

    private managers: EngineManager[];

    private seq = 0;

    private listeners: ((seq: number) => void)[] = [];

    constructor(app: AppBase, managers: EngineManager[]) {
        this.app = app;
        this.managers = managers;
    }

    // Called with the highest patch sequence a completed sort included.
    onSorted(listener: (seq: number) => void) {
        this.listeners.push(listener);
    }

    sorted(seq: number) {
        this.listeners.forEach((listener) => listener(seq));
    }

    // Advance the patch sequence without data, and return it: a sort that
    // reports at least this sequence was posted after every patch and
    // centers update sent before this call. For fencing a centersVersion
    // bump, whose full copy reaches the worker on the next frame, before
    // that frame's sort.
    mark(): number {
        const seq = ++this.seq;
        for (const manager of this.managers) {
            manager.cpuSorter?.worker?.postMessage({
                command: 'patchCenters',
                id: -1,
                offset: 0,
                centers: new ArrayBuffer(0),
                seq
            });
        }
        return seq;
    }

    // Ask every manager for a sort on its next update. The managers sort
    // only when the camera moves or the world changes, and a patch is
    // neither; a request can also be dropped while sorts are in flight, so a
    // caller waiting on a patch repeats this every frame until it lands.
    requestSort() {
        for (const manager of this.managers) {
            manager.sortNeeded = true;
        }
    }

    // Send the centers of splats [start, start + count) of `resource` to
    // every sort worker that holds the resource. Returns the patch sequence
    // to wait for, or null when no worker holds the resource yet (its first
    // full copy will include these centers).
    patch(resource: GSplatResource, start: number, count: number): number | null {
        const centers = resource.centers as Float32Array | undefined;
        if (!centers || count <= 0) {
            return null;
        }
        let posted = false;
        const seq = ++this.seq;
        for (const manager of this.managers) {
            const sorter = manager.cpuSorter;
            if (!sorter?.worker || !sorter.centersSet.has(resource.id)) {
                continue;
            }
            const data = centers.slice(start * 3, (start + count) * 3);
            sorter.worker.postMessage(
                { command: 'patchCenters', id: resource.id, offset: start, centers: data.buffer, seq },
                [data.buffer]
            );
            posted = true;
        }
        return posted ? seq : null;
    }
}

// Replace the CPU sorter of every unified manager with one that accepts
// center patches. Resolves once the managers exist and the swap is done, or
// with null when this engine build does not have the expected internals (the
// engine sorter then stays). The caller must bump the resource's
// centersVersion after it resolves, so the new sorters receive full centers
// and sort parameters.
const installSogstSorter = (app: AppBase): Promise<SogstSorterController | null> => {
    return new Promise((resolve) => {
        let frames = 0;
        const attempt = () => {
            const managers = managersOf(app).filter((m) => m.cpuSorter);
            if (!managers.length) {
                if (++frames > MAX_WAIT_FRAMES) {
                    app.off('frameend', attempt);
                    resolve(null);
                }
                return;
            }
            app.off('frameend', attempt);

            const engineSorter = managers[0].cpuSorter!;
            const source = workerSource();
            const shapeOk =
                source !== null &&
                typeof managers[0].createSorter === 'function' &&
                typeof managers[0].onSorted === 'function' &&
                typeof engineSorter.onSorted === 'function' &&
                engineSorter.centersSet instanceof Set &&
                !!engineSorter.worker;
            if (!shapeOk) {
                console.warn('sogst: engine sorter internals changed; keeping the engine sorter');
                resolve(null);
                return;
            }

            const controller = new SogstSorterController(app, managers);
            const url = URL.createObjectURL(new Blob([source!], { type: 'application/javascript' }));
            const Base = engineSorter.constructor as EngineSorterCtor;
            class SogstSorter extends Base {
                constructor(scene: unknown) {
                    super(scene);
                    // swap the engine's worker for the patch-aware one
                    this.worker?.terminate();
                    this.worker = new Worker(url);
                    this.worker.addEventListener('message', (message: MessageEvent) => {
                        this.onSorted(message);
                        const seq = (message.data as { sogstPatchSeq?: number })?.sogstPatchSeq;
                        if (typeof seq === 'number') {
                            controller.sorted(seq);
                        }
                    });
                }
            }

            // every sorter created from now on, by any manager, is ours
            const proto = Object.getPrototypeOf(managers[0]) as EngineManager;
            proto.createSorter = function (this: EngineManager) {
                const sorter = new SogstSorter(this.scene);
                sorter.on('sorted', (count, version, orderData) => this.onSorted(count, version, orderData));
                return sorter;
            };
            // and the ones that already exist are replaced
            for (const manager of managers) {
                const old = manager.cpuSorter!;
                manager.cpuSorter = manager.createSorter();
                old.destroy();
            }
            app.once('destroy', () => URL.revokeObjectURL(url));
            resolve(controller);
        };
        app.on('frameend', attempt);
    });
};

export { installSogstSorter, SogstSorterController };
