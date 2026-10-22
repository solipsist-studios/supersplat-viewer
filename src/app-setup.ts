import { Asset, Color, createGraphicsDevice, Entity, Keyboard, Mouse, platform, TouchDevice } from 'playcanvas';
import type { AppBase, EventHandler, GSplatResource, TextureHandler } from 'playcanvas';

import { SogstSplatAnimation } from './animation/sogst-splat-animation';
import { App } from './app';
import { fullFileCacheKey, fullFileKeyPrefix } from './core/fetch-splat-anim-buffer';
import { loadSogst, setAabbFromMeta } from './core/load-sogst';
import { setupSplatAnim } from './core/load-splat-anim';
import { observe } from './core/observe';
import { idbDeleteByPrefix, idbGetBuffer, idbGetEntryManifest, idbSetBuffer } from './core/sogst-cache';
import type { SogstData } from './core/sogst-data';
import { SogstPatchFence, SogstSortFence } from './core/sogst-sort-fence';
import { installSogstSorter } from './core/sogst-sorter';
import { replaySogst, streamSogst } from './core/stream-sogst';
import { isSogstFilename } from './parsers/sogst';
import type { Config, Global, LoopMode, State } from './types';

// Shared application bootstrap used by both the standalone web app (index.ts)
// and the embeddable library entry (embed-app.ts).

const loadGsplat = async (app: AppBase, config: Config, progressCallback: (progress: number) => void) => {
    const { contents, contentUrl } = config;
    const c = contents as unknown as ArrayBuffer;
    const filename = config.contentFilename ?? new URL(contentUrl, location.href).pathname.split('/').pop();
    const data = filename.toLowerCase() === 'meta.json' ? await (await contents).json() : undefined;
    const asset = new Asset(filename, 'gsplat', { url: contentUrl, filename, contents: c }, data);

    return new Promise<Entity>((resolve, reject) => {
        asset.on('load', () => {
            const entity = new Entity('gsplat');
            entity.setLocalEulerAngles(0, 0, 180);
            entity.addComponent('gsplat', {
                unified: true,
                asset
            });
            app.root.addChild(entity);
            resolve(entity);
        });

        let watermark = 0;
        asset.on('progress', (received, length) => {
            const progress = Math.min(1, received / length) * 100;
            if (progress > watermark) {
                watermark = progress;
                progressCallback(Math.trunc(watermark));
            }
        });

        asset.on('error', (err) => {
            console.log(err);
            reject(err);
        });

        app.assets.add(asset);
        app.assets.load(asset);
    });
};

// Put a decoded clip's resource on an entity and attach its animation driver
// (shared by the full-buffer and streaming paths).
const setupSogst = (app: AppBase, config: Config, global: Global, data: SogstData) => {
    const resource = data.resource;
    // Exact bounds from the global meta range. The resource was built from
    // empty placeholder data before any splat decoded, so the engine's own
    // bounds describe nothing.
    setAabbFromMeta(data.meta, resource.aabb);

    const animation = new SogstSplatAnimation(data);
    const entity = setupSplatAnim(app, config, global, resource, animation, {
        rotationEulerDeg: config.sogstRotationDeg ?? [0, 0, 0],
        alphaClip: 1 / 1024
    });
    animation.bind(entity, data.cov2dScale);
    return { resource, entity };
};

// Least time between CPU-sorter refreshes while a segment window installs
// segments. A refresh costs ~90 ms (see loadSogstStreaming), so this bounds
// that cost to a fraction of playback time. A new segment waits for the
// next refresh to be sorted before it draws, so this also adds up to this
// long to an install's latency.
const MIN_SORTER_REFRESH_MS = 200;

// The .sogst splat budget in splats, from the config's millions: undefined
// for the automatic budget (unset or not a number), 0 for no budget.
const sogstBudgetSplats = (config: Config): number | undefined => {
    const millions = config.sogstBudget;
    if (millions === undefined || !Number.isFinite(millions)) {
        return undefined;
    }
    return millions > 0 ? Math.round(millions * 1000000) : 0;
};

// Progressive load of a streamed archive: reveal the scene once the
// persistent group and the first temporal segment are decoded, then let
// playback advance as later segments land (playback holds at the loaded
// boundary via data.loadedThrough if the network falls behind).
//
// `open` starts the stream: from the network, or as a replay of the entry
// units an earlier visit cached. The network stream caches itself unit by
// unit as it goes; only a monolithic archive comes back whole for caching.
const loadSogstStreaming = async (
    app: AppBase,
    config: Config,
    global: Global,
    cacheKey: string,
    progressCallback: (progress: number) => void,
    open: (callbacks: Parameters<typeof streamSogst>[2]) => ReturnType<typeof streamSogst>
) => {
    let resource: GSplatResource | null = null;
    let data: SogstData | null = null;

    // Teardown guard for the reveal continuation below. An app.destroy()
    // (an embed host switching scenes while this one is still streaming)
    // can land while the stream is decoding. The decoder and SogstTarget
    // guard their own writes and uploads; this flag keeps the code here from
    // touching a destroyed app.
    let destroyed = false;
    app.on('destroy', () => {
        destroyed = true;
    });

    // The patch-aware CPU sorter's fence (WebGL), once installed: groups and
    // window installs then reach the sorter as center patches.
    let patchFence: SogstPatchFence | null = null;

    // Refreshing the depth sorter (centersVersion) re-clones the whole
    // centers buffer to the sort worker and rebuilds the sorted order —
    // measured at ~90ms of main-thread + driver-sync work per refresh on
    // real GPUs, which read as metronomic playback hitches when done on a
    // cadence during streaming. So the sorter is refreshed exactly once,
    // when the stream completes: until then newly streamed splats render
    // at their correct positions but blend in slightly stale depth order.
    // Once the patch-aware sorter is installed (WebGL, see below), each
    // group's centers reach the sorter as it lands instead, and the final
    // refresh is skipped.
    //
    // The decoder packs and uploads each group before it reports the group
    // ready, so a ready group only moves the playhead limit. A report can
    // arrive before the entity exists; it is held and applied once it does.
    let pendingLoadedThrough: number | null = null;
    let streamEnded = false;
    let centersRefreshed = false;
    const apply = () => {
        if (destroyed || !data || !resource) {
            return;
        }
        if (pendingLoadedThrough !== null) {
            data.loadedThrough = pendingLoadedThrough;
            pendingLoadedThrough = null;
        }
        // With the patch-aware sorter every group was patched as it landed,
        // so the one full refresh at the end is not needed.
        if (streamEnded && !centersRefreshed && !patchFence) {
            centersRefreshed = true;
            resource.centersVersion++;
        }
        app.renderNextFrame = true;
    };

    const sync = (range: [number, number] | null, loadedThrough: number) => {
        pendingLoadedThrough = loadedThrough;
        // resident streaming: send the group's centers to the sorter now, so
        // it draws in order without waiting for the end-of-stream refresh
        if (range && patchFence && resource && !data?.window) {
            patchFence.install(resource, [{ src: range[0], dest: range[0], count: range[1] - range[0] }], 0);
        }
        if (!range) {
            streamEnded = true;
        }
        apply();
    };

    // Deferred SH is already uploaded too; it only needs a redraw.
    const syncSh = () => {
        apply();
    };

    const stream = open({
        onProgress: progressCallback,
        onReady: sync,
        onShReady: syncSh
    });

    data = await stream.reveal;

    // A destroy that lands while the reveal is pending: setupSogst would put
    // the resource on an entity of a dead app. Nothing downstream of a
    // destroyed app can do anything useful, so leave the load promise
    // unsettled rather than handing the caller a scene that belongs to a
    // dead device.
    if (destroyed) {
        return new Promise<Entity>(() => {
            /* never settles: the app is gone */
        });
    }

    const setup = setupSogst(app, config, global, data);
    resource = setup.resource;

    // The CPU sorter (WebGL) sorts from the resource's centers, which it
    // copies to its worker only on a centersVersion bump (54 MB for a 4.5M
    // splat resource). A segment drawn before a sort includes its centers
    // blends in the wrong order and pops into place when the sort lands, so
    // the window holds each new segment back until then.
    //
    // The patch-aware sorter (sogst-sorter.ts) sends just an install's
    // centers and reports the exact sort that includes them. It takes over
    // after the first frame, once the engine has created its sorter; until
    // then, or if this engine build lacks the internals it needs, installs
    // fall back to rate-limited bumps fenced on sort events
    // (SogstSortFence). WebGPU sorts on the GPU from the work buffer and
    // needs none of this.
    if (global.renderer === 'webgl') {
        const segmentWindow = data.window;
        segmentWindow?.useSortFence();
        const bumpFence = segmentWindow ? new SogstSortFence(app, (seq) => segmentWindow.markSorted(seq)) : null;
        let lastBump = 0;
        let bumpTimer: ReturnType<typeof setTimeout> | null = null;
        const scheduleBump = () => {
            if (bumpTimer !== null) {
                return;
            }
            const wait = Math.max(0, lastBump + MIN_SORTER_REFRESH_MS - performance.now());
            bumpTimer = setTimeout(() => {
                bumpTimer = null;
                if (destroyed || !resource || patchFence) {
                    return;
                }
                lastBump = performance.now();
                resource.centersVersion++;
                bumpFence?.fence(segmentWindow!.installSeq);
                app.renderNextFrame = true;
            }, wait);
        };
        segmentWindow?.onInstalled((placement, installSeq) => {
            if (patchFence && resource) {
                patchFence.install(resource, placement, installSeq);
            } else {
                scheduleBump();
            }
        });
        installSogstSorter(app).then((controller) => {
            if (!controller || destroyed || !resource) {
                return;
            }
            patchFence = new SogstPatchFence(
                app,
                controller,
                (seq) => segmentWindow?.markSorted(seq),
                () => segmentWindow?.needsSortSoon() ?? false
            );
            // the new sorters take the whole centers buffer once, which
            // covers everything decoded or installed so far
            resource.centersVersion++;
            patchFence.fenceAll(segmentWindow?.installSeq ?? 0);
            app.renderNextFrame = true;
        });
        app.once('destroy', () => {
            bumpFence?.destroy();
            patchFence?.destroy();
        });
    }

    // apply any group reports that arrived while the entity was being set up
    apply();

    // Once the archive is cached, drop copies cached under an older
    // validator. A monolithic archive still caches as one buffer.
    stream.complete
        .then((buffer) => {
            (buffer ? idbSetBuffer(cacheKey, buffer) : Promise.resolve())
                .then(() => idbDeleteByPrefix(fullFileKeyPrefix(config.contentUrl), cacheKey))
                .catch(() => {
                    /* cache write is best-effort */
                });
        })
        .catch((err: Error) => {
            console.warn('SOGST stream did not complete; partial scene retained:', err);
        });

    return setup.entity;
};

// Load and animate a .sogst (SOG spacetime 4DGS) archive.
const loadSogstGsplat = async (
    app: AppBase,
    config: Config,
    global: Global,
    progressCallback: (progress: number) => void
) => {
    const cacheKey = await fullFileCacheKey(config.contentUrl);
    const fromNetwork = () =>
        loadSogstStreaming(app, config, global, cacheKey, progressCallback, (callbacks) =>
            streamSogst(app, config.contentUrl, callbacks, cacheKey, sogstBudgetSplats(config), config.sogstGpuDecode)
        );

    // A streamed archive cached by unit replays through the streaming path,
    // so a revisit never holds the whole archive in memory either. A replay
    // that fails before the reveal (the browser evicted part of the store)
    // falls back to the network.
    const manifest = await idbGetEntryManifest(cacheKey);
    if (manifest) {
        console.debug('SOGST entry cache hit (idb)', cacheKey);
        try {
            return await loadSogstStreaming(app, config, global, cacheKey, progressCallback, (callbacks) =>
                replaySogst(
                    app,
                    cacheKey,
                    manifest,
                    callbacks,
                    config.contentUrl,
                    sogstBudgetSplats(config),
                    config.sogstGpuDecode
                )
            );
        } catch (err) {
            console.warn('SOGST cache replay failed; streaming from the network instead:', err);
            return fromNetwork();
        }
    }

    // A whole-file entry: a monolithic archive, or one cached by an earlier
    // build. Decode it straight through.
    const cached = await idbGetBuffer(cacheKey);
    if (cached) {
        console.debug('SOGST full-file cache hit (idb)', cacheKey);
        const data = await loadSogst(app, cached, progressCallback);
        return setupSogst(app, config, global, data).entity;
    }
    return fromNetwork();
};

// Load a static 3DGS scene (PLY / LOD / meta.json etc.)
const load3dgs = (app: AppBase, config: Config, progressCallback: (progress: number) => void) =>
    loadGsplat(app, config, progressCallback);

// Extensions the static 3DGS path understands, mirroring the parser table in
// the engine's GSplatHandler ({ply, sog, json} plus lod-meta.json). Variants
// need no entries of their own: `.compressed.ply` is a `.ply`, and both
// `meta.json` and `.lod-meta.json` are `.json`. Anything absent here reaches
// the handler's `?? ply` fallback, which is the fail-slow path this list
// exists to close — keep the two in step when bumping the engine.
const STATIC_3DGS_EXTENSIONS = ['.ply', '.sog', '.json'];

// The scene filename, which is what every format decision is made on.
// `contentFilename` exists because a blob: URL carries no name of its own.
const contentFilename = (config: Config) =>
    (config.contentFilename ?? new URL(config.contentUrl, location.href).pathname.split('/').pop() ?? '').toLowerCase();

// True for formats driven by SplatAnimationBase rather than by the engine's
// gsplat asset handler.
const is4dgsFilename = (filename: string) => isSogstFilename(filename.toLowerCase());

// True for content the eager `contents` prefetch is actually useful for —
// only the static 3DGS handler reads it. Deciding this as "not 4DGS" would
// prefetch unrecognised extensions too, downloading a whole file that
// loadContent then rejects unread.
const isStatic3dgsFilename = (filename: string) => {
    const lower = filename.toLowerCase();
    const dot = lower.lastIndexOf('.');
    return dot <= 0 || STATIC_3DGS_EXTENSIONS.includes(lower.slice(dot));
};

// Dispatch on the filename, rejecting an unrecognised extension here rather
// than letting it reach the gsplat asset handler. That handler downloads the
// whole file before the PLY parser rejects its header, so a mistyped or
// unsupported extension costs a full transfer — 310MB for one of the test
// scenes — to reach a conclusion the filename already supported. An
// extensionless URL carries no evidence either way and keeps the historical
// 3DGS path rather than being rejected on a guess.
const loadContent = (
    app: AppBase,
    config: Config,
    global: Global,
    progressCallback: (progress: number) => void
): Promise<Entity> => {
    const filename = contentFilename(config);
    if (is4dgsFilename(filename)) {
        return loadSogstGsplat(app, config, global, progressCallback);
    }
    // Same predicate the prefetch uses, so the two cannot disagree about
    // which files are worth fetching.
    if (isStatic3dgsFilename(filename)) {
        return load3dgs(app, config, progressCallback);
    }
    const ext = filename.slice(filename.lastIndexOf('.'));
    return Promise.reject(
        new Error(
            `Unsupported content format '${ext}' (${filename}). ` +
                'Supported: .ply, .compressed.ply, .sog, .json (meta.json / lod-meta.json), .sogst'
        )
    );
};

const loadSkybox = (app: AppBase, url: string) => {
    return new Promise<Asset>((resolve, reject) => {
        const asset = new Asset(
            'skybox',
            'texture',
            {
                url
            },
            {
                type: 'rgbp',
                mipmaps: false,
                addressu: 'repeat',
                addressv: 'clamp'
            }
        );

        asset.on('load', () => {
            resolve(asset);
        });

        asset.on('error', (err) => {
            console.log(err);
            reject(err);
        });

        app.assets.add(asset);
        app.assets.load(asset);
    });
};

// The renderer to request when the host names none: WebGPU, except WebGL for
// .sogst clips on a Mac. There, a clip playing through a segment window falls
// behind real time on WebGPU: the engine projects and sorts every splat on
// the GPU each frame, and segment decodes and cache reads slow to around four
// times their WebGL cost behind it (measured on an 8 GB MacBook Air). The
// device is created before the clip's metadata says whether it will play
// windowed, so every .sogst clip takes WebGL; a fully resident one plays at
// full speed on either. visionOS and iPadOS also report a Mac user agent and
// are excluded, since this was measured on neither.
const defaultRenderer = (config: Config): 'webgl' | 'webgpu' => {
    const mac = platform.name === 'osx' && !platform.visionos;
    return mac && is4dgsFilename(contentFilename(config)) ? 'webgl' : 'webgpu';
};

const createApp = async (canvas: HTMLCanvasElement, config: Config) => {
    const requested = config.renderer ?? defaultRenderer(config);
    const useWebGPU = requested === 'webgpu';

    // Create the graphics device. The engine auto-appends WebGL2/null fallbacks
    // when WebGPU isn't supported. Request xrCompatible so the device — WebGPU
    // (via XRGPUBinding) or the WebGL fallback — is usable for AR/VR.
    const device = await createGraphicsDevice(canvas, {
        deviceTypes: useWebGPU ? ['webgpu'] : [],
        antialias: false,
        depth: true,
        stencil: false,
        xrCompatible: true,
        powerPreference: 'high-performance',
        // transparent embedding needs an alpha channel in the backbuffer
        // (WebGPU maps this to alphaMode: 'premultiplied')
        ...(config.transparent ? { alpha: true } : {})
    });

    console.log(`Renderer: ${device.deviceType} (requested ${requested}${config.renderer ? '' : ', automatically'})`);

    // The engine may have fallen back from WebGPU to WebGL2; downstream code
    // (voxel overlay, XR, gsplat renderer selection) needs the *actual* renderer.
    const renderer: 'webgl' | 'webgpu' = device.deviceType === 'webgpu' ? 'webgpu' : 'webgl';

    // Set maxPixelRatio so the XR framebuffer scale factor is computed correctly.
    // Regular rendering bypasses maxPixelRatio via the custom initCanvas sizing.
    device.maxPixelRatio = window.devicePixelRatio;

    // Create the application
    const app = new App(canvas, {
        graphicsDevice: device,
        mouse: new Mouse(canvas),
        touch: new TouchDevice(canvas),
        keyboard: new Keyboard(window)
    });

    // enable anonymous CORS for image loading in safari (must be set before any
    // texture asset starts loading, otherwise the <img> is fetched without the
    // crossorigin attribute and WebGL rejects it with SecurityError)
    (app.loader.getHandler('texture') as TextureHandler).imgParser.crossOrigin = 'anonymous';

    // Create entity hierarchy
    const cameraRoot = new Entity('camera root');
    app.root.addChild(cameraRoot);

    const camera = new Entity('camera');
    cameraRoot.addChild(camera);

    const light = new Entity('light');
    light.setEulerAngles(35, 45, 0);
    light.addComponent('light', {
        color: new Color(1.0, 0.98, 0.957),
        intensity: 1
    });
    app.root.addChild(light);

    app.scene.ambientLight.set(0.51, 0.55, 0.65);

    return { app, camera, renderer };
};

// initialize canvas size and resizing
const initCanvas = (global: Global) => {
    const { app, events, state } = global;
    const { canvas } = app.graphicsDevice;

    // maximum pixel dimension we will allow along the shortest screen dimension based on platform
    const maxPixelDim = platform.mobile ? 1080 : 2160;

    // cap pixel ratio to limit resolution on high-DPI devices
    const calcPixelRatio = () => Math.min(maxPixelDim / Math.min(screen.width, screen.height), window.devicePixelRatio);

    // last known device pixel size (full resolution, before any quality scaling)
    const deviceSize = { width: 0, height: 0 };

    const set = (width: number, height: number) => {
        const ratio = calcPixelRatio();
        deviceSize.width = width * ratio;
        deviceSize.height = height * ratio;
    };

    const apply = () => {
        // don't resize the canvas during XR - the XR system manages its own framebuffers
        // and resetting canvas dimensions can invalidate the XRWebGLLayer
        if (app.xr?.active) return;

        const s = state.performanceMode ? 0.5 : 1.0;
        const w = Math.ceil(deviceSize.width * s);
        const h = Math.ceil(deviceSize.height * s);
        if (w !== canvas.width || h !== canvas.height) {
            canvas.width = w;
            canvas.height = h;
        }
    };

    const resizeObserver = new ResizeObserver((entries: ResizeObserverEntry[]) => {
        const e = entries[0]?.contentBoxSize?.[0];
        if (e) {
            set(e.inlineSize, e.blockSize);
            app.renderNextFrame = true;
        }
    });
    resizeObserver.observe(canvas);

    events.on('performanceMode:changed', () => {
        app.renderNextFrame = true;
    });

    // Resize canvas before render() so the swap chain texture is acquired at the correct size.
    app.on('framerender', apply);

    // Disable the engine's built-in canvas resize — we handle it via ResizeObserver
    (app as unknown as { _allowResize: boolean })._allowResize = false;
    set(canvas.clientWidth, canvas.clientHeight);
    apply();
};

// Translate the config's loop style into the transport's loop mode. The two
// use different vocabularies: the embed API offers 'loop' and 'pingpong',
// while the transport carries the three-way mode the UI button cycles.
//
// The embed sets `noui`, so `loopMode` is the only way a host page can reach
// this setting. A config that never reaches the state therefore reads as
// pingpong being ignored, with playback running in the default mode.
const initialLoopMode = (config?: Config): LoopMode => {
    return config?.animLoopMode === 'pingpong' ? 'pingpong' : 'repeat';
};

// Build the observable viewer state (shared defaults, including the legacy
// `retinaDisplay` -> `performanceMode` localStorage migration).
const createViewerState = (events: EventHandler, config?: Config): State => {
    const legacyRetina = localStorage.getItem('retinaDisplay');
    if (legacyRetina !== null && localStorage.getItem('performanceMode') === null) {
        localStorage.setItem('performanceMode', String(legacyRetina === 'false'));
        localStorage.removeItem('retinaDisplay');
    }
    const storedPerformanceMode = localStorage.getItem('performanceMode');

    return observe(events, {
        loaded: false,
        readyToRender: false,
        performanceMode: storedPerformanceMode !== null ? storedPerformanceMode === 'true' : platform.mobile,
        progress: 0,
        inputMode: platform.mobile ? 'touch' : 'desktop',
        cameraMode: 'orbit',
        hasAnimation: false,
        animationDuration: 0,
        animationTime: 0,
        animationPaused: true,
        animationLoopMode: initialLoopMode(config),
        animationSpeed: 1,
        hasAR: false,
        hasVR: false,
        hasCollision: false,
        hasCollisionOverlay: false,
        walkAllowed: false,
        collisionOverlayEnabled: false,
        isFullscreen: false,
        controlsHidden: false,
        showAnnotations: localStorage.getItem('showAnnotations') !== 'false',
        gamingControls: localStorage.getItem('gamingControls') === 'true'
    });
};

export { createApp, initCanvas, createViewerState, isStatic3dgsFilename, loadContent, loadSkybox };
