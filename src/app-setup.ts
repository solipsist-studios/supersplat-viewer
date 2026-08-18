import {
    Asset,
    Color,
    createGraphicsDevice,
    Entity,
    GSplatResource,
    Keyboard,
    Mouse,
    platform,
    TouchDevice
} from 'playcanvas';
import type { AppBase, EventHandler, TextureHandler } from 'playcanvas';

import { SogstSplatAnimation } from './animation/sogst-splat-animation';
import { App } from './app';
import { fullFileCacheKey, fullFileKeyPrefix } from './core/fetch-splat-anim-buffer';
import { updateGsplatRangeData, updateGsplatSHRange, uploadGsplatRows } from './core/gsplat-range-sync';
import { loadSogst, setAabbFromMeta } from './core/load-sogst';
import { setupSplatAnim } from './core/load-splat-anim';
import { observe } from './core/observe';
import { idbDeleteByPrefix, idbGetBuffer, idbSetBuffer } from './core/sogst-cache';
import type { SogstData } from './core/sogst-data';
import { attachSogstMotion, syncSogstMotionRange, uploadSogstMotionRows } from './core/sogst-motion';
import { streamSogst } from './core/stream-sogst';
import { isSogstFilename } from './parsers/sogst';
import type { Config, Global, State } from './types';

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

// Create resource + entity + animation for decoded archive data (shared by
// full-buffer and streaming paths).
const setupSogst = (app: AppBase, config: Config, global: Global, data: SogstData) => {
    const resource = new GSplatResource(app.graphicsDevice, data.gsplatData);
    attachSogstMotion(resource, data);

    const animation = new SogstSplatAnimation(data);
    const entity = setupSplatAnim(app, config, global, resource, animation, {
        rotationEulerDeg: config.sogstRotationDeg ?? [0, 0, 0],
        alphaClip: 1 / 1024
    });
    animation.bind(entity, data.cov2dScale);
    return { resource, entity };
};

// Progressive load of a streamed archive: reveal the scene once the
// persistent group and the first temporal segment are decoded, then keep
// refreshing GPU data as later segments land (playback holds at the loaded
// boundary via data.loadedThrough if the network falls behind). The
// complete archive is cached for instant revisits.
const loadSogstStreaming = async (
    app: AppBase,
    config: Config,
    global: Global,
    cacheKey: string,
    progressCallback: (progress: number) => void
) => {
    let resource: GSplatResource | null = null;
    let data: Awaited<ReturnType<typeof loadSogst>> | null = null;

    // GPU repack runs in small chunks against a per-slice time budget —
    // chunk counts alone can't bound task length on weak devices (the SH
    // pass dominates at ~45 coeffs per splat, and a throttled CPU can
    // spend >100ms on a chunk that takes 5ms on a fast one). Uploads are
    // NOT per repack chunk: slices only write the CPU level copies, and
    // each queue item ends with row-upload passes bounded to about a
    // megabyte per task — a single multi-MB texSubImage2D batch into
    // actively-sampled textures can stall the driver for a frame or more.
    const SYNC_CHUNK_SPLATS = 256;
    const SYNC_SLICE_MS = 6;
    const UPLOAD_CHUNK_SPLATS = 16384;

    // Refreshing the depth sorter (centersVersion) re-clones the whole
    // centers buffer to the sort worker and rebuilds the sorted order —
    // measured at ~90ms of main-thread + driver-sync work per refresh on
    // real GPUs, which read as metronomic playback hitches when done on a
    // cadence during streaming. So the sorter is refreshed exactly once,
    // when the stream completes: until then newly streamed splats render
    // at their correct positions but blend in slightly stale depth order.
    const bumpCenters = () => {
        if (resource) {
            resource.centersVersion++;
            app.renderNextFrame = true;
        }
    };

    // Per-segment GPU refreshes run through a FIFO queue processed in
    // small slices with yields in between: segments decode behind live
    // playback, and a whole-segment repack in one task reads as a visible
    // hitch on weak devices. data.loadedThrough only advances once a
    // segment's slices have all been pushed to the GPU, so the playhead
    // can never enter a segment that isn't fully renderable yet.
    const queue: { range: [number, number] | null; loadedThrough: number | null; sh: boolean }[] = [];
    let pumping = false;

    // Teardown guard. The pump yields to the UI between slices, so an
    // app.destroy() — an embed host switching scenes while this one is
    // still streaming — lands mid-loop and the continuation runs against a
    // destroyed device. AppBase fires 'destroy' before tearing the device
    // down, so the flag is set while the loop can still observe it. Every
    // resumption point below must re-check it: passing the entry check
    // proves nothing about the state after the next await.
    //
    // The upload half of that is currently absorbed by accident —
    // app.destroy() tears down the gsplat resource first, so the textures
    // are gone and uploadTextureRows() bails on its `!level` check before
    // touching GL. Don't rely on that: it is incidental ordering, not a
    // guard, and removing the `!level` early-return would expose a null-GL
    // write (reported from an embed host as a TypeError on activeTexture).
    let destroyed = false;
    app.on('destroy', () => {
        destroyed = true;
        queue.length = 0;
    });

    const yieldToUi = () =>
        new Promise((resolve) => {
            setTimeout(resolve, 0);
        });

    const drain = async () => {
        while (queue.length > 0) {
            if (destroyed) {
                return;
            }
            const item = queue.shift()!;
            if (item.range && resource && data) {
                const [a, b] = item.range;
                const centers = resource.centers as Float32Array | undefined;
                const x = data.gsplatData.getProp('x') as Float32Array;
                const y = data.gsplatData.getProp('y') as Float32Array;
                const z = data.gsplatData.getProp('z') as Float32Array;
                let s = a;
                while (s < b) {
                    const sliceStart = performance.now();
                    while (s < b) {
                        const e = Math.min(b, s + SYNC_CHUNK_SPLATS);
                        if (item.sh) {
                            // deferred SH arrival: geometry for this range
                            // is already live, only the SH textures change
                            updateGsplatSHRange(resource, data.gsplatData, s, e, false);
                        } else {
                            updateGsplatRangeData(resource, data.gsplatData, s, e, false);
                            syncSogstMotionRange(resource, data, s, e, false);
                            if (centers) {
                                for (let i = s; i < e; i++) {
                                    centers[i * 3 + 0] = x[i];
                                    centers[i * 3 + 1] = y[i];
                                    centers[i * 3 + 2] = z[i];
                                }
                            }
                        }
                        s = e;
                        if (performance.now() - sliceStart >= SYNC_SLICE_MS) {
                            break;
                        }
                    }

                    await yieldToUi();
                    if (destroyed) {
                        return;
                    }
                }
                for (let u = a; u < b; u += UPLOAD_CHUNK_SPLATS) {
                    const e = Math.min(b, u + UPLOAD_CHUNK_SPLATS);
                    uploadGsplatRows(resource, u, e, item.sh);
                    if (!item.sh) {
                        uploadSogstMotionRows(resource, u, e);
                    }

                    await yieldToUi();
                    if (destroyed) {
                        return;
                    }
                }
            }
            if (data && item.loadedThrough !== null) {
                data.loadedThrough = item.loadedThrough;
            }
            if (!item.range) {
                // end of stream: single sorter refresh over the full scene
                bumpCenters();
            }
            app.renderNextFrame = true;
        }
    };

    // Sync wrapper: callers fire and forget. `pumping` is released in a
    // finally so an early bail-out on teardown cannot strand it set, which
    // would silently wedge a later pump.
    const pump = () => {
        if (pumping || destroyed) {
            return;
        }
        pumping = true;
        drain().finally(() => {
            pumping = false;
        });
    };

    const sync = (range: [number, number] | null, loadedThrough: number) => {
        queue.push({ range, loadedThrough, sh: false });
        if (resource && data) {
            pump();
        }
    };

    const syncSh = (range: [number, number]) => {
        queue.push({ range, loadedThrough: null, sh: true });
        if (resource && data) {
            pump();
        }
    };

    const stream = streamSogst(app, config.contentUrl, {
        onProgress: progressCallback,
        onReady: sync,
        onShReady: syncSh
    });

    data = await stream.reveal;

    // Same teardown race as the pump, but on the reveal continuation and
    // with a louder failure: setupSogst builds a GSplatResource against
    // app.graphicsDevice, which app.destroy() has already set to null
    // (TypeError reading 'isWebGPU'). Nothing downstream of a destroyed app
    // can do anything useful, so leave the load promise unsettled rather
    // than handing the caller a scene that belongs to a dead device.
    if (destroyed) {
        return new Promise<Entity>(() => {
            /* never settles: the app is gone */
        });
    }

    const setup = setupSogst(app, config, global, data);
    resource = setup.resource;

    // exact bounds from the global meta range — the arrays are still
    // partially filled, so computed bounds would understate the scene
    setAabbFromMeta(data.meta, resource.aabb);

    // catch up on any groups that decoded while the entity was being set up
    // (the reveal set itself was picked up at resource creation)
    pump();

    // cache the complete archive in the background for instant revisits
    stream.complete
        .then((buffer) => {
            idbSetBuffer(cacheKey, buffer)
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
    const cached = await idbGetBuffer(cacheKey);
    if (cached) {
        // complete archive available locally — decode straight through
        console.debug('SOGST full-file cache hit (idb)', cacheKey);
        const data = await loadSogst(app, cached, progressCallback);
        return setupSogst(app, config, global, data).entity;
    }
    return loadSogstStreaming(app, config, global, cacheKey, progressCallback);
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

const createApp = async (canvas: HTMLCanvasElement, config: Config) => {
    const useWebGPU = config.renderer === 'webgpu';

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

    console.log(`Renderer: ${device.deviceType}`);

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

// Build the observable viewer state (shared defaults, including the legacy
// `retinaDisplay` -> `performanceMode` localStorage migration).
const createViewerState = (events: EventHandler): State => {
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
        animationLoopMode: 'repeat',
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
