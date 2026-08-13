import {
    Asset,
    Color,
    createGraphicsDevice,
    Entity,
    EventHandler,
    GSplatResource,
    Keyboard,
    Mouse,
    platform,
    TouchDevice,
    type TextureHandler,
    type AppBase
} from 'playcanvas';

import { QueenSplatAnimation } from './animation/queen-splat-animation';
import { SogstSplatAnimation } from './animation/sogst-splat-animation';
import { SogstV1SplatAnimation } from './animation/sogst-v1-splat-animation';
import { App } from './app';
import { fetchSplatAnimBuffer, fullFileCacheKey, fullFileKeyPrefix } from './core/fetch-splat-anim-buffer';
import { updateGsplatRangeData, updateGsplatSHRange, uploadGsplatRows } from './core/gsplat-range-sync';
import { isSogstV3, loadSogstV3, setAabbFromV3Meta } from './core/load-sogst-v3';
import { setupSplatAnim } from './core/load-splat-anim';
import { observe } from './core/observe';
import { idbDeleteByPrefix, idbGetBuffer, idbSetBuffer } from './core/sogst-cache';
import { attachSogstMotion, syncSogstMotion, syncSogstMotionRange, uploadSogstMotionRows } from './core/sogst-motion';
import { streamQueenData } from './core/stream-queen';
import { streamSogstV1Data } from './core/stream-sogst-v1';
import { streamSogstV2 } from './core/stream-sogst-v2';
import { streamSogstV3 } from './core/stream-sogst-v3';
import { isSogstFilename, parseSogstV2, readSogstVersion, readSogstV2Header } from './parsers/sogst';
import type { SogstData } from './parsers/sogst';
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

// Fetch the first bytes of a .sogst file (enough for any header variant).
// Uses a byte-range request, but reads through the stream reader and cancels
// so a server that ignores Range headers doesn't trigger a full download.
const fetchSogstHeaderBytes = async (url: string, byteCount: number): Promise<ArrayBuffer> => {
    const response = await fetch(url, { headers: { range: `bytes=0-${byteCount - 1}` } });
    if (!response.ok) {
        throw new Error(`Failed to fetch ${url}: ${response.status} ${response.statusText}`);
    }
    const reader = response.body?.getReader();
    if (!reader) {
        return response.arrayBuffer();
    }
    const out = new Uint8Array(byteCount);
    let filled = 0;
    while (filled < byteCount) {
        // eslint-disable-next-line no-await-in-loop
        const { done, value } = await reader.read();
        if (done) {
            break;
        }
        const take = Math.min(byteCount - filled, value.length);
        out.set(value.subarray(0, take), filled);
        filled += take;
    }
    reader.cancel().catch(() => {});
    return out.buffer;
};

// Create resource + entity + animation for parsed v2 data (shared by the
// full-buffer and streaming paths).
const setupSogst = (app: AppBase, config: Config, global: Global, data: SogstData) => {
    const resource = new GSplatResource(app.graphicsDevice, data.gsplatData);
    attachSogstMotion(resource, data);

    const animation = new SogstSplatAnimation(data);
    const entity = setupSplatAnim(app, config, global, resource, animation, {
        rotationEulerDeg: config.sogstRotationDeg ?? [270, 0, 0],
        alphaClip: 1 / 1024
    });
    animation.bind(entity, data.cov2dScale);
    return { resource, entity };
};

// Progressive load of a streamable (tiled) v2 file: create the resource over
// a prefilled buffer, start rendering after the first ~10% of splats, and
// keep refreshing the GPU data as tiles arrive. The completed buffer is
// cached in standard layout, so revisits take the instant cache path.
const loadSogstV2Streaming = async (
    app: AppBase,
    config: Config,
    global: Global,
    header: ReturnType<typeof readSogstV2Header>,
    cacheKey: string,
    progressCallback: (progress: number) => void
) => {
    let resource: GSplatResource | null = null;
    let data: SogstData | null = null;
    let syncCount = 0;

    // Push everything received so far to the GPU. The engine update methods
    // read straight through the gsplatData views over the streaming buffer;
    // unready splats hold prefilled invisible values.
    const sync = (readySplats: number, done: boolean) => {
        if (!resource || !data) {
            return;
        }
        const r = resource as any;
        r.updateColorData(data.gsplatData);
        r.updateTransformData(data.gsplatData);
        // SH repack is the most expensive — half cadence, but always at the end
        if (header.hasSH && (done || syncCount % 2 === 0)) {
            r.updateSHData(data.gsplatData);
        }
        syncCount++;

        syncSogstMotion(resource, data, readySplats);

        // Refresh depth-sorter centers for the splats received so far.
        const centers = r.centers as Float32Array | undefined;
        if (centers) {
            const x = data.gsplatData.getProp('x') as Float32Array;
            const y = data.gsplatData.getProp('y') as Float32Array;
            const z = data.gsplatData.getProp('z') as Float32Array;
            for (let i = 0; i < readySplats; i++) {
                centers[i * 3 + 0] = x[i];
                centers[i * 3 + 1] = y[i];
                centers[i * 3 + 2] = z[i];
            }
            r.centersVersion++;
        }

        app.renderNextFrame = true;
    };

    const stream = streamSogstV2(config.contentUrl, header, {
        onProgress: progressCallback,
        onReady: sync
    });

    data = stream.data;
    resource = new GSplatResource(app.graphicsDevice, data.gsplatData);
    attachSogstMotion(resource, data);

    await stream.firstBatch;

    // Importance-ordered tiles span the scene early, so bounds computed from
    // the first batch are representative (prefilled splats add the origin).
    data.gsplatData.calcAabb((resource as any).aabb);

    // Cache the completed standard-layout buffer in the background.
    stream.complete
    .then((buffer) => {
        idbSetBuffer(cacheKey, buffer)
        .then(() => idbDeleteByPrefix(fullFileKeyPrefix(config.contentUrl), cacheKey))
        .catch(() => {});
    })
    .catch((err: Error) => {
        console.warn('SOGST stream did not complete; partial scene retained:', err);
    });

    const animation = new SogstSplatAnimation(data);
    const entity = setupSplatAnim(app, config, global, resource, animation, {
        rotationEulerDeg: config.sogstRotationDeg ?? [270, 0, 0],
        alphaClip: 1 / 1024
    });
    animation.bind(entity, data.cov2dScale);
    return entity;
};

// Progressive load of a streamed v3 archive: reveal the scene once the
// persistent group and the first temporal segment are decoded, then keep
// refreshing GPU data as later segments land (playback holds at the loaded
// boundary via data.loadedThrough if the network falls behind). The
// complete archive is cached for instant revisits.
const loadSogstV3Streaming = async (
    app: AppBase,
    config: Config,
    global: Global,
    cacheKey: string,
    progressCallback: (progress: number) => void
) => {
    let resource: GSplatResource | null = null;
    let data: Awaited<ReturnType<typeof loadSogstV3>> | null = null;

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
            (resource as any).centersVersion++;
            app.renderNextFrame = true;
        }
    };

    // Per-segment GPU refreshes run through a FIFO queue processed in
    // small slices with yields in between: segments decode behind live
    // playback, and a whole-segment repack in one task reads as a visible
    // hitch on weak devices. data.loadedThrough only advances once a
    // segment's slices have all been pushed to the GPU, so the playhead
    // can never enter a segment that isn't fully renderable yet.
    const queue: { range: [number, number] | null, loadedThrough: number | null, sh: boolean }[] = [];
    let pumping = false;

    const yieldToUi = () => new Promise((resolve) => {
        setTimeout(resolve, 0);
    });

    const pump = async () => {
        if (pumping) {
            return;
        }
        pumping = true;
        while (queue.length > 0) {
            const item = queue.shift()!;
            if (item.range && resource && data) {
                const r = resource as any;
                const [a, b] = item.range;
                const centers = r.centers as Float32Array | undefined;
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
                            updateGsplatSHRange(r, data.gsplatData, s, e, false);
                        } else {
                            updateGsplatRangeData(r, data.gsplatData, s, e, false);
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
                    // eslint-disable-next-line no-await-in-loop -- deliberate UI yield
                    await yieldToUi();
                }
                for (let u = a; u < b; u += UPLOAD_CHUNK_SPLATS) {
                    const e = Math.min(b, u + UPLOAD_CHUNK_SPLATS);
                    uploadGsplatRows(r, u, e, item.sh);
                    if (!item.sh) {
                        uploadSogstMotionRows(resource, u, e);
                    }
                    // eslint-disable-next-line no-await-in-loop -- deliberate UI yield
                    await yieldToUi();
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
        pumping = false;
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

    const stream = streamSogstV3(app, config.contentUrl, {
        onProgress: progressCallback,
        onReady: sync,
        onShReady: syncSh
    });

    data = await stream.reveal;
    const setup = setupSogst(app, config, global, data);
    resource = setup.resource;

    // exact bounds from the global meta range — the arrays are still
    // partially filled, so computed bounds would understate the scene
    setAabbFromV3Meta(data.meta, (resource as any).aabb);

    // catch up on any groups that decoded while the entity was being set up
    // (the reveal set itself was picked up at resource creation)
    pump();

    // cache the complete archive in the background for instant revisits
    stream.complete
    .then((buffer) => {
        idbSetBuffer(cacheKey, buffer)
        .then(() => idbDeleteByPrefix(fullFileKeyPrefix(config.contentUrl), cacheKey))
        .catch(() => {});
    })
    .catch((err: Error) => {
        console.warn('SOGST v3 stream did not complete; partial scene retained:', err);
    });

    return setup.entity;
};

// Load and animate a .sogst (SOG spacetime 4DGS) file — either extension.
const loadSogstGsplat = async (app: AppBase, config: Config, global: Global, progressCallback: (progress: number) => void) => {
    const headerBytes = await fetchSogstHeaderBytes(config.contentUrl, 40);

    if (isSogstV3(headerBytes)) {
        // v3: SOG-compressed ZIP container (webp textures + codebooks),
        // decoded through the engine's SOG path and played through the
        // whole v2 temporal setup unchanged.
        const cacheKey = await fullFileCacheKey(config.contentUrl);
        const cached = await idbGetBuffer(cacheKey);
        if (cached) {
            // complete archive available locally — decode straight through
            console.debug('SOGST v3 full-file cache hit (idb)', cacheKey);
            const data = await loadSogstV3(app, cached, progressCallback);
            return setupSogst(app, config, global, data).entity;
        }
        return loadSogstV3Streaming(app, config, global, cacheKey, progressCallback);
    }

    const version = readSogstVersion(headerBytes);

    if (version >= 2) {
        // v2: compact temporal format; motion and temporal fade are evaluated
        // on the GPU from a single time uniform.
        const header = readSogstV2Header(headerBytes);

        if (header.tiled) {
            // Streamable layout: serve from the durable cache when possible,
            // otherwise render progressively while downloading.
            const cacheKey = await fullFileCacheKey(config.contentUrl);
            const cached = await idbGetBuffer(cacheKey);
            if (cached) {
                console.debug('SOGST full-file cache hit (idb)', cacheKey);
                progressCallback(100);
                return setupSogst(app, config, global, parseSogstV2(cached)).entity;
            }
            return loadSogstV2Streaming(app, config, global, header, cacheKey, progressCallback);
        }

        // Standard layout: full prefetch (with its own cache handling).
        const buffer = await fetchSplatAnimBuffer(config.contentUrl, progressCallback);
        return setupSogst(app, config, global, parseSogstV2(buffer)).entity;
    }

    // v1: legacy baked per-frame format, streamed in chunks.
    const data = await streamSogstV1Data(config.contentUrl, progressCallback);
    const resource = new GSplatResource(app.graphicsDevice, data.gsplatData);
    const animation = new SogstV1SplatAnimation(data, resource);
    return setupSplatAnim(app, config, global, resource, animation, {
        rotationEulerDeg: config.sogstRotationDeg ?? [270, 0, 0],
        alphaClip: 1 / 1024
    });
};

// Load and animate a .queen (QUEEN-encoded 4D Gaussian Splat) file.
// Waits until initialFrames have been buffered before resolving, so playback
// starts immediately without stutter; remaining frames stream in the background.
const loadQueenGsplat = async (app: AppBase, config: Config, global: Global, progressCallback: (progress: number) => void) => {
    const data = await streamQueenData(config.contentUrl, progressCallback);
    data.loadFrame(0);
    const resource = new GSplatResource(app.graphicsDevice, data.gsplatData);
    const animation = new QueenSplatAnimation(data, resource);
    return setupSplatAnim(app, config, global, resource, animation);
};

// Load a static 3DGS scene (PLY / LOD / meta.json etc.)
const load3dgs = (app: AppBase, config: Config, progressCallback: (progress: number) => void) => loadGsplat(app, config, progressCallback);

// Load and animate a 4DGS file, dispatching to the correct format handler.
const load4dgs = (app: AppBase, config: Config, global: Global, progressCallback: (progress: number) => void): Promise<Entity> => {
    const lowerName = (config.contentFilename ?? new URL(config.contentUrl, location.href).pathname.split('/').pop() ?? '').toLowerCase();
    if (isSogstFilename(lowerName))   return loadSogstGsplat(app, config, global, progressCallback);
    if (lowerName.endsWith('.queen')) return loadQueenGsplat(app, config, global, progressCallback);
    return Promise.reject(new Error(`Unsupported 4DGS format: ${lowerName}`));
};

const loadSkybox = (app: AppBase, url: string) => {
    return new Promise<Asset>((resolve, reject) => {
        const asset = new Asset('skybox', 'texture', {
            url
        }, {
            type: 'rgbp',
            mipmaps: false,
            addressu: 'repeat',
            addressv: 'clamp'
        });

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
    // @ts-ignore
    app._allowResize = false;
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

export { createApp, initCanvas, createViewerState, load3dgs, load4dgs, loadSkybox };
