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

import { Omg4SplatAnimation } from './animation/omg4-splat-animation';
import { Omg4V2SplatAnimation } from './animation/omg4-v2-splat-animation';
import { QueenSplatAnimation } from './animation/queen-splat-animation';
import { App } from './app';
import { fetchSplatAnimBuffer, fullFileCacheKey, fullFileKeyPrefix } from './core/fetch-splat-anim-buffer';
import { setupSplatAnim } from './core/load-splat-anim';
import { observe } from './core/observe';
import { idbDeleteByPrefix, idbGetBuffer, idbSetBuffer } from './core/omg4-cache';
import { attachOmg4V2Motion, syncOmg4V2Motion } from './core/omg4-v2-motion';
import { streamOmg4Data } from './core/stream-omg4';
import { streamOmg4V2 } from './core/stream-omg4-v2';
import { streamQueenData } from './core/stream-queen';
import { parseOmg4V2, readOmg4Version, readOmg4V2Header } from './parsers/omg4';
import type { Omg4V2Data } from './parsers/omg4';
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

// Fetch the first bytes of a .omg4 file (enough for any header variant).
// Uses a byte-range request, but reads through the stream reader and cancels
// so a server that ignores Range headers doesn't trigger a full download.
const fetchOmg4HeaderBytes = async (url: string, byteCount: number): Promise<ArrayBuffer> => {
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
const setupOmg4V2 = (app: AppBase, config: Config, global: Global, data: Omg4V2Data) => {
    const resource = new GSplatResource(app.graphicsDevice, data.gsplatData);
    attachOmg4V2Motion(resource, data);

    const animation = new Omg4V2SplatAnimation(data);
    const entity = setupSplatAnim(app, config, global, resource, animation, {
        rotationEulerDeg: config.omg4RotationDeg ?? [270, 0, 0],
        alphaClip: 1 / 1024
    });
    animation.bind(entity, data.cov2dScale);
    return { resource, entity };
};

// Progressive load of a streamable (tiled) v2 file: create the resource over
// a prefilled buffer, start rendering after the first ~10% of splats, and
// keep refreshing the GPU data as tiles arrive. The completed buffer is
// cached in standard layout, so revisits take the instant cache path.
const loadOmg4V2Streaming = async (
    app: AppBase,
    config: Config,
    global: Global,
    header: ReturnType<typeof readOmg4V2Header>,
    cacheKey: string,
    progressCallback: (progress: number) => void
) => {
    let resource: GSplatResource | null = null;
    let data: Omg4V2Data | null = null;
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

        syncOmg4V2Motion(resource, data, readySplats);

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

    const stream = streamOmg4V2(config.contentUrl, header, {
        onProgress: progressCallback,
        onReady: sync
    });

    data = stream.data;
    resource = new GSplatResource(app.graphicsDevice, data.gsplatData);
    attachOmg4V2Motion(resource, data);

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
            console.warn('OMG4 stream did not complete; partial scene retained:', err);
        });

    const animation = new Omg4V2SplatAnimation(data);
    const entity = setupSplatAnim(app, config, global, resource, animation, {
        rotationEulerDeg: config.omg4RotationDeg ?? [270, 0, 0],
        alphaClip: 1 / 1024
    });
    animation.bind(entity, data.cov2dScale);
    return entity;
};

// Load and animate a .omg4 (OMG4-encoded 4D Gaussian Splat) file.
const loadOmg4Gsplat = async (
    app: AppBase,
    config: Config,
    global: Global,
    progressCallback: (progress: number) => void
) => {
    const headerBytes = await fetchOmg4HeaderBytes(config.contentUrl, 40);
    const version = readOmg4Version(headerBytes);

    if (version >= 2) {
        // v2: compact temporal format; motion and temporal fade are evaluated
        // on the GPU from a single time uniform.
        const header = readOmg4V2Header(headerBytes);

        if (header.tiled) {
            // Streamable layout: serve from the durable cache when possible,
            // otherwise render progressively while downloading.
            const cacheKey = await fullFileCacheKey(config.contentUrl);
            const cached = await idbGetBuffer(cacheKey);
            if (cached) {
                console.debug('OMG4 full-file cache hit (idb)', cacheKey);
                progressCallback(100);
                return setupOmg4V2(app, config, global, parseOmg4V2(cached)).entity;
            }
            return loadOmg4V2Streaming(app, config, global, header, cacheKey, progressCallback);
        }

        // Standard layout: full prefetch (with its own cache handling).
        const buffer = await fetchSplatAnimBuffer(config.contentUrl, progressCallback);
        return setupOmg4V2(app, config, global, parseOmg4V2(buffer)).entity;
    }

    // v1: legacy baked per-frame format, streamed in chunks.
    const data = await streamOmg4Data(config.contentUrl, progressCallback);
    const resource = new GSplatResource(app.graphicsDevice, data.gsplatData);
    const animation = new Omg4SplatAnimation(data, resource);
    return setupSplatAnim(app, config, global, resource, animation, {
        rotationEulerDeg: config.omg4RotationDeg ?? [270, 0, 0],
        alphaClip: 1 / 1024
    });
};

// Load and animate a .queen (QUEEN-encoded 4D Gaussian Splat) file.
// Waits until initialFrames have been buffered before resolving, so playback
// starts immediately without stutter; remaining frames stream in the background.
const loadQueenGsplat = async (
    app: AppBase,
    config: Config,
    global: Global,
    progressCallback: (progress: number) => void
) => {
    const data = await streamQueenData(config.contentUrl, progressCallback);
    data.loadFrame(0);
    const resource = new GSplatResource(app.graphicsDevice, data.gsplatData);
    const animation = new QueenSplatAnimation(data, resource);
    return setupSplatAnim(app, config, global, resource, animation);
};

// Load a static 3DGS scene (PLY / LOD / meta.json etc.)
const load3dgs = (app: AppBase, config: Config, progressCallback: (progress: number) => void) =>
    loadGsplat(app, config, progressCallback);

// Load and animate a 4DGS file, dispatching to the correct format handler.
const load4dgs = (
    app: AppBase,
    config: Config,
    global: Global,
    progressCallback: (progress: number) => void
): Promise<Entity> => {
    const lowerName = (
        config.contentFilename ??
        new URL(config.contentUrl, location.href).pathname.split('/').pop() ??
        ''
    ).toLowerCase();
    if (lowerName.endsWith('.omg4')) return loadOmg4Gsplat(app, config, global, progressCallback);
    if (lowerName.endsWith('.queen')) return loadQueenGsplat(app, config, global, progressCallback);
    return Promise.reject(new Error(`Unsupported 4DGS format: ${lowerName}`));
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
        hasAR: false,
        hasVR: false,
        hasCollision: false,
        hasCollisionOverlay: false,
        walkAllowed: false,
        collisionOverlayEnabled: false,
        isFullscreen: false,
        controlsHidden: false,
        gamingControls: localStorage.getItem('gamingControls') === 'true'
    });
};

export { createApp, initCanvas, createViewerState, load3dgs, load4dgs, loadSkybox };
