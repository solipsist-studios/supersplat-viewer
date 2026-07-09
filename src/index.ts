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
    type Texture,
    type TextureHandler,
    type AppBase,
    revision as engineRevision,
    version as engineVersion
} from 'playcanvas';

import { Omg4SplatAnimation } from './animation/omg4-splat-animation';
import { Omg4V2SplatAnimation } from './animation/omg4-v2-splat-animation';
import { QueenSplatAnimation } from './animation/queen-splat-animation';
import { App } from './app';
import { MeshCollision, loadVoxelCollision } from './collision';
import type { Collision } from './collision';
import { version as appVersion } from '../package.json';
import { fetchSplatAnimBuffer } from './core/fetch-splat-anim-buffer';
import { setupSplatAnim } from './core/load-splat-anim';
import { observe } from './core/observe';
import { attachOmg4V2Motion } from './core/omg4-v2-motion';
import { streamOmg4Data } from './core/stream-omg4';
import { streamQueenData } from './core/stream-queen';
import { initLocalization } from './localization';
import { parseOmg4V2, readOmg4Version } from './parsers/omg4';
import { importSettings } from './settings';
import type { Config, Global } from './types';
import { initPoster, initUI } from './ui';
import { Viewer } from './viewer';
import { initXr } from './xr';

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

// Read the format version of a .omg4 file with a small byte-range request
// (falls back gracefully when the server ignores Range headers).
const fetchOmg4Version = async (url: string): Promise<number> => {
    const response = await fetch(url, { headers: { range: 'bytes=0-31' } });
    if (!response.ok) {
        throw new Error(`Failed to fetch ${url}: ${response.status} ${response.statusText}`);
    }
    const buffer = await response.arrayBuffer();
    return readOmg4Version(buffer);
};

// Load and animate a .omg4 (OMG4-encoded 4D Gaussian Splat) file.
const loadOmg4Gsplat = async (app: AppBase, config: Config, global: Global, progressCallback: (progress: number) => void) => {
    const version = await fetchOmg4Version(config.contentUrl);

    if (version >= 2) {
        // v2: compact temporal format. Small enough to load fully; motion and
        // temporal fade are evaluated on the GPU from a single time uniform.
        const buffer = await fetchSplatAnimBuffer(config.contentUrl, progressCallback);
        const data = parseOmg4V2(buffer);
        const resource = new GSplatResource(app.graphicsDevice, data.gsplatData);
        attachOmg4V2Motion(resource, data);

        const animation = new Omg4V2SplatAnimation(data);
        const entity = setupSplatAnim(app, config, global, resource, animation, {
            rotationEulerDeg: config.omg4RotationDeg ?? [270, 0, 0],
            alphaClip: 1 / 1024
        });
        animation.bind(entity, data.cov2dScale);
        return entity;
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
    if (lowerName.endsWith('.omg4'))  return loadOmg4Gsplat(app, config, global, progressCallback);
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
    // when WebGPU isn't supported, so request xrCompatible so the WebGL fallback
    // is also usable for AR/VR.
    const device = await createGraphicsDevice(canvas, {
        deviceTypes: useWebGPU ? ['webgpu'] : [],
        antialias: false,
        depth: true,
        stencil: false,
        xrCompatible: true,
        powerPreference: 'high-performance'
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

const main = async (canvas: HTMLCanvasElement, settingsJson: any, config: Config) => {
    const { app, camera, renderer } = await createApp(canvas, config);

    // create events
    const events = new EventHandler();

    // migrate legacy `retinaDisplay` preference (inverted) to `performanceMode`
    const legacyRetina = localStorage.getItem('retinaDisplay');
    if (legacyRetina !== null && localStorage.getItem('performanceMode') === null) {
        localStorage.setItem('performanceMode', String(legacyRetina === 'false'));
        localStorage.removeItem('retinaDisplay');
    }
    const storedPerformanceMode = localStorage.getItem('performanceMode');

    const state = observe(events, {
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

    const global: Global = {
        app,
        settings: importSettings(settingsJson),
        config,
        state,
        events,
        camera,
        renderer
    };

    initCanvas(global);

    // start the application
    app.start();

    // Initialize the load-time poster
    if (config.poster) {
        initPoster(events);
    }

    camera.addComponent('camera');

    // Initialize XR support (availability detection always runs so the UI can offer
    // a reload into WebGL when the user requests AR/VR under WebGPU)
    initXr(global);

    // Initialize user interface
    initLocalization();
    initUI(global);

    // Load model
    const filename = config.contentFilename ?? new URL(config.contentUrl, location.href).pathname.split('/').pop() ?? '';
    const lowerFilename = filename.toLowerCase();
    const progressCallback = (progress: number) => {
        state.progress = progress;
    };
    const is4dgs = lowerFilename.endsWith('.omg4') || lowerFilename.endsWith('.queen');
    const gsplatLoad = is4dgs ?
        load4dgs(app, config, global, progressCallback) :
        load3dgs(app, config, progressCallback);

    // Load skybox (continue without if it fails — e.g. CORS, 404)
    const skyboxLoad = config.skyboxUrl &&
        loadSkybox(app, config.skyboxUrl).then((asset) => {
            app.scene.envAtlas = asset.resource as Texture;
        }).catch((err: Error) => {
            console.warn('Failed to load skybox:', err);
        });

    // Load collision data (type determined by file extension)
    let collisionLoad: Promise<Collision> | undefined;
    if (config.collisionUrl) {
        const ext = new URL(config.collisionUrl, location.href).pathname.split('.').pop()?.toLowerCase();
        if (ext === 'glb') {
            collisionLoad = MeshCollision.fromGlb(app, config.collisionUrl).catch((err: Error): null => {
                console.warn('Failed to load mesh collision:', err);
                return null;
            });
        } else {
            collisionLoad = loadVoxelCollision(config.collisionUrl).catch((err: Error): null => {
                console.warn('Failed to load voxel data:', err);
                return null;
            });
        }
    }

    // Load and play sound
    if (global.settings.soundUrl) {
        const sound = new Audio(global.settings.soundUrl);
        sound.crossOrigin = 'anonymous';
        document.body.addEventListener('click', () => {
            if (sound) {
                sound.play();
            }
        }, {
            capture: true,
            once: true
        });
    }

    // Create the viewer
    return new Viewer(global, gsplatLoad, skyboxLoad, collisionLoad);
};

console.log(`SuperSplat Viewer v${appVersion} | Engine v${engineVersion} (${engineRevision})`);

export { main };
