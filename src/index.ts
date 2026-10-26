import { EventHandler, platform, revision as engineRevision, version as engineVersion } from 'playcanvas';
import type { Texture } from 'playcanvas';

import {
    createApp,
    createViewerState,
    initCanvas,
    isPrefetchable,
    loadContent,
    loadSkybox,
    measureCanvas,
    resizeCanvas
} from './app-setup';
import { MeshCollision, loadVoxelCollision } from './collision';
import type { Collision } from './collision';
import { initEmbed } from './embed';
import { initLocalization } from './localization';
import type { CreateViewerOptions } from './options';
import { persistPreferences, readPreferences } from './preferences';
import { importSettings } from './settings';
import type { Config, Global, ViewerHandle } from './types';
import { initPoster, initUI } from './ui';
import uiHtml from './ui.html';
import { Viewer } from './viewer';
import { version as appVersion } from '../package.json';

const createImage = (url: string) => {
    const img = new Image();
    img.src = url;
    return img;
};

// the options with every default applied
const resolveConfig = (options: CreateViewerOptions): Config => ({
    contentUrl: options.contentUrl,
    contentFilename: options.contentFilename,
    posterUrl: options.posterUrl,
    skyboxUrl: options.skyboxUrl,
    collisionUrl: options.collisionUrl,
    poster: options.poster ?? (options.posterUrl ? createImage(options.posterUrl) : undefined),
    contents:
        options.contents ??
        (isPrefetchable(options.contentUrl, options.contentFilename) ? fetch(options.contentUrl) : undefined),
    renderer: options.renderer,
    ui: options.ui ?? true,
    reticle: options.reticle ?? false,
    noanim: options.noanim ?? false,
    nofx: options.nofx ?? false,
    hpr: options.hpr,
    ministats: options.ministats ?? false,
    colorize: options.colorize ?? false,
    fullload: options.fullload ?? false,
    aa: options.aa ?? false,
    budget: options.budget,
    heatmap: options.heatmap ?? false,
    debug: options.debug ?? false,
    lang: options.lang,
    transparent: options.transparent ?? false,
    embed: options.embed ?? false,
    sogstBudget: options.sogstBudget,
    sogstGpuDecode: options.sogstGpuDecode ?? true,
    sogstRotationDeg: options.sogstRotationDeg,
    animLoopMode: options.animLoopMode,
    exposeGlobals: options.exposeGlobals ?? false
});

const createViewer = async (options: CreateViewerOptions): Promise<ViewerHandle> => {
    const { container } = options;
    const config = resolveConfig(options);

    // the instance root. The canvas and the ui markup are siblings under it, which scopes
    // everything the viewer looks up or attaches in the dom; the viewer owns it outright, so
    // nothing on the host's own element is read or written, and destroy() removes it whole
    const root = document.createElement('div');
    root.className = 'sse-viewer';
    if (config.ui) {
        root.innerHTML = uiHtml;
    } else {
        root.appendChild(document.createElement('canvas'));
    }

    container.appendChild(root);
    const canvas = root.querySelector('canvas');

    // create events
    const events = new EventHandler();

    // the poster covers the hidden canvas from the first moment, before the graphics device
    // exists. It is part of the ui, so a headless instance shows the canvas from the start and
    // its host covers the wait however it likes
    if (config.poster && config.ui) {
        initPoster(root, config.poster, events);
    }

    // resolve settings after showing the poster, including a fetch started by the document
    const settingsJson =
        typeof options.settings === 'string' ? await (await fetch(options.settings)).json() : await options.settings;

    const preferences = readPreferences(platform.mobile);

    // size the canvas backbuffer before the graphics device is created, so the swap
    // chain and any backbuffer-sized resources start at the correct resolution instead
    // of being recreated on the first frame's resize. a hidden embed keeps the default
    // canvas size here; the resize observer sizes it on reveal
    resizeCanvas(canvas, measureCanvas(canvas), preferences.performanceMode);

    const { app, camera, renderer } = await createApp(canvas, config);

    // translate the markup and get this instance's string lookup, before the ui reads any
    const localize = initLocalization(config.lang, root);

    // shared with the embed entry point, so the defaults live in one place and the two
    // bundles cannot drift (a missing field reads as `undefined` at runtime)
    const state = createViewerState(events, config);

    const global: Global = {
        app,
        settings: importSettings(settingsJson),
        config,
        state,
        events,
        camera,
        renderer,
        root,
        localize
    };

    const disposeCanvas = initCanvas(global);

    // start the application
    app.start();

    camera.addComponent('camera');

    // a load continuation can outlive a destroy, so anything that resumes after an await checks
    // this before touching the app
    let destroyed = false;

    // Load model: loadContent picks the handler from the filename and rejects an unrecognised
    // extension without fetching the body
    const gsplatLoad = loadContent(
        app,
        config,
        global,
        (progress: number) => {
            state.progress = progress;
        },
        () => destroyed
    );

    // Load skybox (continue without if it fails — e.g. CORS, 404)
    const skyboxLoad =
        config.skyboxUrl &&
        loadSkybox(app, config.skyboxUrl)
            .then((asset) => {
                app.scene.envAtlas = asset.resource as Texture;
            })
            .catch((err: Error) => {
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
    let disposeAudio: (() => void) | undefined;
    if (global.settings.soundUrl) {
        const sound = new Audio(global.settings.soundUrl);
        sound.crossOrigin = 'anonymous';
        const unlock = () => {
            if (sound) {
                sound.play();
            }
        };
        root.addEventListener('click', unlock, {
            capture: true,
            once: true
        });
        disposeAudio = () => {
            root.removeEventListener('click', unlock, { capture: true });
            sound.pause();
        };
    }

    // Create the viewer
    const viewer = new Viewer(global, gsplatLoad, skyboxLoad, collisionLoad);
    viewer.onDestroy(persistPreferences(events));

    // Enable the host-page bridge when embedded
    if (config.embed) {
        initEmbed(global, viewer);
    }

    const handle: ViewerHandle = {
        app,
        state,
        events,
        annotations: global.settings.annotations,
        captureFrame: (captureOptions) => viewer.captureFrame(captureOptions),
        seek: (time) => viewer.seek(time),
        frameScene: () => viewer.frameScene(),
        resetCamera: () => viewer.resetCamera(),
        toggleWalk: () => viewer.toggleWalk(),
        selectAnnotation: (index) => viewer.selectAnnotation(index),
        setMoveInput: (x, z) => viewer.setMoveInput(x, z),
        requestFullscreen: () => viewer.requestFullscreen(),
        exitFullscreen: () => viewer.exitFullscreen(),
        startXR: (mode) => viewer.startXR(mode),
        endXR: () => viewer.endXR(),
        destroy: () => viewer.destroy()
    };

    // The built-in controls use the same handle returned to an embedding host.
    const disposeUI = config.ui ? initUI(global, handle, !!viewer.cameraFrame) : null;
    viewer.onDestroy(() => {
        destroyed = true;
    });
    viewer.onDestroy(disposeCanvas);
    if (disposeUI) {
        viewer.onDestroy(disposeUI);
    }
    if (disposeAudio) {
        viewer.onDestroy(disposeAudio);
    }

    return handle;
};

console.log(`SuperSplat Viewer v${appVersion} | Engine v${engineVersion} (${engineRevision})`);

export type { CaptureResult } from './capture';
export type { CreateViewerOptions, ViewerAssets, ViewerFlags } from './options';
export type { CaptureOptions, ViewerHandle, ViewerState, XrMode } from './types';
export { createViewer };
