import {
    EventHandler,
    type Texture,
    revision as engineRevision,
    version as engineVersion
} from 'playcanvas';

import { version as appVersion } from '../package.json';
import { createApp, createViewerState, initCanvas, load3dgs, load4dgs, loadSkybox } from './app-setup';
import { MeshCollision, loadVoxelCollision } from './collision';
import type { Collision } from './collision';
import { initEmbed } from './embed';
import { initLocalization } from './localization';
import { importSettings } from './settings';
import type { Config, Global } from './types';
import { initPoster, initUI } from './ui';
import { Viewer } from './viewer';
import { initXr } from './xr';

const main = async (canvas: HTMLCanvasElement, settingsJson: any, config: Config) => {
    const { app, camera, renderer } = await createApp(canvas, config);

    // create events and observable state
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
    initLocalization(config.lang);
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
    const viewer = new Viewer(global, gsplatLoad, skyboxLoad, collisionLoad);

    // Enable the host-page bridge when embedded
    if (config.embed) {
        initEmbed(global, viewer);
    }

    return viewer;
};

console.log(`SuperSplat Viewer v${appVersion} | Engine v${engineVersion} (${engineRevision})`);

export { main };
