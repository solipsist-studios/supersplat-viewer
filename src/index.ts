import { EventHandler, revision as engineRevision, version as engineVersion } from 'playcanvas';
import type { Texture } from 'playcanvas';

import { version as appVersion } from '../package.json';
import { createApp, createViewerState, initCanvas, loadContent, loadSkybox } from './app-setup';
import { MeshCollision, loadVoxelCollision } from './collision';
import type { Collision } from './collision';
import { initEmbed } from './embed';
import { initLocalization } from './localization';
import { importSettings } from './settings';
import type { Config, Global } from './types';
import { initPoster, initUI } from './ui';
import { Viewer } from './viewer';
import { initXr } from './xr';

const main = async (canvas: HTMLCanvasElement, settingsJson: unknown, config: Config) => {
    const { app, camera, renderer } = await createApp(canvas, config);

    // create events and observable state
    const events = new EventHandler();

    // shared with the embed entry point — keep the defaults in one place so the two
    // bundles cannot drift (a missing field here reads as `undefined` at runtime)
    const state = createViewerState(events, config);

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

    // Initialize XR support (any backend; when the current device can't host a
    // session the UI can offer a reload into WebGL instead)
    initXr(global);

    // Initialize user interface
    initLocalization(config.lang);
    initUI(global);

    // Load model — loadContent picks the handler from the filename and
    // rejects an unrecognised extension without fetching the body
    const progressCallback = (progress: number) => {
        state.progress = progress;
    };
    const gsplatLoad = loadContent(app, config, global, progressCallback);

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
    if (global.settings.soundUrl) {
        const sound = new Audio(global.settings.soundUrl);
        sound.crossOrigin = 'anonymous';
        document.body.addEventListener(
            'click',
            () => {
                if (sound) {
                    sound.play();
                }
            },
            {
                capture: true,
                once: true
            }
        );
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
