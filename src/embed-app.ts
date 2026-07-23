import { EventHandler } from 'playcanvas';

import { createApp, createViewerState, initCanvas, load3dgs, load4dgs } from './app-setup';
import { EmbedInputDevice } from './input/devices/external';
import { importSettings } from './settings';
import type { Config, Global, State } from './types';
import { Viewer } from './viewer';
import { initXr } from './xr';

// Embeddable library entry: creates a full viewer (engine, loaders, camera
// pipeline, XR) on a host-page canvas with NO HTML UI and no iframe. The
// host drives playback/input/XR through the returned handle and observes
// state via `state` + `events` ('<prop>:changed'). Because everything runs
// in the top-level document, XR sessions can start directly from the host's
// own click handlers (WebKit refuses immersive sessions in iframes).

type EmbedViewerOptions = {
    canvas: HTMLCanvasElement;
    /** URL of the scene file (.ply/.sog/.compressed.ply/.omg4/.queen/meta.json). */
    contentUrl: string;
    /** Original filename when contentUrl has no extension (e.g. blob URLs). */
    contentFilename?: string;
    /** Experience settings JSON (v1 or v2); defaults to a minimal setup. */
    settings?: object;
    /** Requested renderer; XR sessions require 'webgl'. Default 'webgpu'. */
    renderer?: 'webgl' | 'webgpu';
    /** Render with a transparent background for blending with the host page. */
    transparent?: boolean;
    /** Start with animation paused. */
    noanim?: boolean;
    /** OMG4 content rotation in degrees (default [270, 0, 0]). */
    omg4RotationDeg?: [number, number, number];
    /** 4DGS playback loop style (default 'loop'). */
    loopMode?: 'loop' | 'pingpong';
    /**
     * World-space position for the content entity. Splats are authored at
     * arbitrary origins/scales, so each needs its own adjustment — this also
     * determines where content sits relative to the floor in AR (y=0).
     */
    position?: [number, number, number];
    /** Scale for the content entity (real-world size in AR). */
    scale?: [number, number, number];
};

type XrMode = 'AR' | 'VR';

type EmbedViewer = {
    /** Live observed viewer state (loaded, progress, animation*, hasAR/VR…). */
    state: State;
    /** Fires '<prop>:changed' for every state change, plus 'xrError'. */
    events: EventHandler;
    /** The actual renderer in use after any engine fallback. */
    renderer: 'webgl' | 'webgpu';
    play: () => void;
    pause: () => void;
    restart: () => void;
    seek: (time: number) => void;
    resetCamera: () => void;
    /** Queue orbit/zoom pixel deltas (same semantics as the site's flipbook). */
    input: (deltas: { rotate?: [number, number]; zoom?: number }) => void;
    /** Whether an XR session of this mode can start right now. */
    canStartXr: (mode: XrMode) => boolean;
    /** Start an XR session. Call synchronously from a user-gesture handler. */
    startXr: (mode: XrMode) => boolean;
    destroy: () => void;
};

const defaultSettings = {
    version: 2,
    tonemapping: 'none',
    highPrecisionRendering: false,
    background: { color: [0, 0, 0] },
    postEffectSettings: {
        sharpness: { enabled: false, amount: 0 },
        bloom: { enabled: false, intensity: 1, blurLevel: 2 },
        grading: { enabled: false, brightness: 0, contrast: 1, saturation: 1, tint: [1, 1, 1] },
        vignette: { enabled: false, intensity: 0.5, inner: 0.3, outer: 0.75, curvature: 1 },
        fringing: { enabled: false, intensity: 0.5 }
    },
    animTracks: [] as unknown[],
    cameras: [] as unknown[],
    annotations: [] as unknown[],
    startMode: 'default'
};

const createEmbedViewer = async (options: EmbedViewerOptions): Promise<EmbedViewer> => {
    const config: Config = {
        contentUrl: options.contentUrl,
        contentFilename: options.contentFilename,
        contents: fetch(options.contentUrl),
        omg4RotationDeg: options.omg4RotationDeg,
        animLoopMode: options.loopMode,
        noui: true,
        noanim: !!options.noanim,
        embed: true,
        transparent: !!options.transparent,
        nofx: false,
        ministats: false,
        colorize: false,
        fullload: false,
        aa: false,
        renderer: options.renderer ?? 'webgpu',
        heatmap: false,
        debug: false
    };

    const { app, camera, renderer } = await createApp(options.canvas, config);

    const events = new EventHandler();
    const state = createViewerState(events);

    const global: Global = {
        app,
        settings: importSettings(options.settings ?? defaultSettings),
        config,
        state,
        events,
        camera,
        renderer
    };

    initCanvas(global);
    app.start();
    camera.addComponent('camera');
    initXr(global);

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

    const viewer = new Viewer(global, gsplatLoad, undefined, undefined);

    // Host-fed input flows through the shared controller pipeline; the
    // input controller exists once loading completes.
    const device = new EmbedInputDevice();
    events.once('firstFrame', () => {
        viewer.inputController?.extraDevices.push(device);
    });

    // Re-fire XR lifecycle on the shared event handler so hosts have a
    // single subscription point.
    app.xr?.on('error', (err: Error) => {
        events.fire('xrError', err);
    });
    app.xr?.on('start', () => events.fire('xrStart'));
    app.xr?.on('end', () => events.fire('xrEnd'));

    // Position/scale the content entity (splats are authored at arbitrary
    // origins and scales)
    if (options.position || options.scale) {
        const { position, scale } = options;
        gsplatLoad.then((entity) => {
            if (position) {
                entity.setLocalPosition(position[0], position[1], position[2]);
            }
            if (scale) {
                entity.setLocalScale(scale[0], scale[1], scale[2]);
            }
        }).catch(() => {});
    }

    // mirrors the play/pause buttons in ui.ts: 3D content switches to the
    // camera track, 4DGS content toggles file playback
    const setPaused = (paused: boolean) => {
        if (!state.hasAnimation) {
            state.cameraMode = 'anim';
        }
        state.animationPaused = paused;
    };

    const canStartXr = (mode: XrMode) => {
        // the XR start rig (xr.ts) only runs under the WebGL renderer
        const type = mode === 'AR' ? 'immersive-ar' : 'immersive-vr';
        return renderer === 'webgl' && (app.xr?.isAvailable(type) ?? false);
    };

    return {
        state,
        events,
        renderer,
        play: () => setPaused(false),
        pause: () => setPaused(true),
        restart: () => {
            events.fire('scrubAnim', 0);
            setPaused(false);
        },
        seek: (time: number) => {
            events.fire('scrubAnim', time);
        },
        resetCamera: () => {
            events.fire('inputEvent', 'reset');
        },
        input: (deltas: { rotate?: [number, number]; zoom?: number }) => {
            if (deltas.rotate) {
                device.queueRotate(deltas.rotate[0], deltas.rotate[1]);
            }
            if (typeof deltas.zoom === 'number') {
                device.queueZoom(deltas.zoom);
            }
            app.renderNextFrame = true;
        },
        canStartXr,
        startXr: (mode: XrMode) => {
            if (!canStartXr(mode)) {
                return false;
            }
            events.fire(mode === 'AR' ? 'startAR' : 'startVR');
            return true;
        },
        destroy: () => {
            app.destroy();
        }
    };
};

export { createEmbedViewer };
export type { EmbedViewer, EmbedViewerOptions, XrMode };
