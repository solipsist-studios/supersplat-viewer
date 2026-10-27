import type { EventHandler } from 'playcanvas';

import { createViewer } from './index';
import type { CameraPose, CreateViewerOptions, ViewerHandle, ViewerState } from './index';

// Embeddable library entry: creates a full viewer (engine, loaders, camera pipeline, XR) inside a
// host-page element with NO built-in UI and no iframe. It is a thin adapter over `createViewer`,
// keeping the shape hosts such as GaussPlay already use (play, pause, seek, input, XR) while the
// viewer itself is upstream's. The host drives playback, input and XR through the returned handle
// and observes state via `state` + `events` ('<prop>:changed'). Because everything runs in the
// top-level document, XR sessions can start directly from the host's own click handlers (WebKit
// refuses immersive sessions in iframes).

type EmbedViewerOptions = {
    /**
     * The element the viewer builds its subtree in (a canvas, plus an empty wrapper). The host
     * sizes it; the viewer fills it.
     */
    container?: HTMLElement;
    /**
     * @deprecated Pass `container`. The viewer creates its own canvas, so a host canvas cannot be
     * rendered into: given instead of `container`, the viewer builds in this canvas's parent and
     * hides the canvas, restoring its display when the viewer is destroyed.
     */
    canvas?: HTMLCanvasElement;
    /** URL of the scene file (.ply/.sog/.compressed.ply/.sogst/meta.json). */
    contentUrl: string;
    /** Original filename when contentUrl has no extension (e.g. blob URLs). */
    contentFilename?: string;
    /** Experience settings JSON (v1 or v2); defaults to a minimal setup. */
    settings?: object;
    /**
     * Requested renderer; XR sessions require 'webgl'. Unset, the viewer
     * picks: WebGPU, except WebGL for .sogst clips on a Mac or Vision Pro.
     */
    renderer?: 'webgl' | 'webgpu';
    /**
     * Antialias the splats (the GSPLAT_AA shader path). Off by default. The
     * viewer fixes this at load time, so a later change has no effect.
     */
    aa?: boolean;
    /** Render with a transparent background for blending with the host page. */
    transparent?: boolean;
    /** Start with animation paused. */
    noanim?: boolean;
    /** SOGST content rotation in degrees (default [0, 0, 0]). */
    sogstRotationDeg?: [number, number, number];
    /** 4DGS playback loop style (default 'loop'). */
    loopMode?: 'loop' | 'pingpong';
    /**
     * Splat budget for .sogst content, in millions of splats. A segmented
     * clip larger than this plays through a segment window: only the
     * segments near the playhead stay decoded, and the rest are re-read from
     * the IndexedDB cache (or the network) as playback reaches them. Unset,
     * the viewer picks a budget from the device type and the clip's cost per
     * splat; 0 plays every clip fully resident.
     */
    sogstBudget?: number;
    /**
     * Decode .sogst content on the GPU (WebGL2 or WebGPU). Default true;
     * false forces the CPU pack workers.
     */
    sogstGpuDecode?: boolean;
    /**
     * World-space position for the content entity. Splats are authored at
     * arbitrary origins/scales, so each needs its own adjustment — this also
     * determines where content sits relative to the floor in AR (y=0).
     */
    position?: [number, number, number];
    /**
     * Scale for the content entity (real-world size in AR). Uniform scales
     * are exact; .sogst motion follows the world transform's lossy scale.
     */
    scale?: [number, number, number];
};

type XrMode = 'AR' | 'VR';

type EmbedViewer = {
    /** Live observed viewer state (loaded, progress, animation*, hasAR/VR…). */
    state: ViewerState;
    /** Fires '<prop>:changed' for every state change, plus 'xrError', 'xrStart' and 'xrEnd'. */
    events: EventHandler;
    /** The actual renderer in use after any engine fallback. */
    renderer: 'webgl' | 'webgpu';
    /** The underlying `createViewer` handle, for everything this adapter does not wrap. */
    handle: ViewerHandle;
    play: () => void;
    pause: () => void;
    restart: () => void;
    seek: (time: number) => void;
    resetCamera: () => void;
    /**
     * Read the live camera pose, or null before the camera exists (it is built once the scene
     * finishes loading). This is an authoring aid: a host can orbit to the framing it wants and
     * read out the numbers to bake in, instead of guessing settings values and reloading.
     */
    getCameraPose: () => CameraPose | null;
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

// The layout the viewer's stylesheet gives its root and canvas, which an embed does not load: the
// root fills its container and the canvas fills the root. Injected once, scoped to the viewer's own
// class, so a host page's own styles are untouched.
const injectLayoutStyle = () => {
    if (document.querySelector('style[data-sse-embed]')) {
        return;
    }
    const style = document.createElement('style');
    style.dataset.sseEmbed = '';
    style.textContent =
        '.sse-viewer{position:relative;width:100%;height:100%;overflow:hidden;isolation:isolate;touch-action:none}' +
        '.sse-viewer>canvas{position:absolute;top:0;left:0;width:100%;height:100%}';
    document.head.appendChild(style);
};

const createEmbedViewer = async (options: EmbedViewerOptions): Promise<EmbedViewer> => {
    const { canvas } = options;
    const container = options.container ?? canvas?.parentElement;
    if (!container) {
        throw new Error('createEmbedViewer: pass a container element');
    }

    // a host canvas cannot be rendered into (the viewer creates its own), so step it aside
    const canvasDisplay = canvas?.style.display;
    if (!options.container && canvas) {
        canvas.style.display = 'none';
    }
    injectLayoutStyle();

    const viewerOptions: CreateViewerOptions = {
        container,
        settings: options.settings ?? defaultSettings,
        contentUrl: options.contentUrl,
        contentFilename: options.contentFilename,
        renderer: options.renderer,
        aa: options.aa,
        transparent: options.transparent,
        noanim: options.noanim,
        sogstRotationDeg: options.sogstRotationDeg,
        animLoopMode: options.loopMode,
        sogstBudget: options.sogstBudget,
        sogstGpuDecode: options.sogstGpuDecode,
        position: options.position,
        scale: options.scale,
        // the host owns the page, the controls and all input
        ui: false,
        hostInput: true
    };

    let handle: ViewerHandle;
    try {
        handle = await createViewer(viewerOptions);
    } catch (err) {
        if (!options.container && canvas) {
            canvas.style.display = canvasDisplay;
        }
        throw err;
    }
    const { app, state, events } = handle;
    const renderer = app.graphicsDevice.deviceType === 'webgpu' ? 'webgpu' : 'webgl';

    // Re-fire XR lifecycle on the shared event handler so hosts have a single subscription point.
    app.xr?.on('error', (err: Error) => {
        events.fire('xrError', err);
    });
    app.xr?.on('start', () => events.fire('xrStart'));
    app.xr?.on('end', () => events.fire('xrEnd'));

    // the handle's commands throw before the first frame; these have never thrown, since a host
    // calls them from its own controls whenever they are pressed
    const whenLoaded = (run: () => void) => {
        if (state.loaded) {
            run();
        }
    };

    // mirrors the play/pause buttons of the built-in ui: 3D content switches to the camera
    // track, 4DGS content toggles file playback
    const setPaused = (paused: boolean) => {
        if (!state.has4DAnimation) {
            state.cameraMode = 'anim';
        }
        state.animationPaused = paused;
    };

    const canStartXr = (mode: XrMode) => (mode === 'AR' ? state.canStartAR : state.canStartVR);

    return {
        state,
        events,
        renderer,
        handle,
        play: () => setPaused(false),
        pause: () => setPaused(true),
        restart: () => {
            whenLoaded(() => handle.seek(0));
            setPaused(false);
        },
        seek: (time: number) => whenLoaded(() => state.hasAnimation && handle.seek(time)),
        resetCamera: () => whenLoaded(() => handle.resetCamera()),
        getCameraPose: () => handle.getCameraPose(),
        input: (deltas) => handle.input(deltas),
        canStartXr,
        startXr: (mode: XrMode) => {
            if (!canStartXr(mode)) {
                return false;
            }
            handle.startXR(mode === 'AR' ? 'ar' : 'vr').catch((err: Error) => events.fire('xrError', err));
            return true;
        },
        destroy: () => {
            handle.destroy();
            if (!options.container && canvas) {
                canvas.style.display = canvasDisplay;
            }
        }
    };
};

export { createEmbedViewer };
export type { CameraPose, EmbedViewer, EmbedViewerOptions, XrMode };
