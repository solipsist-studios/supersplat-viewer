/* eslint-disable no-unused-vars */
interface Window {
    sse: {
        poster?: HTMLImageElement,
        settings: Promise<object>,
        contentUrl: string,
        contents: ArrayBuffer,
        params: Record<string, string>
    }

    firstFrame?: () => void;

    app?: import('playcanvas').AppBase;

    scrubTo?: (time: number) => Promise<void>;

    captureFrame?: (options?: { time?: number; width?: number; height?: number; supersample?: number }) => Promise<{ width: number; height: number; data: string }>;

    animationDuration?: number;

    // Embed mode: start an XR session. Exposed as a window function so a
    // same-origin host can call it SYNCHRONOUSLY from its own click handler —
    // WebKit tracks user activation per call stack, so a postMessage task hop
    // would lose the gesture and requestSession would be rejected.
    startXr?: (mode: 'AR' | 'VR') => void;

    getCameraState?: () => {
        position: [number, number, number];
        angles: [number, number, number];
        distance: number;
        fov: number;
        mode: 'orbit' | 'anim' | 'fly' | 'walk';
    };

    setCameraState?: (snapshot: {
        position: [number, number, number];
        angles: [number, number, number];
        distance: number;
        fov: number;
        mode: 'orbit' | 'anim' | 'fly' | 'walk';
    }) => void;
}

declare module 'playcanvas/scripts/esm/xr/xr-controllers.mjs' {
    const XrControllers: typeof pc.Script;
    export { XrControllers };
}

declare module 'playcanvas/scripts/esm/xr/xr-navigation.mjs' {
    const XrNavigation: typeof pc.Script;
    export { XrNavigation };
}

declare module '*.html' {
    const content: string;
    export default content;
}

declare module '*.css' {
    const content: string;
    export default content;
}

declare module '*.js' {
    const content: string;
    export default content;
}

// Engine internals the .sogst sorter (src/core/sogst-sorter.ts) builds its
// worker from. Not part of the engine's public API: they are imported by
// path from the package's build tree, and only their source text and the
// bin-weight helper the worker needs are used.
declare module 'playcanvas/build/playcanvas/src/scene/gsplat-unified/gsplat-unified-sort-worker.js' {
    export function UnifiedSortWorker(): void;
}

declare module 'playcanvas/build/playcanvas/src/scene/gsplat-unified/gsplat-sort-bin-weights.js' {
    export class GSplatSortBinWeights {}
}
