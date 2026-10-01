import type { GSplatResource } from 'playcanvas';

import type { SogstMeta, SogstSegments } from '../parsers/sogst';

// A playable .sogst clip: its manifest, timing and segment table, and the
// GSplatResource the decoder packs splats into. The per-splat attributes live
// only in that resource's textures (and the motion textures attached to it);
// nothing keeps a whole-clip CPU copy of them.
//
// `SogstDecoder.buildData()` produces one. app-setup.ts puts the resource on
// an entity, and sogst-splat-animation.ts drives playback from it.
class SogstData {
    // Temporal segment table for per-segment culling (see parsers/sogst.ts).
    segments?: SogstSegments;

    // Highest absolute clip time the decoder has finished. A streaming load
    // advances this once per segment, and the animation driver holds the
    // playhead at it. It becomes Infinity when everything is loaded.
    loadedThrough = Infinity;

    readonly meta: SogstMeta;

    readonly numSplats: number;

    readonly timeMin: number;

    readonly timeMax: number;

    readonly fps: number;

    cov2dScale: [number, number] | null = null;

    readonly resource: GSplatResource;

    // Degree-2 motion present (meta.accel), which selects the shader variant.
    readonly hasAccel: boolean;

    constructor(meta: SogstMeta, resource: GSplatResource) {
        this.meta = meta;
        this.numSplats = meta.count;
        this.timeMin = meta.time?.min ?? 0;
        this.timeMax = meta.time?.max ?? 0;
        this.fps = meta.time?.fps ?? 30;
        this.cov2dScale = meta.cov2d_scale ? [meta.cov2d_scale[0], meta.cov2d_scale[1]] : null;
        if (meta.segments?.list?.length && meta.segments.persistent) {
            this.segments = meta.segments as SogstSegments;
        }
        this.resource = resource;
        this.hasAccel = !!meta.accel;
    }

    get duration(): number {
        return Math.max(0, this.timeMax - this.timeMin);
    }
}

export { SogstData };
