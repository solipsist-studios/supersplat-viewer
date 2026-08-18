import type { GSplatData } from 'playcanvas';

import type { SogstMeta, SogstSegments } from '../parsers/sogst';

// A decoded, playable .sogst clip: the engine's own GSplatData for the
// static attributes, plus the per-splat temporal attributes that have no
// engine equivalent (velocity, optional acceleration, and the radial-basis
// window's centre and sigma). `SogstDecoder.buildData()` produces one, and
// everything downstream — GSplatResource creation in app-setup.ts, the GPU
// modifier in sogst-motion.ts, the driver in sogst-splat-animation.ts —
// consumes this shape.
class SogstData {
    // Temporal segment table for per-segment culling (see parsers/sogst.ts).
    segments?: SogstSegments;

    // Highest absolute clip time that is fully decoded — streaming loads
    // advance this per segment and the animation driver holds the playhead
    // at it. Infinity once (or when) everything is loaded.
    loadedThrough = Infinity;

    readonly meta: SogstMeta;

    readonly numSplats: number;

    readonly timeMin: number;

    readonly timeMax: number;

    readonly fps: number;

    cov2dScale: [number, number] | null = null;

    readonly gsplatData: GSplatData;

    readonly velocityX: Float32Array;

    readonly velocityY: Float32Array;

    readonly velocityZ: Float32Array;

    readonly tCenter: Float32Array;

    readonly tSigma: Float32Array;

    // Degree-2 motion: quadratic coefficient arrays (units/sec^2), or null
    // on degree-1 content. These are the raw dt^2 coefficients, not
    // half-acceleration (see parsers/sogst.ts).
    accelX: Float32Array | null = null;

    accelY: Float32Array | null = null;

    accelZ: Float32Array | null = null;

    constructor(
        meta: SogstMeta,
        gsplatData: GSplatData,
        velocity: [Float32Array, Float32Array, Float32Array],
        tCenter: Float32Array,
        tSigma: Float32Array,
        accel: [Float32Array, Float32Array, Float32Array] | null = null
    ) {
        this.meta = meta;
        this.numSplats = meta.count;
        this.timeMin = meta.time?.min ?? 0;
        this.timeMax = meta.time?.max ?? 0;
        this.fps = meta.time?.fps ?? 30;
        this.cov2dScale = meta.cov2d_scale ? [meta.cov2d_scale[0], meta.cov2d_scale[1]] : null;
        if (meta.segments?.list?.length && meta.segments.persistent) {
            this.segments = meta.segments as SogstSegments;
        }
        this.gsplatData = gsplatData;
        [this.velocityX, this.velocityY, this.velocityZ] = velocity;
        this.tCenter = tCenter;
        this.tSigma = tSigma;
        if (accel) {
            [this.accelX, this.accelY, this.accelZ] = accel;
        }
    }

    get duration(): number {
        return Math.max(0, this.timeMax - this.timeMin);
    }
}

export { SogstData };
