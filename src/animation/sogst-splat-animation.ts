import { Quat, Vec3, type Entity } from 'playcanvas';

import { Playhead } from './playhead';
import type { SogstData } from '../core/load-sogst';
import { bindSogstModifier, setSogstParams } from '../core/sogst-motion';
import type { Global } from '../types';

// Animation driver for .sogst v2 content. Unlike the per-frame formats there
// is nothing to upload per frame: playback is a pair of uniforms (time and
// entity rotation) evaluated by the GPU in the unified work-buffer pass, so
// time is continuous and never gated on fetches or texture uploads. Depth
// sorting picks up the motion-displaced centers automatically because the
// modifier runs before the work buffer is sorted.
class SogstSplatAnimation {
    private data: SogstData;

    private entity: Entity | null = null;

    private camera: Entity | null = null;

    private cov2dScale: [number, number] | null = null;

    private lastTime = NaN;

    private lastRotation = new Quat(NaN, NaN, NaN, NaN);

    private lastCamRotation = new Quat(NaN, NaN, NaN, NaN);

    private lastCamPosition = new Vec3(NaN, NaN, NaN);

    constructor(data: SogstData) {
        this.data = data;
    }

    get duration(): number {
        return this.data.duration;
    }

    get numFrames(): number {
        return Math.max(1, Math.round(this.duration * (this.data.fps || 30)));
    }

    // Install the GPU modifier on the entity's gsplat component. Called once
    // the entity exists (after setupSplatAnim).
    bind(entity: Entity, cov2dScale: [number, number] | null = null) {
        this.entity = entity;
        this.cov2dScale = cov2dScale;
        bindSogstModifier(entity, cov2dScale, !!this.data.segments, !!this.data.accelX);
    }

    // Active splat-index bounds for segmented (v3) content at an absolute
    // clip time: persistent splats plus the contiguous run of segments
    // whose time coverage contains t. Null when the file has no segments.
    private cullRanges(absTime: number): [number, number, number] | null {
        const segments = this.data.segments;
        if (!segments) {
            return null;
        }
        let lo = -1;
        let hi = -1;
        for (const s of segments.list) {
            if (s.t0 <= absTime && absTime <= s.t1) {
                lo = lo < 0 ? s.range[0] : Math.min(lo, s.range[0]);
                hi = Math.max(hi, s.range[1]);
            }
        }
        if (lo < 0) {
            lo = 0;
            hi = 0;
        }
        return [segments.persistent[1], lo, hi];
    }

    // Push uniforms if the time, entity rotation or (when covariance
    // compensation is active) camera rotation changed - each push marks the
    // work buffer render-dirty, so avoid redundant updates.
    private apply(animTime: number): boolean {
        if (!this.entity) {
            return false;
        }
        const rotation = this.entity.getRotation();
        const camRotation = this.camera?.getRotation();
        const camPosition = this.camera?.getPosition();
        const camChanged = !!(this.cov2dScale && camRotation && camPosition &&
            (!camRotation.equals(this.lastCamRotation) || !camPosition.equals(this.lastCamPosition)));
        if (animTime === this.lastTime && rotation.equals(this.lastRotation) && !camChanged) {
            return false;
        }
        this.lastTime = animTime;
        this.lastRotation.copy(rotation);
        if (camRotation) {
            this.lastCamRotation.copy(camRotation);
        }
        if (camPosition) {
            this.lastCamPosition.copy(camPosition);
        }
        const absTime = this.data.timeMin + animTime;
        setSogstParams(this.entity, absTime, this.camera ?? undefined, this.cullRanges(absTime));
        return true;
    }

    attach(global: Global): () => void {
        const { app, state, events } = global;
        this.camera = global.camera;
        const playhead = new Playhead();

        const onUpdate = (dt: number) => {
            if (!state.animationPaused) {
                if (playhead.advance(dt, this.duration, state)) {
                    state.animationPaused = true;
                }
                // Streaming loads: hold the playhead at the last fully decoded
                // time until the next segment arrives. Clamp the playhead
                // itself, not just the reported time, so `apply()` below does
                // not push a time past the loaded region. `loadedThrough` is
                // Infinity for non-streaming sources, making this a no-op.
                const loadedLimit = this.data.loadedThrough - this.data.timeMin;
                if (playhead.time > loadedLimit) {
                    playhead.time = Math.max(0, loadedLimit);
                }
                state.animationTime = playhead.time;
            } else {
                // Honour external scrubs that write to state.animationTime directly.
                playhead.time = state.animationTime;
            }
            if (this.apply(playhead.time)) {
                app.renderNextFrame = true;
            }
        };

        const onScrub = (time: number) => {
            playhead.seek(time, this.duration);
            state.animationTime = playhead.time;
            if (this.apply(playhead.time)) {
                app.renderNextFrame = true;
            }
        };

        // Leaving pingpong mode resumes normal forward playback.
        const onLoopModeChanged = () => {
            playhead.resetDirection();
        };

        app.on('update', onUpdate);
        events.on('scrubAnim', onScrub);
        events.on('animationLoopMode:changed', onLoopModeChanged);

        return () => {
            app.off('update', onUpdate);
            events.off('scrubAnim', onScrub);
            events.off('animationLoopMode:changed', onLoopModeChanged);
        };
    }
}

export { SogstSplatAnimation };
