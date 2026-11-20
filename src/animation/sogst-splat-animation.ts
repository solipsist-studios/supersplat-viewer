import { Quat, Vec3 } from 'playcanvas';
import type { Entity } from 'playcanvas';

import type { SogstData } from '../core/sogst-data';
import { bindSogstModifier, setSogstCullPages, setSogstParams } from '../core/sogst-motion';
import { PAGE_SPLATS } from '../core/sogst-window';
import type { Global } from '../types';

import { Playhead } from './playhead';

// Animation driver for .sogst content. This driver uploads nothing per
// frame, which the per-frame formats must do.
//
// Playback is a pair of uniforms, the time and the entity rotation, and the
// GPU evaluates them in the unified work-buffer pass. Time is therefore
// continuous, and it never waits for a fetch or a texture upload. Depth
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

    // Windowed clips: the active pages last pushed to the shader, as a
    // comparable key.
    private lastPagesKey = '';

    // Set when the segment window's resident set changed, which can change
    // the active pages without the time changing (an install while paused).
    private cullDirty = true;

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
        bindSogstModifier(entity, cov2dScale, !!this.data.segments, this.data.hasAccel);
    }

    // Active splat-index bounds for segmented content at an absolute clip
    // time: persistent splats plus the contiguous run of segments whose
    // time coverage contains t. Null when the file has no segment table.
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

    // Push the uniforms when the time or the entity rotation changed. Also
    // push them when the camera rotation changed and covariance compensation
    // is active. Each push marks the work buffer render-dirty, so do not push
    // the same values twice.
    private apply(animTime: number): boolean {
        if (!this.entity) {
            return false;
        }
        const rotation = this.entity.getRotation();
        const camRotation = this.camera?.getRotation();
        const camPosition = this.camera?.getPosition();
        const camChanged = !!(
            this.cov2dScale &&
            camRotation &&
            camPosition &&
            (!camRotation.equals(this.lastCamRotation) || !camPosition.equals(this.lastCamPosition))
        );
        if (animTime === this.lastTime && rotation.equals(this.lastRotation) && !camChanged && !this.cullDirty) {
            return false;
        }
        this.cullDirty = false;
        this.lastTime = animTime;
        this.lastRotation.copy(rotation);
        if (camRotation) {
            this.lastCamRotation.copy(camRotation);
        }
        if (camPosition) {
            this.lastCamPosition.copy(camPosition);
        }
        const absTime = this.data.timeMin + animTime;
        const segmentWindow = this.data.window;
        if (segmentWindow) {
            // windowed: cull by arena page, uploading the mask only when the
            // active pages change
            setSogstParams(this.entity, absTime, this.camera ?? undefined);
            const pages = segmentWindow.activePages(absTime);
            const key = pages.join(',');
            if (key !== this.lastPagesKey) {
                this.lastPagesKey = key;
                setSogstCullPages(this.entity, this.data.resource, segmentWindow.persistentEnd, PAGE_SPLATS, pages);
            }
        } else {
            setSogstParams(this.entity, absTime, this.camera ?? undefined, this.cullRanges(absTime));
        }
        return true;
    }

    attach(global: Global): () => void {
        const { app, state, events } = global;
        this.camera = global.camera;
        const playhead = new Playhead();
        const segmentWindow = this.data.window;
        segmentWindow?.onChange(() => {
            this.cullDirty = true;
            app.renderNextFrame = true;
        });

        const onUpdate = (dt: number) => {
            if (!state.animationPaused) {
                const before = playhead.save();
                if (playhead.advance(dt, this.duration, state)) {
                    state.animationPaused = true;
                }
                // Windowed clips: hold the playhead where it was while the
                // segments at the new time are still being installed. This
                // covers both pingpong directions, since the window answers
                // for any time.
                if (segmentWindow && !segmentWindow.isReady(this.data.timeMin + playhead.time)) {
                    playhead.restore(before);
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
            segmentWindow?.setPlayback({
                time: this.data.timeMin + playhead.time,
                direction: playhead.playDirection,
                speed: state.animationSpeed ?? 1,
                mode: state.animationLoopMode
            });
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
