import { Quat, type Entity } from 'playcanvas';

import { bindOmg4V2Modifier, setOmg4V2Params } from '../core/omg4-v2-motion';
import type { Omg4V2Data } from '../parsers/omg4';
import type { Global } from '../types';

// Animation driver for .omg4 v2 content. Unlike the per-frame formats there
// is nothing to upload per frame: playback is a pair of uniforms (time and
// entity rotation) evaluated by the GPU in the unified work-buffer pass, so
// time is continuous and never gated on fetches or texture uploads. Depth
// sorting picks up the motion-displaced centers automatically because the
// modifier runs before the work buffer is sorted.
class Omg4V2SplatAnimation {
    private data: Omg4V2Data;

    private entity: Entity | null = null;

    private lastTime = NaN;

    private lastRotation = new Quat(NaN, NaN, NaN, NaN);

    constructor(data: Omg4V2Data) {
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
    bind(entity: Entity) {
        this.entity = entity;
        bindOmg4V2Modifier(entity);
    }

    // Push uniforms if the time or entity rotation changed (each push marks
    // the work buffer render-dirty, so avoid redundant updates).
    private apply(animTime: number): boolean {
        if (!this.entity) {
            return false;
        }
        const rotation = this.entity.getRotation();
        if (animTime === this.lastTime && rotation.equals(this.lastRotation)) {
            return false;
        }
        this.lastTime = animTime;
        this.lastRotation.copy(rotation);
        setOmg4V2Params(this.entity, this.data.timeMin + animTime);
        return true;
    }

    attach(global: Global): () => void {
        const { app, state, events } = global;
        let animTime = 0;

        const onUpdate = (dt: number) => {
            if (!state.animationPaused) {
                animTime += dt;
                if (animTime > this.duration) {
                    animTime = this.duration > 0 ? animTime % this.duration : 0;
                }
                state.animationTime = animTime;
            } else {
                // Honour external scrubs that write to state.animationTime directly.
                animTime = state.animationTime;
            }
            if (this.apply(animTime)) {
                app.renderNextFrame = true;
            }
        };

        const onScrub = (time: number) => {
            animTime = Math.max(0, Math.min(this.duration, time));
            state.animationTime = animTime;
            if (this.apply(animTime)) {
                app.renderNextFrame = true;
            }
        };

        app.on('update', onUpdate);
        events.on('scrubAnim', onScrub);

        return () => {
            app.off('update', onUpdate);
            events.off('scrubAnim', onScrub);
        };
    }
}

export { Omg4V2SplatAnimation };
