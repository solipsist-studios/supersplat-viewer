import type { Global } from '../types';

import { Playhead } from './playhead';

// The transport of a clip that plays itself: this owns only the playhead (play, pause, loop mode, speed, scrubbing) and writes it
// to `state.animationTime`. Something else reads that time and moves the splats: for .asog, animated-sog's player, which is given
// `() => state.animationTime` as its clock. Nothing is uploaded or applied here.
class ClockAnimation {
    readonly duration: number;

    constructor(duration: number) {
        this.duration = duration;
    }

    attach(global: Global): () => void {
        const { app, state, events } = global;
        const playhead = new Playhead();

        const onUpdate = (dt: number) => {
            if (!state.animationPaused) {
                if (playhead.advance(dt, this.duration, state)) {
                    state.animationPaused = true;
                }
                state.animationTime = playhead.time;
            } else {
                // Honour external scrubs that write to state.animationTime directly.
                playhead.time = state.animationTime;
            }
        };

        const onScrub = (time: number) => {
            playhead.seek(time, this.duration);
            state.animationTime = playhead.time;
            app.renderNextFrame = true;
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

export { ClockAnimation };
