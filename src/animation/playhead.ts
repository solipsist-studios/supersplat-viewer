import type { State } from '../types';

// Shared transport playhead for animated-splat drivers. Owns the current
// playback time and the pingpong direction, advancing them according to the
// transport's speed and loop mode so every format — per-frame formats driven
// by SplatAnimationBase as well as the GPU-evaluated .omg4 v2 driver —
// responds identically to the UI controls.
class Playhead {
    time = 0;

    // pingpong playback direction: +1 forward, -1 on the return leg
    private direction = 1;

    // Advance by dt seconds. Returns true once a 'none' loop has run past the
    // end, signalling the caller to pause the transport.
    advance(dt: number, duration: number, state: State): boolean {
        this.time += dt * state.animationSpeed * this.direction;

        if (duration <= 0) {
            this.time = 0;
            return false;
        }

        // still inside the track — nothing to wrap
        if (this.time >= 0 && this.time <= duration) {
            return false;
        }

        switch (state.animationLoopMode) {
            case 'none':
                // play through once, then hold on the last frame
                this.time = duration;
                return true;
            case 'repeat':
                this.time = ((this.time % duration) + duration) % duration;
                return false;
            case 'pingpong':
                // bounce off whichever end was crossed
                if (this.time > duration) {
                    this.time = 2 * duration - this.time;
                    this.direction = -1;
                } else {
                    this.time = -this.time;
                    this.direction = 1;
                }
                return false;
        }

        return false;
    }

    // Jump to an absolute time, clamped to the track, resuming forward playback.
    seek(time: number, duration: number) {
        this.time = Math.max(0, Math.min(duration, time));
        this.direction = 1;
    }

    // Leaving pingpong mode resumes normal forward playback.
    resetDirection() {
        this.direction = 1;
    }
}

export { Playhead };
