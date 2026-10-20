import type { State } from '../types';

// Shared transport playhead for animated-splat drivers. It owns the current
// playback time and the pingpong direction, and it advances both from the
// transport's speed and loop mode.
//
// Two kinds of driver share it: the per-frame formats that SplatAnimationBase
// drives, and the GPU-evaluated .sogst driver. Both therefore respond to the
// UI controls in the same way.
class Playhead {
    time = 0;

    // pingpong playback direction: +1 forward, -1 on the return leg
    private direction = 1;

    // Advance by dt seconds. Returns true once a 'none' loop has run past the
    // end, signalling the caller to pause the transport.
    advance(dt: number, duration: number, state: State): boolean {
        // `?? 1` guards against a host state object that has no transport
        // fields. A NaN here would corrupt `time` permanently, because NaN
        // fails every range test below and fires observe() without end.
        this.time += dt * (state.animationSpeed ?? 1) * this.direction;

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

    // pingpong playback direction, for planning which segments come next
    get playDirection(): 1 | -1 {
        return this.direction < 0 ? -1 : 1;
    }

    // Snapshot and restore of the whole playhead, so a driver can undo an
    // advance into content that is not loaded yet (direction included: an
    // advance can bounce a pingpong).
    save(): { time: number; direction: number } {
        return { time: this.time, direction: this.direction };
    }

    restore(saved: { time: number; direction: number }) {
        this.time = saved.time;
        this.direction = saved.direction;
    }
}

export { Playhead };
