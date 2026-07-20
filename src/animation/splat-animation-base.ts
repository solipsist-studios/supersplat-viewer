import { Playhead } from './playhead';
import type { Global } from '../types';

// Abstract base for all animated Gaussian-splat drivers.
//
// Subclasses supply format-specific frame access (duration, numFrames,
// getFrameIndex, setFrame). This class owns the shared update/scrub
// event-loop that is identical for every format.
abstract class SplatAnimationBase {
    abstract get duration(): number;

    abstract get numFrames(): number;

    // Map a playback time [0..duration] to a frame index.
    abstract getFrameIndex(time: number): number;

    // Switch to the given frame. Returns true if the frame changed (GPU upload needed).
    abstract setFrame(frameIndex: number): boolean | Promise<boolean>;

    // Attach the frame-advance loop to the application update event.
    // Returns a cleanup function that removes the listeners.
    attach(global: Global): () => void {
        const { app, state, events, camera } = global;
        const playhead = new Playhead();
        let destroyed = false;
        let applyingFrame = false;
        let queuedFrame: number | null = null;

        const sortAndRender = () => {
            const instance = (app.root.findComponents('gsplat') as any[])[0]?.instance;
            if (instance) {
                instance.sort(camera);
            }
            app.renderNextFrame = true;
        };

        const applyQueuedFrame = async () => {
            if (applyingFrame || destroyed) {
                return;
            }

            applyingFrame = true;
            try {
                while (queuedFrame !== null && !destroyed) {
                    const frameIdx = queuedFrame;
                    queuedFrame = null;
                    if (await this.setFrame(frameIdx)) {
                        sortAndRender();
                    }
                }
            } catch (err) {
                console.error('Failed to update animation frame', err);
            } finally {
                applyingFrame = false;
            }
        };

        const requestFrame = (frameIdx: number) => {
            queuedFrame = frameIdx;
            void applyQueuedFrame();
        };

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

            const frameIdx = this.getFrameIndex(playhead.time);
            requestFrame(frameIdx);
        };

        // Handle timeline scrubbing from the UI.
        const onScrub = (time: number) => {
            playhead.seek(time, this.duration);
            state.animationTime = playhead.time;

            const frameIdx = this.getFrameIndex(playhead.time);
            requestFrame(frameIdx);
        };

        // Leaving pingpong mode resumes normal forward playback.
        const onLoopModeChanged = () => {
            playhead.resetDirection();
        };

        app.on('update', onUpdate);
        events.on('scrubAnim', onScrub);
        events.on('animationLoopMode:changed', onLoopModeChanged);

        return () => {
            destroyed = true;
            app.off('update', onUpdate);
            events.off('scrubAnim', onScrub);
            events.off('animationLoopMode:changed', onLoopModeChanged);
        };
    }
}

export { SplatAnimationBase };
