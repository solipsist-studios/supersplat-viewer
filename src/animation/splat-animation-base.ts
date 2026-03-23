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
    abstract setFrame(frameIndex: number): boolean;

    // Attach the frame-advance loop to the application update event.
    // Returns a cleanup function that removes the listeners.
    attach(global: Global): () => void {
        const { app, state, events, camera } = global;
        let animTime = 0;

        const sortAndRender = () => {
            const instance = (app.root.findComponents('gsplat') as any[])[0]?.instance;
            if (instance) {
                instance.sort(camera);
            }
            app.renderNextFrame = true;
        };

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

            const frameIdx = this.getFrameIndex(animTime);
            if (this.setFrame(frameIdx)) {
                sortAndRender();
            }
        };

        // Handle timeline scrubbing from the UI.
        const onScrub = (time: number) => {
            animTime = Math.max(0, Math.min(this.duration, time));
            state.animationTime = animTime;

            const frameIdx = this.getFrameIndex(animTime);
            if (this.setFrame(frameIdx)) {
                sortAndRender();
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

export { SplatAnimationBase };
