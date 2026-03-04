import { GSplatResource } from 'playcanvas';

import type { Omg4Data } from '../parsers/omg4';
import type { Global } from '../types';

// Drives per-frame GPU texture updates for an OMG4 animated scene.
// After calling setFrame(), the GSplatResource's transform and colour textures
// are updated to reflect the requested frame.
class Omg4SplatAnimation {
    private data: Omg4Data;

    private resource: GSplatResource;

    private currentFrame: number = -1;

    constructor(data: Omg4Data, resource: GSplatResource) {
        this.data     = data;
        this.resource = resource;
    }

    get duration(): number {
        return this.data.duration;
    }

    get numFrames(): number {
        return this.data.numFrames;
    }

    // Map a playback time [0..duration] to a frame index.
    getFrameIndex(time: number): number {
        return this.data.getFrameIndex(time);
    }

    // Switch to the given frame. Returns true if the frame changed (GPU upload needed).
    setFrame(frameIndex: number): boolean {
        if (frameIndex === this.currentFrame) return false;
        this.currentFrame = frameIndex;

        this.data.loadFrame(frameIndex);
        this.resource.updateTransformData(this.data.gsplatData);
        this.resource.updateColorData(this.data.gsplatData);

        return true;
    }

    // Attach the frame-advance loop to the application update event.
    // Returns a cleanup function that removes the listener.
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

export { Omg4SplatAnimation };
