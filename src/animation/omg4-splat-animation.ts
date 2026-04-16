import { GSplatResource } from 'playcanvas';

import { SplatAnimationBase } from './splat-animation-base';
import type { Omg4FrameData } from '../parsers/omg4';

// Drives per-frame GPU texture updates for an OMG4 animated scene.
// After calling setFrame(), the GSplatResource's transform and colour textures
// are updated to reflect the requested frame.
class Omg4SplatAnimation extends SplatAnimationBase {
    private data: Omg4FrameData;

    private resource: GSplatResource;

    private currentFrame: number = -1;

    constructor(data: Omg4FrameData, resource: GSplatResource) {
        super();
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
    async setFrame(frameIndex: number): Promise<boolean> {
        if (frameIndex === this.currentFrame) return false;

        try {
            await this.data.loadFrame(frameIndex);
        } catch (err) {
            // Transient cache/prefetch miss: keep current frame and try again next tick.
            return false;
        }
        this.currentFrame = frameIndex;
        this.resource.updateTransformData(this.data.gsplatData);
        this.resource.updateColorData(this.data.gsplatData);
        this.data.prefetchFrame?.(Math.min(this.numFrames - 1, frameIndex + 1));
        this.data.prefetchFrame?.(Math.min(this.numFrames - 1, frameIndex + 2));
        this.data.prefetchFrame?.(Math.min(this.numFrames - 1, frameIndex + 3));
        this.data.prefetchFrame?.(Math.min(this.numFrames - 1, frameIndex + 4));

        return true;
    }
}

export { Omg4SplatAnimation };
