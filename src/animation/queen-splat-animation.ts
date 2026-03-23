import { GSplatResource } from 'playcanvas';

import { SplatAnimationBase } from './splat-animation-base';
import type { QueenData } from '../parsers/queen';

// Drives per-frame GPU texture updates for a QUEEN animated scene.
// After calling setFrame(), the GSplatResource's transform and colour textures
// are updated to reflect the requested frame.
class QueenSplatAnimation extends SplatAnimationBase {
    private data: QueenData;

    private resource: GSplatResource;

    private currentFrame: number = -1;

    constructor(data: QueenData, resource: GSplatResource) {
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
    setFrame(frameIndex: number): boolean {
        if (frameIndex === this.currentFrame) return false;
        this.currentFrame = frameIndex;

        this.data.loadFrame(frameIndex);
        this.resource.updateTransformData(this.data.gsplatData);
        this.resource.updateColorData(this.data.gsplatData);

        return true;
    }
}

export { QueenSplatAnimation };
