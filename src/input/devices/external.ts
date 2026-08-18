import { Vec3 } from 'playcanvas';

import { DISPLACEMENT_SCALE, flipZForOrbit } from '../shared';
import type { CameraInputFrame, InputDevice, UpdateContext } from '../shared';

const tmpV1 = new Vec3();

/**
 * Input device fed by a host page (via the postMessage bridge or the
 * embeddable library API). Accumulated pixel deltas are flushed into the
 * shared InputFrame each update, so external input flows through the same
 * controller pipeline (damping, clamping, mode handling) as native
 * mouse/touch input.
 */
class EmbedInputDevice implements InputDevice {
    /** Orbit rotation, in degrees per pixel of drag. */
    rotateSpeed = 0.3;

    /** Zoom, applied like wheel deltas (drag down = zoom out). */
    zoomSpeed = 0.4;

    private _rotate = { x: 0, y: 0 };

    private _zoom = 0;

    queueRotate(dx: number, dy: number) {
        this._rotate.x += dx;
        this._rotate.y += dy;
    }

    queueZoom(dz: number) {
        this._zoom += dz;
    }

    attach(): void {}

    detach(): void {}

    update(ctx: UpdateContext, frame: CameraInputFrame): void {
        const { deltas } = frame;

        deltas.rotate.append([this._rotate.x * this.rotateSpeed, this._rotate.y * this.rotateSpeed, 0]);

        const v = tmpV1.set(0, 0, -this._zoom * this.zoomSpeed * DISPLACEMENT_SCALE);
        deltas.move.append([v.x, v.y, flipZForOrbit(ctx.mode, v.z)]);

        this._rotate.x = 0;
        this._rotate.y = 0;
        this._zoom = 0;
    }
}

export { EmbedInputDevice };
