import { Vec3 } from 'playcanvas';

import { DISPLACEMENT_SCALE, flipZForOrbit } from './input/shared';
import type { CameraInputFrame, InputDevice, UpdateContext } from './input/shared';
import type { Global } from './types';
import type { Viewer } from './viewer';

const tmpV1 = new Vec3();

/**
 * Input device fed by the host page via postMessage. Accumulated pixel
 * deltas are flushed into the shared InputFrame each update, so external
 * input flows through the same controller pipeline (damping, clamping,
 * mode handling) as native mouse/touch input.
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

        deltas.rotate.append([
            this._rotate.x * this.rotateSpeed,
            this._rotate.y * this.rotateSpeed,
            0
        ]);

        const v = tmpV1.set(0, 0, -this._zoom * this.zoomSpeed * DISPLACEMENT_SCALE);
        deltas.move.append([v.x, v.y, flipZForOrbit(ctx.mode, v.z)]);

        this._rotate.x = 0;
        this._rotate.y = 0;
        this._zoom = 0;
    }
}

/**
 * postMessage bridge for embedding the viewer in a host page (enabled with
 * the `embed` URL param). The host drives input and transport:
 *
 *   { type: 'ssv:play' | 'ssv:pause' | 'ssv:restart' | 'ssv:cameraReset' }
 *   { type: 'ssv:seek', time: number }
 *   { type: 'ssv:input', rotate?: [dx, dy], zoom?: dz }   // pixel deltas
 *   { type: 'ssv:startAr' | 'ssv:startVr' }
 *
 * and receives state snapshots (sent on change, time throttled to ~10Hz):
 *
 *   { type: 'ssv:state', loaded, hasAnimation, duration, time, paused, progress,
 *     hasAR, hasVR, arDirect, vrDirect }
 *
 * plus XR notifications:
 *
 *   { type: 'ssv:xrState', active }          // session started/ended
 *   { type: 'ssv:xrNeedsWebgl', mode }       // session needs a reload with ?webgl
 * @param global - The global app context.
 * @param viewer - The viewer instance (provides the input controller once loaded).
 */
const initEmbed = (global: Global, viewer: Viewer) => {
    const { events, state } = global;

    if (window.parent === window) {
        return;
    }

    const device = new EmbedInputDevice();

    // the input controller is created once loading completes
    events.once('firstFrame', () => {
        viewer.inputController?.extraDevices.push(device);
    });

    const send = (data: object) => {
        window.parent.postMessage(data, location.origin);
    };

    const sendState = () => {
        send({
            type: 'ssv:state',
            loaded: state.loaded,
            hasAnimation: state.hasAnimation,
            duration: state.animationDuration,
            time: state.animationTime,
            paused: state.animationPaused,
            progress: state.progress,
            // hasAR/hasVR include sessions that would work after a reload into
            // WebGL; arDirect/vrDirect can start on the current device now.
            hasAR: state.hasAR,
            hasVR: state.hasVR,
            arDirect: global.app.xr?.isAvailable('immersive-ar') ?? false,
            vrDirect: global.app.xr?.isAvailable('immersive-vr') ?? false
        });
    };

    events.on('firstFrame', sendState);
    ['hasAnimation', 'animationDuration', 'animationPaused', 'progress', 'hasAR', 'hasVR'].forEach((prop) => {
        events.on(`${prop}:changed`, sendState);
    });
    global.app.xr?.on('available', sendState);
    global.app.xr?.on('start', () => send({ type: 'ssv:xrState', active: true }));
    global.app.xr?.on('end', () => send({ type: 'ssv:xrState', active: false }));

    // mirrors handleXrClick in ui.ts: start directly when the current device
    // supports the session, otherwise ask the host to reload us with ?webgl
    const startXr = (mode: 'AR' | 'VR') => {
        if (global.app.xr?.isAvailable(mode === 'AR' ? 'immersive-ar' : 'immersive-vr')) {
            events.fire(mode === 'AR' ? 'startAR' : 'startVR');
        } else {
            send({ type: 'ssv:xrNeedsWebgl', mode });
        }
    };

    let lastTimeSent = 0;
    events.on('animationTime:changed', () => {
        const now = performance.now();
        if (now - lastTimeSent > 100) {
            lastTimeSent = now;
            sendState();
        }
    });

    // mirrors the play/pause buttons in ui.ts: 3D content switches to the
    // camera track, 4DGS content toggles file playback
    const setPaused = (paused: boolean) => {
        if (!state.hasAnimation) {
            state.cameraMode = 'anim';
        }
        state.animationPaused = paused;
    };

    window.addEventListener('message', (event: MessageEvent) => {
        if (event.origin !== location.origin || !event.data || typeof event.data.type !== 'string') {
            return;
        }

        switch (event.data.type) {
            case 'ssv:play':
                setPaused(false);
                break;
            case 'ssv:pause':
                setPaused(true);
                break;
            case 'ssv:restart':
                events.fire('scrubAnim', 0);
                setPaused(false);
                break;
            case 'ssv:seek':
                if (typeof event.data.time === 'number') {
                    events.fire('scrubAnim', event.data.time);
                }
                break;
            case 'ssv:cameraReset':
                events.fire('inputEvent', 'reset');
                break;
            case 'ssv:startAr':
                startXr('AR');
                break;
            case 'ssv:startVr':
                startXr('VR');
                break;
            case 'ssv:input':
                if (Array.isArray(event.data.rotate)) {
                    device.queueRotate(event.data.rotate[0], event.data.rotate[1]);
                }
                if (typeof event.data.zoom === 'number') {
                    device.queueZoom(event.data.zoom);
                }
                global.app.renderNextFrame = true;
                break;
        }
    });

    send({ type: 'ssv:ready' });
};

export { initEmbed };
