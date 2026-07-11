import { Entity, GSplatResource, type AppBase } from 'playcanvas';

import type { Config, Global } from '../types';

type SetupSplatAnimOptions = {
    rotationEulerDeg?: [number, number, number];
    alphaClip?: number;
};

// Structural interface satisfied by SplatAnimationBase subclasses and by
// Omg4V2SplatAnimation (which drives a GPU time uniform instead of frames).
interface SplatAnimation {
    readonly duration: number;
    attach(global: Global): () => void;
}

// Shared entity-creation and state-wiring used by every animated-splat loader.
//
// The caller is responsible for:
//   1. Fetching the binary data.
//   2. Parsing the format-specific data and loading frame 0.
//   3. Creating a GSplatResource from the parsed GSplatData.
//   4. Constructing the format-specific SplatAnimationBase subclass.
//
// This function handles everything after that: entity setup, scene attachment,
// animation event-loop wiring, and firstFrame state propagation.
const setupSplatAnim = (
    app: AppBase,
    config: Config,
    global: Global,
    resource: GSplatResource,
    animation: SplatAnimation,
    options: SetupSplatAnimOptions = {}
): Entity => {
    const rotationEulerDeg = options.rotationEulerDeg ?? [0, 0, 180];
    const alphaClip = options.alphaClip ?? (1 / 255);

    const entity = new Entity('gsplat');
    entity.setLocalEulerAngles(rotationEulerDeg[0], rotationEulerDeg[1], rotationEulerDeg[2]);
    entity.addComponent('gsplat', {});
    entity.gsplat.resource = resource;

    // The gsplat instance (and its material) is only created once the entity
    // joins the hierarchy, so add it before touching the material.
    app.root.addChild(entity);

    const material = entity.gsplat.material;
    if (material) {
        material.setDefine('GSPLAT_AA', config.aa);
        // The forward pass reads alphaClipForward; alphaClip covers the
        // shadow/pick/prepass variants.
        material.setParameter('alphaClip', alphaClip);
        material.setParameter('alphaClipForward', alphaClip);
    }

    const detachAnim = animation.attach(global);

    // Set synchronously (not deferred to 'firstFrame') so CameraManager,
    // constructed immediately after this resolves, already knows the
    // content is 4DGS and skips its camera-anim-track defaults.
    global.state.hasAnimation = true;

    const onFirstFrame = () => {
        global.state.animationDuration = animation.duration;
    };
    global.events.once('firstFrame', onFirstFrame);

    // Clean up animation listeners when the entity is destroyed (e.g. page unload).
    entity.once('destroy', () => {
        detachAnim();
        global.events.off('firstFrame', onFirstFrame);
    });

    return entity;
};

export { setupSplatAnim };
