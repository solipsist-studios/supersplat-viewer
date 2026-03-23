import { Entity, GSplatResource, type AppBase } from 'playcanvas';

import { SplatAnimationBase } from '../animation/splat-animation-base';
import type { Config, Global } from '../types';

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
    animation: SplatAnimationBase
): Entity => {
    const entity = new Entity('gsplat');
    entity.setLocalEulerAngles(0, 0, 180);
    entity.addComponent('gsplat', {});
    entity.gsplat.resource = resource;

    const material = entity.gsplat.material;
    if (material) {
        material.setDefine('GSPLAT_AA', config.aa);
        material.setParameter('alphaClip', 1 / 255);
    }

    app.root.addChild(entity);

    const detachAnim = animation.attach(global);

    const onFirstFrame = () => {
        global.state.hasAnimation = true;
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
