import { AsogPlayer } from 'animated-sog/playcanvas';
import type { Entity, AppBase  } from 'playcanvas';


import { ClockAnimation } from '../animation/clock-animation';
import type { Config, Global } from '../types';

// .asog loading: animated-sog's own PlayCanvas player (MIT, github.com/arrival-space/animated-sog, `animated-sog/playcanvas`)
// decodes the file while it downloads and moves the splats in a work-buffer modifier. It reads its clock from a function, so it
// is handed the viewer's `state.animationTime`; this module adds the viewer's transport (ClockAnimation: play, pause, loop mode,
// speed, the timeline) and the entity the rest of the viewer looks for.
//
// The player stands a clip up with a half turn about z, as TRiGS files are y down. A .sogst or .trigsbin here carries its whole
// rotation in `sogstRotationDeg` (Studio computes it from the rig), so the player's own turn is undone and the entity named
// 'gsplat' carries the configured one, the same as for those formats.
//
// The player needs the engine's unified gsplat renderer and some of its internals (it names PlayCanvas 2.22.6 and later).

const loadAsog = async (app: AppBase, config: Config, global: Global, onProgress: (progress: number) => void): Promise<Entity> => {
    const { state, events } = global;
    const player = await AsogPlayer.load(app, config.contentUrl, app.root, {
        audio: false,
        getTime: () => state.animationTime,
        onProgress: (p: number) => onProgress(Math.min(100, Math.trunc(p * 100)))
    });
    // the viewer looks for the gsplat component on the entity this returns, so that is the player's own entity
    const root = player.entity!;
    root.name = 'gsplat';
    const [x, y, z] = config.sogstRotationDeg ?? [0, 0, 0];
    root.setLocalEulerAngles(x, y, z);

    // the same material settings setupSplatAnim gives every animated clip
    const material = root.gsplat?.material;
    if (material) {
        material.setDefine('GSPLAT_AA', config.aa);
        material.setParameter('alphaClip', 1 / 1024);
        material.setParameter('alphaClipForward', 1 / 1024);
    }

    const animation = new ClockAnimation(player.duration);
    const detach = animation.attach(global);

    // Set synchronously (not deferred to 'firstFrame') so CameraManager, constructed right after this resolves, already knows the
    // content is 4DGS and skips its camera-anim-track defaults.
    state.has4DAnimation = true;
    state.hasAnimation = true;
    const onFirstFrame = () => {
        state.animationDuration = animation.duration;
    };
    events.once('firstFrame', onFirstFrame);

    root.once('destroy', () => {
        detach();
        events.off('firstFrame', onFirstFrame);
        player.destroy();
    });
    return root;
};

export { loadAsog };
