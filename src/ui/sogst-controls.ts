import { isSogstFilename } from '../parsers/sogst';
import type { Global } from '../types';

// The `.sogst` rotation block of the settings panel (rotate the clip in 90 degree steps, with the
// result mirrored into the page url as `sogstrot`) and its cache-clearing button. The block is
// shown only for `.sogst` content.
const initSogstControls = (global: Global) => {
    const { app, config, events, root } = global;
    const element = (name: string) => root.querySelector<HTMLElement>(`.sse-${name}`);

    const block = element('sogstRotationBlock');
    const value = element('sogstRotationValue');

    const defaultRotation: [number, number, number] = config.sogstRotationDeg ?? [0, 0, 0];
    const rotation: [number, number, number] = [...defaultRotation];

    const isSogstContent = isSogstFilename(config.contentFilename ?? config.contentUrl ?? '');
    block.classList.toggle('sse-hidden', !isSogstContent);

    const normalizeDegrees = (degrees: number) => {
        const normalized = degrees % 360;
        return normalized < 0 ? normalized + 360 : normalized;
    };

    const apply = () => {
        const entity = app.root.findByName('gsplat');
        if (!entity) {
            return;
        }

        entity.setLocalEulerAngles(rotation[0], rotation[1], rotation[2]);
        value.textContent = `SOGST Rotation: ${rotation.join(', ')}`;

        // keep the page url a faithful link to this view; an embedded viewer owns no url
        if (global.config.exposeGlobals) {
            const url = new URL(window.location.href);
            url.searchParams.set('sogstrot', rotation.join(','));
            window.history.replaceState({}, '', url);
        }
        app.renderNextFrame = true;
    };

    const rotate = (axis: 0 | 1 | 2, delta: number) => {
        rotation[axis] = normalizeDegrees(rotation[axis] + delta);
        apply();
    };

    const onClick = (name: string, handler: () => void) => {
        const button = element(name);
        button.addEventListener('click', handler);
        return () => button.removeEventListener('click', handler);
    };

    const disposers = [
        onClick('sogstRotateXNeg', () => rotate(0, -90)),
        onClick('sogstRotateXPos', () => rotate(0, 90)),
        onClick('sogstRotateYNeg', () => rotate(1, -90)),
        onClick('sogstRotateYPos', () => rotate(1, 90)),
        onClick('sogstRotateZNeg', () => rotate(2, -90)),
        onClick('sogstRotateZPos', () => rotate(2, 90)),
        onClick('sogstRotateReset', () => {
            rotation[0] = defaultRotation[0];
            rotation[1] = defaultRotation[1];
            rotation[2] = defaultRotation[2];
            apply();
        }),
        onClick('sogstClearCache', async () => {
            // `supersplat-sogst-chunks` is the only store the viewer writes.
            if (typeof indexedDB !== 'undefined') {
                await new Promise<void>((resolve) => {
                    const request = indexedDB.deleteDatabase('supersplat-sogst-chunks');
                    request.onsuccess = () => resolve();
                    request.onerror = () => resolve();
                    request.onblocked = () => resolve();
                });
            }
        })
    ];

    // the clip's entity exists once the scene has loaded
    const loaded = events.on('loaded:changed', () => {
        if (isSogstContent) {
            apply();
        }
    });

    return () => {
        loaded.off();
        for (const dispose of disposers) {
            dispose();
        }
    };
};

export { initSogstControls };
