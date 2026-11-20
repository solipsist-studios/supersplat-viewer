import type { EventHandler } from 'playcanvas';

import type { LoopMode, State } from '../types';

const loopModes: LoopMode[] = ['repeat', 'pingpong', 'none'];

// Playback speed bounds; they match the custom field's min and max in ui.html.
const minSpeed = 0.1;
const maxSpeed = 10;

// The transport controls beyond play, pause and the timeline: the loop-mode toggle (loop, bounce,
// play once) and the playback speed dropdown with its free-form entry field. They drive
// `state.animationLoopMode` and `state.animationSpeed`, which the camera manager and the 4DGS
// animations read.
const initPlaybackExtras = (state: State, events: EventHandler, root: HTMLElement) => {
    const element = <T extends HTMLElement>(name: string) => root.querySelector<T>(`.sse-${name}`);

    const loopButton = element('loopMode');
    const loopSvgs: Record<LoopMode, HTMLElement> = {
        repeat: element('loopRepeatSvg'),
        pingpong: element('loopPingpongSvg'),
        none: element('loopNoneSvg')
    };

    const speedButton = element('playbackSpeed');
    const speedMenu = element('playbackSpeedMenu');
    const speedInput = element<HTMLInputElement>('playbackSpeedCustom');
    const speedPresets = Array.from(speedMenu.querySelectorAll<HTMLElement>('.sse-speedPreset'));

    const updateLoopMode = () => {
        for (const mode of loopModes) {
            loopSvgs[mode].classList.toggle('sse-hidden', state.animationLoopMode !== mode);
        }
    };

    const onLoopClick = () => {
        state.animationLoopMode = loopModes[(loopModes.indexOf(state.animationLoopMode) + 1) % loopModes.length];
    };

    // Trim float noise (1.0500000000000003 to 1.05) so the label stays readable.
    const formatSpeed = (speed: number) => parseFloat(speed.toFixed(2)).toString();

    const updateSpeed = () => {
        speedButton.textContent = `${formatSpeed(state.animationSpeed)}×`;
        for (const preset of speedPresets) {
            preset.classList.toggle('sse-active', parseFloat(preset.dataset.speed) === state.animationSpeed);
        }
    };

    const isSpeedMenuOpen = () => !speedMenu.classList.contains('sse-hidden');

    const closeSpeedMenu = () => {
        speedMenu.classList.add('sse-hidden');
        speedButton.classList.remove('sse-menuOpen');
    };

    const openSpeedMenu = () => {
        // seed the field with the live value so it can be tweaked from there
        speedInput.value = formatSpeed(state.animationSpeed);
        speedMenu.classList.remove('sse-hidden');
        speedButton.classList.add('sse-menuOpen');
    };

    const onSpeedClick = () => (isSpeedMenuOpen() ? closeSpeedMenu() : openSpeedMenu());

    const presetHandlers = speedPresets.map((preset) => {
        const handler = () => {
            state.animationSpeed = parseFloat(preset.dataset.speed);
            closeSpeedMenu();
        };
        preset.addEventListener('click', handler);
        return () => preset.removeEventListener('click', handler);
    });

    // Apply the typed speed, ignoring junk and clamping to the supported range.
    const applyCustomSpeed = () => {
        const value = parseFloat(speedInput.value);
        if (!isFinite(value)) {
            speedInput.value = formatSpeed(state.animationSpeed);
            return;
        }
        const clamped = Math.min(maxSpeed, Math.max(minSpeed, value));
        state.animationSpeed = parseFloat(clamped.toFixed(2));
        speedInput.value = formatSpeed(state.animationSpeed);
    };

    const onSpeedKeydown = (event: KeyboardEvent) => {
        if (event.key === 'Enter') {
            applyCustomSpeed();
            closeSpeedMenu();
        }
        // keep viewer shortcuts (space, escape, …) from firing while typing
        event.stopPropagation();
    };

    loopButton.addEventListener('click', onLoopClick);
    speedButton.addEventListener('click', onSpeedClick);
    speedInput.addEventListener('change', applyCustomSpeed);
    speedInput.addEventListener('keydown', onSpeedKeydown);

    const subscriptions = [
        events.on('animationLoopMode:changed', updateLoopMode),
        events.on('animationSpeed:changed', updateSpeed)
    ];
    updateLoopMode();
    updateSpeed();

    return {
        isSpeedMenuOpen,
        closeSpeedMenu,
        dispose: () => {
            for (const subscription of subscriptions) subscription.off();
            for (const dispose of presetHandlers) dispose();
            loopButton.removeEventListener('click', onLoopClick);
            speedButton.removeEventListener('click', onSpeedClick);
            speedInput.removeEventListener('change', applyCustomSpeed);
            speedInput.removeEventListener('keydown', onSpeedKeydown);
        }
    };
};

export { initPlaybackExtras };
