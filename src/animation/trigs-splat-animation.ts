import { Quat } from 'playcanvas';
import type { Entity } from 'playcanvas';

import type { TrigsData } from '../core/load-trigsbin';
import { evalCenters } from '../core/trigs-math';
import { bindTrigsModifier, setTrigsParams } from '../core/trigs-motion';
import type { Global } from '../types';

import { Playhead } from './playhead';

// Least time between two refreshes of the CPU sorter's centers (WebGL).
const CENTERS_INTERVAL_MS = 120;

// Animation driver for .trigsbin content. Like the .sogst driver it uploads nothing per frame: playback is a time
// uniform and the entity's rotation, and the GPU evaluates the model in the unified work-buffer pass
// (core/trigs-motion.ts). Time is therefore continuous and never waits for a fetch.
//
// Depth sorting is the one thing the work buffer does not settle on every backend. On WebGPU the engine sorts on the GPU
// from the work buffer, after the modifier, so the order follows the moving splats. On WebGL it sorts on the CPU from
// `resource.centers`, which holds the canonical positions; a splat that turns about an anchor moves far from there, so
// the order would be that of the canonical pose. This driver therefore evaluates the centers at the current time on the
// CPU (core/trigs-math.ts evalCenters) and hands them to the sorter, at most every CENTERS_INTERVAL_MS, with a final
// refresh once playback settles.
class TrigsSplatAnimation {
    private entity: Entity | null = null;
    private lastTime = NaN;
    private lastRotation = new Quat(NaN, NaN, NaN, NaN);
    private readonly data: TrigsData;
    private lastCentersAt = 0;
    private centersTime = NaN;
    private centersTimer: ReturnType<typeof setTimeout> | null = null;
    private global: Global | null = null;
    private scratch: Float32Array | null = null;

    constructor(data: TrigsData) {
        this.data = data;
    }

    get duration(): number {
        return this.data.duration;
    }

    get numFrames(): number {
        return Math.max(1, Math.round(this.data.duration * (this.data.fps || 30)) + 1);
    }

    // Install the GPU modifier on the entity's gsplat component. Called once the entity exists (after setupSplatAnim).
    bind(entity: Entity) {
        this.entity = entity;
        bindTrigsModifier(entity);
    }

    // True when the engine sorts on the CPU from the resource's centers.
    private get cpuSorted(): boolean {
        return !!this.global && !this.global.app.graphicsDevice.isWebGPU;
    }

    // Write the centers at `time` into the resource and tell the sorter they changed.
    private refreshCenters(time: number) {
        const global = this.global;
        const centers = this.data.resource.centers as Float32Array | undefined;
        if (!global || !centers) {
            return;
        }
        this.scratch ??= new Float32Array(centers.length);
        evalCenters(this.data.arrays, this.data.numSplats, time, this.scratch);
        centers.set(this.scratch);
        this.data.resource.centersVersion++;
        this.centersTime = time;
        this.lastCentersAt = performance.now();
        const instance = (global.app.root.findComponents('gsplat') as { instance?: { sort(camera: unknown): void } }[])[0]?.instance;
        instance?.sort(global.camera);
        global.app.renderNextFrame = true;
    }

    private scheduleCenters(time: number) {
        if (!this.cpuSorted || time === this.centersTime) {
            return;
        }
        const wait = CENTERS_INTERVAL_MS - (performance.now() - this.lastCentersAt);
        if (wait <= 0) {
            this.refreshCenters(time);
            return;
        }
        // throttled: make sure the last time asked for is sorted once playback stops
        if (this.centersTimer === null) {
            this.centersTimer = setTimeout(() => {
                this.centersTimer = null;
                this.refreshCenters(this.lastTime);
            }, wait);
        }
    }

    // Push the uniforms when the time or the entity rotation changed (each push marks the work buffer render-dirty).
    private apply(animTime: number): boolean {
        if (!this.entity) {
            return false;
        }
        const rotation = this.entity.getRotation();
        if (animTime === this.lastTime && rotation.equals(this.lastRotation)) {
            return false;
        }
        this.lastTime = animTime;
        this.lastRotation.copy(rotation);
        setTrigsParams(this.entity, animTime);
        this.scheduleCenters(animTime);
        return true;
    }

    attach(global: Global): () => void {
        const { app, state, events } = global;
        this.global = global;
        const playhead = new Playhead();

        const onUpdate = (dt: number) => {
            if (!state.animationPaused) {
                if (playhead.advance(dt, this.duration, state)) {
                    state.animationPaused = true;
                }
                state.animationTime = playhead.time;
            } else {
                // Honour external scrubs that write to state.animationTime directly.
                playhead.time = state.animationTime;
            }
            if (this.apply(playhead.time)) {
                app.renderNextFrame = true;
            }
        };

        const onScrub = (time: number) => {
            playhead.seek(time, this.duration);
            state.animationTime = playhead.time;
            if (this.apply(playhead.time)) {
                app.renderNextFrame = true;
            }
        };

        // Leaving pingpong mode resumes normal forward playback.
        const onLoopModeChanged = () => {
            playhead.resetDirection();
        };

        app.on('update', onUpdate);
        events.on('scrubAnim', onScrub);
        events.on('animationLoopMode:changed', onLoopModeChanged);

        return () => {
            if (this.centersTimer !== null) {
                clearTimeout(this.centersTimer);
                this.centersTimer = null;
            }
            app.off('update', onUpdate);
            events.off('scrubAnim', onScrub);
            events.off('animationLoopMode:changed', onLoopModeChanged);
        };
    }
}

export { TrigsSplatAnimation };
