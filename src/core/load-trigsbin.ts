import { GSplatData, GSplatResource } from 'playcanvas';
import type { AppBase, BoundingBox } from 'playcanvas';

import { parseTrigsbin } from '../parsers/trigsbin';
import type { TrigsArrays, TrigsFile } from '../parsers/trigsbin';

import { evalSplat } from './trigs-math';
import { createTrigsTextures } from './trigs-motion';

// .trigsbin loading: one parsed file becomes a GSplatResource whose splats sit at their canonical position (the
// resource's own textures carry colour, scale and peak opacity, exactly as for a .ply) plus the eight motion
// textures of trigs-motion.ts. Playback is then a time uniform, with no per-frame upload.
//
// One clip (what Studio writes: one trainer window) plus the optional static group is supported. A file of several clips
// is refused with a message that says so, rather than drawn wrongly.

type TrigsData = {
    resource: GSplatResource;
    file: TrigsFile;
    // the splat-ordered tensors the textures were made from, kept for the CPU sorter's centers (WebGL)
    arrays: TrigsArrays;
    numSplats: number;
    // seconds of playback: the time of the last frame
    duration: number;
    fps: number;
};

// The clip's tensors and the static group, in one splat-ordered set of arrays, with the motion zeroed (and a wide
// window) for the static ones.
const concatenate = (file: TrigsFile): { count: number; arrays: TrigsArrays } => {
    if (file.clips.length !== 1) {
        throw new Error(
            `trigsbin: this file has ${file.clips.length} clips; only a file of one clip plays. ` +
                'Studio writes one clip per model.'
        );
    }
    const clip = file.clips[0].arrays;
    const stat = file.static;
    const nClip = clip.means.length / 3;
    const nStatic = stat.means ? stat.means.length / 3 : 0;
    const n = nClip + nStatic;
    const sizes: Record<string, number> = {
        means: 3,
        scales: 3,
        quats: 4,
        opacities: 1,
        sh0: 3,
        times: 1,
        log_durations: 1,
        anchors: 3,
        zeta_base_r: 3,
        zeta_base_t: 3,
        zeta_c0_r: 3,
        zeta_c0_t: 3,
        zeta_c1_r: 3,
        zeta_c1_t: 3,
        zeta_c2_r: 3,
        zeta_c2_t: 3
    };
    const shN = clip.shN ?? new Float32Array(0);
    const k = nClip ? shN.length / (nClip * 3) : 0;
    const out: TrigsArrays = {};
    for (const [name, per] of Object.entries(sizes)) {
        out[name] = new Float32Array(n * per);
        out[name].set(clip[name], 0);
        if (nStatic && stat[name]) {
            out[name].set(stat[name], nClip * per);
        }
    }
    // a static splat does not move and is always there: no twist, and a window far wider than any clip
    for (let i = nClip; i < n; i++) {
        out.log_durations[i] = 20;
        out.anchors.set(
            [stat.means[(i - nClip) * 3], stat.means[(i - nClip) * 3 + 1], stat.means[(i - nClip) * 3 + 2]],
            i * 3
        );
    }
    out.shN = new Float32Array(n * k * 3);
    out.shN.set(shN, 0);
    if (nStatic && stat.shN && stat.shN.length === nStatic * k * 3) {
        out.shN.set(stat.shN, nClip * k * 3);
    }
    return { count: n, arrays: out };
};

const propertyNames = (shCoefficients: number): string[] => {
    const names = [
        'x',
        'y',
        'z',
        'f_dc_0',
        'f_dc_1',
        'f_dc_2',
        'opacity',
        'scale_0',
        'scale_1',
        'scale_2',
        'rot_0',
        'rot_1',
        'rot_2',
        'rot_3'
    ];
    for (let i = 0; i < 3 * shCoefficients; i++) {
        names.push(`f_rest_${i}`);
    }
    return names;
};

// The canonical splats as PLY-convention data (log scales, logit opacity, wxyz rotation, channel-major f_rest), which
// is what the engine builds its textures from.
const buildGsplatData = (count: number, a: TrigsArrays, shCoefficients: number): GSplatData => {
    const f = () => new Float32Array(count);
    const cols: Record<string, Float32Array> = {};
    for (const name of propertyNames(shCoefficients)) {
        cols[name] = f();
    }
    for (let i = 0; i < count; i++) {
        cols.x[i] = a.means[i * 3];
        cols.y[i] = a.means[i * 3 + 1];
        cols.z[i] = a.means[i * 3 + 2];
        for (let c = 0; c < 3; c++) {
            cols[`f_dc_${c}`][i] = a.sh0[i * 3 + c];
            cols[`scale_${c}`][i] = a.scales[i * 3 + c];
        }
        cols.opacity[i] = a.opacities[i];
        // the file stores xyzw, the PLY convention is wxyz
        cols.rot_0[i] = a.quats[i * 4 + 3];
        cols.rot_1[i] = a.quats[i * 4];
        cols.rot_2[i] = a.quats[i * 4 + 1];
        cols.rot_3[i] = a.quats[i * 4 + 2];
        for (let k = 0; k < shCoefficients; k++) {
            for (let c = 0; c < 3; c++) {
                cols[`f_rest_${c * shCoefficients + k}`][i] = a.shN[(i * shCoefficients + k) * 3 + c];
            }
        }
    }
    return new GSplatData([
        {
            name: 'vertex',
            count,
            properties: Object.entries(cols).map(([name, storage]) => ({
                name,
                type: 'float' as const,
                byteSize: 4,
                storage
            }))
        }
    ]);
};

// A shell that reports only the count and the SH band count, so the full-length arrays can be released once the engine
// has built its textures from them (it reads nothing else from this object afterwards).
const shellData = (count: number, shCoefficients: number): GSplatData =>
    new GSplatData([
        {
            name: 'vertex',
            count,
            properties: propertyNames(shCoefficients).map((name) => ({
                name,
                type: 'float' as const,
                byteSize: 4,
                storage: new Float32Array(0)
            }))
        }
    ]);

// Bounds of the whole clip: the canonical positions, and where a spread of splats goes over a handful of times.
const sceneBounds = (count: number, a: TrigsArrays, duration: number) => {
    const min = [Infinity, Infinity, Infinity];
    const max = [-Infinity, -Infinity, -Infinity];
    const take = (p: number[]) => {
        for (let k = 0; k < 3; k++) {
            min[k] = Math.min(min[k], p[k]);
            max[k] = Math.max(max[k], p[k]);
        }
    };
    const step = Math.max(1, Math.floor(count / 20000));
    for (let i = 0; i < count; i += step) {
        take([a.means[i * 3], a.means[i * 3 + 1], a.means[i * 3 + 2]]);
        for (const f of [0, 0.25, 0.5, 0.75, 1]) {
            const s = evalSplat(a, i, f * duration);
            if (s.alpha > 0.02 && s.mean.every((v) => Number.isFinite(v))) {
                take(s.mean);
            }
        }
    }
    return { min, max };
};

const setAabb = (aabb: BoundingBox, bounds: { min: number[]; max: number[] }) => {
    aabb.center.set(
        (bounds.min[0] + bounds.max[0]) / 2,
        (bounds.min[1] + bounds.max[1]) / 2,
        (bounds.min[2] + bounds.max[2]) / 2
    );
    aabb.halfExtents.set(
        (bounds.max[0] - bounds.min[0]) / 2,
        (bounds.max[1] - bounds.min[1]) / 2,
        (bounds.max[2] - bounds.min[2]) / 2
    );
};

const loadTrigsbin = (app: AppBase, buffer: ArrayBuffer, onProgress?: (progress: number) => void): TrigsData => {
    const file = parseTrigsbin(buffer);
    onProgress?.(40);
    const { count, arrays } = concatenate(file);
    if (count === 0) {
        throw new Error('trigsbin: the file holds no splats');
    }
    const coefficients = arrays.shN.length / (count * 3);
    const data = buildGsplatData(count, arrays, coefficients);
    const resource = new GSplatResource(app.graphicsDevice, data);
    resource.gsplatData = shellData(count, coefficients);
    createTrigsTextures(resource, count, arrays);
    onProgress?.(80);
    setAabb(resource.aabb, sceneBounds(count, arrays, file.duration));
    onProgress?.(100);
    return { resource, file, arrays, numSplats: count, duration: file.duration, fps: file.fps };
};

export { concatenate, loadTrigsbin };
export type { TrigsData };
