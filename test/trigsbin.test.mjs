import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { describe, it } from 'node:test';
import { fileURLToPath } from 'node:url';

// The reader and the CPU model are plain TypeScript with no engine imports, so node runs them as they are.
import { evalCenters, evalSplat } from '../src/core/trigs-math.ts';
import { f16ToF32, isTrigsbinFilename, parseTrigsbin } from '../src/parsers/trigsbin.ts';

const fixture = (name) => readFileSync(fileURLToPath(new URL(`./fixtures/trigsbin/${name}`, import.meta.url)));
const toArrayBuffer = (buf) => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);

const file = parseTrigsbin(toArrayBuffer(fixture('small.trigsbin')));
const expected = JSON.parse(fixture('expected.json').toString());

describe('.trigsbin reader', () => {
    it('names the file by its extension', () => {
        assert.equal(isTrigsbinFilename('a/b/Final.TRIGSBIN'), true);
        assert.equal(isTrigsbinFilename('final.sogst'), false);
    });

    it('reads the header, the timeline and the one clip', () => {
        assert.equal(file.fps, expected.fps);
        assert.equal(file.clips.length, 1);
        assert.equal(file.totalFrames, 31);
        assert.ok(Math.abs(file.duration - 1.0) < 1e-9, 'the last frame is at 1 s');
        assert.equal(file.clips[0].arrays.means.length, expected.count * 3);
        assert.equal(file.clips[0].core[0], 0);
        assert.equal(file.clips[0].core[1], 31);
    });

    it('reads every tensor the model needs, as float32', () => {
        for (const key of ['means', 'scales', 'quats', 'opacities', 'sh0', 'shN', 'times', 'log_durations', 'anchors',
            'zeta_base_r', 'zeta_base_t', 'zeta_c0_r', 'zeta_c0_t', 'zeta_c1_r', 'zeta_c1_t', 'zeta_c2_r', 'zeta_c2_t']) {
            assert.ok(file.clips[0].arrays[key] instanceof Float32Array, key);
        }
    });

    it('refuses what is not a trigsbin', () => {
        assert.throws(() => parseTrigsbin(new ArrayBuffer(10)), /shorter than a header/);
        const bad = new Uint8Array(200);
        bad.set(new TextEncoder().encode('NOTTRIGS'));
        assert.throws(() => parseTrigsbin(bad.buffer), /not a .trigsbin/);
    });

    it('widens half floats exactly', () => {
        const halves = new Uint16Array([0x3c00, 0xc000, 0x0000, 0x3555, 0x7bff, 0x0001]);
        const got = f16ToF32(halves);
        assert.deepEqual(Array.from(got.slice(0, 3)), [1, -2, 0]);
        assert.ok(Math.abs(got[3] - 0.333251953125) < 1e-12);
        assert.equal(got[4], 65504);
        assert.ok(Math.abs(got[5] - 2 ** -24) < 1e-20);
    });
});

describe('the CPU model', () => {
    it('slices as cumuli_core.trigs does, at every test time, for every splat', () => {
        const arrays = file.clips[0].arrays;
        let worst = 0;
        for (const slice of expected.slices) {
            for (let i = 0; i < expected.count; i++) {
                const got = evalSplat(arrays, i, slice.t);
                for (let k = 0; k < 3; k++) {
                    worst = Math.max(worst, Math.abs(got.mean[k] - slice.means[i][k]));
                }
                // a quaternion and its negative are one rotation
                const want = slice.quats[i];
                const sign = got.quat.reduce((s, v, k) => s + v * want[k], 0) < 0 ? -1 : 1;
                for (let k = 0; k < 4; k++) {
                    worst = Math.max(worst, Math.abs(sign * got.quat[k] - want[k]));
                }
                worst = Math.max(worst, Math.abs(got.alpha - slice.alpha[i]));
            }
        }
        assert.ok(worst < 1e-9, `the largest difference from the Python reference was ${worst}`);
    });

    it('gives the sorter the same centers as the full model, without allocating per splat', () => {
        const arrays = file.clips[0].arrays;
        const out = new Float32Array(expected.count * 3);
        for (const slice of expected.slices) {
            evalCenters(arrays, expected.count, slice.t, out);
            for (let i = 0; i < expected.count; i++) {
                for (let k = 0; k < 3; k++) {
                    assert.ok(Math.abs(out[i * 3 + k] - slice.means[i][k]) < 2e-5, `splat ${i} at t=${slice.t}`);
                }
            }
        }
    });

    it('leaves a splat where it is at its own time centre', () => {
        const arrays = file.clips[0].arrays;
        const got = evalSplat(arrays, 5, arrays.times[5]);
        for (let k = 0; k < 3; k++) {
            assert.ok(Math.abs(got.mean[k] - arrays.means[5 * 3 + k]) < 1e-6);
        }
    });
});
