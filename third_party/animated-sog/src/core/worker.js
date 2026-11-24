/**
 * Source of a worker that re-evaluates every splat's centre for a time, and depth sorts them, for
 * renderers that sort on the CPU. Engine-agnostic: it is built from the motion.js and sort.js
 * functions' own source text.
 *
 * Protocol:
 *   → { init: true, n, centers, A, B, clips }   ArrayBuffers: canonical centres (f32 xyz), motionA, motionB, clip table
 *   → { time, out }                              out: an ArrayBuffer of n * 3 floats to fill
 *   ← { time, out }                              out filled with the centres at `time`
 *   → { sort: true, time, depthRow, indices, margin }  indices: an ArrayBuffer of n uint32 to fill; depthRow: 4 numbers;
 *                                                margin: seconds (sortSplats)
 *                                                (sortSplats); the centres are re-evaluated only when `time` changed
 *   ← { sort: true, time, count, indices }       indices: the splats visible at `time`, back to front (sortSplats)
 */
import { halfTable } from "./half.js";
import { clipWeight, createMotion, evaluateCenters, evaluateSplat, readMotion, temporalWeight } from "./motion.js";
import { sortSplats } from "./sort.js";

function sortWorkerMain() {
    const half = halfTable();
    let state = null;
    self.onmessage = (e) => {
        const msg = e.data;
        if (msg.init) {
            state = {
                n: msg.n,
                centers: new Float32Array(msg.centers),
                A: new Uint32Array(msg.A),
                B: new Uint16Array(msg.B),
                clips: new Float32Array(msg.clips),
                moved: null,
                movedTime: NaN,
                scratch: {},
            };
            return;
        }
        if (msg.sort) {
            const s = state;
            if (!s.moved) s.moved = new Float32Array(s.n * 3);
            if (msg.time !== s.movedTime) {
                evaluateCenters(s.n, s.centers, s.A, s.B, s.clips, half, msg.time, s.moved);
                s.movedTime = msg.time;
            }
            const indices = new Uint32Array(msg.indices);
            const count = sortSplats(s.n, s.moved, s.A, s.clips, msg.time, msg.depthRow, indices, s.scratch, msg.margin ?? 0);
            self.postMessage({ sort: true, time: msg.time, count, indices: indices.buffer }, [indices.buffer]);
            return;
        }
        const out = new Float32Array(msg.out);
        evaluateCenters(state.n, state.centers, state.A, state.B, state.clips, half, msg.time, out);
        self.postMessage({ time: msg.time, out: out.buffer }, [out.buffer]);
    };
}

/**
 * The worker's JavaScript source (for `new Worker(URL.createObjectURL(new Blob([source])))`).
 * @returns {string}
 */
export function sortWorkerSource() {
    const parts = [halfTable, temporalWeight, clipWeight, createMotion, readMotion, evaluateSplat, evaluateCenters, sortSplats];
    return `${parts.map(String).join("\n")}\n(${sortWorkerMain})();\n`;
}
