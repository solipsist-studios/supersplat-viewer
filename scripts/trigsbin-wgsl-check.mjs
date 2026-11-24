// Prepares a page that runs the viewer's WGSL motion code (src/core/trigs-motion.ts) as a compute shader on the GPU and
// compares it with the CPU model (src/core/trigs-math.ts) on test/fixtures/trigsbin/small.trigsbin: displacement, rotation
// turn and temporal factor of every splat at five times. Serve <out dir> and open check.html in a browser with WebGPU; the
// page sets window.__result = { ok, text }. Headless Chrome needs --enable-unsafe-webgpu --use-angle=vulkan, and the page
// asks for 10 storage buffers per stage (the harness binds the engine's eight textures as buffers).
//
//   node scripts/trigsbin-wgsl-check.mjs <out dir>
import { readFileSync, writeFileSync, copyFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

import ts from 'typescript';

const repo = fileURLToPath(new URL('..', import.meta.url));
const [out] = process.argv.slice(2);
mkdirSync(out, { recursive: true });
const transpile = (file, name) => {
    const src = readFileSync(join(repo, file), 'utf8');
    const js = ts.transpileModule(src, { compilerOptions: { module: ts.ModuleKind.ES2022, target: ts.ScriptTarget.ES2022 } }).outputText;
    writeFileSync(join(out, name), js);
};
transpile('src/core/trigs-math.ts', 'trigs-math.js');
transpile('src/parsers/trigsbin.ts', 'trigsbin.js');

// the WGSL chunk, as the viewer ships it, turned into something a compute shader can include: the textures become
// storage buffers indexed by the splat, the engine's `uniform` block becomes a struct
const motion = readFileSync(join(repo, 'src/core/trigs-motion.ts'), 'utf8');
const wgsl = /const wgslModifyChunk = \/\* wgsl \*\/ `([\s\S]*?)`;/.exec(motion)[1];
let body = wgsl
    .replace(/var (trigs[A-H]): texture_2d<uff>;/g, '')
    .replace(/uniform trigsTime: f32;\nuniform trigsModelRotation: vec4f;\nuniform trigsModelScale: vec3f;/, '')
    .replace(/textureLoad\((trigs[A-H]), splat\.uv, 0\)/g, '$1[gidx]')
    .replace(/uniform\./g, 'uni.');
// drop the engine hooks (modifySplat*), which need engine types; the check calls trigsEval directly
body = body.slice(0, body.indexOf('fn modifySplatCenter'));
const header = `
var<private> gidx: u32;
struct Uni { trigsTime: f32, trigsModelRotation: vec4f, trigsModelScale: vec3f }
@group(0) @binding(0) var<storage, read> trigsA: array<vec4f>;
@group(0) @binding(1) var<storage, read> trigsB: array<vec4f>;
@group(0) @binding(2) var<storage, read> trigsC: array<vec4f>;
@group(0) @binding(3) var<storage, read> trigsD: array<vec4f>;
@group(0) @binding(4) var<storage, read> trigsE: array<vec4f>;
@group(0) @binding(5) var<storage, read> trigsF: array<vec4f>;
@group(0) @binding(6) var<storage, read> trigsG: array<vec4f>;
@group(0) @binding(7) var<storage, read> trigsH: array<vec4f>;
@group(0) @binding(8) var<storage, read_write> result: array<vec4f>;
@group(0) @binding(9) var<uniform> uni: Uni;
`;
const footer = `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= arrayLength(&trigsA)) { return; }
    gidx = id.x;
    let s = trigsEval();
    result[id.x * 2u] = vec4f(s.disp, s.gamma);
    result[id.x * 2u + 1u] = s.turn;
}
`;
writeFileSync(join(out, 'shader.wgsl'), header + body + footer);
copyFileSync(join(repo, 'test/fixtures/trigsbin/small.trigsbin'), join(out, 'small.trigsbin'));

writeFileSync(join(out, 'check.html'), `<!doctype html><meta charset="utf-8"><body style="background:#fff;color:#000;font:14px monospace"><pre id="log">running</pre>
<script type="module">
import { parseTrigsbin } from './trigsbin.js';
import { evalSplat, quatMul } from './trigs-math.js';
const log = (s) => { document.getElementById('log').textContent += '\\n' + s; };
const done = (ok, s) => { log(s); window.__result = { ok, text: document.getElementById('log').textContent }; };
try {
    const file = parseTrigsbin(await (await fetch('./small.trigsbin')).arrayBuffer());
    const a = file.clips[0].arrays;
    const n = a.means.length / 3;
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice({ requiredLimits: { maxStorageBuffersPerShaderStage: 10 } }); device.addEventListener('uncapturederror', (e) => log('GPU error: ' + e.error.message.slice(0, 300)));
    const code = await (await fetch('./shader.wgsl')).text();
    const module = device.createShaderModule({ code });
    const info = await module.getCompilationInfo();
    for (const m of info.messages) log(m.type + ': ' + m.message + ' @' + m.lineNum);
    if (info.messages.some((m) => m.type === 'error')) { done(false, 'WGSL did not compile'); throw new Error('stop'); }

    // the eight textures' contents, as the loader lays them out (trigs-motion.ts createTrigsTextures)
    const buf = (name) => new Float32Array(n * 4);
    const T = ['A','B','C','D','E','F','G','H'].map(buf);
    for (let i = 0; i < n; i++) {
        const o = i * 3;
        const put = (k, arr, w) => { T[k][i*4] = arr ? arr[o] : 0; T[k][i*4+1] = arr ? arr[o+1] : 0; T[k][i*4+2] = arr ? arr[o+2] : 0; T[k][i*4+3] = w; };
        put(0, a.zeta_base_r, a.times[i]); put(1, a.zeta_base_t, Math.exp(a.log_durations[i]));
        put(2, a.zeta_c0_r, a.anchors[o]); put(3, a.zeta_c0_t, a.anchors[o+1]); put(4, a.zeta_c1_r, a.anchors[o+2]);
        put(5, a.zeta_c1_t, a.means[o]); put(6, a.zeta_c2_r, a.means[o+1]); put(7, a.zeta_c2_t, a.means[o+2]);
    }
    const mk = (data, usage) => { const b = device.createBuffer({ size: data.byteLength, usage: usage | GPUBufferUsage.COPY_DST }); device.queue.writeBuffer(b, 0, data); return b; };
    const inputs = T.map((d) => mk(d, GPUBufferUsage.STORAGE));
    const resultBuf = device.createBuffer({ size: n * 2 * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const readBuf = device.createBuffer({ size: n * 2 * 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const uniBuf = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    let worst = 0, worstWhat = ''; const bad = [];
    for (const t of [0.0, 0.2, 0.5, 0.77, 1.0]) {
        device.queue.writeBuffer(uniBuf, 0, new Float32Array([t, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]));
        const entries = [...inputs.map((b, i) => ({ binding: i, resource: { buffer: b } })), { binding: 8, resource: { buffer: resultBuf } }, { binding: 9, resource: { buffer: uniBuf } }];
        const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass(); pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(Math.ceil(n / 64)); pass.end();
        enc.copyBufferToBuffer(resultBuf, 0, readBuf, 0, n * 32);
        device.queue.submit([enc.finish()]);
        await readBuf.mapAsync(GPUMapMode.READ);
        const r = new Float32Array(readBuf.getMappedRange().slice(0));
        readBuf.unmap();
        for (let i = 0; i < n; i++) {
            const cpu = evalSplat(a, i, t);
            const mu = [a.means[i*3], a.means[i*3+1], a.means[i*3+2]];
            const q = [a.quats[i*4], a.quats[i*4+1], a.quats[i*4+2], a.quats[i*4+3]];
            const conj = [-q[0], -q[1], -q[2], q[3]];
            const turn = quatMul(cpu.quat, conj);          // cpu quat = turn * q
            const gpuDisp = [r[i*8], r[i*8+1], r[i*8+2]], gpuGamma = r[i*8+3], gpuTurn = [r[i*8+4], r[i*8+5], r[i*8+6], r[i*8+7]];
            const sign = gpuTurn.reduce((s, v, k) => s + v * turn[k], 0) < 0 ? -1 : 1;
            const checks = [
                ['disp', Math.max(...gpuDisp.map((v, k) => Math.abs(v - (cpu.mean[k] - mu[k]))))],
                ['turn', Math.max(...gpuTurn.map((v, k) => Math.abs(sign * v - turn[k])))],
                ['gamma', Math.abs(gpuGamma - (cpu.alpha / (1 / (1 + Math.exp(-a.opacities[i])))))]
            ];
            for (const [what, err] of checks) { if (err > worst) { worst = err; worstWhat = what + ' splat ' + i + ' t=' + t; } if (err > 2e-3) bad.push({ i, t, what, err: +err.toFixed(4), gpu: gpuDisp.map((v)=>+v.toFixed(4)), cpu: cpu.mean.map((v,k)=>+(v-mu[k]).toFixed(4)), gamma: [gpuGamma, cpu.alpha], om: [a.zeta_base_r[i*3], a.zeta_base_r[i*3+1], a.zeta_base_r[i*3+2]].map((v)=>+v.toFixed(3)) }); }
        }
    }
    log('bad entries: ' + bad.length + ' ' + JSON.stringify(bad.slice(0, 6))); done(worst < 2e-3, 'worst difference between the WGSL and the CPU model: ' + worst.toExponential(2) + ' (' + worstWhat + ')');
} catch (e) { if (e.message !== 'stop') done(false, 'error: ' + e); }
</script>`);
console.log('prepared', out);
