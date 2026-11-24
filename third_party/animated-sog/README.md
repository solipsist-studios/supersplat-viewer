# animated-sog (JavaScript)

Reads, decodes and plays **Animated SOG** (`.asog`) clips: 4D Gaussian splats with TRiGS rigid-body
motion and an optional soundtrack. The format is specified in [`../docs/format.md`](../docs/format.md).

- `animated-sog`: engine-agnostic core (browser and Node 18+): streaming reader, inflate, stream
  decoding, dequantisation, GPU-ready layouts, a per-splat accessor and a pure-JS reference of the motion.
- `animated-sog/playcanvas`: `AsogPlayer` for PlayCanvas 2.22.6+ (WebGPU and WebGL 2).
- `animated-sog/three`: `AsogThreePlayer` for three.js 0.187+ (`WebGLRenderer`, WebGL 2).

ES modules, no build step; types are JSDoc.

## Install

```sh
npm install animated-sog
npm install playcanvas   # only for animated-sog/playcanvas
npm install three        # only for animated-sog/three
```

## PlayCanvas

```js
import * as pc from "playcanvas";
import { AsogPlayer } from "animated-sog/playcanvas";

const player = await AsogPlayer.load(app, "clip.asog", app.root, {
    getTime: () => performance.now() / 1000, // the clock; the clip loops on its duration
    audio: true,                              // play the soundtrack if the file has one
    volume: 1,
    positional: true, refDistance: 3, maxDistance: 30,
    onProgress: (p) => console.log(`${Math.round(p * 100)} %`),
});
// browsers start audio only after a user gesture:
button.onclick = () => app.systems.sound.context.resume();

player.time;     // current clip time (s)
player.duration; // clip length (s)
player.bounds;   // { min, max }: box of most splat centres, model space (y down)
player.entity;   // the clip's entity, rotated 180° about z to show it y up
player.destroy();
```

The source can be a URL, a `Response`, a `Blob`, a `ReadableStream`, or an `ArrayBuffer` / typed array.
Use a shared clock (e.g. server time) in `getTime` and every viewer sees the same frame; a clock that
stands still pauses the soundtrack, a jump seeks it.

How it works: the file is decoded while it downloads, one stream at a time; every texture is uploaded
and its data freed as soon as it is complete, and the decoder yields to the event loop at least every
25 ms. The splats render through the engine's `GSplatSogResource` (SOG v2, SH included); the motion
runs in a gsplat work-buffer modifier (GLSL and WGSL, `ASOG_CHUNK_GLSL` / `ASOG_CHUNK_WGSL`) every frame.
WebGPU sorts the moved splats on the GPU; WebGL 2 sorts on the CPU from centres a worker re-evaluates
for the current time. The soundtrack is restarted with each loop of the clip and seeks when it drifts
more than 60 ms from it.

Limits: up to 4,194,304 splats (the motion texture is 4096 texels wide, 4 texels a splat). The player
does not survive a lost WebGL context (textures keep no CPU copy).

## three.js

```js
import * as THREE from "three";
import { AsogThreePlayer } from "animated-sog/three";

const player = await AsogThreePlayer.load(renderer, "clip.asog", scene, {
    getTime: () => performance.now() / 1000, // the clock; the clip loops on its duration
    audio: true,                              // play the soundtrack if the file has one
    listener,                                 // a THREE.AudioListener (default: one added to the rendering camera)
    volume: 1,
    positional: true, refDistance: 3, maxDistance: 30,
    onProgress: (p) => console.log(`${Math.round(p * 100)} %`),
});
// browsers start audio only after a user gesture:
button.onclick = () => player.listener.context.resume();

player.time;     // current clip time (s)
player.duration; // clip length (s)
player.bounds;   // { min, max }: box of most splat centres, model space (y down)
player.object;   // the clip's THREE.Mesh, rotated 180° about z to show it y up
player.destroy();
```

Same sources and the same clock, sound and loading behaviour as `AsogPlayer`. The player updates itself in
the object's `onBeforeRender`: nothing to call per frame, just `renderer.render(scene, camera)`.

How it works: a small Gaussian splat renderer of its own, in two passes (`src/three/shaders.js`).

1. A work pass draws one fragment a splat into two RGBA32UI targets: it decodes the SOG v2 textures,
   moves the splat with the same motion modifier the PlayCanvas player runs (`ASOG_CHUNK_GLSL`, the GPU
   twin of the core motion reference) and evaluates its colour with SH (degree 1–3) for the camera
   position. It runs when the time or the camera position changed.
2. One instanced quad a splat projects the 3D covariance (EWA, the PlayCanvas gsplat formulas) and blends
   back to front with premultiplied alpha, in the order a worker gives: the core sort worker re-evaluates
   the moved centres for the current time and counting-sorts the splats visible then by view depth
   (invisible splats are left out of the draw).

The GLSL decodes and projects exactly as PlayCanvas 2.22 does, so the two players draw the same image:
the same clip, camera and time differ by 1 / 255 on average over the subject (rubloff, SH 3, 44.5 dB PSNR;
the 900k-splat test clip 49.5 dB), about what PlayCanvas's own WebGL and WebGPU renderers differ by.
With 900,946 splats a frame takes 2.5 ms (work pass + draw, 1060 × 900, RTX 5080); the sort worker takes
~37 ms, so the order follows the motion at ~25 Hz.

Why not [Spark](https://sparkjs.dev): Spark is a capable three.js splat renderer (MIT) with GPU
modifiers, but it stores splats in its own 16-byte packing (float16 centres, 8-bit octahedral axis-angle
rotations, 6–8-bit SH), so feeding it an `.asog` means repacking every splat on the CPU and quantising
the already quantised SOG data a second time: no visual parity with the PlayCanvas player, no streamed
upload of the SOG textures as they decode, and the motion (two motion textures read per splat) would have
to live in its shader graph through custom GLSL. The renderer here takes the decoder's textures
unchanged, shares the motion GLSL with the PlayCanvas player and adds no dependency besides three.

Limits: WebGL 2 only (no `WebGPURenderer`); the order is sorted for the last camera that rendered the
clip; textures keep no CPU copy, so a lost context is not recovered.

## Core

```js
import { decodeAsog, readSplat, evaluateSplatAt } from "animated-sog";

const data = await decodeAsog("clip.asog");   // or bytes, a Response, a Blob ...
data.count;            // splats
data.sog;              // SOG v2 RGBA8 texture data: means_l, means_u, quats, scales, sh0, sh_labels, sh_centroids
data.motion.A;         // Uint32Array, RGBA32U, 4096 × rowsA texels: mu, sigma, anchor, flags
data.motion.B;         // Uint16Array, RGBA16F, 4096 × rowsB texels: Omega and G control points
data.clips;            // Float32Array clip table
data.centers;          // Float32Array, canonical centres xyz
data.audio;            // { mime, data } or null

const s = readSplat(data, 0);                 // position, rotation (x y z w), scale, color, opacity, sh, motion
const e = evaluateSplatAt(data, 0, 0.5);      // { center, rotation, weight } at t = 0.5 s
```

The layouts are documented in [`src/core/layout.js`](src/core/layout.js). A renderer that wants to
upload and free as it goes (as `AsogPlayer` does) passes `onArray` and `keep`:

```js
import { openAsog, decodeAsog } from "animated-sog";

const file = await openAsog(url, { onProgress });   // header + manifest first
const data = await decodeAsog(file, {
    keep: [],                                       // free every array after onArray
    onArray: async (name, array, { width, height, format }) => {
        // name: means_l | means_u | quats | scales | sh0 | sh_labels | sh_centroids | motionA | motionB | centers
        // format: rgba8 | rgba32u | rgba16f | xyz32f (centers)
    },
    signal,                                         // AbortSignal
});
```

Lower-level pieces, each usable on its own: `parseHeader`, `parseManifest`, `openReader`, `inflate`
(`DecompressionStream` in browsers, `node:zlib` in Node), `decodeChannel` / `decodeStream` (byte planes,
zigzag, delta), the `dequant*` functions and `gStep` (spec 3.3 and 4.2), `floatToHalf`, `clipTable`,
`clipWeight`, `temporalWeight`, `evaluateSplat` (the motion of spec 4.1 in plain JS, the GPU modifier's
twin), `evaluateCenters` (all centres at a time), `sortSplats` (the splats visible at a time, back to front)
and `sortWorkerSource()` (a worker for CPU depth sorting, built from those functions: it returns the
centres at a time, or with `{ sort: true, time, depthRow, indices }` the sorted visible splats).

## Viewer

[`../examples/playcanvas`](../examples/playcanvas) and [`../examples/three`](../examples/three): serve the
repository root and open `/examples/playcanvas/` (`?src=<url>`, `?webgpu=0|1`) or `/examples/three/`
(`?src=<url>`).

```sh
npx http-server .. -p 8765 -c-1
```

## Tests

```sh
npm test
```

Decodes `../test-data/rubloff-1s.asog` and checks the container, every stream's size, the GPU layouts,
value ranges and the dequantisation against the spec's formulas, plus the motion reference, the depth
sort and the sort worker.
