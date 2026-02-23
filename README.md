# SuperSplat Viewer

[![NPM Version](https://img.shields.io/npm/v/@playcanvas/supersplat-viewer)](https://www.npmjs.com/package/@playcanvas/supersplat-viewer)
[![NPM Downloads](https://img.shields.io/npm/dw/@playcanvas/supersplat-viewer)](https://npmtrends.com/@playcanvas/supersplat-viewer)
[![License](https://img.shields.io/npm/l/@playcanvas/supersplat-viewer)](https://github.com/playcanvas/supersplat-viewer/blob/main/LICENSE)
[![Discord](https://img.shields.io/badge/Discord-5865F2?style=flat&logo=discord&logoColor=white&color=black)](https://discord.gg/RSaMRzg)
[![Reddit](https://img.shields.io/badge/Reddit-FF4500?style=flat&logo=reddit&logoColor=white&color=black)](https://www.reddit.com/r/PlayCanvas)
[![X](https://img.shields.io/badge/X-000000?style=flat&logo=x&logoColor=white&color=black)](https://x.com/intent/follow?screen_name=playcanvas)

| [User Manual](https://developer.playcanvas.com/user-manual/gaussian-splatting/editing/supersplat/import-export/#html-viewer-htmlzip) | [Blog](https://blog.playcanvas.com) | [Forum](https://forum.playcanvas.com) |

This is the official viewer for [SuperSplat](https://superspl.at).

<img width="1114" height="739" alt="supersplat-viewer" src="https://github.com/user-attachments/assets/15d2c654-9484-4265-a279-99acb65e38c9" />

The web app compiles to a simple, self-contained static website.

The app supports a few useful URL parameters (though please note these are subject to change):

| Parameter | Description | Default |
| --------- | ----------- | ------- |
| `settings` | URL of the `settings.json` file | `./settings.json` |
| `content` | URL of the scene file (`.ply`, `.sog`, `.meta.json`, `.lod-meta.json`) | `./scene.compressed.ply` |
| `skybox` | URL of an equirectangular skybox image | |
| `poster` | URL of an image to show while loading | |
| `noui` | Hide UI | |
| `noanim` | Start with animation paused | |
| `ministats` | Show runtime CPU/GPU performance graphs | |
| `unified` | Force unified rendering mode | |
| `aa` | Enable antialiasing (not supported in unified mode) | |

The web app source files are available as strings for templating when you import the package from npm:

```ts
import { html, css, js } from '@playcanvas/supersplat-viewer';

// logs the source of index.html
console.log(html);

// logs the source of index.css
console.log(css);

// logs the source of index.js
console.log(js);
```

## Local Development

To initialize a local development environment for SuperSplat Viewer, ensure you have [Node.js](https://nodejs.org/) 18 or later installed. Follow these steps:

1. Clone the repository:

   ```sh
   git clone https://github.com/playcanvas/supersplat-viewer.git
   cd supersplat-viewer
   ```

2. Install dependencies:

   ```sh
   npm install
   ```

3. Start the development build and local web server:

   ```sh
   npm run develop
   ```

4. Open your browser at http://localhost:3000.

## Settings Schema

The `settings.json` file has the following schema (defined in TypeScript and taken from the SuperSplat editor):


```typescript
type AnimTrack = {
    name: string,
    duration: number,
    frameRate: number,
    target: 'camera',
    loopMode: 'none' | 'repeat' | 'pingpong',
    interpolation: 'step' | 'spline',
    smoothness: number,
    keyframes: {
        times: number[],
        values: {
            position: number[],
            target: number[],
        }
    }
};

type ExperienceSettings = {
    camera: {
        fov?: number,
        position?: number[],
        target?: number[],
        startAnim: 'none' | 'orbit' | 'animTrack',
        animTrack: string
    },
    background: {
        color?: number[]
    },
    animTracks: AnimTrack[]
};
```

### Example settings.json

```json
{
  "background": {"color": [0,0,0,0]},
  "camera": {
    "fov": 1.0,
    "position": [0,1,-1],
    "target": [0,0,0],
    "startAnim": "orbit"
  },
  "animTracks": []
}
```

---

## 4D Gaussian Splatting — OMG4 format (`.4dgs`)

The viewer supports animated 4D Gaussian Splat scenes produced by the
[OMG4](https://github.com/MinShirley/OMG4) training pipeline.

### What is `.4dgs`?

`.4dgs` is a web-friendly binary container that stores pre-baked per-frame
Gaussian attributes (position, rotation, scale, opacity, colour) so the
browser can play back a 4DGS animation at runtime without any GPU-side MLP
inference.

### Converting an OMG4 `.xz` checkpoint

Run the provided converter on the machine where you trained the model (a CUDA
GPU is required to evaluate the neural MLPs):

```bash
python scripts/omg4_to_4dgs.py \
    --input  output/my_scene/comp.xz \
    --output public/scene.4dgs \
    --frames 30        \  # number of animation frames to bake
    --fps    24        \  # playback frame rate
    --time_min -0.5    \  # temporal range used during training
    --time_max  0.5
```

The converter requires the full OMG4 Python environment:
```
torch  numpy  dahuffman  tinycudann  lzma  pickle
```

### Loading a `.4dgs` file in the viewer

Pass the file URL via the `content` query parameter as with any other format:

```
https://example.com/viewer/?content=scene.4dgs
```

The viewer will display a play/pause button and a timeline scrubber, just like
camera animation.  The user can orbit/fly around the scene while the 4DGS
animation plays.

### File-size guidance

| Gaussians (N) | Frames (F) | Approx. file size |
|---------------|------------|-------------------|
| 50 000        | 30         | ~42 MB            |
| 100 000       | 30         | ~84 MB            |
| 100 000       | 50         | ~140 MB           |

Standard gzip compression (e.g. `gzip -k scene.4dgs`) and serving with
`Content-Encoding: gzip` typically halves the transfer size.

### Binary format specification

```
Header (28 bytes, all values little-endian):
  uint32  magic = 0x53474434  ("4DGS")
  uint32  version = 1
  uint32  numSplats
  uint32  numFrames
  float32 fps
  float32 timeDurationMin
  float32 timeDurationMax

Per-frame record (repeated numFrames times):
  float32          timestamp
  float32[N × 14]  per-splat data, AoS layout per splat:
    x  y  z                    — world-space position
    rot_0  rot_1  rot_2  rot_3  — quaternion (w, x, y, z), raw
    scale_0  scale_1  scale_2   — log-space scales (renderer applies exp)
    opacity                     — logit-space opacity (renderer applies sigmoid)
    f_dc_0  f_dc_1  f_dc_2      — raw SH DC coefficients
                                  (renderer computes 0.5 + val × SH_C0)
```
