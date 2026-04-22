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

## URL Parameters

The app supports a number of URL parameters (these are subject to change):

### Content

| Parameter | Description | Default |
| --------- | ----------- | ------- |
| `settings` | URL of the `settings.json` file | `./settings.json` |
| `content` | URL of the scene file (`.ply`, `.sog`, `.compressed.ply`, `.meta.json`, `.lod-meta.json`) | `./scene.compressed.ply` |
| `skybox` | URL of an equirectangular skybox image | |
| `poster` | URL of an image to show while loading | |
| `collision` | URL of a collision asset (`.glb` mesh, or voxel data). `voxel` is accepted as an alias. | |

### UI

| Parameter | Description |
| --------- | ----------- |
| `noui` | Hide the UI overlay |
| `noanim` | Start with animation paused |
| `ministats` | Show runtime CPU/GPU performance graphs |

### Renderer

By default the viewer uses WebGL. The flags below opt into WebGPU rendering paths:

| Parameter | Description |
| --------- | ----------- |
| `compute` | Use the WebGPU compute renderer |
| `gpu` | Use the WebGPU GPU sort renderer |
| `cpu` | Use the WebGPU CPU sort renderer |
| `aa` | Enable antialiasing (WebGL only) |
| `nofx` | Disable post effects |
| `hpr` | Override `highPrecisionRendering` from settings (`?hpr`, `?hpr=1`, `?hpr=true`, `?hpr=enable` to enable) |
| `budget` | Override the splat budget, in millions of splats |
| `colorize` | Render with LOD colorization |
| `fullload` | Load all streaming LOD data before the first frame |
| `heatmap` | Render the heatmap debug overlay (WebGPU only) |

## NPM Package

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

The package also exports the settings schema types and helpers via the `/settings` subpath, which is useful for generating, validating or migrating a `settings.json` file:

```ts
import {
    importSettings,
    validateSettings,
    type ExperienceSettings
} from '@playcanvas/supersplat-viewer/settings';

// throws on invalid input
validateSettings(json);

// migrates a v1 settings object to the latest schema
const settings: ExperienceSettings = importSettings(json);
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

The `settings.json` file uses the schema below (defined in TypeScript and exported from `@playcanvas/supersplat-viewer/settings`). Legacy v1 settings produced by older SuperSplat releases are automatically migrated to v2 on load.

```typescript
type AnimTrack = {
    name: string,
    duration: number,
    frameRate: number,
    loopMode: 'none' | 'repeat' | 'pingpong',
    interpolation: 'step' | 'spline',
    smoothness: number,
    keyframes: {
        times: number[],
        values: {
            position: number[],
            target: number[],
            fov: number[],
        }
    }
};

type CameraPose = {
    position: [number, number, number],
    target: [number, number, number],
    fov: number
};

type Camera = {
    initial: CameraPose
};

type Annotation = {
    position: [number, number, number],
    title: string,
    text: string,
    extras?: any,
    camera: Camera
};

type PostEffectSettings = {
    sharpness: { enabled: boolean, amount: number },
    bloom:     { enabled: boolean, intensity: number, blurLevel: number },
    grading:   { enabled: boolean, brightness: number, contrast: number, saturation: number, tint: [number, number, number] },
    vignette:  { enabled: boolean, intensity: number, inner: number, outer: number, curvature: number },
    fringing:  { enabled: boolean, intensity: number }
};

type ExperienceSettings = {
    version: 2,
    tonemapping: 'none' | 'linear' | 'filmic' | 'hejl' | 'aces' | 'aces2' | 'neutral',
    highPrecisionRendering: boolean,
    soundUrl?: string,
    background: {
        color: [number, number, number],
        skyboxUrl?: string
    },
    postEffectSettings: PostEffectSettings,
    animTracks: AnimTrack[],
    cameras: Camera[],
    annotations: Annotation[],
    startMode: 'default' | 'animTrack' | 'annotation'
};
```

### Example settings.json

```json
{
    "version": 2,
    "tonemapping": "none",
    "highPrecisionRendering": false,
    "background": {
        "color": [0, 0, 0]
    },
    "postEffectSettings": {
        "sharpness": { "enabled": false, "amount": 0 },
        "bloom":     { "enabled": false, "intensity": 1, "blurLevel": 2 },
        "grading":   { "enabled": false, "brightness": 0, "contrast": 1, "saturation": 1, "tint": [1, 1, 1] },
        "vignette":  { "enabled": false, "intensity": 0.5, "inner": 0.3, "outer": 0.75, "curvature": 1 },
        "fringing":  { "enabled": false, "intensity": 0.5 }
    },
    "animTracks": [],
    "cameras": [
        {
            "initial": {
                "position": [0, 1, -1],
                "target": [0, 0, 0],
                "fov": 60
            }
        }
    ],
    "annotations": [],
    "startMode": "default"
}
```

---

## 4D Gaussian Splatting — OMG4 format (`.omg4`)

The viewer supports animated 4D Gaussian Splat scenes produced by the
[OMG4](https://github.com/MinShirley/OMG4) training pipeline.

### What is `.omg4`?

`.omg4` is a web-friendly binary container that stores pre-baked per-frame
Gaussian attributes (position, rotation, scale, opacity, colour) so the
browser can play back an OMG4 animation at runtime without any GPU-side MLP
inference.

### Converting an OMG4 `.xz` checkpoint

Use the converter in [playcanvas/splat-transform](https://github.com/playcanvas/splat-transform)
to produce a `.omg4` file from a trained OMG4 model. A CUDA GPU is required to
evaluate the neural MLPs during conversion.

### Loading a `.omg4` file in the viewer

Pass the file URL via the `content` query parameter as with any other format:

```
https://example.com/viewer/?content=scene.omg4
```

The viewer will display a play/pause button and a timeline scrubber, just like
camera animation.  The user can orbit/fly around the scene while the OMG4
animation plays. Playback timing follows the per-frame timestamps baked into
the `.omg4` file, so custom `--omg4-time-min` / `--omg4-time-max` exports from
`splat-transform` are respected.

### File-size guidance

| Gaussians (N) | Frames (F) | Approx. file size |
|---------------|------------|-------------------|
| 50 000        | 30         | ~42 MB            |
| 100 000       | 30         | ~84 MB            |
| 100 000       | 50         | ~140 MB           |

Standard gzip compression (e.g. `gzip -k scene.omg4`) and serving with
`Content-Encoding: gzip` typically halves the transfer size.

### Binary format specification

```
Header (28 bytes, all values little-endian):
  uint32  magic = 0x34474D4F  ("OMG4")
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
