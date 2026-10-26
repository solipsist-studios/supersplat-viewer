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
| `content` | URL of the scene file (`.ply`, `.compressed.ply`, `.sog`, `.json` incl. `meta.json` / `lod-meta.json`, `.sogst`). An unrecognised extension is rejected before the file is fetched; a URL with no extension is treated as static 3DGS. | `./scene.compressed.ply` |
| `skybox` | URL of an equirectangular skybox image | |
| `poster` | URL of an image to show while loading | |
| `collision` | URL of a collision asset (`.glb` mesh, or voxel data). `voxel` is accepted as an alias. | |

### UI

| Parameter | Description |
| --------- | ----------- |
| `noui` | Hide the UI overlay (the programmatic form is `ui: false`, which skips building it) |
| `reticle` | Show the centre reticle in captured desktop fly/walk mode (off by default; programmatic form: `reticle: true`) |
| `noanim` | Start with animation paused |
| `ministats` | Show runtime CPU/GPU performance graphs |
| `lang` | Override the UI language (`de`, `en`, `es`, `fr`, `ja`, `ko`, `pt-BR`, `ru`, `zh-CN`; default: detect from browser) |

### Renderer

By default the viewer uses WebGPU when available (falling back automatically when not), except for `.sogst` clips on a Mac or Vision Pro (and iPad, which reports itself the same way), which use WebGL: there, clips that play through a segment window keep up with real time only on WebGL. The flags below force a renderer:

| Parameter | Description |
| --------- | ----------- |
| `webgl` | Force the WebGL renderer (required for AR/VR) |
| `webgpu` | Force the WebGPU renderer, including for `.sogst` clips on a Mac or Vision Pro |
| `aa` | Enable antialiasing (WebGL only) |
| `nofx` | Disable post effects |
| `hpr` | Override `highPrecisionRendering` from settings (`?hpr`, `?hpr=1`, `?hpr=true`, `?hpr=enable` to enable) |
| `budget` | Override the splat budget, in millions of splats |
| `sogstbudget` | Splat budget for `.sogst` content, in millions of splats. A segmented clip larger than this keeps only the segments near the playhead decoded, and re-reads the rest from the cache (or the network) as playback reaches them. Unset, the viewer picks a budget from the device type (Apple touch devices, other mobile, desktop) and the clip's memory cost per splat, so large clips play windowed on phones and headsets. `0` plays every clip fully resident. |
| `sogstcpu` | Decode `.sogst` content with the CPU pack workers instead of on the GPU. GPU decode is the default under both WebGL2 and WebGPU. |
| `colorize` | Render with LOD colorization |
| `fullload` | Load all streaming LOD data before the first frame |
| `heatmap` | Use heatmap mode for the voxel collision debug overlay. Requires WebGPU and voxel collision data; press `V` or use the collision toolbar button to show the overlay. |
| `debug` | Open the developer debug panel on load (`Ctrl+Shift+D` to toggle) |

## NPM Package

See the [API reference](docs/api.md) for embedding, runtime control, UI customization and settings.

## Local Development

To initialize a local development environment for SuperSplat Viewer, ensure you have [Node.js](https://nodejs.org/) 20 or later installed. Follow these steps:

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

### Testing over HTTPS

WebGPU needs a secure context: `https://`, or `http://localhost`. A page opened from another machine over plain http (a phone, a headset, a second computer) falls back to WebGL2. To serve over HTTPS with a self-signed certificate:

```sh
npm run develop:https
```

The first run creates `.certs/` (ignored by git) with a certificate for `localhost`, the machine's hostname and its IPv4 addresses, using `openssl`. Open `https://<hostname>:3000` and accept the browser's certificate warning once per origin. Delete `.certs/` to regenerate the certificate after the addresses change. `npm run serve:https` serves `public/` without the watcher.

### Debug engine build

By default the viewer links against the release build of the PlayCanvas engine. Set `ENGINE=debug` to link against the engine's debug build instead, which includes runtime assertions and unminified, readable source for easier debugging:

```sh
ENGINE=debug npm run develop
```

This also works with `npm run build` and `npm run watch`.


## 4D Gaussian Splatting — SOG spacetime (`.sogst`)

The viewer supports animated 4D Gaussian Splat scenes in the `.sogst` format.

> **Previously `.omg4`.** The old name came from the
> [OMG4](https://github.com/MinShirley/OMG4) training pipeline, whose
> compression stage the encoder first consumed. Nothing in the container comes
> from that work. The representation is spacetime-shaped, the container is
> PlayCanvas SOG, and the segment streaming belongs to this project. The name
> `.sogst` records that: SOG plus spacetime.
>
> The development-era `.omg4` containers were a flat binary format that carried
> the ASCII magic `OMG4`. We **never released them, and the viewer no longer
> reads them**.

### What is `.sogst`?

`.sogst` is a web-friendly container for 4D (space-time) Gaussians. It is a ZIP
archive of lossless-WebP attribute textures and a `meta.json` manifest. The
manifest must be the first entry.

`meta.version` is `1` and `meta.format` is `"sogst"`. Both are required, and the
viewer rejects any other value.

Static attributes follow the PlayCanvas **SOG v2** conventions byte for byte, so
an existing SOG decoder reconstructs them unchanged.

The spacetime extension stores each Gaussian once. Each one carries its
position, its sliced 3D covariance, its colour, and its temporal parameters:
linear velocity, temporal centre and temporal standard deviation. The viewer
then evaluates motion and temporal fade **on the GPU** every frame:

```
position(t) = position + velocity · (t − t_center)
alpha(t)    = sigmoid(opacity) · exp(−0.5 · ((t − t_center) / t_sigma)²)
```

Playback is continuous in time. There are no baked frames and no per-frame
texture uploads. A full 10-second Neural-3D-Video scene fits in ~36 MB, or
~11 MB without view-dependent SH.

Motion may also carry an optional second-order term (`motion.degree == 2`),
where `position(t) = position + velocity · dt + accel · dt²`.

Two details are easy to get wrong. `accel` is the raw `dt²` coefficient, **not**
half-acceleration. The temporal factor above is **unnormalised**, so there is no
`1/√(2πσ²)` term.

The file orders splats as `[ persistent | segment 0 | segment 1 | … ]` and
groups them by temporal centre. A player can therefore cull by time, and the
archive can stream.

Playback starts once the persistent group and the first segment have arrived. If
the network is too slow, the playhead holds at the decoded boundary.

### Converting an OMG4 `.xz` checkpoint

Use the converter in [playcanvas/splat-transform](https://github.com/playcanvas/splat-transform)
to produce a `.sogst` file from a trained OMG4 model. A CUDA GPU is required to
evaluate the neural MLPs during conversion.

### Loading a `.sogst` file in the viewer

Pass the file URL via the `content` query parameter as with any other format:

```
https://example.com/viewer/?content=scene.sogst
```

The viewer will display a play/pause button and a timeline scrubber, just like
camera animation.  The user can orbit/fly around the scene while the animation
plays. The clip time range comes from the file header
(`--time_min` / `--time_max` at export time).

### File-size guidance

| Gaussians (N) | Without SH | With 3-band SH |
|---------------|------------|----------------|
| 100 000       | ~7.6 MB    | ~26 MB         |
| 150 000       | ~11 MB     | ~38 MB         |

File size is independent of clip duration. The WebP payloads are already
compressed, so the archive does not benefit meaningfully from transport gzip.

### Format specification

[`docs/sogst-format.md`](https://github.com/solipsist-studios/cumuli/blob/main/docs/sogst-format.md)
specifies the container normatively. This README is a summary. Where the two
differ, the specification is correct.

Within the viewer:

- `src/parsers/sogst.ts` — what the player relies on
- `src/core/zip.ts` — reads the container
- `src/core/sogst-texels.ts` — decodes the WebP payloads
- `src/core/sogst-decoder.ts` — fills the attribute arrays
- `src/core/load-sogst.ts` — whole-file entry point
- `src/core/stream-sogst.ts` — streaming entry point
