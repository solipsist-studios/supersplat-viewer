# `.sogst` Decode Performance

Measurements of the `.sogst` decode and playback pipeline, kept so later changes
can be compared against them. Each section names the code it measured. Re-run
the same configuration (see [Method](#method)) before comparing numbers.

## Summary

| Pipeline                           | Where unpack and pack run | Clip             | Budget | Playback rate       | Main-thread long tasks |
| ---------------------------------- | ------------------------- | ---------------- | ------ | ------------------- | ---------------------- |
| Phase 3 window, main-thread decode | main thread, time-sliced  | `karasuba_rin_4` | 4.5M   | ~0.1× real time     | (not measured)         |
| Phase 3 window, pack workers       | 4 workers                 | `thrroog`        | 4.5M   | **~0.9× real time** | **none**               |
| Phase 3 window, GPU decode (spike) | fragment passes on GPU    | `thrroog`        | 4.5M   | **1.0× real time**¹ | **none**               |

¹ Before the paged arena, until the window stalled on arena fragmentation
(see [Paged Arena](#paged-arena-2026-10-02)); with it, sustained.

Moving unpack and pack off the main thread made windowed playback usable;
the remaining limit there is worker compute, dominated by SH packing. Doing
the same work in GPU passes costs a few milliseconds per million splats,
leaving the browser's WebP decode and the centers readback as the costs.

## Test Clips

| Clip                   | Splats | Persistent | In segments | SH bands | Segments | Duration | Size     |
| ---------------------- | ------ | ---------- | ----------- | -------- | -------- | -------- | -------- |
| `karasuba_rin_4.sogst` | 11.18M | 2.35M      | 8.83M       | 2        | 50       | 5.0 s    | 216.7 MB |
| `thrroog.sogst`        | 11.25M | 3.20M      | 8.05M       | 3        | 50       | 5.0 s    | 223.8 MB |

At most 4 segments of `karasuba_rin_4` are active at one instant (1.14M
splats); its segments range from 5k to 436k splats.

## Environment

- **Desktop:** AMD Ryzen 9 9950X3D (16 cores, 32 threads), NVIDIA RTX 5090,
  60 GB RAM, Linux. Brave (Chromium 152), WebGPU renderer. Brave reports
  `navigator.hardwareConcurrency` as 15, so the pack pool ran its cap of 4
  workers.
- **Headset:** Apple Vision Pro, Safari 26.6 (desktop macOS user agent). The
  splatxxx site forces the WebGL2 renderer there (`forceWebGL`, for one-tap
  AR), so the CPU sorter is in use.

## Method

- **Pages:** the standalone viewer build served by the splatxxx dev server,
  `/supersplat/index.html?content=<clip>&sogstbudget=<millions>`. The
  `sogstbudget` parameter selects windowed playback when the clip is larger
  than the budget.
- **Cold cache:** every cold run used a fresh filename (a symlink to the
  clip), because a warm IndexedDB entry switches the load to cache replay.
- **Visible tab:** Chrome throttles a hidden tab's timers to about once a
  second and stops `requestAnimationFrame`, which stalls both decoding and
  playback. Check `document.visibilityState === 'visible'` before trusting
  any number.
- **Playback rate:** the work-buffer modifier's `sogstTime` parameter,
  sampled every 500 ms for 20 s; frame rate from `requestAnimationFrame`
  counts; long tasks from a `PerformanceObserver` on `longtask`.
- **Per-job and per-install timings:** temporary instrumentation (not
  committed) recorded, for each pack job, its splat count, the time inside
  the worker, the time from submit to result, and the main-thread time for
  copying rows and uploading them; and for each window install, its fetch
  and total time and the evictions it caused.

## Phase 3: Main-Thread Decode (2026-10-01)

Code: segment window with unpack and pack on the main thread, in 6 ms slices
that yield to rendering. `karasuba_rin_4`, 4.5M budget, desktop WebGPU,
cache replay.

- **Playback:** clip time advanced 0.1 s per wall-clock second (one segment
  per second), about 0.1× real time. Raising the budget to 8.5M did not
  change it.
- **Install cost:** one 397k-splat segment took 620 ms: 55 ms WebP decode in
  a worker, 264 ms unpacking attributes, 189 ms packing for the GPU, 111 ms
  uploading. Throughput was about 600k splats/s before rendering competed for
  the main thread, and far less once the scene was up, because every slice
  waited behind a frame rendering 4–8M splats.
- **Upload path:** before a fix in the same work, WebGPU re-uploaded whole
  stream textures for every row range, because the engine's WebGPU textures
  have no partial `write()`. Writing the rows through `queue.writeTexture`
  halved install time.
- **Stall:** the persistent group's deferred SH (2.35M splats) held the
  decoder for about 7 s, blocking every install.

## Phase 3: Pack Workers (2026-10-02)

Code: `feature/sogst-window` at `025f52b` plus the timing instrumentation.
Unpack and pack run in a pool of 4 workers; the main thread copies finished
rows and uploads them. `thrroog`, desktop WebGPU, cold cache.

|                         | Windowed, 4.5M budget                      | Resident, no budget |
| ----------------------- | ------------------------------------------ | ------------------- |
| Reveal after navigation | 3.75 s                                     | 5.23 s              |
| All geometry decoded    | (continuous)                               | 14.3 s              |
| Playback rate           | ~0.9× (wraps every 5.5–6 s for a 5 s clip) | 1×                  |
| Frame rate              | 60 fps                                     | —                   |
| Main-thread long tasks  | none                                       | —                   |

Windowed detail, over 31 s of playback:

| Measure                                        | Value                                                         |
| ---------------------------------------------- | ------------------------------------------------------------- |
| Installs                                       | 260, with 255 evictions, up to 4 in flight                    |
| Sustained install throughput                   | 1.33M splats/s                                                |
| Install latency (cache read + decode + upload) | 180 ms average, of which 7.6 ms cache read                    |
| Worker compute, segment jobs (geometry + SH3)  | 1.41 s per million splats per worker                          |
| Worker compute, deferred-SH-only jobs          | 0.59 s per million splats per worker                          |
| Worker compute, resident load                  | 1.80 s per million splats per worker                          |
| Main thread, copying rows                      | 15 ms per million splats                                      |
| Main thread, upload                            | ~160 ms per million splats of wall time, paced, no long tasks |

Analysis:

- **Required rate:** at a 4.5M budget about 1.3M segment splats stay
  resident, so nearly the whole 8.05M of segments is re-decoded each loop:
  about 1.6M splats/s for real time. The pool sustained 1.33M/s, hence ~0.9×.
- **SH dominates worker time.** SH-only jobs cost about 40% of a full
  segment job.
- **Duplicate WebP decoding.** A group split across workers is WebP-decoded
  once per job. The 3.2M-splat persistent group was decoded four times, which
  is why the resident load costs more per splat than windowed installs.
- **First load is serial by group,** so the workers are partly idle during a
  resident load.

## GPU Decode Spike (2026-10-02)

Code: `feature/sogst-window` plus `sogst-gpu-decode.ts` (uncommitted spike),
opt-in with the `sogstgpu` URL flag, WebGL2 only (GLSL). The browser decodes
each WebP into an RGBA8 texture (`createImageBitmap`, no CPU readback), and
three fragment passes, limited to the destination rows, write the existing
stream layouts: geometry (colour, transformA, transformB), temporal (motion,
temporal, accel) and SH (up to four words). The CPU sorter still needs
centers, so the spike reads transformA's rows back. `thrroog`, desktop,
`?webgl`, cold cache.

|                                   | CPU pack workers (WebGPU) | GPU decode (WebGL2)             |
| --------------------------------- | ------------------------- | ------------------------------- |
| Resident: reveal after navigation | 5.23 s                    | 4.48 s                          |
| Resident: all groups decoded      | 14.3 s                    | **6.25 s**                      |
| Windowed 4.5M: playback rate      | ~0.9×                     | **1.0×**, until the stall below |
| Windowed 4.5M: install throughput | 1.33M splats/s            | 1.53M splats/s                  |
| Windowed 4.5M: average install    | 180 ms                    | **47 ms**                       |
| Frame rate / long tasks           | 60 fps / none             | 60 fps / none                   |

Cost per million splats:

| Stage                      | CPU pack workers         | GPU decode              |
| -------------------------- | ------------------------ | ----------------------- |
| Unpack and pack            | 1,410 ms per worker      | **3–7 ms** (the passes) |
| WebP decode                | (inside the worker cost) | 73–102 ms (browser)     |
| Centers for the CPU sorter | (inside the worker cost) | 115–120 ms (readback)   |

- **Correctness:** at t = 2.0 s the GPU-decoded frame matches the CPU frame
  visually. It is not byte-identical: the GPU rounds float to half with its
  own rounding mode and computes in float32.
- **What is left:** the WebP decode and the readback. The readback exists
  only for the CPU sorter (WebGL); WebGPU sorts on the GPU and would not
  need it.
- **Not yet measured:** WebGPU (needs WGSL versions of the passes) and the
  Vision Pro.

## Paged Arena (2026-10-02)

Code: `feature/sogst-window` with the arena split into 16,384-splat pages and
a per-page cull mask (uncommitted), fixing the fragmentation stall below.
`thrroog`, 4.5M budget (79 pages), desktop, cold cache, window focused. 35 s
of playback each.

|                         | GPU decode (WebGL2) | CPU pack workers (WebGPU) |
| ----------------------- | ------------------- | ------------------------- |
| Reveal after navigation | 2.38 s              | 3.87 s                    |
| Playback rate           | **1.0×**, 7 loops   | ~0.85×, 7 loops           |
| Longest playhead hold   | 0 ms                | 0 ms                      |
| Frame rate              | 59.7 fps            | 59.8 fps                  |
| Installs / average      | 358 / 70 ms         | 306 / 280 ms              |
| Failed installs         | 0                   | 0                         |

Segments took 2–15 pages laid out in 1–3 runs, so a placement costs one to
three passes or row copies. The configuration that stalled within 25 s with
contiguous slots played without a hold.

**Measurement trap:** a Chrome window that is visible but not focused can
be paced to about 1 fps on this Linux desktop (`document.visibilityState`
still reports `visible`; `document.hasFocus()` is false). The engine caps a
frame's time step at 0.1 s, so playback then advances 0.1 s per frame and
looks like a decode stall. Check the frame rate, not only visibility.

## Known Issues

**Window stall from arena fragmentation (fixed by the paged arena).** The window places each segment
in one contiguous slot and never evicts an active segment. With a tight
arena the free space can end up split into holes that are each too small.
Measured with `thrroog` at 4.5M (1.30M-splat arena): at t = 1.06 s three
active segments held 710k splats, and the next segment (242,314 splats)
needed a slot, but evicting the one evictable segment left holes of
237,900, 115,013 and 231,875 splats, 585k in total and none large enough.
`choose()` returns no plan and the playhead waits forever. The CPU run at
the same budget avoided it only through placement luck; this is a window
design bug, not a GPU-path bug.

## Candidate Improvements

CPU path (the fallback, kept regardless of a GPU path):

1. **SH lookup table per worker.** A splat's packed SH words depend only on
   its 16-bit centroid label. Packing all 65,536 centroids once per worker
   (about 3.4 MB for SH3) turns per-splat SH into a copy. Output stays
   byte-identical.
2. **Byte-indexed tables** for colour, opacity and scale half-floats.
3. **One WebP decode per group,** shared by the group's jobs.
4. **Pool size from the device,** instead of a cap of 4.
5. **Overlapping groups on first load.**

GPU path: upload the decoded SOG textures and unpack and pack them in a GPU
pass (or decode in the render shader, as the engine's `GSplatSogResource`
does), so the CPU only decodes WebP. Measured separately.

## Earlier Findings (Vision Pro)

- **Crash cause:** `karasuba_rin_4` lost the WebGL context at the reveal
  (`WebGL: context lost`; the page survived) while the decoder kept
  full-length per-splat arrays (about 1.9 GB at 43 floats per splat).
- **Fixed by Phase 2:** decoding straight into a capacity-sized GPU resource,
  with no whole-clip CPU arrays, let `karasuba_rin_4` play fully resident on
  the headset, AR included.
