# `.sogst` Decode Performance

Measurements of the `.sogst` decode and playback pipeline, kept so later changes
can be compared against them. Each section names the code it measured. Re-run
the same configuration (see [Method](#method)) before comparing numbers.

## Summary

| Pipeline                           | Where unpack and pack run | Clip             | Budget | Playback rate       | Main-thread long tasks |
| ---------------------------------- | ------------------------- | ---------------- | ------ | ------------------- | ---------------------- |
| Phase 3 window, main-thread decode | main thread, time-sliced  | `karasuba_rin_4` | 4.5M   | ~0.1× real time     | (not measured)         |
| Phase 3 window, pack workers       | 4 workers                 | `thrroog`        | 4.5M   | **~0.9× real time** | **none**               |

Moving unpack and pack off the main thread made windowed playback usable.
The remaining limit is worker compute, dominated by SH packing.

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
