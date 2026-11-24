/**
 * AsogPlayer: plays an Animated SOG (.asog) clip in PlayCanvas (engine 2.22.6+).
 *
 * The file is decoded while it downloads (core decodeAsog), each texture uploaded and its data freed
 * as soon as it is complete, so the whole clip is never held in memory at once. The splats render
 * through the engine's GSplatSogResource (SOG v2, SH included); the motion runs in a work-buffer
 * modifier (chunks.js) re-evaluated every frame from two motion textures.
 *
 * Depth order: on WebGPU the engine sorts the moved work buffer on the GPU by itself. WebGL 2 sorts
 * on the CPU from resource.centers, which a worker re-evaluates for the current time (core
 * sortWorkerSource) and swaps in (centersVersion).
 *
 * The clip is shown y up: the player's entity is rotated 180° about z (TRiGS files are y down).
 */
import * as pc from "playcanvas";
import { openAsog, decodeAsog } from "../core/decode.js";
import { centerBounds } from "../core/splats.js";
import { sortWorkerSource } from "../core/worker.js";
import { ASOG_CHUNK_GLSL, ASOG_CHUNK_WGSL } from "./chunks.js";

const UPLOAD_BAND = 16 * 1024 * 1024; // bytes per WebGPU texture write, each copied into the GPU process
const AUDIO_MAX_DRIFT = 0.06;          // seconds the sound may drift from the clip before it seeks

const PIXEL_FORMATS = {
    rgba8: pc.PIXELFORMAT_RGBA8,
    rgba32u: pc.PIXELFORMAT_RGBA32U,
    rgba16f: pc.PIXELFORMAT_RGBA16F,
    rgba32f: pc.PIXELFORMAT_RGBA32F,
};

/**
 * @typedef {object} AsogPlayerOptions
 * @property {() => number} [getTime] - the clock in seconds; the clip plays at getTime() modulo its
 *   duration. Default performance.now() / 1000. A clock shared between viewers (server time) shows
 *   everyone the same frame.
 * @property {boolean} [audio] - play the soundtrack, if the file has one (default true). Browsers start
 *   audio only after a user gesture: resume app.systems.sound.context (or app.soundManager) on one.
 * @property {number} [volume] - soundtrack volume (default 1)
 * @property {boolean} [positional] - soundtrack as a positional sound at the clip (default true)
 * @property {number} [refDistance] - positional: full volume within (default 3)
 * @property {number} [maxDistance] - positional: silent beyond (default 30)
 * @property {number[]} [layers] - layer ids for the gsplat component (default: the component's default)
 * @property {(progress: number) => void} [onProgress] - download progress 0..1 (when the size is known)
 * @property {string} [name] - asset name (default "asog")
 */

export class AsogPlayer {
    /**
     * Downloads (or reads) an .asog clip and starts playing it under `parent`.
     * @param {pc.AppBase} app
     * @param {string | URL | Response | Blob | ArrayBuffer | ArrayBufferView} source
     * @param {pc.Entity} parent
     * @param {AsogPlayerOptions} [options]
     * @returns {Promise<AsogPlayer>}
     */
    static async load(app, source, parent, options) {
        const player = new AsogPlayer(app, options);
        await player.load(source, parent);
        return player;
    }

    /**
     * @param {pc.AppBase} app
     * @param {AsogPlayerOptions} [options]
     */
    constructor(app, options = {}) {
        this.app = app;
        /** @type {Required<Omit<AsogPlayerOptions, "layers" | "onProgress">> & AsogPlayerOptions} */
        this.options = {
            getTime: () => performance.now() / 1000,
            audio: true, volume: 1, positional: true, refDistance: 3, maxDistance: 30, name: "asog",
            ...options,
        };
        /** @type {pc.Entity | null} the clip's entity (child of the parent passed to load) */
        this.entity = null;
        /** @type {import("../core/container.js").AsogManifest | null} */
        this.manifest = null;
        this.duration = 0;
        this.numSplats = 0;
        /** current clip time (seconds) */
        this.time = 0;
        /** @type {{ fileBytes: number, decodeMs: number, halfOverflow: number, loadMs: number } | null} */
        this.stats = null;
        /** @type {{ min: number[], max: number[] } | null} model-space box of most splats' centres (centerBounds) */
        this.bounds = null;
        /** soundtrack: the drift from the clip measured on the last frame (seconds), and the seeks made */
        this.audioDrift = 0;
        this.audioSeeks = 0;
        /** @type {Error | null} why the soundtrack could not be played (the clip plays without it) */
        this.audioError = null;

        this._abort = new AbortController();
        this._destroyed = false;
        /** @type {pc.Texture[]} textures this player owns (the SOG textures belong to the resource) */
        this._textures = [];
        this._sog = null;
        this._asset = null;
        this._audioAsset = null;
        this._audioSlot = null;
        this._audioTarget = 0;    // the sound position asked for on the last frame
        this._worker = null;
        this._sortSpare = null; // the centres buffer the worker gives back between jobs
        this._sortBusy = false;
        this._sortTime = NaN;     // the time of the centres last asked of the worker
        this._prevTime = NaN;
        this._modelInv = new pc.Mat4();
        this._onUpdate = this._update.bind(this);
    }

    /**
     * Loads a clip and starts playing it under `parent`. Rejects (and cleans up) on failure or when
     * destroy() is called while loading.
     * @param {string | URL | Response | Blob | ArrayBuffer | ArrayBufferView} source
     * @param {pc.Entity} parent
     */
    async load(source, parent) {
        try {
            await this._load(source, parent);
        } catch (e) {
            this.destroy();
            throw e;
        }
        return this;
    }

    async _load(source, parent) {
        const t0 = performance.now();
        const device = this.app.graphicsDevice;
        const signal = this._abort.signal;
        const file = await openAsog(source, { onProgress: this.options.onProgress });
        const { manifest } = file;
        this.manifest = manifest;
        this.duration = manifest.timeline.durationSeconds;
        this.numSplats = manifest.count;

        // only WebGL's CPU sorter needs the centres and the motion on the CPU (in the worker)
        const cpuSort = !device.isWebGPU;
        const sog = new pc.GSplatSogData();
        sog.meta = { version: 2, count: manifest.count, means: manifest.means, scales: manifest.scales, sh0: manifest.sh0, shN: manifest.shN };
        sog.numSplats = manifest.count;
        this._sog = sog;
        /** @type {Record<string, pc.Texture>} */
        const motion = {};
        const data = await decodeAsog(file, {
            signal,
            audio: this.options.audio && !!this.app.systems.sound,
            keep: cpuSort ? ["centers", "motionA", "motionB"] : [],
            onArray: async (name, array, { width, height, format }) => {
                if (name === "centers") {
                    this.bounds = centerBounds(/** @type {Float32Array} */ (array));
                    return;
                }
                const own = name === "motionA" || name === "motionB";
                const tex = this._makeTexture(name, width, height, PIXEL_FORMATS[format], own);
                if (own) motion[name] = tex;
                else sog[name] = tex;
                await this._upload(tex, array);
            },
        });
        if (this._destroyed) throw new Error("AsogPlayer destroyed while loading");

        sog.shBands = pc.GSplatSogData.calcBands(sog.sh_centroids?.width);
        sog.prepareCodebook();
        const resource = new pc.GSplatSogResource(device, sog, { prepareCenters: false });
        this._sog = null; // the resource owns the SOG textures now
        this._asset = new pc.Asset(this.options.name, "gsplat", null);
        this._asset.resource = resource;
        this._asset.loaded = true;
        this.app.assets.add(this._asset);

        const clipTex = this._makeTexture("asogClips", 2, manifest.clips.length, pc.PIXELFORMAT_RGBA32F, true);
        await this._upload(clipTex, data.clips);

        this.entity = new pc.Entity("AsogClip", this.app); // the app, not the last one created: pages may run several
        this.entity.setLocalEulerAngles(0, 0, 180);
        parent.addChild(this.entity);
        this.entity.addComponent("gsplat", {
            asset: this._asset,
            unified: true,
            ...(this.options.layers ? { layers: this.options.layers } : {}),
        });
        const gs = this.entity.gsplat;
        gs.setWorkBufferModifier({ glsl: ASOG_CHUNK_GLSL, wgsl: ASOG_CHUNK_WGSL });
        gs.setParameter("uAsogA", motion.motionA);
        gs.setParameter("uAsogB", motion.motionB);
        gs.setParameter("uAsogClips", clipTex);
        gs.workBufferUpdate = pc.WORKBUFFER_UPDATE_ALWAYS;

        if (cpuSort) {
            const n = this.numSplats;
            resource.centers = data.centers;
            this._worker = new Worker(URL.createObjectURL(new Blob([sortWorkerSource()], { type: "application/javascript" })));
            this._worker.onmessage = (e) => this._onSorted(e.data);
            const canonical = data.centers.slice();
            const clips = data.clips.slice();
            this._worker.postMessage({
                init: true, n, centers: canonical.buffer, A: data.motion.A.buffer, B: data.motion.B.buffer, clips: clips.buffer,
            }, [canonical.buffer, data.motion.A.buffer, data.motion.B.buffer, clips.buffer]);
            this._sortSpare = new Float32Array(n * 3).buffer;
        }

        this.stats = { ...data.stats, loadMs: performance.now() - t0 };
        this._update();
        this.app.on("update", this._onUpdate);
        if (data.audio) {
            try {
                await this._startAudio(data.audio.data);
            } catch (e) {
                this.audioError = e;
            }
        }
    }

    /** soundtrack volume */
    get volume() {
        return this.options.volume;
    }

    set volume(value) {
        this.options.volume = value;
        if (this._audioSlot) this._audioSlot.volume = value;
    }

    /** true once the soundtrack is decoded and attached */
    get hasAudio() {
        return !!this._audioSlot;
    }

    // The soundtrack plays on the entity, cut to the clip's length, and is restarted with every loop of
    // the clip (_syncAudio). Not a looping sound: a sound file is rarely exactly as long as the clip
    // (an mp3 of a 7.1 s clip decodes to 7.082 s), and a looping one would run ahead by the difference
    // every loop.
    async _startAudio(bytes) {
        const buffer = await this.app.systems.sound.context.decodeAudioData(bytes.buffer);
        if (this._destroyed) return;
        const o = this.options;
        this._audioAsset = new pc.Asset(`${o.name}-audio`, "audio", null);
        this._audioAsset.resource = new pc.Sound(buffer);
        this._audioAsset.loaded = true;
        this.app.assets.add(this._audioAsset);
        this.entity.addComponent("sound", {
            positional: o.positional,
            refDistance: o.refDistance,
            maxDistance: o.maxDistance,
            distanceModel: pc.DISTANCE_LINEAR,
        });
        this._audioSlot = this.entity.sound.addSlot("track", {
            asset: this._audioAsset.id,
            loop: false,
            autoPlay: false,
            duration: this.duration,
            volume: o.volume,
        });
    }

    _syncAudio() {
        const slot = this._audioSlot;
        // the audio clock stands still until a user gesture unlocks audio, and an instance started
        // before that does not survive it
        const context = this.app.systems.sound.context;
        if (!slot || !this.entity.enabled || context.state !== "running") return;
        // a clock that stands still (paused) pauses the sound
        if (this.time === this._prevTime) {
            if (slot.isPlaying) slot.pause();
            return;
        }
        if (slot.isPaused) slot.resume();
        // what is heard now left the audio graph outputLatency ago
        const target = (this.time + (context.baseLatency || 0) + (context.outputLatency || 0)) % this.duration;
        const wrapped = target < this._audioTarget; // the clip looped (or was scrubbed back): start over
        this._audioTarget = target;
        // slot.duration: the sound's length, at most the clip's; a shorter sound is silent for the rest of the loop
        if (target >= slot.duration) return;
        if (wrapped) slot.stop();
        if (!slot.isPlaying) {
            slot.play().currentTime = target;
            return;
        }
        const instance = slot.instances[0];
        const drift = target - instance.currentTime;
        this.audioDrift = drift;
        if (Math.abs(drift) > AUDIO_MAX_DRIFT) {
            instance.currentTime = target;
            this.audioSeeks++;
        }
    }

    // own: destroyed with the player; the SOG textures belong to their resource
    _makeTexture(name, width, height, format, own) {
        const tex = new pc.Texture(this.app.graphicsDevice, {
            name, width, height, format,
            mipmaps: false,
            minFilter: pc.FILTER_NEAREST,
            magFilter: pc.FILTER_NEAREST,
            addressU: pc.ADDRESS_CLAMP_TO_EDGE,
            addressV: pc.ADDRESS_CLAMP_TO_EDGE,
        });
        if (own) this._textures.push(tex);
        return tex;
    }

    // Uploads a texture's whole RGBA data. The texture keeps no CPU copy (that would only serve a
    // lost WebGL context, which this player does not recover from). WebGPU gets it in bands, as each
    // write is copied into the GPU process before it is done (engine 2.22 has no Texture.write for it).
    async _upload(tex, data) {
        const device = this.app.graphicsDevice;
        if (!device.isWebGPU) {
            tex._levels[0] = data;
            tex.upload();
            tex._levels[0] = null;
            return;
        }
        const rowLength = tex.width * 4;
        const rows = Math.max(1, Math.floor(UPLOAD_BAND / (rowLength * data.BYTES_PER_ELEMENT)));
        for (let y = 0; y < tex.height; y += rows) {
            const h = Math.min(rows, tex.height - y);
            device.wgpu.queue.writeTexture({ texture: tex.impl.gpuTexture, origin: [0, y, 0] },
                data.subarray(y * rowLength, (y + h) * rowLength),
                { bytesPerRow: rowLength * data.BYTES_PER_ELEMENT, rowsPerImage: h }, [tex.width, h, 1]);
            await device.wgpu.queue.onSubmittedWorkDone();
        }
    }

    _update() {
        const d = this.duration;
        this._prevTime = this.time;
        this.time = ((this.options.getTime() % d) + d) % d;
        const gs = this.entity.gsplat;
        gs.setParameter("uAsogTime", this.time);
        this._modelInv.copy(this.entity.getWorldTransform()).invert();
        gs.setParameter("uAsogModelInv", this._modelInv.data);
        // new centres only when the time moved (the sorter re-sorts camera moves by itself)
        if (this._worker && !this._sortBusy && this.time !== this._sortTime && this._usesCpuSorter()) {
            this._sortBusy = true;
            this._sortTime = this.time;
            const out = this._sortSpare;
            this._sortSpare = null;
            this._worker.postMessage({ time: this.time, out }, [out]);
        }
        if (this._audioSlot) this._syncAudio();
    }

    // Is a CPU sorter drawing this splat? (engine internals: the unified renderer's sorter for each camera and layer)
    _usesCpuSorter() {
        const id = this._asset.resource.id;
        for (const camData of this.app.renderer.gsplatDirector?.camerasMap?.values() ?? []) {
            for (const layerData of camData.layersMap?.values() ?? []) {
                if (layerData.gsplatManager?.cpuSorter?.centersSet?.has(id)) return true;
            }
        }
        return false;
    }

    _onSorted({ out }) {
        this._sortBusy = false;
        this._sortSpare = out;
        const res = this._asset?.resource;
        if (!res) return;
        res.centers.set(new Float32Array(out));
        res.centersVersion++;
    }

    /** Stops playing and frees everything (also while loading: load() then rejects). */
    destroy() {
        if (this._destroyed) return;
        this._destroyed = true;
        this._abort.abort(new Error("AsogPlayer destroyed while loading"));
        this.app.off("update", this._onUpdate);
        this._worker?.terminate();
        this._worker = null;
        this._audioSlot = null;
        const { entity, _asset: asset, _audioAsset: audioAsset, _sog: sog } = this;
        const textures = this._textures;
        this.entity = this._asset = this._audioAsset = this._sog = null;
        this._textures = [];
        const release = () => {
            entity?.destroy();
            if (audioAsset) this.app.assets.remove(audioAsset);
            if (asset) {
                asset.unload();
                this.app.assets.remove(asset);
            }
            sog?.destroy(); // SOG textures made before the resource was
            for (const t of textures) t.destroy();
        };
        if (!entity) {
            release();
            return;
        }
        // Engine 2.22.6: destroying a gsplat entity nulls its placement's resource at once, while the
        // unified renderer still lists the placement until it next reconciles the layer; with the CPU
        // sorter (WebGL) its streaming update then throws every frame. So the clip leaves the scene
        // first and is freed two frames later, when the renderer has let go of it and its textures
        // (one frame is not enough on WebGPU: the next frame's submit still binds them).
        entity.parent?.removeChild(entity);
        this.app.once("frameend", () => this.app.once("frameend", release));
    }
}
