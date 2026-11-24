/**
 * AsogThreePlayer: plays an Animated SOG (.asog) clip in three.js (WebGLRenderer, WebGL 2).
 *
 * The file is decoded while it downloads (core decodeAsog), each texture uploaded as soon as it is
 * complete and its data freed. The splats are drawn by a small Gaussian splat renderer of its own
 * (shaders.js): a work pass decodes, moves (the motion modifier shared with the PlayCanvas player)
 * and colours every splat into two RGBA32UI targets when the time or the camera position changed,
 * then one instanced quad a splat is projected and blended back to front (premultiplied alpha).
 * The order comes from a worker (core sortWorkerSource) that re-evaluates the centres for the
 * current time and depth sorts every splat of the active clips; the ones not visible at the time drawn are
 * dropped in the vertex shader (zero opacity from the work pass).
 *
 * The clip is shown y up: the player's object is rotated 180° about z (TRiGS files are y down).
 */
import * as THREE from "three";
import { openAsog, decodeAsog } from "../core/decode.js";
import { centerBounds } from "../core/splats.js";
import { sortWorkerSource } from "../core/worker.js";
import { SPLAT_FS, SPLAT_VS, WORK_FS, WORK_VS } from "./shaders.js";

const AUDIO_MAX_DRIFT = 0.06; // seconds the sound may drift from the clip before it seeks

/**
 * @typedef {object} AsogThreePlayerOptions
 * @property {() => number} [getTime] - the clock in seconds; the clip plays at getTime() modulo its
 *   duration. Default performance.now() / 1000. A clock shared between viewers (server time) shows
 *   everyone the same frame.
 * @property {boolean} [audio] - play the soundtrack, if the file has one (default true). Browsers start
 *   audio only after a user gesture: resume the listener's context (THREE.AudioContext.getContext()) on one.
 * @property {THREE.AudioListener} [listener] - the listener for the soundtrack (default: one the
 *   player makes and adds to the camera that renders the clip)
 * @property {number} [volume] - soundtrack volume (default 1)
 * @property {boolean} [positional] - soundtrack as a THREE.PositionalAudio at the clip (default true)
 * @property {number} [refDistance] - positional: full volume within (default 3)
 * @property {number} [maxDistance] - positional: silent beyond (default 30, linear fall-off)
 * @property {(progress: number) => void} [onProgress] - download progress 0..1 (when the size is known)
 * @property {string} [name] - name of the clip's object (default "asog")
 */

export class AsogThreePlayer {
    /**
     * Downloads (or reads) an .asog clip and starts playing it under `parent`.
     * @param {THREE.WebGLRenderer} renderer
     * @param {string | URL | Response | Blob | ArrayBuffer | ArrayBufferView} source
     * @param {THREE.Object3D} parent
     * @param {AsogThreePlayerOptions} [options]
     * @returns {Promise<AsogThreePlayer>}
     */
    static async load(renderer, source, parent, options) {
        const player = new AsogThreePlayer(renderer, options);
        await player.load(source, parent);
        return player;
    }

    /**
     * @param {THREE.WebGLRenderer} renderer
     * @param {AsogThreePlayerOptions} [options]
     */
    constructor(renderer, options = {}) {
        this.renderer = renderer;
        /** @type {Required<Omit<AsogThreePlayerOptions, "listener" | "onProgress">> & AsogThreePlayerOptions} */
        this.options = {
            getTime: () => performance.now() / 1000,
            audio: true, volume: 1, positional: true, refDistance: 3, maxDistance: 30, name: "asog",
            ...options,
        };
        /** @type {THREE.Mesh | null} the clip's object (child of the parent passed to load), rotated 180° about z */
        this.object = null;
        /** @type {import("../core/container.js").AsogManifest | null} */
        this.manifest = null;
        this.duration = 0;
        this.numSplats = 0;
        /** current clip time (seconds) */
        this.time = 0;
        /** splats drawn: those of the clips active at the time of the last sort */
        this.numVisible = 0;
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
        /** @type {THREE.Texture[]} */
        this._textures = [];
        /** @type {THREE.WebGLRenderTarget | null} */
        this._work = null;
        this._workScene = null;
        this._workTime = NaN;
        this._workCamera = new THREE.Camera();
        this._workCameraPos = new THREE.Vector3(NaN, NaN, NaN);
        this._worker = null;
        this._sortSpare = null; // the index buffer the worker fills next
        this._sortBusy = false;
        this._sortTime = NaN;
        this._sortRow = [NaN, NaN, NaN, NaN];
        this._prevTime = NaN;
        /** @type {THREE.Audio | THREE.PositionalAudio | null} */
        this._sound = null;
        this._soundLength = 0;
        this._listener = null;
        this._ownListener = false;
        this._audioStart = 0;   // context time at which the sound's position 0 played
        this._audioTarget = 0;  // the sound position asked for on the last frame
        this._modelView = new THREE.Matrix4();
        this._modelInv = new THREE.Matrix4();
        this._cameraPos = new THREE.Vector3();
        this._viewport = new THREE.Vector4();
    }

    /**
     * Loads a clip and starts playing it under `parent`. Rejects (and cleans up) on failure or when
     * destroy() is called while loading.
     * @param {string | URL | Response | Blob | ArrayBuffer | ArrayBufferView} source
     * @param {THREE.Object3D} parent
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
        const signal = this._abort.signal;
        const file = await openAsog(source, { onProgress: this.options.onProgress });
        const { manifest } = file;
        this.manifest = manifest;
        this.duration = manifest.timeline.durationSeconds;
        this.numSplats = manifest.count;

        /** @type {Record<string, THREE.DataTexture>} */
        const tex = {};
        const data = await decodeAsog(file, {
            signal,
            audio: this.options.audio,
            keep: ["centers", "motionA", "motionB"], // for the sort worker
            onArray: (name, array, { width, height, format }) => {
                if (name === "centers") this.bounds = centerBounds(/** @type {Float32Array} */ (array));
                else tex[name] = this._texture(array, width, height, format);
            },
        });
        if (this._destroyed) throw new Error("AsogThreePlayer destroyed while loading");

        const n = this.numSplats;
        const codebook = new Float32Array(256 * 4);
        for (let i = 0; i < 256; i++) {
            codebook[i * 4] = manifest.scales.codebook[i];
            codebook[i * 4 + 1] = manifest.sh0.codebook[i];
            codebook[i * 4 + 2] = manifest.shN?.codebook[i] ?? 0;
        }
        const codebookTex = this._texture(codebook, 256, 1, "rgba32f");
        const clipsTex = this._texture(data.clips.slice(), 2, manifest.clips.length, "rgba32f");

        // work pass: one fragment a splat, in the SOG texture layout
        const { width, height } = data.sog;
        this._work = new THREE.WebGLRenderTarget(width, height, {
            count: 2,
            format: THREE.RGBAIntegerFormat,
            type: THREE.UnsignedIntType,
            internalFormat: "RGBA32UI",
            minFilter: THREE.NearestFilter,
            magFilter: THREE.NearestFilter,
            generateMipmaps: false,
            depthBuffer: false,
            stencilBuffer: false,
        });
        const shBands = data.shBands;
        const workMaterial = new THREE.RawShaderMaterial({
            glslVersion: THREE.GLSL3,
            vertexShader: WORK_VS,
            fragmentShader: WORK_FS,
            defines: { SH_BANDS: shBands },
            uniforms: {
                means_l: { value: tex.means_l },
                means_u: { value: tex.means_u },
                quats: { value: tex.quats },
                scales: { value: tex.scales },
                sh0: { value: tex.sh0 },
                sh_labels: { value: tex.sh_labels ?? null },
                sh_centroids: { value: tex.sh_centroids ?? null },
                sogCodebook: { value: codebookTex },
                means_mins: { value: new THREE.Vector3().fromArray(manifest.means.mins) },
                means_maxs: { value: new THREE.Vector3().fromArray(manifest.means.maxs) },
                uNumSplats: { value: n },
                uCameraPosition: { value: new THREE.Vector3() },
                uAsogA: { value: tex.motionA },
                uAsogB: { value: tex.motionB },
                uAsogClips: { value: clipsTex },
                uAsogTime: { value: 0 },
                uAsogModelInv: { value: new THREE.Matrix4() },
            },
            depthTest: false,
            depthWrite: false,
            blending: THREE.NoBlending,
        });
        const triangle = new THREE.BufferGeometry();
        triangle.setAttribute("position", new THREE.Float32BufferAttribute([-1, -1, 0, 3, -1, 0, -1, 3, 0], 3));
        const workMesh = new THREE.Mesh(triangle, workMaterial);
        workMesh.frustumCulled = false;
        this._workScene = new THREE.Scene();
        this._workScene.add(workMesh);

        // splat pass: an instanced quad a splat, in the sorted order
        const geometry = new THREE.InstancedBufferGeometry();
        geometry.setAttribute("position", new THREE.Float32BufferAttribute([-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0], 3));
        geometry.setIndex([0, 1, 2, 0, 2, 3]);
        const order = new THREE.InstancedBufferAttribute(new Uint32Array(n), 1);
        order.setUsage(THREE.DynamicDrawUsage);
        geometry.setAttribute("splatIndex", order);
        geometry.instanceCount = 0;
        const material = new THREE.RawShaderMaterial({
            glslVersion: THREE.GLSL3,
            vertexShader: SPLAT_VS,
            fragmentShader: SPLAT_FS,
            uniforms: {
                uWork0: { value: this._work.textures[0] },
                uWork1: { value: this._work.textures[1] },
                uViewport: { value: new THREE.Vector4() },
                uMinPixelSize: { value: 2 },
                uAlphaClip: { value: 1 / 255 },
            },
            transparent: true,
            side: THREE.DoubleSide, // the corner basis (v1, v2) is mirrored
            depthTest: true,
            depthWrite: false,
            blending: THREE.CustomBlending,
            blendEquation: THREE.AddEquation,
            blendSrc: THREE.OneFactor,
            blendDst: THREE.OneMinusSrcAlphaFactor,
            blendSrcAlpha: THREE.OneFactor,
            blendDstAlpha: THREE.OneMinusSrcAlphaFactor,
        });
        const object = new THREE.Mesh(geometry, material);
        object.name = this.options.name;
        object.frustumCulled = false;
        object.rotation.z = Math.PI;
        object.onBeforeRender = (renderer, scene, camera) => this._update(camera);
        this.object = object;
        // compile both programs off the main thread (KHR_parallel_shader_compile) rather than on the first frame
        await Promise.all([
            this.renderer.compileAsync(this._workScene, this._workCamera),
            this.renderer.compileAsync(object, this._workCamera),
        ]);
        if (this._destroyed) throw new Error("AsogThreePlayer destroyed while loading");

        this._worker = new Worker(URL.createObjectURL(new Blob([sortWorkerSource()], { type: "application/javascript" })));
        this._worker.onmessage = (e) => this._onSorted(e.data);
        const clips = data.clips.slice();
        this._worker.postMessage({
            init: true, n, centers: data.centers.buffer, A: data.motion.A.buffer, B: data.motion.B.buffer, clips: clips.buffer,
        }, [data.centers.buffer, data.motion.A.buffer, data.motion.B.buffer, clips.buffer]);
        this._sortSpare = new Uint32Array(n).buffer;

        this.stats = { ...data.stats, loadMs: performance.now() - t0 };
        parent.add(object);
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
        this._sound?.setVolume(value);
    }

    /** true once the soundtrack is decoded and attached */
    get hasAudio() {
        return !!this._sound;
    }

    /** the soundtrack's listener (null without a soundtrack) */
    get listener() {
        return this._listener;
    }

    // Uploads one decoded array as a texture now (the decoder frees the array after onArray) and drops
    // the CPU copy. The texture is not re-uploaded after that; the player does not survive a lost context.
    _texture(array, width, height, format) {
        const T = THREE;
        const [fmt, type] = {
            rgba8: [T.RGBAFormat, T.UnsignedByteType],
            rgba32u: [T.RGBAIntegerFormat, T.UnsignedIntType],
            rgba16f: [T.RGBAFormat, T.HalfFloatType],
            rgba32f: [T.RGBAFormat, T.FloatType],
        }[format];
        const tex = new THREE.DataTexture(array, width, height, fmt, type);
        if (format === "rgba32u") tex.internalFormat = "RGBA32UI";
        tex.minFilter = tex.magFilter = THREE.NearestFilter;
        tex.generateMipmaps = false;
        tex.needsUpdate = true;
        this.renderer.initTexture(tex);
        tex.image = { width, height, data: null };
        this._textures.push(tex);
        return tex;
    }

    // The soundtrack plays at the clip, restarted with every loop of the clip (_syncAudio). Not a
    // looping sound: a sound file is rarely exactly as long as the clip (an mp3 of a 7.1 s clip decodes
    // to 7.082 s), and a looping one would run ahead by the difference every loop.
    async _startAudio(bytes) {
        const o = this.options;
        this._ownListener = !o.listener;
        const listener = o.listener ?? new THREE.AudioListener();
        this._listener = listener;
        const own = bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength ? bytes.buffer : bytes.slice().buffer;
        const buffer = await listener.context.decodeAudioData(own);
        if (this._destroyed) return;
        const sound = o.positional ? new THREE.PositionalAudio(listener) : new THREE.Audio(listener);
        sound.setBuffer(buffer);
        sound.setLoop(false);
        sound.setVolume(o.volume);
        if (o.positional) {
            const p = /** @type {THREE.PositionalAudio} */ (sound);
            p.setDistanceModel("linear");
            p.setRefDistance(o.refDistance);
            p.setMaxDistance(o.maxDistance);
        }
        this.object.add(sound);
        this._sound = sound;
        this._soundLength = Math.min(buffer.duration, this.duration);
    }

    _syncAudio() {
        const sound = this._sound;
        const context = sound.context;
        // the audio clock stands still until a user gesture unlocks audio
        if (context.state !== "running" || !this.object.visible) return;
        // a clock that stands still (paused) stops the sound; it restarts at the clip time
        if (this.time === this._prevTime) {
            if (sound.isPlaying) sound.stop();
            return;
        }
        // what is heard now left the audio graph outputLatency ago
        const target = (this.time + (context.baseLatency || 0) + (context.outputLatency || 0)) % this.duration;
        const wrapped = target < this._audioTarget; // the clip looped (or was scrubbed back): start over
        this._audioTarget = target;
        // a sound shorter than the clip is silent for the rest of the loop
        if (target >= this._soundLength) return;
        if (wrapped && sound.isPlaying) sound.stop();
        if (!sound.isPlaying) {
            this._playAudioAt(target);
            return;
        }
        const drift = target - (context.currentTime - this._audioStart);
        this.audioDrift = drift;
        if (Math.abs(drift) > AUDIO_MAX_DRIFT) {
            this._playAudioAt(target);
            this.audioSeeks++;
        }
    }

    _playAudioAt(t) {
        const sound = this._sound;
        if (sound.isPlaying) sound.stop();
        sound.offset = t;
        sound.play();
        this._audioStart = sound.context.currentTime - t;
    }

    // per frame (onBeforeRender of the clip's object): time, work pass, sort request, soundtrack
    _update(camera) {
        const d = this.duration;
        this._prevTime = this.time;
        this.time = ((this.options.getTime() % d) + d) % d;
        const object = this.object;
        const renderer = this.renderer;

        // the camera position in model space, for the SH colours
        this._modelInv.copy(object.matrixWorld).invert();
        this._cameraPos.setFromMatrixPosition(camera.matrixWorld).applyMatrix4(this._modelInv);
        if (this.time !== this._workTime || !this._cameraPos.equals(this._workCameraPos)) {
            this._workTime = this.time;
            this._workCameraPos.copy(this._cameraPos);
            this._runWorkPass();
        }
        renderer.getCurrentViewport(this._viewport);
        const vp = this._viewport;
        object.material.uniforms.uViewport.value.set(vp.z, vp.w, 1 / vp.z, 1 / vp.w);

        // a new order when the time or the view moved and the worker is free
        this._modelView.multiplyMatrices(camera.matrixWorldInverse, object.matrixWorld);
        const e = this._modelView.elements;
        const row = [-e[2], -e[6], -e[10], -e[14]];
        if (!this._sortBusy && (this.time !== this._sortTime || row.some((v, i) => Math.abs(v - this._sortRow[i]) > 1e-4))) {
            this._sortBusy = true;
            this._sortTime = this.time;
            this._sortRow = row;
            const indices = this._sortSpare;
            this._sortSpare = null;
            // every splat of the active clips, not only those visible now: an order is drawn until the next one
            // arrives (tens to hundreds of ms, more while scrubbing), and the work pass gives a splat that is not
            // visible at the current time zero opacity, so the vertex shader drops it
            this._worker.postMessage({ sort: true, time: this.time, depthRow: row, indices, margin: Infinity }, [indices]);
        }

        if (this._sound) {
            if (this._ownListener && this._listener.parent !== camera) camera.add(this._listener);
            this._syncAudio();
        }
    }

    // like three's Reflector: a nested render into the work targets, then back to the current target
    _runWorkPass() {
        const renderer = this.renderer;
        const uniforms = this._workScene.children[0].material.uniforms;
        uniforms.uAsogTime.value = this.time;
        uniforms.uCameraPosition.value.copy(this._cameraPos);
        const target = renderer.getRenderTarget();
        const autoClear = renderer.autoClear;
        const xr = renderer.xr.enabled;
        const shadows = renderer.shadowMap.autoUpdate;
        renderer.autoClear = false; // the pass writes every texel (and an integer target cannot be cleared to a float colour)
        renderer.xr.enabled = false;
        renderer.shadowMap.autoUpdate = false;
        renderer.setRenderTarget(this._work);
        renderer.render(this._workScene, this._workCamera);
        renderer.setRenderTarget(target);
        renderer.autoClear = autoClear;
        renderer.xr.enabled = xr;
        renderer.shadowMap.autoUpdate = shadows;
    }

    _onSorted({ count, indices }) {
        this._sortBusy = false;
        if (!this.object) return;
        const geometry = this.object.geometry;
        const order = /** @type {THREE.InstancedBufferAttribute} */ (geometry.getAttribute("splatIndex"));
        // the previous order was uploaded by the frame that posted this sort: it is free for the next one
        this._sortSpare = order.array.buffer;
        order.array = new Uint32Array(indices);
        order.clearUpdateRanges();
        order.addUpdateRange(0, count);
        order.needsUpdate = true;
        geometry.instanceCount = count;
        this.numVisible = count;
    }

    /** Stops playing and frees everything (also while loading: load() then rejects). */
    destroy() {
        if (this._destroyed) return;
        this._destroyed = true;
        this._abort.abort(new Error("AsogThreePlayer destroyed while loading"));
        this._worker?.terminate();
        this._worker = null;
        if (this._sound) {
            if (this._sound.isPlaying) this._sound.stop();
            this._sound.disconnect();
            this._sound.removeFromParent();
            this._sound = null;
        }
        if (this._ownListener) this._listener?.removeFromParent();
        this._listener = null;
        const object = this.object;
        if (object) {
            object.removeFromParent();
            object.geometry.dispose();
            object.material.dispose();
            this.object = null;
        }
        if (this._workScene) {
            const mesh = this._workScene.children[0];
            mesh.geometry.dispose();
            mesh.material.dispose();
            this._workScene = null;
        }
        this._work?.dispose();
        this._work = null;
        for (const t of this._textures) t.dispose();
        this._textures = [];
    }
}
