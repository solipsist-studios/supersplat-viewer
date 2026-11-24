/**
 * Decodes an .asog file into GPU-ready arrays (layout.js), one stream at a time while it downloads.
 */
import { HEADER_SIZE, parseHeader, parseManifest, shBandsFromCoeffs } from "./container.js";
import {
    dequantAnchor, dequantG, dequantMu, dequantOmega, dequantPosition, dequantSigma,
    gStep, lifeSpan, motionWeights, omegaStep,
} from "./dequant.js";
import { HALF_MAX, floatToHalf, getHalfTable } from "./half.js";
import { inflate } from "./inflate.js";
import {
    ARRAY_FORMATS, MOTION_B_TEXELS, MOTION_TEX_WIDTH, clipTable, motionTextureRows, shCentroidsSize, sogTextureSize,
} from "./layout.js";
import { decodeChannel } from "./planes.js";
import { openReader } from "./reader.js";
import { createPacer, free } from "./util.js";

/** The streams every v1 file has (the two SH streams are optional). */
const REQUIRED_STREAMS = ["means", "quats", "quatmode", "scales", "sh0", "opacity", "mu", "logsigma", "anchor", "omega", "g", "flags"];

/** Streams whose decoding reads what another stream decoded. */
const STREAM_NEEDS = { anchor: ["means"], omega: ["anchor", "logsigma"], g: ["omega", "logsigma"] };

/** The streams each output array is made of; it is emitted once they are all decoded. */
const ARRAY_STREAMS = {
    means_l: ["means"],
    means_u: ["means"],
    quats: ["quats", "quatmode"],
    scales: ["scales"],
    sh0: ["sh0", "opacity"],
    sh_labels: ["shlabels"],
    sh_centroids: ["shpalette"],
    motionA: ["mu", "logsigma", "anchor", "flags"],
    motionB: ["omega", "g"],
    centers: ["means", "anchor"], // the anchor's Omega step reads the centres
};

/**
 * @typedef {object} AsogFile
 * @property {import("./reader.js").ByteReader} reader
 * @property {import("./container.js").AsogHeader} header
 * @property {import("./container.js").AsogManifest} manifest
 */

/**
 * @typedef {keyof typeof ARRAY_FORMATS} AsogArrayName
 */

/**
 * @typedef {object} AsogArrayInfo
 * @property {number} width - texels per row (splats for `centers`)
 * @property {number} height - rows
 * @property {"rgba8" | "rgba32u" | "rgba16f" | "xyz32f"} format
 */

/**
 * @typedef {object} AsogDecodeOptions
 * @property {(progress: number) => void} [onProgress] - download progress 0..1 (when the size is known)
 * @property {RequestInit} [fetchOptions] - for a URL source
 * @property {(name: AsogArrayName, data: Uint8Array | Uint16Array | Uint32Array | Float32Array, info: AsogArrayInfo) => void | Promise<void>} [onArray]
 *   called as soon as an array is complete (e.g. to upload it as a texture); awaited
 * @property {AsogArrayName[]} [keep] - arrays to keep in the result; the others are freed right after
 *   onArray. Default: keep all.
 * @property {boolean} [audio] - read the soundtrack (default true)
 * @property {AbortSignal} [signal]
 * @property {number} [yieldMs] - longest stretch of decoding between yields to the event loop (default 25)
 */

/**
 * @typedef {object} AsogData
 * @property {import("./container.js").AsogManifest} manifest
 * @property {import("./container.js").AsogHeader} header
 * @property {number} count - splats
 * @property {number} duration - timeline.durationSeconds
 * @property {number} shBands - 0..3
 * @property {{ width: number, height: number, shWidth: number, shHeight: number,
 *   means_l?: Uint8Array, means_u?: Uint8Array, quats?: Uint8Array, scales?: Uint8Array, sh0?: Uint8Array,
 *   sh_labels?: Uint8Array, sh_centroids?: Uint8Array }} sog - SOG v2 RGBA8 texture data
 * @property {{ width: number, rowsA: number, rowsB: number, A?: Uint32Array, B?: Uint16Array }} motion - motion textures
 * @property {Float32Array} clips - clip table (layout.js)
 * @property {Float32Array | null} centers - canonical centres, xyz per splat
 * @property {{ mime: string, data: Uint8Array } | null} audio - the soundtrack file
 * @property {{ fileBytes: number, decodeMs: number, halfOverflow: number }} stats - halfOverflow: motion values
 *   beyond the float16 range, clamped to +-65504
 */

/**
 * Opens a source and reads its header and manifest.
 * @param {import("./reader.js").AsogSource} source
 * @param {Pick<AsogDecodeOptions, "onProgress" | "fetchOptions">} [options]
 * @returns {Promise<AsogFile>}
 */
export async function openAsog(source, { onProgress, fetchOptions } = {}) {
    const reader = await openReader(source, { onProgress, fetchOptions });
    try {
        const header = parseHeader(await reader.read(0, HEADER_SIZE));
        const manifest = parseManifest(await reader.read(header.manifestOffset, header.manifestLength));
        return { reader, header, manifest };
    } catch (e) {
        reader.cancel();
        throw e;
    }
}

/**
 * Decodes a whole .asog file.
 * @param {import("./reader.js").AsogSource | AsogFile} source - a source, or a file from openAsog()
 * @param {AsogDecodeOptions} [options]
 * @returns {Promise<AsogData>}
 */
export async function decodeAsog(source, options = {}) {
    const file = /** @type {any} */ (source)?.reader && /** @type {any} */ (source).manifest
        ? /** @type {AsogFile} */ (source)
        : await openAsog(/** @type {import("./reader.js").AsogSource} */ (source), options);
    try {
        return await decodeFile(file, options);
    } finally {
        file.reader.cancel();
    }
}

/**
 * @param {AsogFile} file
 * @param {AsogDecodeOptions} options
 * @returns {Promise<AsogData>}
 */
async function decodeFile({ reader, header, manifest }, { onArray, keep, audio = true, signal, yieldMs = 25 }) {
    const t0 = performance.now();
    const n = manifest.count;
    const present = new Set(manifest.streams.map((s) => s.name));
    for (const name of REQUIRED_STREAMS) {
        if (!present.has(name)) throw new Error(`asog: the file has no "${name}" stream`);
    }
    const hasSH = !!manifest.shN && present.has("shlabels") && present.has("shpalette");

    const { width, height } = sogTextureSize(n);
    const { rowsA, rowsB } = motionTextureRows(n);
    const shSize = hasSH ? shCentroidsSize(manifest.shN) : { width: 0, height: 0 };
    /** @type {Record<string, [number, number]>} */
    const dims = {
        means_l: [width, height], means_u: [width, height], quats: [width, height], scales: [width, height],
        sh0: [width, height], sh_labels: [width, height], sh_centroids: [shSize.width, shSize.height],
        motionA: [MOTION_TEX_WIDTH, rowsA], motionB: [MOTION_TEX_WIDTH, rowsB], centers: [n, 1],
    };
    /** @type {Record<string, any>} */
    const arrays = {};
    const get = (name) => {
        if (!arrays[name]) {
            const [w, h] = dims[name];
            const format = ARRAY_FORMATS[name];
            arrays[name] = format === "rgba8" ? new Uint8Array(w * h * 4)
                : format === "rgba32u" ? new Uint32Array(w * h * 4)
                : format === "rgba16f" ? new Uint16Array(w * h * 4)
                : new Float32Array(n * 3);
        }
        return arrays[name];
    };

    /** @type {AsogData} */
    const result = {
        manifest, header, count: n,
        duration: manifest.timeline.durationSeconds,
        shBands: hasSH ? shBandsFromCoeffs(manifest.shN.coeffs) : 0,
        sog: { width, height, shWidth: shSize.width, shHeight: shSize.height },
        motion: { width: MOTION_TEX_WIDTH, rowsA, rowsB },
        clips: clipTable(manifest),
        centers: null,
        audio: null,
        stats: { fileBytes: header.payloadOffset + header.payloadLength, decodeMs: 0, halfOverflow: 0 },
    };
    const keepSet = keep ? new Set(keep) : null;
    const remaining = {};
    for (const [name, streams] of Object.entries(ARRAY_STREAMS)) {
        if (hasSH || !name.startsWith("sh_")) remaining[name] = streams.length;
    }
    const decoded = new Set();

    const emit = async (name) => {
        const data = arrays[name];
        arrays[name] = null;
        const [w, h] = dims[name];
        if (onArray) await onArray(/** @type {AsogArrayName} */ (name), data, { width: w, height: h, format: ARRAY_FORMATS[name] });
        if (keepSet && !keepSet.has(name)) free(data);
        else if (name === "motionA") result.motion.A = data;
        else if (name === "motionB") result.motion.B = data;
        else if (name === "centers") result.centers = data;
        else result.sog[name] = data;
    };
    const complete = async (stream) => {
        decoded.add(stream);
        for (const [name, streams] of Object.entries(ARRAY_STREAMS)) {
            if (name in remaining && streams.includes(stream) && --remaining[name] === 0) await emit(name);
        }
    };

    const pace = createPacer(yieldMs);
    const half = getHalfTable();
    const v = new Float64Array(Math.max(n, hasSH ? manifest.shN.count : 0));
    const channel = async (s, raw, ch) => {
        await pace();
        decodeChannel(s, raw, ch, v);
    };
    const toHalf = (x) => {
        if (Math.abs(x) > HALF_MAX) result.stats.halfOverflow++;
        return floatToHalf(x);
    };
    const { mins, maxs } = manifest.means;
    const mo = manifest.motion;
    const { wO, wG } = motionWeights(mo);
    const BS = MOTION_B_TEXELS * 4; // halves per splat in motionB
    /** @type {Float32Array} */ let L;      // visible half-life per splat
    /** @type {Float32Array} */ let step;   // Omega step, then G step
    const floatsA = () => new Float32Array(get("motionA").buffer);

    /** @type {Record<string, (s: import("./container.js").AsogStreamDesc, raw: Uint8Array) => Promise<void>>} */
    const decoders = {
        async means(s, raw) {
            const lo = get("means_l"), hi = get("means_u"), centers = get("centers");
            for (let c = 0; c < 3; c++) {
                await channel(s, raw, c);
                for (let i = 0; i < n; i++) {
                    lo[i * 4 + c] = v[i] & 255;
                    hi[i * 4 + c] = v[i] >> 8;
                    centers[i * 3 + c] = dequantPosition(v[i], mins[c], maxs[c]);
                }
            }
        },
        async quats(s, raw) {
            const d = get("quats");
            for (let c = 0; c < 3; c++) {
                await channel(s, raw, c);
                for (let i = 0; i < n; i++) d[i * 4 + c] = v[i];
            }
        },
        async quatmode(s, raw) {
            const d = get("quats");
            await channel(s, raw, 0);
            for (let i = 0; i < n; i++) d[i * 4 + 3] = 252 + v[i];
        },
        async scales(s, raw) {
            const d = get("scales");
            for (let c = 0; c < 3; c++) {
                await channel(s, raw, c);
                for (let i = 0; i < n; i++) d[i * 4 + c] = v[i];
            }
        },
        async sh0(s, raw) {
            const d = get("sh0");
            for (let c = 0; c < 3; c++) {
                await channel(s, raw, c);
                for (let i = 0; i < n; i++) d[i * 4 + c] = v[i];
            }
        },
        async opacity(s, raw) {
            const d = get("sh0");
            await channel(s, raw, 0);
            for (let i = 0; i < n; i++) d[i * 4 + 3] = v[i];
        },
        async shlabels(s, raw) {
            const d = get("sh_labels");
            await channel(s, raw, 0);
            for (let i = 0; i < n; i++) {
                d[i * 4] = v[i] & 255;
                d[i * 4 + 1] = v[i] >> 8;
            }
        },
        async shpalette(s, raw) {
            const { count, coeffs } = manifest.shN;
            const pw = shSize.width, d = get("sh_centroids");
            for (let ch = 0; ch < coeffs * 3; ch++) {
                await channel(s, raw, ch);
                const j = Math.floor(ch / 3), c = ch % 3;
                for (let k = 0; k < count; k++) d[((k >> 6) * pw + (k & 63) * coeffs + j) * 4 + c] = v[k];
            }
        },
        async mu(s, raw) {
            const Af = floatsA();
            await channel(s, raw, 0);
            for (let i = 0; i < n; i++) Af[i * 4] = dequantMu(v[i], manifest);
        },
        async logsigma(s, raw) {
            const Af = floatsA();
            L = new Float32Array(n);
            await channel(s, raw, 0);
            for (let i = 0; i < n; i++) {
                const sigma = dequantSigma(v[i], manifest);
                Af[i * 4 + 1] = sigma;
                L[i] = lifeSpan(sigma, mo);
            }
        },
        async anchor(s, raw) {
            const A = get("motionA"), centers = arrays.centers;
            step = new Float32Array(n); // |x - a|^2 first
            for (let c = 0; c < 3; c++) {
                await channel(s, raw, c);
                for (let i = 0; i < n; i++) {
                    const a = dequantAnchor(v[i], mo), d = centers[i * 3 + c] - a;
                    step[i] += d * d;
                    if (c < 2) A[i * 4 + 2] |= toHalf(a) << (16 * c);
                    else A[i * 4 + 3] |= toHalf(a);
                }
            }
            for (let i = 0; i < n; i++) step[i] = omegaStep(Math.sqrt(step[i]), mo);
        },
        async omega(s, raw) {
            const B = get("motionB");
            for (let ch = 0; ch < 9; ch++) {
                await channel(s, raw, ch);
                const k = Math.floor(ch / 3);
                for (let i = 0; i < n; i++) B[i * BS + ch] = toHalf(dequantOmega(v[i], k, step[i], L[i], wO));
            }
            // the G step, from the control points as stored (float16), as spec 4.2 asks of reduced-precision decoders
            const O = new Float64Array(9);
            for (let i = 0; i < n; i++) {
                if ((i & 0x3fff) === 0) await pace();
                for (let k = 0; k < 9; k++) O[k] = half[B[i * BS + k]];
                step[i] = gStep(O, L[i], mo);
            }
        },
        async g(s, raw) {
            const B = get("motionB");
            for (let j = 0; j < 5; j++) {
                await channel(s, raw, j);
                for (let i = 0; i < n; i++) B[i * BS + 9 + j] = toHalf(dequantG(v[i], j, step[i], L[i], wG));
            }
            free(step, L);
        },
        async flags(s, raw) {
            const A = get("motionA");
            await channel(s, raw, 0);
            for (let i = 0; i < n; i++) A[i * 4 + 3] |= ((v[i] & 1) << 16) | ((v[i] >> 1) << 17);
        },
    };

    for (const s of [...manifest.streams].sort((a, b) => a.byteOffset - b.byteOffset)) {
        if (signal?.aborted) throw signal.reason ?? new Error("asog: aborted");
        const decode = decoders[s.name];
        if (!decode || (!hasSH && (s.name === "shlabels" || s.name === "shpalette"))) continue;
        const rows = s.name === "shpalette" ? manifest.shN.count : n;
        const channels = s.name === "shpalette" ? manifest.shN.coeffs * 3 : { means: 3, quats: 3, scales: 3, sh0: 3, anchor: 3, omega: 9, g: 5 }[s.name] ?? 1;
        if (s.count !== rows || s.channels !== channels) {
            throw new Error(`asog: stream "${s.name}" is ${s.count} x ${s.channels}, expected ${rows} x ${channels}`);
        }
        for (const need of STREAM_NEEDS[s.name] ?? []) {
            if (!decoded.has(need)) throw new Error(`asog: stream "${s.name}" comes before "${need}", which it needs`);
        }
        const bytes = await reader.read(header.payloadOffset + s.byteOffset, s.byteLength);
        const raw = await inflate(bytes, s.count * s.channels * s.planes);
        free(bytes);
        await decode(s, raw);
        free(raw);
        await complete(s.name);
    }

    if (manifest.audio && audio) {
        if (signal?.aborted) throw signal.reason ?? new Error("asog: aborted");
        const data = await reader.read(header.payloadOffset + manifest.audio.byteOffset, manifest.audio.byteLength);
        result.audio = { mime: manifest.audio.mime, data };
    }
    result.stats.decodeMs = performance.now() - t0;
    return result;
}
