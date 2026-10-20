// Worker that turns a .sogst group's compressed payloads into the exact
// texel words the GPU resource stores, off the main thread.
//
// A job carries a group's WebP payloads and a splat range within the group.
// The worker decodes the WebP textures, unpacks the SOG attributes (the same
// math as the engine's GSplatSogIterator) and the temporal attributes, and
// packs them into the stream textures' layouts (the same math as the
// engine's GSplatResource updates). The main thread only copies the returned
// rows into the textures' CPU copies and uploads them.
//
// The previous pipeline unpacked into Float32Arrays and packed from those,
// so every value passed through float32 between the two steps. Math.fround
// below marks each of those points; without it a few half-float words would
// differ by one ulp from before.
//
// sogstPackWorker must stay self-contained: the pool runs it from its own
// source text (`(${sogstPackWorker})()`), so it can reference globals only,
// never anything else in this module.

type PackMeta = {
    means: { mins: number[]; maxs: number[] };
    scales: { codebook: number[] };
    sh0: { codebook: number[] };
    shN: { codebook: number[]; bands: number } | null;
    motion: { mins: number[]; maxs: number[] };
    accel: { mins: number[]; maxs: number[] } | null;
    trbf: { center: number[]; sigma: number[] };
};

type PackJobMessage = {
    type: 'pack';
    id: number;
    // payloads keyed by bare canonical name: WebP bytes, or texels the main
    // thread already decoded (when the worker has no WebGL2)
    files: Record<string, Uint8Array | RawTexels>;
    start: number;
    count: number;
    // unpack geometry and temporal attributes (false: SH only)
    geometry: boolean;
    // unpack SH (needs shN_labels.webp in files)
    sh: boolean;
    // 'texels': only decode the WebP files and return their texels (plus
    // centers for [start, start + count) when the means pair is in files),
    // for the GPU decode path (sogst-gpu-decode.ts), which unpacks and packs
    // on the GPU
    mode?: 'pack' | 'texels';
};

type RawTexels = { width: number; data: Uint8Array };

type PackInitMessage = {
    type: 'init';
    meta: PackMeta;
    centroids: Uint8Array | RawTexels | null;
};

type PackedRows = {
    color?: Uint16Array;
    transformA?: Uint32Array;
    transformB?: Uint16Array;
    centers?: Float32Array;
    motion?: Float32Array;
    temporal?: Float32Array;
    accel?: Float32Array;
    // One per engine SH stream texture (splatSH_1to3 ... splatSH_12to15).
    // Degree 3, the highest SOG stores, fills all four; lower degrees leave
    // the later ones unset.
    sh1to3?: Uint32Array;
    sh4to7?: Uint32Array;
    sh8to11?: Uint32Array;
    sh12to15?: Uint32Array;
    // 'texels' mode: decoded payloads keyed by file name
    texels?: Record<string, RawTexels>;
};

function sogstPackWorker() {
    const ctx = self as unknown as {
        onmessage: (e: MessageEvent) => void;
        postMessage: (message: unknown, transfer?: Transferable[]) => void;
    };

    // -- WebP decode -------------------------------------------------------
    // WebGL readback, not a 2D canvas: getImageData premultiplies alpha, and
    // sh0 stores opacity in alpha.
    let gl: WebGL2RenderingContext | null = null;
    let tex: WebGLTexture | null = null;
    let fbo: WebGLFramebuffer | null = null;
    const decodeWebp = async (payload: Uint8Array | RawTexels): Promise<RawTexels> => {
        if (!(payload instanceof Uint8Array)) {
            return payload;
        }
        const bytes = payload;
        const bitmap = await createImageBitmap(new Blob([bytes as unknown as ArrayBuffer], { type: 'image/webp' }), {
            premultiplyAlpha: 'none',
            colorSpaceConversion: 'none'
        });
        const w = bitmap.width;
        const h = bitmap.height;
        if (!gl) {
            const canvas = new OffscreenCanvas(1, 1);
            gl = canvas.getContext('webgl2', { antialias: false, depth: false }) as WebGL2RenderingContext | null;
            if (!gl) {
                throw new Error('no webgl2 in worker');
            }
            tex = gl.createTexture();
            fbo = gl.createFramebuffer();
        }
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
        gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.NONE);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, bitmap);
        bitmap.close();
        gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
        const data = new Uint8Array(w * h * 4);
        gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, data);
        return { width: w, data };
    };

    // -- float32 to float16 bits ------------------------------------------
    // The engine's FloatPacking.float2Half, with its constants named. The
    // worker must produce the same bits as the engine's own packing, but
    // cannot import it (see the header).
    // Rounds to nearest, ties away from zero, from the first dropped bit.
    const floatView = new Float32Array(1);
    const int32View = new Int32Array(floatView.buffer);
    // float32 exponent bias (127) minus float16's (15)
    const EXP_REBIAS = 112;
    const HALF_SIGN = 0x8000;
    const HALF_INFINITY = 0x7c00;
    const HALF_MANTISSA_BITS = 10;
    const FLOAT_MANTISSA_BITS = 23;
    const FLOAT_MANTISSA_MASK = 0x7fffff;
    const FLOAT_EXP_NAN = 255;
    // Biased float32 exponents: below this the value rounds to zero, at or
    // below this it is subnormal as a half, above this it overflows.
    const HALF_ZERO_BELOW = EXP_REBIAS - HALF_MANTISSA_BITS + 1;
    const HALF_SUBNORMAL_MAX = EXP_REBIAS;
    const HALF_NORMAL_MAX = EXP_REBIAS + 30;
    const float2Half = (value: number) => {
        floatView[0] = value;
        const x = int32View[0];
        const sign = (x >> 16) & HALF_SIGN;
        const e = (x >> FLOAT_MANTISSA_BITS) & 0xff;
        // the half's 10 mantissa bits plus one rounding bit below them
        let m = (x >> (FLOAT_MANTISSA_BITS - HALF_MANTISSA_BITS - 1)) & 0x7ff;
        if (e < HALF_ZERO_BELOW) {
            return sign;
        }
        if (e > HALF_NORMAL_MAX) {
            // Infinity. Like the engine, a finite overflow also keeps the
            // float's low mantissa bits, and NaN becomes infinity.
            return sign | HALF_INFINITY | (e === FLOAT_EXP_NAN ? 0 : x & FLOAT_MANTISSA_MASK);
        }
        if (e <= HALF_SUBNORMAL_MAX) {
            // subnormal: restore the implicit leading 1, then shift right
            // by how far the exponent is below the half's normal range
            m |= 1 << (HALF_MANTISSA_BITS + 1);
            const shift = HALF_SUBNORMAL_MAX + 2 - e;
            return sign | ((m >> shift) + ((m >> (shift - 1)) & 1));
        }
        return (sign | ((e - EXP_REBIAS) << HALF_MANTISSA_BITS) | (m >> 1)) + (m & 1);
    };

    const f32 = Math.fround;
    const SH_C0 = 0.28209479177387814;
    const OPACITY_LOGIT_LIMIT = 40;
    const SH_COEFFS: Record<number, number> = { 1: 3, 2: 8, 3: 15 };
    // smallest-three rotation; see QUAT_MODE_BASE in sogst-gpu-decode.ts
    const QUAT_MODE_BASE = 252;
    // split-16 attributes: a 16-bit fraction as low and high bytes
    const U16_MAX = 65535;
    // shN_centroids.webp: centroids per row, one texel per coefficient
    const CENTROIDS_PER_ROW = 64;
    // 11:10:11 SH word layout (see gsplat-range-sync.ts)
    const SH_R_MAX = 2047;
    const SH_G_MAX = 1023;
    const SH_B_MAX = 2047;
    const SH_R_SHIFT = 21;
    const SH_G_SHIFT = 11;

    let meta: PackMeta | null = null;
    let centroidsBytes: Uint8Array | RawTexels | null = null;
    let centroids: RawTexels | null = null;

    const lerp = (a: number, b: number, t: number) => a * (1 - t) + b * t;
    const unlog = (v: number) => Math.sign(v) * (Math.exp(Math.abs(v)) - 1);

    const pack = async (job: PackJobMessage): Promise<PackedRows> => {
        const m = meta!;
        const n = job.count;
        const s0 = job.start;
        if (job.mode === 'texels') {
            const texels: Record<string, RawTexels> = {};
            for (const name of Object.keys(job.files)) {
                texels[name] = await decodeWebp(job.files[name]);
            }
            const out: PackedRows = { texels };
            const meansL = texels['means_l.webp']?.data;
            const meansU = texels['means_u.webp']?.data;
            if (meansL && meansU && n > 0) {
                // the same positions the geometry pass computes, for the CPU
                // sorter, which reads them from the resource's centers
                const { mins, maxs } = m.means;
                const centers = new Float32Array(n * 3);
                for (let j = 0; j < n; j++) {
                    const i = (s0 + j) * 4;
                    for (let ch = 0; ch < 3; ch++) {
                        centers[j * 3 + ch] = f32(
                            unlog(lerp(mins[ch], maxs[ch], ((meansU[i + ch] << 8) + meansL[i + ch]) / U16_MAX))
                        );
                    }
                }
                out.centers = centers;
            }
            return out;
        }
        const tex8: Record<string, Uint8Array> = {};
        for (const name of Object.keys(job.files)) {
            tex8[name] = (await decodeWebp(job.files[name])).data;
        }
        const out: PackedRows = {};

        if (job.geometry) {
            const meansL = tex8['means_l.webp'];
            const meansU = tex8['means_u.webp'];
            const quats = tex8['quats.webp'];
            const scales = tex8['scales.webp'];
            const sh0 = tex8['sh0.webp'];
            const color = new Uint16Array(n * 4);
            const transformA = new Uint32Array(n * 4);
            const transformAf = new Float32Array(transformA.buffer);
            const transformB = new Uint16Array(n * 4);
            const centers = new Float32Array(n * 3);
            const { mins, maxs } = m.means;
            const scb = m.scales.codebook;
            const ccb = m.sh0.codebook;
            for (let j = 0; j < n; j++) {
                const i = (s0 + j) * 4;
                // position
                const x = f32(unlog(lerp(mins[0], maxs[0], ((meansU[i] << 8) + meansL[i]) / U16_MAX)));
                const y = f32(unlog(lerp(mins[1], maxs[1], ((meansU[i + 1] << 8) + meansL[i + 1]) / U16_MAX)));
                const z = f32(unlog(lerp(mins[2], maxs[2], ((meansU[i + 2] << 8) + meansL[i + 2]) / U16_MAX)));
                // rotation, smallest-three
                const a = (quats[i] / 255 - 0.5) * Math.SQRT2;
                const b = (quats[i + 1] / 255 - 0.5) * Math.SQRT2;
                const c = (quats[i + 2] / 255 - 0.5) * Math.SQRT2;
                const d = Math.sqrt(Math.max(0, 1 - (a * a + b * b + c * c)));
                let qx = 0;
                let qy = 0;
                let qz = 0;
                let qw = 0;
                switch (quats[i + 3] - QUAT_MODE_BASE) {
                    case 0:
                        qx = a;
                        qy = b;
                        qz = c;
                        qw = d;
                        break;
                    case 1:
                        qx = d;
                        qy = b;
                        qz = c;
                        qw = a;
                        break;
                    case 2:
                        qx = b;
                        qy = d;
                        qz = c;
                        qw = a;
                        break;
                    case 3:
                        qx = b;
                        qy = c;
                        qz = d;
                        qw = a;
                        break;
                }
                qx = f32(qx);
                qy = f32(qy);
                qz = f32(qz);
                qw = f32(qw);
                const len = Math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
                if (len === 0) {
                    qx = 0;
                    qy = 0;
                    qz = 0;
                    qw = 1;
                } else {
                    const inv = 1 / len;
                    qx *= inv;
                    qy *= inv;
                    qz *= inv;
                    qw *= inv;
                }
                if (qw < 0) {
                    qx = -qx;
                    qy = -qy;
                    qz = -qz;
                }
                // scale (log), colour and opacity, through the old round trip
                const sx = f32(scb[scales[i]]);
                const sy = f32(scb[scales[i + 1]]);
                const sz = f32(scb[scales[i + 2]]);
                const dc0 = f32((0.5 + ccb[sh0[i]] * SH_C0 - 0.5) / SH_C0);
                const dc1 = f32((0.5 + ccb[sh0[i + 1]] * SH_C0 - 0.5) / SH_C0);
                const dc2 = f32((0.5 + ccb[sh0[i + 2]] * SH_C0 - 0.5) / SH_C0);
                const alpha = sh0[i + 3] / 255;
                const logit = f32(
                    alpha <= 0 ? -OPACITY_LOGIT_LIMIT : alpha >= 1 ? OPACITY_LOGIT_LIMIT : -Math.log(1 / alpha - 1)
                );

                const o = j * 4;
                color[o] = float2Half(dc0 * SH_C0 + 0.5);
                color[o + 1] = float2Half(dc1 * SH_C0 + 0.5);
                color[o + 2] = float2Half(dc2 * SH_C0 + 0.5);
                color[o + 3] = float2Half(1 / (1 + Math.exp(-logit)));
                transformAf[o] = x;
                transformAf[o + 1] = y;
                transformAf[o + 2] = z;
                transformA[o + 3] = float2Half(qx) | (float2Half(qy) << 16);
                transformB[o] = float2Half(Math.exp(sx));
                transformB[o + 1] = float2Half(Math.exp(sy));
                transformB[o + 2] = float2Half(Math.exp(sz));
                transformB[o + 3] = float2Half(qz);
                centers[j * 3] = x;
                centers[j * 3 + 1] = y;
                centers[j * 3 + 2] = z;
            }
            Object.assign(out, { color, transformA, transformB, centers });

            // temporal attributes: split-16 velocity (and accel), log space
            const split16 = (lo: Uint8Array, hi: Uint8Array, i: number, ch: number, min: number, max: number) => {
                const o = i * 4 + ch;
                return f32(unlog(min + ((max - min) * ((hi[o] << 8) + lo[o])) / U16_MAX));
            };
            const motionL = tex8['motion_l.webp'];
            const motionU = tex8['motion_u.webp'];
            const trbf = tex8['trbf.webp'];
            const motion = new Float32Array(n * 4);
            const temporal = new Float32Array(n);
            const vm = m.motion;
            for (let j = 0; j < n; j++) {
                const i = s0 + j;
                motion[j * 4] = split16(motionL, motionU, i, 0, vm.mins[0], vm.maxs[0]);
                motion[j * 4 + 1] = split16(motionL, motionU, i, 1, vm.mins[1], vm.maxs[1]);
                motion[j * 4 + 2] = split16(motionL, motionU, i, 2, vm.mins[2], vm.maxs[2]);
                motion[j * 4 + 3] = m.trbf.center[trbf[i * 4]];
                temporal[j] = m.trbf.sigma[trbf[i * 4 + 1]];
            }
            Object.assign(out, { motion, temporal });
            if (m.accel) {
                const accelL = tex8['accel_l.webp'];
                const accelU = tex8['accel_u.webp'];
                const accel = new Float32Array(n * 4);
                const am = m.accel;
                for (let j = 0; j < n; j++) {
                    const i = s0 + j;
                    for (let ch = 0; ch < 3; ch++) {
                        accel[j * 4 + ch] = split16(accelL, accelU, i, ch, am.mins[ch], am.maxs[ch]);
                    }
                }
                out.accel = accel;
            }
        }

        if (job.sh && m.shN) {
            if (!centroids) {
                centroids = await decodeWebp(centroidsBytes!);
            }
            const bands = m.shN.bands;
            const coeffs = SH_COEFFS[bands];
            const labels = tex8['shN_labels.webp'];
            const cb = m.shN.codebook;
            const cw = centroids.width;
            const cdata = centroids.data;
            const sh1to3 = new Uint32Array(n * 4);
            const sh4to7 = bands > 1 ? new Uint32Array(n * 4) : null;
            const sh8to11 = bands > 1 ? new Uint32Array(n * (bands > 2 ? 4 : 1)) : null;
            const sh12to15 = bands > 2 ? new Uint32Array(n * 4) : null;
            const c = new Array<number>(coeffs * 3).fill(0);
            const f32w = new Float32Array(1);
            const u32w = new Uint32Array(f32w.buffer);
            for (let j = 0; j < n; j++) {
                const i = (s0 + j) * 4;
                const label = labels[i] + (labels[i + 1] << 8);
                const u = (label % CENTROIDS_PER_ROW) * coeffs;
                const v = Math.floor(label / CENTROIDS_PER_ROW);
                // channel-interleaved, as the repack reads it: c[k*3 + ch]
                for (let ch = 0; ch < 3; ch++) {
                    for (let k = 0; k < coeffs; k++) {
                        c[k * 3 + ch] = f32(cb[cdata[(u + k) * 4 + ch + v * cw * 4]]);
                    }
                }
                let max = c[0];
                for (let k = 1; k < coeffs * 3; k++) {
                    max = Math.max(max, Math.abs(c[k]));
                }
                if (max === 0) {
                    continue; // words stay zero, which decodes to zero SH
                }
                for (let k = 0; k < coeffs; k++) {
                    c[k * 3] = Math.max(
                        0,
                        Math.min(SH_R_MAX, Math.floor(((c[k * 3] / max) * 0.5 + 0.5) * SH_R_MAX + 0.5))
                    );
                    c[k * 3 + 1] = Math.max(
                        0,
                        Math.min(SH_G_MAX, Math.floor(((c[k * 3 + 1] / max) * 0.5 + 0.5) * SH_G_MAX + 0.5))
                    );
                    c[k * 3 + 2] = Math.max(
                        0,
                        Math.min(SH_B_MAX, Math.floor(((c[k * 3 + 2] / max) * 0.5 + 0.5) * SH_B_MAX + 0.5))
                    );
                }
                const word = (k: number) => (c[k] << SH_R_SHIFT) | (c[k + 1] << SH_G_SHIFT) | c[k + 2];
                const o = j * 4;
                f32w[0] = max;
                sh1to3[o] = u32w[0];
                sh1to3[o + 1] = word(0);
                sh1to3[o + 2] = word(3);
                sh1to3[o + 3] = word(6);
                if (sh4to7 && sh8to11) {
                    sh4to7[o] = word(9);
                    sh4to7[o + 1] = word(12);
                    sh4to7[o + 2] = word(15);
                    sh4to7[o + 3] = word(18);
                    if (sh12to15) {
                        sh8to11[o] = word(21);
                        sh8to11[o + 1] = word(24);
                        sh8to11[o + 2] = word(27);
                        sh8to11[o + 3] = word(30);
                        sh12to15[o] = word(33);
                        sh12to15[o + 1] = word(36);
                        sh12to15[o + 2] = word(39);
                        sh12to15[o + 3] = word(42);
                    } else {
                        sh8to11[j] = word(21);
                    }
                }
            }
            out.sh1to3 = sh1to3;
            if (sh4to7) {
                out.sh4to7 = sh4to7;
            }
            if (sh8to11) {
                out.sh8to11 = sh8to11;
            }
            if (sh12to15) {
                out.sh12to15 = sh12to15;
            }
        }
        return out;
    };

    ctx.onmessage = async (e: MessageEvent) => {
        const msg = e.data as PackInitMessage | PackJobMessage;
        if (msg.type === 'init') {
            meta = msg.meta;
            centroidsBytes = msg.centroids;
            centroids = null;
            return;
        }
        try {
            const rows = await pack(msg);
            const transfer: ArrayBuffer[] = [];
            for (const value of Object.values(rows)) {
                if (ArrayBuffer.isView(value)) {
                    transfer.push(value.buffer as ArrayBuffer);
                } else if (value) {
                    for (const texels of Object.values(value as Record<string, RawTexels>)) {
                        transfer.push(texels.data.buffer as ArrayBuffer);
                    }
                }
            }
            ctx.postMessage({ id: msg.id, rows }, transfer);
        } catch (err) {
            ctx.postMessage({ id: msg.id, error: String(err) });
        }
    };
}

export { sogstPackWorker };
export type { PackedRows, PackInitMessage, PackJobMessage, PackMeta, RawTexels };
