import { GSplatData } from 'playcanvas';

// .queen binary format
// ─────────────────────────────────────────────────────────────────────────────
// Header (32 bytes, little-endian):
//   [0-3]   uint32  magic = 0x4E455551 ("QUEN")
//   [4-7]   uint32  version = 1
//   [8-11]  uint32  numSplats  (N)
//   [12-15] uint32  numFrames  (F)
//   [16-19] float32 fps
//   [20-23] float32 timeDurationMin
//   [24-27] float32 timeDurationMax
//   [28-31] uint32  shDegree   (0–3; controls f_rest channel count)
//
// Per-attribute decoder block (6 decoders, fixed order: xyz, f_dc, f_rest, sc, rot, op):
//   For each attribute a in [0..5]:
//     uint8   type:  0 = identity, 1 = latent, 2 = latent_res
//     uint32  latentDim   (input dimensionality; equals featureDim for identity)
//     uint32  featureDim  (output floats per splat for this attribute)
//     if type > 0 and featureDim > 0:
//       float32[latentDim × featureDim]  weight  (row-major)
//       uint8                             hasBias
//       if hasBias: float32[featureDim]   bias
//
// Frame 0 (dense base frame):
//   float32[N × stride]  per-splat data in AoS layout; stride = sum of featureDims
//     attribute order within stride: xyz, f_dc, f_rest, sc, rot, op
//
// Frames 1..F-1 (residual frames):
//   For each attribute a in [0..5] with featureDim > 0:
//     int16[N × latentDim]  quantLatents  (signed, little-endian)
//     float32               quantScale    (dequantize: float = int16 / quantScale)
//
// Decoder types:
//   identity  (0): output = dequantized latents (no matrix multiply; latentDim = featureDim)
//   latent    (1): output = dequant(latents) @ weight + bias
//   latent_res(2): same as latent, then output += previous frame's decoded output
// ─────────────────────────────────────────────────────────────────────────────

const MAGIC = 0x4E455551;   // little-endian uint32 for "QUEN"
const HEADER_SIZE = 32;

const DECODER_IDENTITY  = 0;
const DECODER_LATENT    = 1;
const DECODER_LATENT_RES = 2;

// Fixed attribute group indices.
const ATTR_XYZ   = 0;   // featureDim = 3
const ATTR_FDC   = 1;   // featureDim = 3
const ATTR_FREST = 2;   // featureDim = shRestCount (0 when shDegree = 0)
const ATTR_SC    = 3;   // featureDim = 3
const ATTR_ROT   = 4;   // featureDim = 4
const ATTR_OP    = 5;   // featureDim = 1

const NUM_ATTRS = 6;

interface QueenHeader {
    version: number;
    numSplats: number;
    numFrames: number;
    fps: number;
    timeDurationMin: number;
    timeDurationMax: number;
    shDegree: number;
}

interface AttributeDecoder {
    type: number;
    latentDim: number;
    featureDim: number;
    weight: Float32Array | null;    // [latentDim × featureDim], null for identity / empty attr
    bias: Float32Array | null;      // [featureDim] or null
}

// Number of f_rest SH channels for a given shDegree.
const computeShRestCount = (shDegree: number): number => (shDegree + 1) * (shDegree + 1) * 3 - 3;

// Returns the minimum number of bytes required to construct a QueenData instance (i.e. to have
// base frame 0 fully available), or -1 if the buffer doesn't yet contain enough data to determine
// this.  Passes through the content of `buf` up to `available` bytes only.
const computeFrame0MinBytes = (buf: ArrayBuffer, available: number): number => {
    if (available < HEADER_SIZE) return -1;

    const dv = new DataView(buf);

    const magic = dv.getUint32(0, true);
    if (magic !== MAGIC) return -1;

    const numSplats = dv.getUint32(8, true);
    const shDegree  = dv.getUint32(28, true);
    const frestCount = computeShRestCount(shDegree);

    // Walk the decoder block to find where it ends.
    let cursor = HEADER_SIZE;
    let stride = 0;

    for (let a = 0; a < NUM_ATTRS; a++) {
        // type(1) + latentDim(4) + featureDim(4) = 9 bytes minimum
        if (cursor + 9 > available) return -1;

        const type       = dv.getUint8(cursor); cursor += 1;
        const latentDim  = dv.getUint32(cursor, true); cursor += 4;
        const featureDim = dv.getUint32(cursor, true); cursor += 4;
        stride += featureDim;

        if (type !== DECODER_IDENTITY && featureDim > 0) {
            const weightBytes = latentDim * featureDim * 4;
            if (cursor + weightBytes > available) return -1;
            cursor += weightBytes;

            if (cursor + 1 > available) return -1;
            const hasBias = dv.getUint8(cursor); cursor += 1;

            if (hasBias) {
                const biasBytes = featureDim * 4;
                if (cursor + biasBytes > available) return -1;
                cursor += biasBytes;
            }
        }
    }

    // Validate expected feature dimensions against parsed stride.
    const expectedFeatureDims = [3, 3, frestCount, 3, 4, 1];
    const expectedStride = expectedFeatureDims.reduce((s, v) => s + v, 0);
    if (stride !== expectedStride) return -1;

    return cursor + numSplats * stride * 4;
};

class QueenData {
    // Growable backing store.  rawBytes always owns the bytes; _buffer === rawBytes.buffer.
    private rawBytes: Uint8Array;

    private _buffer: ArrayBuffer;

    private bytesAvailable: number;

    readonly header: QueenHeader;

    private decoders: AttributeDecoder[];

    private baseFrameByteOffset: number;

    // Byte size of one residual frame (all attributes combined).
    private residualFrameByteSize: number;

    // Byte offset to the start of the residual frame region.
    private residualFramesStartOffset: number;

    // Per-attribute pre-allocated decode buffers (reused each frame to avoid GC pressure).
    private decodeBufs: Float32Array[];

    // Pre-allocated scratch space for dequantised latents (avoids per-frame allocation).
    private floatLatentsBuf: Float32Array;

    // Temporal residual state: decoded output of the previous frame, per attribute.
    // Non-null only for DECODER_LATENT_RES attributes.
    private prevDecoded: (Float32Array | null)[];

    // Which frame index is currently reflected in the work arrays (-1 = none).
    private lastDecodedFrame: number = -1;

    // Mutable per-splat typed arrays that back a single reusable GSplatData.
    private work: {
        x: Float32Array;
        y: Float32Array;
        z: Float32Array;
        rot0: Float32Array;
        rot1: Float32Array;
        rot2: Float32Array;
        rot3: Float32Array;
        scale0: Float32Array;
        scale1: Float32Array;
        scale2: Float32Array;
        opacity: Float32Array;
        fdc0: Float32Array;
        fdc1: Float32Array;
        fdc2: Float32Array;
    };

    // Optional f_rest arrays (one per SH channel, empty for shDegree = 0).
    private frestArrays: Float32Array[];

    // A single GSplatData whose property storages are the mutable work arrays.
    // After calling loadFrame() the GSplatData reflects that frame's data.
    private _gsplatData!: GSplatData;

    get gsplatData(): GSplatData {
        return this._gsplatData;
    }

    constructor(buffer: ArrayBuffer) {
        // Copy the supplied buffer into a growable backing store.
        this.rawBytes = new Uint8Array(buffer);
        this._buffer = this.rawBytes.buffer as ArrayBuffer;
        this.bytesAvailable = buffer.byteLength;

        const dv = new DataView(this._buffer);

        // Validate magic bytes.
        const magic = dv.getUint32(0, true);
        if (magic !== MAGIC) {
            throw new Error(`Invalid .queen file: expected magic 0x${MAGIC.toString(16)}, got 0x${magic.toString(16)}`);
        }

        this.header = {
            version: dv.getUint32(4, true),
            numSplats: dv.getUint32(8, true),
            numFrames: dv.getUint32(12, true),
            fps: dv.getFloat32(16, true),
            timeDurationMin: dv.getFloat32(20, true),
            timeDurationMax: dv.getFloat32(24, true),
            shDegree: dv.getUint32(28, true)
        };

        this.initialize();
    }

    // Parses the decoder block, validates feature dimensions, allocates all working buffers, and
    // constructs the GSplatData.  Called exactly once from the constructor.
    private initialize(): void {
        const N = this.header.numSplats;
        const frestCount = computeShRestCount(this.header.shDegree);
        const dv = new DataView(this._buffer);

        // Parse decoder block.
        let cursor = HEADER_SIZE;
        const decoders: AttributeDecoder[] = [];
        for (let a = 0; a < NUM_ATTRS; a++) {
            const type       = dv.getUint8(cursor); cursor += 1;
            const latentDim  = dv.getUint32(cursor, true); cursor += 4;
            const featureDim = dv.getUint32(cursor, true); cursor += 4;

            let weight: Float32Array | null = null;
            let bias: Float32Array | null = null;

            if (type !== DECODER_IDENTITY && featureDim > 0) {
                weight = QueenData._readFloat32(this._buffer, cursor, latentDim * featureDim);
                cursor += latentDim * featureDim * 4;

                const hasBias = dv.getUint8(cursor); cursor += 1;
                if (hasBias) {
                    bias = QueenData._readFloat32(this._buffer, cursor, featureDim);
                    cursor += featureDim * 4;
                }
            }

            decoders.push({ type, latentDim, featureDim, weight, bias });
        }
        this.decoders = decoders;

        // Validate that featureDims match the expected attribute layout.
        const expectedFeatureDims = [3, 3, frestCount, 3, 4, 1];
        for (let a = 0; a < NUM_ATTRS; a++) {
            if (decoders[a].featureDim !== expectedFeatureDims[a]) {
                throw new Error(
                    `.queen attribute ${a}: featureDim ${decoders[a].featureDim}, expected ${expectedFeatureDims[a]}`
                );
            }
        }

        // Compute byte offsets.
        this.baseFrameByteOffset = cursor;
        const stride = decoders.reduce((s, d) => s + d.featureDim, 0);
        cursor += N * stride * 4;

        this.residualFramesStartOffset = cursor;
        this.residualFrameByteSize = decoders.reduce((s, d) => {
            if (d.featureDim === 0) return s;
            return s + N * d.latentDim * 2 + 4;   // int16 data + float32 quantScale
        }, 0);

        // Allocate per-splat work arrays.
        this.work = {
            x: new Float32Array(N),
            y: new Float32Array(N),
            z: new Float32Array(N),
            rot0: new Float32Array(N),
            rot1: new Float32Array(N),
            rot2: new Float32Array(N),
            rot3: new Float32Array(N),
            scale0: new Float32Array(N),
            scale1: new Float32Array(N),
            scale2: new Float32Array(N),
            opacity: new Float32Array(N),
            fdc0: new Float32Array(N),
            fdc1: new Float32Array(N),
            fdc2: new Float32Array(N)
        };

        this.frestArrays = [];
        for (let k = 0; k < frestCount; k++) {
            this.frestArrays.push(new Float32Array(N));
        }

        // Pre-allocate decode buffers, the dequantised-latents scratch buffer, and temporal state.
        this.decodeBufs      = decoders.map(d => new Float32Array(Math.max(1, N * d.featureDim)));
        const maxLatentN      = decoders.reduce((m, d) => Math.max(m, N * d.latentDim), 0);
        this.floatLatentsBuf  = new Float32Array(Math.max(1, maxLatentN));
        this.prevDecoded = decoders.map(d => ((d.type === DECODER_LATENT_RES && d.featureDim > 0) ? new Float32Array(N * d.featureDim) : null)
        );

        // Build GSplatData backed by the work arrays.
        const prop = (name: string, arr: Float32Array) => ({
            type: 'float' as const,
            name,
            storage: arr,
            byteSize: 4
        });

        const properties = [
            prop('x', this.work.x),
            prop('y', this.work.y),
            prop('z', this.work.z),
            prop('rot_0', this.work.rot0),
            prop('rot_1', this.work.rot1),
            prop('rot_2', this.work.rot2),
            prop('rot_3', this.work.rot3),
            prop('scale_0', this.work.scale0),
            prop('scale_1', this.work.scale1),
            prop('scale_2', this.work.scale2),
            prop('opacity', this.work.opacity),
            prop('f_dc_0', this.work.fdc0),
            prop('f_dc_1', this.work.fdc1),
            prop('f_dc_2', this.work.fdc2)
        ];

        for (let k = 0; k < frestCount; k++) {
            properties.push(prop(`f_rest_${k}`, this.frestArrays[k]));
        }

        this._gsplatData = new GSplatData([{
            name: 'vertex',
            count: N,
            properties
        }]);
    }

    // Append a newly received chunk to the internal buffer.
    // Called by the streaming loader as each HTTP response chunk arrives.
    appendChunk(chunk: Uint8Array): void {
        const needed = this.bytesAvailable + chunk.length;
        if (needed > this.rawBytes.length) {
            // Double-or-fit growth strategy.
            let newSize = this.rawBytes.length * 2;
            while (newSize < needed) newSize *= 2;
            const newBytes = new Uint8Array(newSize);
            newBytes.set(this.rawBytes.subarray(0, this.bytesAvailable));
            this.rawBytes = newBytes;
            this._buffer  = newBytes.buffer as ArrayBuffer;
        }
        this.rawBytes.set(chunk, this.bytesAvailable);
        this.bytesAvailable += chunk.length;
    }

    // Number of frames for which we have enough data to decode.
    // During streaming this grows from 1 (base frame) up to numFrames as chunks arrive.
    get availableFrames(): number {
        if (this.header.numFrames <= 1) return this.header.numFrames;
        const residualBytes = this.bytesAvailable - this.residualFramesStartOffset;
        if (residualBytes <= 0) return 1;
        return Math.min(this.header.numFrames, 1 + Math.floor(residualBytes / this.residualFrameByteSize));
    }

    get numFrames(): number {
        return this.header.numFrames;
    }

    // Total playback duration in seconds.
    get duration(): number {
        const { numFrames, fps } = this.header;
        return numFrames > 1 ? (numFrames - 1) / fps : 0;
    }

    // Map a playback time [0..duration] to a frame index [0..availableFrames-1].
    // Clamps to the highest frame index for which data has arrived so partial-buffer
    // playback advances only as far as the streamed data allows.
    getFrameIndex(time: number): number {
        const { numFrames, fps } = this.header;
        const maxFrame = Math.min(numFrames - 1, this.availableFrames - 1);
        return Math.max(0, Math.min(maxFrame, Math.round(time * fps)));
    }

    // Decode the given frame into the work arrays (and therefore into gsplatData).
    loadFrame(frameIndex: number): void {
        if (frameIndex === this.lastDecodedFrame) return;

        if (frameIndex === 0) {
            this._decodeBaseFrame();
            return;
        }

        // For latent_res decoders the output of frame f depends on frame f-1, so frames
        // must be decoded sequentially. For purely identity/latent decoders every frame
        // can be decoded independently.
        const needsSequential = this.decoders.some(d => d.type === DECODER_LATENT_RES);

        if (!needsSequential) {
            // Independent frames: jump directly to the requested frame.
            this._decodeResidualFrame(frameIndex);
        } else if (this.lastDecodedFrame < 0 || frameIndex <= this.lastDecodedFrame) {
            // Uninitialized or backward seek: restart from the base frame.
            this._decodeBaseFrame();
            for (let f = 1; f <= frameIndex; f++) {
                this._decodeResidualFrame(f);
            }
        } else {
            // Forward seek: decode incrementally from the last decoded frame.
            for (let f = this.lastDecodedFrame + 1; f <= frameIndex; f++) {
                this._decodeResidualFrame(f);
            }
        }
    }

    private _decodeBaseFrame(): void {
        const { numSplats: N } = this.header;
        const stride = this.decoders.reduce((s, d) => s + d.featureDim, 0);

        // Read the dense base frame (copies to handle potential buffer alignment).
        const baseData = QueenData._readFloat32(this._buffer, this.baseFrameByteOffset, N * stride);

        // Compute per-attribute starting offsets within the interleaved stride.
        const attrOffset: number[] = [];
        let off = 0;
        for (let a = 0; a < NUM_ATTRS; a++) {
            attrOffset.push(off);
            off += this.decoders[a].featureDim;
        }

        for (let a = 0; a < NUM_ATTRS; a++) {
            const { featureDim } = this.decoders[a];
            if (featureDim === 0) continue;

            const buf = this.decodeBufs[a];
            const ao  = attrOffset[a];
            for (let i = 0; i < N; i++) {
                for (let j = 0; j < featureDim; j++) {
                    buf[i * featureDim + j] = baseData[i * stride + ao + j];
                }
            }

            // Initialise prevDecoded for latent_res attributes so frame 1 can add residuals.
            if (this.decoders[a].type === DECODER_LATENT_RES) {
                this.prevDecoded[a]!.set(buf.subarray(0, N * featureDim));
            }

            this._writeAttr(a, N);
        }

        this.lastDecodedFrame = 0;
    }

    private _decodeResidualFrame(frameIndex: number): void {
        const { numSplats: N } = this.header;
        const dv = new DataView(this._buffer);

        let byteOffset = this.residualFramesStartOffset + (frameIndex - 1) * this.residualFrameByteSize;

        for (let a = 0; a < NUM_ATTRS; a++) {
            const decoder = this.decoders[a];
            const { type, latentDim, featureDim } = decoder;

            if (featureDim === 0) continue;

            // Read quantised latents (copies to handle alignment).
            const quantLatents = QueenData._readInt16(this._buffer, byteOffset, N * latentDim);
            byteOffset += N * latentDim * 2;

            // Read dequantisation scale.  Use the reciprocal to replace per-element divisions
            // with multiplications in the tight decode loops below.
            const invScale = 1 / dv.getFloat32(byteOffset, true);
            byteOffset += 4;

            const buf = this.decodeBufs[a];

            if (type === DECODER_IDENTITY) {
                // Identity: bulk dequantise into output buffer (no matrix multiply).
                for (let i = 0; i < N * featureDim; i++) {
                    buf[i] = quantLatents[i] * invScale;
                }
            } else if (type === DECODER_LATENT || type === DECODER_LATENT_RES) {
                // Latent / latent_res: pre-dequantise into the scratch buffer, then apply the
                // linear decoder.  Splitting the dequantisation into a separate pass allows the
                // JIT to vectorise both loops independently and keeps the inner matrix-multiply
                // loop free of division.
                const { weight, bias } = decoder;
                const floatLatents = this.floatLatentsBuf;

                for (let i = 0; i < N * latentDim; i++) {
                    floatLatents[i] = quantLatents[i] * invScale;
                }

                for (let i = 0; i < N; i++) {
                    for (let j = 0; j < featureDim; j++) {
                        let val = bias ? bias[j] : 0;
                        for (let k = 0; k < latentDim; k++) {
                            val += floatLatents[i * latentDim + k] * weight![k * featureDim + j];
                        }
                        buf[i * featureDim + j] = val;
                    }
                }
            }

            // Apply temporal residual and update the stored previous-frame output.
            if (type === DECODER_LATENT_RES) {
                const prev = this.prevDecoded[a]!;
                for (let i = 0; i < N * featureDim; i++) {
                    buf[i] += prev[i];
                }
                prev.set(buf.subarray(0, N * featureDim));
            }

            this._writeAttr(a, N);
        }

        this.lastDecodedFrame = frameIndex;
    }

    // Copy decodeBufs[a] into the appropriate typed work arrays.
    private _writeAttr(a: number, N: number): void {
        const buf = this.decodeBufs[a];
        const fd  = this.decoders[a].featureDim;
        const { work } = this;

        switch (a) {
            case ATTR_XYZ:
                for (let i = 0; i < N; i++) {
                    work.x[i]     = buf[i * fd + 0];
                    work.y[i]     = buf[i * fd + 1];
                    work.z[i]     = buf[i * fd + 2];
                }
                break;

            case ATTR_FDC:
                for (let i = 0; i < N; i++) {
                    work.fdc0[i]  = buf[i * fd + 0];
                    work.fdc1[i]  = buf[i * fd + 1];
                    work.fdc2[i]  = buf[i * fd + 2];
                }
                break;

            case ATTR_FREST: {
                const frc = this.frestArrays.length;
                if (frc === 0) break;
                for (let i = 0; i < N; i++) {
                    for (let k = 0; k < frc; k++) {
                        this.frestArrays[k][i] = buf[i * frc + k];
                    }
                }
                break;
            }

            case ATTR_SC:
                for (let i = 0; i < N; i++) {
                    work.scale0[i] = buf[i * fd + 0];
                    work.scale1[i] = buf[i * fd + 1];
                    work.scale2[i] = buf[i * fd + 2];
                }
                break;

            case ATTR_ROT:
                for (let i = 0; i < N; i++) {
                    work.rot0[i]  = buf[i * fd + 0];
                    work.rot1[i]  = buf[i * fd + 1];
                    work.rot2[i]  = buf[i * fd + 2];
                    work.rot3[i]  = buf[i * fd + 3];
                }
                break;

            case ATTR_OP:
                for (let i = 0; i < N; i++) {
                    work.opacity[i] = buf[i];
                }
                break;
        }
    }

    // Read count float32 values from buffer at byteOffset into a fresh Float32Array.
    // Copies the bytes to guarantee correct handling of unaligned offsets.
    private static _readFloat32(buf: ArrayBuffer, byteOffset: number, count: number): Float32Array {
        const dst = new Float32Array(count);
        new Uint8Array(dst.buffer).set(new Uint8Array(buf, byteOffset, count * 4));
        return dst;
    }

    // Read count int16 values from buffer at byteOffset into a fresh Int16Array.
    // Copies the bytes to guarantee correct handling of unaligned offsets.
    private static _readInt16(buf: ArrayBuffer, byteOffset: number, count: number): Int16Array {
        const dst = new Int16Array(count);
        new Uint8Array(dst.buffer).set(new Uint8Array(buf, byteOffset, count * 2));
        return dst;
    }
}

// Parse a .queen ArrayBuffer and return a QueenData instance.
const parseQueen = (buffer: ArrayBuffer): QueenData => new QueenData(buffer);

export { QueenData, parseQueen, computeFrame0MinBytes };
