import { GSplatData } from 'playcanvas';

// .4dgs binary format
// ─────────────────────────────────────────────────────────────────────────────
// Header (28 bytes, little-endian):
//   [0-3]   uint32  magic = 0x53474434 ("4DGS")
//   [4-7]   uint32  version = 1
//   [8-11]  uint32  numSplats  (N)
//   [12-15] uint32  numFrames  (F)
//   [16-19] float32 fps
//   [20-23] float32 timeDurationMin
//   [24-27] float32 timeDurationMax
//
// Per-frame record (for each f in [0, F)):
//   [0-3]       float32          timestamp
//   [4...]      float32[N * 14]  per-splat data, AoS layout:
//     stride-14:  x  y  z  rot_0  rot_1  rot_2  rot_3  scale_0  scale_1  scale_2  opacity  f_dc_0  f_dc_1  f_dc_2
//       rot_*  : quaternion (w, x, y, z) stored raw (renderer normalises)
//       scale_* : log-space (renderer applies exp)
//       opacity : logit-space (renderer applies sigmoid)
//       f_dc_*  : raw SH DC coefficients (renderer applies 0.5 + val * SH_C0)
// ─────────────────────────────────────────────────────────────────────────────

const MAGIC = 0x53474434;   // little-endian uint32 of "4DGS"
const HEADER_SIZE = 28;
const FLOATS_PER_SPLAT = 14;

interface Omg4Header {
    version: number;
    numSplats: number;
    numFrames: number;
    fps: number;
    timeDurationMin: number;
    timeDurationMax: number;
}

// Mutable per-splat typed arrays that back a single reusable GSplatData.
interface WorkArrays {
    x: Float32Array;
    y: Float32Array;
    z: Float32Array;
    rot0: Float32Array;  // w
    rot1: Float32Array;  // x
    rot2: Float32Array;  // y
    rot3: Float32Array;  // z
    scale0: Float32Array;
    scale1: Float32Array;
    scale2: Float32Array;
    opacity: Float32Array;
    fdc0: Float32Array;
    fdc1: Float32Array;
    fdc2: Float32Array;
}

class Omg4Data {
    private buffer: ArrayBuffer;

    readonly header: Omg4Header;

    private frameByteSize: number;

    private work: WorkArrays;

    // A single GSplatData whose property storages are the mutable WorkArrays.
    // After calling loadFrame() the GSplatData reflects that frame's data.
    readonly gsplatData: GSplatData;

    constructor(buffer: ArrayBuffer) {
        this.buffer = buffer;
        const view = new DataView(buffer);

        // Validate magic bytes
        const magic = view.getUint32(0, true);
        if (magic !== MAGIC) {
            throw new Error(`Invalid .4dgs file: expected magic 0x${MAGIC.toString(16)}, got 0x${magic.toString(16)}`);
        }

        this.header = {
            version: view.getUint32(4, true),
            numSplats: view.getUint32(8, true),
            numFrames: view.getUint32(12, true),
            fps: view.getFloat32(16, true),
            timeDurationMin: view.getFloat32(20, true),
            timeDurationMax: view.getFloat32(24, true)
        };

        const N = this.header.numSplats;
        this.frameByteSize = 4 + N * FLOATS_PER_SPLAT * 4;

        // Allocate working arrays once; they are reused across all frames.
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

        const prop = (name: string, arr: Float32Array) => ({
            type: 'float' as const,
            name,
            storage: arr,
            byteSize: 4
        });

        const element = {
            name: 'vertex',
            count: N,
            properties: [
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
            ]
        };

        this.gsplatData = new GSplatData([element]);
    }

    get numFrames(): number {
        return this.header.numFrames;
    }

    // Total playback duration in seconds.
    get duration(): number {
        const { numFrames, fps } = this.header;
        return numFrames > 1 ? (numFrames - 1) / fps : 0;
    }

    // Map a playback time [0..duration] to a frame index [0..numFrames-1].
    getFrameIndex(time: number): number {
        const { numFrames, fps } = this.header;
        return Math.max(0, Math.min(numFrames - 1, Math.round(time * fps)));
    }

    // Unpack the given frame's data into the mutable working arrays.
    // After this call, this.gsplatData reflects the requested frame.
    loadFrame(frameIndex: number): void {
        const N = this.header.numSplats;
        const frameOffset = HEADER_SIZE + frameIndex * this.frameByteSize;
        // Skip the 4-byte timestamp and create a view over the per-splat data.
        const floats = new Float32Array(this.buffer, frameOffset + 4, N * FLOATS_PER_SPLAT);

        const { x, y, z, rot0, rot1, rot2, rot3, scale0, scale1, scale2, opacity, fdc0, fdc1, fdc2 } = this.work;
        for (let i = 0; i < N; i++) {
            const b = i * FLOATS_PER_SPLAT;
            x[i]       = floats[b];
            y[i]       = floats[b + 1];
            z[i]       = floats[b + 2];
            rot0[i]    = floats[b + 3];   // w
            rot1[i]    = floats[b + 4];   // x
            rot2[i]    = floats[b + 5];   // y
            rot3[i]    = floats[b + 6];   // z
            scale0[i]  = floats[b + 7];
            scale1[i]  = floats[b + 8];
            scale2[i]  = floats[b + 9];
            opacity[i] = floats[b + 10];
            fdc0[i]    = floats[b + 11];
            fdc1[i]    = floats[b + 12];
            fdc2[i]    = floats[b + 13];
        }
    }
}

// Parse a .4dgs ArrayBuffer and return an Omg4Data instance.
const parseOmg4 = (buffer: ArrayBuffer): Omg4Data => new Omg4Data(buffer);

export { Omg4Data, parseOmg4 };
