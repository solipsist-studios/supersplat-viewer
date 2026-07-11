import { GSplatData } from 'playcanvas';

// .omg4 binary format
// ─────────────────────────────────────────────────────────────────────────────
// Header (28 bytes, little-endian):
//   [0-3]   uint32  magic = 0x34474D4F ("OMG4")
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

// Version 2 (compact temporal splats, no per-frame data)
// ─────────────────────────────────────────────────────────────────────────────
// Header (32 bytes, little-endian):
//   [0-3]   uint32  magic = 0x34474D4F ("OMG4")
//   [4-7]   uint32  version = 2
//   [8-11]  uint32  numSplats (N)
//   [12-15] uint32  flags (bit 0: file includes 45 f_rest SH arrays)
//   [16-19] float32 timeMin  (seconds)
//   [20-23] float32 timeMax  (seconds)
//   [24-27] float32 fps      (advisory, UI only)
//   [28-31] uint32  reserved
//
// Data: 19 SoA float32[N] arrays, in order:
//   x y z                    position at t = t_center
//   rot_0..rot_3             quaternion (w,x,y,z) of the sliced 3D covariance
//   scale_0..scale_2         log-space scales
//   opacity                  logit-space peak opacity
//   f_dc_0..f_dc_2           raw SH DC coefficients
//   vx vy vz                 linear velocity (scene units / second)
//   t_center                 temporal centre (seconds)
//   t_sigma                  temporal std-dev (seconds)
//
// If flags bit 0 is set, 45 further float32[N] arrays follow: f_rest_0..44,
// standard 3-band spherical harmonics in PLY channel-major order (baked from
// the OMG4 view MLP at each splat's temporal centre).
//
// Reconstruction at time t (done on the GPU by the viewer):
//   position(t) = (x,y,z) + (vx,vy,vz) * (t - t_center)
//   alpha(t)    = sigmoid(opacity) * exp(-0.5 * ((t - t_center) / t_sigma)^2)
// ─────────────────────────────────────────────────────────────────────────────

const MAGIC = 0x34474D4F;   // little-endian uint32 of "OMG4"
const HEADER_SIZE = 28;
const FLOATS_PER_SPLAT = 14;

const V2_HEADER_SIZE = 32;
const V2_NUM_FIELDS = 19;

interface Omg4Header {
    version: number;
    numSplats: number;
    numFrames: number;
    fps: number;
    timeDurationMin: number;
    timeDurationMax: number;
}

interface Omg4FrameData {
    readonly gsplatData: GSplatData;
    readonly numFrames: number;
    readonly duration: number;
    getFrameIndex(time: number): number;
    loadFrame(frameIndex: number): void | Promise<void>;
    prefetchFrame?(frameIndex: number): void;
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

    private frameTimestamps: Float32Array;

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
            throw new Error(`Invalid .omg4 file: expected magic 0x${MAGIC.toString(16)}, got 0x${magic.toString(16)}`);
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
        const expectedSize = HEADER_SIZE + this.header.numFrames * this.frameByteSize;
        if (buffer.byteLength < expectedSize) {
            throw new Error(`Invalid .omg4 file: expected at least ${expectedSize} bytes, got ${buffer.byteLength}`);
        }

        this.frameTimestamps = new Float32Array(this.header.numFrames);
        for (let i = 0; i < this.header.numFrames; i++) {
            this.frameTimestamps[i] = view.getFloat32(HEADER_SIZE + i * this.frameByteSize, true);
        }

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
        if (numFrames <= 1) return 0;

        const timestampDuration = this.frameTimestamps[numFrames - 1] - this.frameTimestamps[0];
        return timestampDuration > 0 ? timestampDuration : (fps > 0 ? (numFrames - 1) / fps : 0);
    }

    // Map a playback time [0..duration] to the nearest baked frame timestamp.
    getFrameIndex(time: number): number {
        const { numFrames, fps } = this.header;
        if (numFrames <= 1) return 0;

        const duration = this.duration;
        if (duration <= 0) {
            return fps > 0 ? Math.max(0, Math.min(numFrames - 1, Math.round(time * fps))) : 0;
        }

        const clampedTime = Math.max(0, Math.min(duration, time));
        const targetTimestamp = this.frameTimestamps[0] + clampedTime;

        let lo = 0;
        let hi = numFrames - 1;
        while (lo < hi) {
            const mid = (lo + hi) >> 1;
            if (this.frameTimestamps[mid] < targetTimestamp) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        const upper = lo;
        const lower = Math.max(0, upper - 1);
        return Math.abs(this.frameTimestamps[upper] - targetTimestamp) < Math.abs(targetTimestamp - this.frameTimestamps[lower]) ? upper : lower;
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

// Parse a .omg4 ArrayBuffer and return an Omg4Data instance.
const parseOmg4 = (buffer: ArrayBuffer): Omg4Data => new Omg4Data(buffer);

// Version-2 data: static splat attributes plus per-splat temporal parameters
// (velocity, temporal centre, temporal std-dev). There is no per-frame data;
// the viewer evaluates motion and temporal fade on the GPU each frame.
class Omg4V2Data {
    readonly numSplats: number;

    readonly timeMin: number;

    readonly timeMax: number;

    readonly fps: number;

    // Optional screen-space 2D covariance scale (kx, ky) from the header.
    cov2dScale: [number, number] | null = null;

    readonly gsplatData: GSplatData;

    // Per-splat temporal parameters
    readonly velocityX: Float32Array;

    readonly velocityY: Float32Array;

    readonly velocityZ: Float32Array;

    readonly tCenter: Float32Array;

    readonly tSigma: Float32Array;

    constructor(buffer: ArrayBuffer) {
        const view = new DataView(buffer);

        const magic = view.getUint32(0, true);
        if (magic !== MAGIC) {
            throw new Error(`Invalid .omg4 file: expected magic 0x${MAGIC.toString(16)}, got 0x${magic.toString(16)}`);
        }
        const version = view.getUint32(4, true);
        if (version !== 2) {
            throw new Error(`Omg4V2Data: expected version 2, got ${version}`);
        }

        const N = view.getUint32(8, true);
        this.numSplats = N;
        const flags = view.getUint32(12, true);
        const hasSH = (flags & 1) !== 0;
        this.timeMin = view.getFloat32(16, true);
        this.timeMax = view.getFloat32(20, true);
        this.fps = view.getFloat32(24, true);

        // flags bit 1: reserved word carries a screen-space 2D-covariance
        // scale (kx, ky as 2 x float16) that the renderer must apply.
        if ((flags & 2) !== 0) {
            const halfToFloat = (h: number) => {
                const s = (h & 0x8000) ? -1 : 1;
                const e = (h >> 10) & 0x1f;
                const m = h & 0x3ff;
                if (e === 0) return s * m * 2 ** -24;
                if (e === 31) return m ? NaN : s * Infinity;
                return s * (1 + m / 1024) * 2 ** (e - 15);
            };
            const reserved = view.getUint32(28, true);
            this.cov2dScale = [halfToFloat(reserved & 0xffff), halfToFloat(reserved >>> 16)];
        }

        const numFields = V2_NUM_FIELDS + (hasSH ? 45 : 0);
        const expectedSize = V2_HEADER_SIZE + numFields * N * 4;
        if (buffer.byteLength < expectedSize) {
            throw new Error(`Invalid .omg4 v2 file: expected at least ${expectedSize} bytes, got ${buffer.byteLength}`);
        }

        // SoA float32[N] arrays; zero-copy views over the fetched buffer.
        const field = (i: number) => new Float32Array(buffer, V2_HEADER_SIZE + i * N * 4, N);

        const prop = (name: string, arr: Float32Array) => ({
            type: 'float' as const,
            name,
            storage: arr,
            byteSize: 4
        });

        const properties = [
            prop('x', field(0)),
            prop('y', field(1)),
            prop('z', field(2)),
            prop('rot_0', field(3)),
            prop('rot_1', field(4)),
            prop('rot_2', field(5)),
            prop('rot_3', field(6)),
            prop('scale_0', field(7)),
            prop('scale_1', field(8)),
            prop('scale_2', field(9)),
            prop('opacity', field(10)),
            prop('f_dc_0', field(11)),
            prop('f_dc_1', field(12)),
            prop('f_dc_2', field(13))
        ];
        if (hasSH) {
            for (let i = 0; i < 45; i++) {
                properties.push(prop(`f_rest_${i}`, field(V2_NUM_FIELDS + i)));
            }
        }

        this.gsplatData = new GSplatData([{
            name: 'vertex',
            count: N,
            properties
        }]);

        this.velocityX = field(14);
        this.velocityY = field(15);
        this.velocityZ = field(16);
        this.tCenter = field(17);
        this.tSigma = field(18);
    }

    get duration(): number {
        return Math.max(0, this.timeMax - this.timeMin);
    }
}

const parseOmg4V2 = (buffer: ArrayBuffer): Omg4V2Data => new Omg4V2Data(buffer);

// Read the format version from the first bytes of a .omg4 file (throws on bad magic).
const readOmg4Version = (buffer: ArrayBuffer): number => {
    const view = new DataView(buffer);
    const magic = view.getUint32(0, true);
    if (magic !== MAGIC) {
        throw new Error(`Invalid .omg4 file: expected magic 0x${MAGIC.toString(16)}, got 0x${magic.toString(16)}`);
    }
    return view.getUint32(4, true);
};

export { Omg4Data, Omg4V2Data, parseOmg4, parseOmg4V2, readOmg4Version };
export type { Omg4FrameData };
