// .sogst — SOG spacetime. What a .sogst file *is*; `core/load-sogst.ts`
// decodes one.
// ─────────────────────────────────────────────────────────────────────────────
// A .sogst file is a ZIP archive (`PK\x03\x04`) of lossless-WebP attribute
// textures plus a `meta.json` manifest, which MUST be the first entry. Static
// attributes follow the PlayCanvas SOG v2 conventions byte for byte, so an
// existing SOG decoder reconstructs them unmodified; the spacetime extension
// adds per-splat linear motion, an optional second-order term, and a temporal
// radial-basis window.
//
// `meta.version` is 1 and `meta.format` is "sogst"; both are REQUIRED and
// anything else is rejected.
//
// At clip time t (seconds, absolute — not normalised), a splat evaluates as:
//
//   dt       = t - t_center
//   mean(t)  = xyz + v*dt                 (motion.degree == 1)
//   mean(t)  = xyz + v*dt + a*dt*dt       (motion.degree == 2)
//   alpha(t) = sigmoid(opacity) * exp(-0.5 * (dt / t_sigma)^2)
//
// Three things about that are easy to get wrong and all fail silently: the
// temporal factor is **unnormalised** (no 1/sqrt(2*pi*sigma^2)); `t_sigma` is a
// standard deviation in seconds, not a variance; and `a` is the raw dt^2
// coefficient, **not** half-acceleration.
// ─────────────────────────────────────────────────────────────────────────────

// The only extension.
const SOGST_EXTENSIONS = ['.sogst'];

// Required `meta.json` identity. A conforming player rejects anything else.
const SOGST_META_VERSION = 1;
const SOGST_META_FORMAT = 'sogst';

// True if `filename` names a .sogst container.
const isSogstFilename = (filename: string): boolean => {
    const lower = filename.toLowerCase();
    return SOGST_EXTENSIONS.some((ext) => lower.endsWith(ext));
};

// Temporal segment table. Splats are ordered [ persistent | seg 0 | seg 1 | … ];
// persistent splats are always drawn and the rest are bucketed by t_center.
//
// **All index ranges are half-open, [first, last).** `persistent` is [0, P) and
// each segment `range` is [first, last); an empty segment has first == last and
// is legal. `t0`/`t1` are the actual time coverage of a segment's members, not
// the bucket bounds, so they overlap adjacent segments and `t0` may precede
// `time.min`.
//
// The drawing rule at time t: draw [0, P), plus — over every segment whose
// [t0, t1] contains t — the single span [ min(range[0]), max(range[1]) ).
// Segments are time-ordered and their coverage overlaps, so the active set is
// contiguous and this is two ranges, never a scatter.
type SogstSegments = {
    duration: number;
    persistent: [number, number];
    list: { t0: number; t1: number; range: [number, number] }[];
};

// The `meta.json` manifest, as far as this player reads it.
//
// Deliberately a description of what we consume, not of the whole format: the
// spec requires a player to **ignore** manifest keys it does not recognise
// rather than reject them, so an archive carrying more than this is conforming
// and must still load. TypeScript agrees — extra properties are structurally
// fine on a parsed value — so adding a field here is only ever about letting
// this code read it, never about tightening what we accept.
//
// Quantised attributes follow SOG v2: `mins`/`maxs` are split-plane endpoints
// stored in log space (T = sign(x)*ln(1+|x|)), so a reader comparing them
// against real-world values has to invert that first.
type SogstMeta = {
    version: number;
    format: string;
    count: number;
    means: { mins: number[]; maxs: number[] };
    motion: { mins: number[]; maxs: number[]; degree?: number; files?: string[] };
    trbf: { center: { codebook: number[] }; sigma: { codebook: number[] } };
    /** Present only when motion.degree == 2. */
    accel?: { mins: number[]; maxs: number[]; files?: string[] };
    /** Absent on a still; defaults are 0 / 0 / 30. */
    time?: { min: number; max: number; fps: number };
    cov2d_scale?: number[];
    shN?: { bands: number; files: string[] };
    segments?: SogstSegments;
    /** Present only on streamable archives; absent means one monolithic group. */
    streams?: {
        persistent: string;
        segments: (string | null)[];
        reveal_bytes: number;
        geometry_bytes: number;
        sh_deferred?: boolean;
    };
};

export { isSogstFilename, SOGST_EXTENSIONS, SOGST_META_VERSION, SOGST_META_FORMAT };
export type { SogstSegments, SogstMeta };
