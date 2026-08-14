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
// anything else is rejected. (There is no version 2 or 3: the container was
// renumbered from 3 to 1 when the development-era binary formats — the ones
// carrying the ASCII magic "OMG4" — were removed. Nothing had shipped, so
// nothing reads them any more.)
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

// The only extension. `.omg4` was the development-era spelling and is no
// longer accepted — those containers were never released.
const SOGST_EXTENSIONS = ['.sogst'];

// Required `meta.json` identity. A conforming player rejects anything else.
const SOGST_META_VERSION = 1;
const SOGST_META_FORMAT = 'sogst';

// True if `filename` names a .sogst container.
const isSogstFilename = (filename: string): boolean => {
    const lower = filename.toLowerCase();
    return SOGST_EXTENSIONS.some(ext => lower.endsWith(ext));
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
interface SogstSegments {
    duration: number;
    persistent: [number, number];
    list: { t0: number, t1: number, range: [number, number] }[];
}

export { isSogstFilename, SOGST_EXTENSIONS, SOGST_META_VERSION, SOGST_META_FORMAT };
export type { SogstSegments };
