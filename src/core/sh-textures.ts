// The engine's SH stream textures, and how an SH degree maps onto them.
//
// Two numbering schemes meet here. The SH degree (`bands` in the SOG
// format, `shBands` on the engine's resource) is 1, 2 or 3. The engine's
// texture names count *coefficients* instead: splatSH_1to3 holds the first
// three non-DC coefficients, which happen to be all of degree 1. So
// "splatSH_8to11" names coefficients 8 to 11, not degrees. Code in this
// repository refers to the textures by their index in SH_TEXTURES, and only
// this table spells out the engine's names.
//
// Each splat's SH is one scale word (the largest coefficient magnitude, as
// float bits) followed by one 11:10:11 word per coefficient, four words per
// RGBA32U texel:
//
//   degree 1:  3 coefficients,  4 words: texture 0
//   degree 2:  8 coefficients,  9 words: textures 0-2 (texture 2 is R32U)
//   degree 3: 15 coefficients, 16 words: textures 0-3
//
// sogst-pack-worker.ts packs the same layout but cannot import this module;
// keep the two in step.

const SH_TEXTURES = ['splatSH_1to3', 'splatSH_4to7', 'splatSH_8to11', 'splatSH_12to15'] as const;

// Words per texel of an RGBA32U texture.
const WORDS_PER_TEXEL = 4;

// SH coefficients per colour channel at a degree, excluding the DC term.
const shCoefficients = (bands: number) => (bands > 0 ? (bands + 1) * (bands + 1) - 1 : 0);

// Packed words per splat at a degree: the scale word plus one per coefficient.
const shWords = (bands: number) => (bands > 0 ? 1 + shCoefficients(bands) : 0);

// How many of SH_TEXTURES a degree uses.
const shTextureCount = (bands: number) => Math.ceil(shWords(bands) / WORDS_PER_TEXEL);

// Words per splat in SH texture `index` at a degree: 4, or fewer in a last
// texture the degree only partly fills.
const shTextureWords = (bands: number, index: number) =>
    Math.max(0, Math.min(WORDS_PER_TEXEL, shWords(bands) - index * WORDS_PER_TEXEL));

export { SH_TEXTURES, shCoefficients, shTextureCount, shTextureWords, shWords, WORDS_PER_TEXEL };
