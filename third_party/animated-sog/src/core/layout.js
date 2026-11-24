/**
 * GPU-ready layouts the decoder produces.
 *
 * SOG v2 textures (RGBA8, `width × height` texels, splat i at texel i, row-major), exactly as a SOG
 * v2 file's images decode, so a SOG renderer takes them unchanged:
 *   means_l / means_u  low / high byte of the 16-bit log-mapped position, rgb = xyz
 *   quats              rgb = the three smallest components (w x y z order), a = 252 + index of the largest
 *   scales             rgb = scales codebook indices
 *   sh0                rgb = sh0 codebook indices, a = opacity · 255
 *   sh_labels          rg = 16-bit SH palette entry (low, high byte)
 *   sh_centroids       entry k, coefficient j at texel ((k mod 64) · coeffs + j, ⌊k / 64⌋), rgb = shN codebook indices
 *
 * Motion textures, MOTION_TEX_WIDTH texels a row:
 *   motionA  RGBA32U, one texel a splat:
 *            x = μ (f32 bits), y = σ (f32 bits), z = half(anchor.x) | half(anchor.y) << 16,
 *            w = half(anchor.z) | isStatic << 16 | clip << 17
 *   motionB  RGBA16F, MOTION_B_TEXELS (4) texels a splat, 16 halves:
 *            O0.xyz O1.xyz O2.xyz G0 G1 G2 G3 G4, 2 unused
 *
 * Clip table (RGBA32F, 2 × clips): per clip (timeOffset, activeStart, activeEnd, 0),
 * (coreStart, coreEnd, fadeIn, fadeOut), all in seconds.
 */

export const MOTION_TEX_WIDTH = 4096;
export const MOTION_B_TEXELS = 4;
export const CLIP_FLOATS = 8;

/** Texture format of every array the decoder emits (`centers` is plain xyz float32, not a texture). */
export const ARRAY_FORMATS = /** @type {const} */ ({
    means_l: "rgba8",
    means_u: "rgba8",
    quats: "rgba8",
    scales: "rgba8",
    sh0: "rgba8",
    sh_labels: "rgba8",
    sh_centroids: "rgba8",
    motionA: "rgba32u",
    motionB: "rgba16f",
    centers: "xyz32f",
});

/**
 * Size of the SOG textures for n splats (as SOG v2 files lay them out).
 * @param {number} n
 * @returns {{ width: number, height: number }}
 */
export function sogTextureSize(n) {
    const width = Math.max(1, Math.ceil(Math.sqrt(n)));
    return { width, height: Math.max(1, Math.ceil(n / width)) };
}

/**
 * Rows of the two motion textures for n splats.
 * @param {number} n
 * @returns {{ rowsA: number, rowsB: number }}
 */
export function motionTextureRows(n) {
    return {
        rowsA: Math.max(1, Math.ceil(n / MOTION_TEX_WIDTH)),
        rowsB: Math.max(1, Math.ceil(n * MOTION_B_TEXELS / MOTION_TEX_WIDTH)),
    };
}

/**
 * Size of the SH palette texture.
 * @param {{ count: number, coeffs: number }} shN - manifest.shN
 * @returns {{ width: number, height: number }}
 */
export function shCentroidsSize(shN) {
    return { width: 64 * shN.coeffs, height: Math.ceil(shN.count / 64) };
}

/**
 * The clip table (format spec 5) from the manifest.
 * @param {import("./container.js").AsogManifest} manifest
 * @returns {Float32Array} CLIP_FLOATS per clip
 */
export function clipTable(manifest) {
    const fps = manifest.timeline.frameRate;
    const table = new Float32Array(manifest.clips.length * CLIP_FLOATS);
    manifest.clips.forEach((clip, c) => {
        const G = clip.global, L = clip.local, R = clip.render;
        table.set([
            (G.frameStart - L.frameStart) / fps, G.activeTimeStart, G.activeTimeEnd, 0,
            G.coreTimeStart, G.coreTimeEnd, (R?.fadeInFrames ?? 0) / fps, (R?.fadeOutFrames ?? 0) / fps,
        ], c * CLIP_FLOATS);
    });
    return table;
}
