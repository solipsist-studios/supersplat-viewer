/**
 * animated-sog: engine-agnostic reading and decoding of Animated SOG (.asog) 4D Gaussian splat clips.
 * The PlayCanvas player is `animated-sog/playcanvas`.
 */
export { ASOG_MAGIC, HEADER_SIZE, parseHeader, parseManifest, shBandsFromCoeffs } from "./core/container.js";
export { BufferReader, StreamReader, openReader } from "./core/reader.js";
export { inflate } from "./core/inflate.js";
export { decodeChannel, decodeStream } from "./core/planes.js";
export {
    SH_C0, dequantAnchor, dequantG, dequantMu, dequantOmega, dequantPosition, dequantQuat, dequantSigma,
    gStep, lifeSpan, motionWeights, omegaStep,
} from "./core/dequant.js";
export { HALF_MAX, floatToHalf, getHalfTable, halfTable, halfToFloat } from "./core/half.js";
export {
    ARRAY_FORMATS, CLIP_FLOATS, MOTION_B_TEXELS, MOTION_TEX_WIDTH, clipTable, motionTextureRows, shCentroidsSize, sogTextureSize,
} from "./core/layout.js";
export {
    TEMPORAL_THRESHOLD, clipWeight, createMotion, evaluateCenters, evaluateSplat, quatMultiply, readMotion, temporalWeight,
} from "./core/motion.js";
export { decodeAsog, openAsog } from "./core/decode.js";
export { centerBounds, evaluateSplatAt, readSplat, readSplatMotion } from "./core/splats.js";
export { sortSplats } from "./core/sort.js";
export { sortWorkerSource } from "./core/worker.js";
export { createPacer, free } from "./core/util.js";
