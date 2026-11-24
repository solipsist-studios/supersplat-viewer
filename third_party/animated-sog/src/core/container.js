/**
 * The .asog container (format spec section 1) and manifest (section 2).
 */

export const ASOG_MAGIC = "ASOG";
export const HEADER_SIZE = 64;

/**
 * @typedef {object} AsogHeader
 * @property {number} manifestOffset - byte offset of the UTF-8 JSON manifest
 * @property {number} manifestLength - manifest size in bytes
 * @property {number} payloadOffset - byte offset of the payload; every manifest byteOffset is relative to it
 * @property {number} payloadLength - payload size in bytes
 */

/**
 * @typedef {object} AsogStreamDesc
 * @property {string} name
 * @property {number} count - rows (splats, or SH palette entries for `shpalette`)
 * @property {number} channels
 * @property {number} planes - bytes per value
 * @property {boolean} delta
 * @property {boolean} zigzag
 * @property {number} byteOffset - relative to the payload
 * @property {number} byteLength - compressed size
 */

/**
 * @typedef {object} AsogManifest
 * @property {"asog"} format
 * @property {1} version
 * @property {number} count
 * @property {{ frameRate: number, durationSeconds: number }} timeline
 * @property {Array<{ global: any, local: any, render?: any }>} clips
 * @property {{ mins: number[], maxs: number[] }} means
 * @property {{ codebook: number[] }} scales
 * @property {{ codebook: number[] }} sh0
 * @property {{ codebook: number[], count: number, coeffs: number }} [shN]
 * @property {{ scale: number }} mu
 * @property {{ min: number, max: number, bits: number }} logsigma
 * @property {{ qp: number, qa: number, sm: number, midO: number, midG: number, mref_samples: number, r_min: number }} motion
 * @property {{ mime: string, byteOffset: number, byteLength: number }} [audio]
 * @property {AsogStreamDesc[]} streams
 */

/**
 * Parses the 64-byte header.
 * @param {Uint8Array} bytes - the file's first 64 bytes
 * @returns {AsogHeader}
 */
export function parseHeader(bytes) {
    if (bytes.length < HEADER_SIZE) throw new Error("asog: header too short");
    const magic = String.fromCharCode(bytes[0], bytes[1], bytes[2], bytes[3]);
    if (magic !== ASOG_MAGIC) throw new Error(`asog: not an .asog file (magic "${magic}")`);
    const dv = new DataView(bytes.buffer, bytes.byteOffset, HEADER_SIZE);
    const u64 = (o) => Number(dv.getBigUint64(o, true));
    return {
        manifestOffset: u64(16),
        manifestLength: u64(24),
        payloadOffset: u64(32),
        payloadLength: u64(40),
    };
}

/**
 * Parses and checks the JSON manifest.
 * @param {Uint8Array} bytes
 * @returns {AsogManifest}
 */
export function parseManifest(bytes) {
    const manifest = JSON.parse(new TextDecoder().decode(bytes));
    if (manifest.format !== "asog" || manifest.version !== 1) {
        throw new Error(`asog: unsupported format ${manifest.format} v${manifest.version}`);
    }
    return manifest;
}

/**
 * SH bands (0-3) for a number of SH coefficients per channel (0, 3, 8, 15).
 * @param {number} coeffs
 * @returns {number}
 */
export function shBandsFromCoeffs(coeffs) {
    return { 3: 1, 8: 2, 15: 3 }[coeffs] ?? 0;
}
