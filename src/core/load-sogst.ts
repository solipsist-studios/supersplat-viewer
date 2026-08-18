import type { AppBase, BoundingBox } from 'playcanvas';

import { SOGST_META_FORMAT, SOGST_META_VERSION } from '../parsers/sogst';
import type { SogstMeta } from '../parsers/sogst';

import type { SogstData } from './sogst-data';
import { SogstDecoder, enumerateSogstGroups, groupFileList } from './sogst-decoder';
import { inflateRaw, parseZipEntries, ZIP_LOCAL_MAGIC } from './zip';

// .sogst loading: SOG-compressed temporal splats. parsers/sogst.ts describes
// the container. This file turns a complete archive into playable data, and
// stream-sogst.ts does the same work incrementally off the network.
//
// The file is a ZIP archive that holds a meta.json and lossless-webp
// attribute textures. The leading "PK\x03\x04" magic identifies it.
//
// The static attributes follow the PlayCanvas SOG v2 conventions exactly, so
// the engine's own GSplatSogData decoder reconstructs them unchanged:
//
//   means_l/means_u  : 16-bit split, log-transformed
//   quats            : smallest-three
//   scales, sh0      : 256-entry codebook indices, opacity in sh0's alpha
//   shN_centroids,
//   shN_labels       : optional, vector-quantised higher-order SH
//
// Two more textures carry the temporal model:
//
//   motion_l/motion_u : per-axis 16-bit split of velocity, with the same
//                       sign(x)*ln(1+|x|) transform and mins/maxs as means
//   trbf              : R = index into trbf.center codebook (t_center, s),
//                       G = index into trbf.sigma codebook (t_sigma, s)
//
// The other pieces are next door. zip.ts reads the container, sogst-texels.ts
// decodes the webp payloads, sogst-decoder.ts fills the attribute arrays, and
// sogst-data.ts holds the decoded result the playback path reads.

// True if the buffer starts with the ZIP local-file magic, because the
// container is a ZIP.
const isSogstArchive = (buffer: ArrayBuffer): boolean => {
    return buffer.byteLength >= 4 && new DataView(buffer).getUint32(0, true) === ZIP_LOCAL_MAGIC;
};

const parseSogstMeta = (bytes: Uint8Array | undefined): SogstMeta => {
    if (!bytes) {
        throw new Error('sogst: meta.json not found in archive');
    }
    const meta = JSON.parse(new TextDecoder().decode(bytes));
    // Both keys are REQUIRED. This rejects any other value, and it also
    // rejects a missing `format`.
    if (meta.version !== SOGST_META_VERSION) {
        throw new Error(`sogst: expected meta version ${SOGST_META_VERSION}, got ${meta.version}`);
    }
    if (meta.format !== SOGST_META_FORMAT) {
        throw new Error(`sogst: expected meta format '${SOGST_META_FORMAT}', got ${JSON.stringify(meta.format)}`);
    }
    return meta;
};

// Decode a complete archive into playable data. This accepts either layout.
// Two callers use it: non-streamed archives, and cache hits on streamed
// archives.
const loadSogst = async (
    app: AppBase,
    buffer: ArrayBuffer,
    onProgress?: (progress: number) => void
): Promise<SogstData> => {
    const report = (value: number) => onProgress?.(Math.min(100, Math.round(value)));

    const entries = parseZipEntries(buffer);
    const files = new Map<string, Uint8Array>();
    for (const entry of entries) {
        files.set(entry.filename, entry.deflated ? await inflateRaw(entry.data) : entry.data);
    }

    const meta = parseSogstMeta(files.get('meta.json'));
    const decoder = new SogstDecoder(app, meta);
    if (meta.shN) {
        const centroidsName = meta.streams ? 'shN_centroids.webp' : meta.shN.files[0];
        decoder.setCentroids(files.get(centroidsName)!);
    }

    const groups = enumerateSogstGroups(meta);
    const names = groupFileList(meta);
    let done = 0;
    for (const group of groups) {
        const groupBytes = new Map<string, Uint8Array>();
        for (const name of names) {
            const stored = files.get(group.prefix ? `${group.prefix}/${name}` : name);
            if (stored) {
                groupBytes.set(name, stored);
            }
        }
        const m = group.range[1] - group.range[0];
        const base = done;

        await decoder.decodeGroup(group, groupBytes, (frac) => report(((base + frac * m) / meta.count) * 100));
        done += m;
    }

    const data = decoder.buildData();
    decoder.destroy();
    report(100);
    return data;
};

// Set exact scene bounds from the archive's global means range. A streaming
// load fills the attribute arrays only in part, so bounds computed from those
// arrays would understate the scene. The mins and maxs are in the SOG
// log-transformed space. Invert them with sign(v) * (e^|v| - 1).
const setAabbFromMeta = (meta: SogstMeta, aabb: BoundingBox) => {
    const mins = meta.means.mins as number[];
    const maxs = meta.means.maxs as number[];
    const map = (v: number) => Math.sign(v) * (Math.exp(Math.abs(v)) - 1);
    aabb.center.set(
        (map(mins[0]) + map(maxs[0])) * 0.5,
        (map(mins[1]) + map(maxs[1])) * 0.5,
        (map(mins[2]) + map(maxs[2])) * 0.5
    );
    aabb.halfExtents.set(
        (map(maxs[0]) - map(mins[0])) * 0.5,
        (map(maxs[1]) - map(mins[1])) * 0.5,
        (map(maxs[2]) - map(mins[2])) * 0.5
    );
};

export { isSogstArchive, loadSogst, parseSogstMeta, setAabbFromMeta };
