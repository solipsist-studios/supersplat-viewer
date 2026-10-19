import { idbGetEntryUnit } from './sogst-cache';
import { ZIP_LFH, ZIP_LOCAL_MAGIC } from './zip';

// Re-fetch of a segment's archive entries after the segment window evicted
// the decoded segment. The stream records, for every entry it passes on,
// which cache unit holds it and where it sits in the archive. A later fetch
// reads the unit back from IndexedDB, and falls back to HTTP Range requests
// against the archive when the unit is missing (quota, private browsing, or
// a unit the writer has not committed yet).

type IndexedEntry = {
    // index of the cache unit (sogst-cache.ts) that holds the entry
    unit: number;
    // archive offset of the entry's local file header
    offset: number;
    // payload size in bytes
    size: number;
};

class SogstEntryIndex {
    private entries = new Map<string, IndexedEntry>();

    record(unit: number, entries: { name: string; offset: number; size: number }[]) {
        for (const { name, offset, size } of entries) {
            this.entries.set(name, { unit, offset, size });
        }
    }

    get(name: string): IndexedEntry | undefined {
        return this.entries.get(name);
    }
}

class SogstSegmentSource {
    private url: string;

    private cacheKey: string | null;

    readonly index: SogstEntryIndex;

    constructor(url: string, cacheKey: string | null, index: SogstEntryIndex) {
        this.url = url;
        this.cacheKey = cacheKey;
        this.index = index;
    }

    // True once every named entry of the group has passed through the stream.
    has(prefix: string, names: string[]): boolean {
        return names.every((name) => this.index.get(`${prefix}/${name}`));
    }

    // The group's payloads keyed by bare name, or null when an entry has not
    // been seen yet. `optional` names (deferred SH labels) are included when
    // indexed and skipped otherwise. Throws when an entry cannot be read
    // from either the cache or the network.
    async get(prefix: string, names: string[], optional: string[] = []): Promise<Map<string, Uint8Array> | null> {
        if (!this.has(prefix, names)) {
            return null;
        }
        const wanted = [...names, ...optional.filter((name) => this.index.get(`${prefix}/${name}`))];
        const files = new Map<string, Uint8Array>();

        // cache units first, one read per unit
        if (this.cacheKey) {
            const units = new Set(wanted.map((name) => this.index.get(`${prefix}/${name}`)!.unit));
            for (const unit of units) {
                const entries = await idbGetEntryUnit(this.cacheKey, unit);
                for (const entry of entries ?? []) {
                    if (entry.name.startsWith(`${prefix}/`)) {
                        const bare = entry.name.slice(prefix.length + 1);
                        if (wanted.includes(bare)) {
                            files.set(bare, new Uint8Array(entry.bytes));
                        }
                    }
                }
            }
        }

        // whatever the cache could not supply comes from the archive
        for (const name of wanted) {
            if (!files.has(name)) {
                files.set(name, await this.fetchEntry(`${prefix}/${name}`, this.index.get(`${prefix}/${name}`)!));
            }
        }
        return files;
    }

    private async fetchRange(start: number, end: number): Promise<Uint8Array> {
        const response = await fetch(this.url, { headers: { Range: `bytes=${start}-${end - 1}` } });
        // 200 would be the whole archive: refuse it rather than download it
        if (response.status !== 206) {
            throw new Error(`sogst: range request for ${this.url} returned ${response.status}, not 206`);
        }
        const bytes = new Uint8Array(await response.arrayBuffer());
        if (bytes.length !== end - start) {
            throw new Error(`sogst: range request for ${this.url} returned ${bytes.length} of ${end - start} bytes`);
        }
        return bytes;
    }

    // Two requests: the local header, which gives the extra-field length the
    // index does not record, then the payload.
    private async fetchEntry(name: string, entry: IndexedEntry): Promise<Uint8Array> {
        const header = await this.fetchRange(entry.offset, entry.offset + ZIP_LFH.size);
        const view = new DataView(header.buffer, header.byteOffset, header.byteLength);
        if (view.getUint32(0, true) !== ZIP_LOCAL_MAGIC) {
            throw new Error(`sogst: no local header for ${name} at archive offset ${entry.offset}`);
        }
        const nameLength = view.getUint16(ZIP_LFH.nameLength, true);
        const extraLength = view.getUint16(ZIP_LFH.extraLength, true);
        const start = entry.offset + ZIP_LFH.size + nameLength + extraLength;
        return this.fetchRange(start, start + entry.size);
    }
}

export { SogstEntryIndex, SogstSegmentSource };
