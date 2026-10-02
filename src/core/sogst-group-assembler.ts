import type { SogstMeta } from '../parsers/sogst';

import { parseSogstMeta } from './load-sogst';
import type { CachedEntry } from './sogst-cache';
import { enumerateSogstGroups, groupBaseNames, groupFileList } from './sogst-decoder';
import type { SogstGroup } from './sogst-decoder';

// Turns a streamed archive's entries, in archive order, into the steps a
// decoder acts on: the manifest, the SH centroids, each complete group, and
// each deferred SH labels payload. It decodes nothing and calls nothing; it
// only copies the bytes it keeps. Feeding it entries is all a test needs.
//
// Alongside each step it hands back the step's cache unit: the run of
// entries that step consumed, for the IndexedDB writer (sogst-cache.ts).
// Unit boundaries depend only on the archive, so a replay of the cache
// reproduces them exactly.

type AssemblerStep =
    | { kind: 'meta'; meta: SogstMeta; monolithic: boolean }
    | { kind: 'centroids'; bytes: Uint8Array }
    | { kind: 'group'; index: number; group: SogstGroup; files: Map<string, Uint8Array> }
    | { kind: 'labels'; group: SogstGroup; bytes: Uint8Array };

type AssemblerResult = {
    // null for an entry the stream ignores (a stray or unknown entry, or any
    // entry of a monolithic archive), or one still filling its group
    step: AssemblerStep | null;
    // the entries that make up a completed step, ready to cache
    unit: CachedEntry[] | null;
};

const LABELS_SUFFIX = '/shN_labels.webp';

class SogstGroupAssembler {
    meta: SogstMeta | null = null;

    monolithic = false;

    groups: SogstGroup[] = [];

    // index of the group whose entries are arriving
    private groupIdx = 0;

    // entry names a group needs before it can decode
    private needed: string[] = [];

    private pending = new Map<string, Uint8Array>();

    private unit: CachedEntry[] = [];

    private takeUnit(): CachedEntry[] | null {
        if (!this.unit.length) {
            return null;
        }
        const unit = this.unit;
        this.unit = [];
        return unit;
    }

    // `data` may be a view into the source's buffer; anything kept is copied.
    accept(name: string, data: Uint8Array, offset: number): AssemblerResult {
        if (name === 'meta.json') {
            const meta = parseSogstMeta(data.slice());
            this.meta = meta;
            if (!meta.streams) {
                // monolithic: the caller decodes the whole file at the end
                this.monolithic = true;
                return { step: { kind: 'meta', meta, monolithic: true }, unit: null };
            }
            this.unit.push({ name, offset, bytes: data.slice().buffer });
            this.groups = enumerateSogstGroups(meta);
            // sh-deferred archives ship labels behind all geometry, so a
            // geometry group completes on the base texture set alone
            this.needed = meta.streams.sh_deferred ? [...groupBaseNames(meta)] : groupFileList(meta);
            return { step: { kind: 'meta', meta, monolithic: false }, unit: null };
        }
        if (!this.meta) {
            throw new Error(`sogst: unexpected entry ${name} before meta.json`);
        }
        if (this.monolithic) {
            return { step: null, unit: null };
        }
        if (name === 'shN_centroids.webp') {
            const bytes = data.slice();
            this.unit.push({ name, offset, bytes: bytes.buffer });
            return { step: { kind: 'centroids', bytes }, unit: null };
        }
        if (this.meta.streams.sh_deferred && name.endsWith(LABELS_SUFFIX)) {
            const prefix = name.slice(0, -LABELS_SUFFIX.length);
            const group = this.groups.find((g) => g.prefix === prefix);
            if (!group) {
                return { step: null, unit: null };
            }
            const bytes = data.slice();
            this.unit.push({ name, offset, bytes: bytes.buffer });
            return { step: { kind: 'labels', group, bytes }, unit: this.takeUnit() };
        }
        const group = this.groups[this.groupIdx];
        if (!group || !name.startsWith(`${group.prefix}/`)) {
            return { step: null, unit: null }; // stray entry, or all groups done
        }
        const bare = name.slice(group.prefix!.length + 1);
        // Ignore an entry that backs a group member this build does not
        // know. Spec section 3.1 requires this: additive groups ship under
        // version 1, and a player must degrade the way it does for shN and
        // accel.
        //
        // The filter also keeps the completion test correct. That test
        // counts entries. An unrecognised entry would raise the count to the
        // needed size while a required name was still missing, and the
        // decode would start one file short. Whether that happened depended
        // on the order in which the encoder wrote the entries.
        if (!this.needed.includes(bare)) {
            return { step: null, unit: null };
        }
        const bytes = data.slice();
        this.pending.set(bare, bytes);
        this.unit.push({ name, offset, bytes: bytes.buffer });
        if (this.pending.size < this.needed.length) {
            return { step: null, unit: null };
        }
        return this.completeGroup();
    }

    private completeGroup(): AssemblerResult {
        const index = this.groupIdx++;
        const files = this.pending;
        this.pending = new Map();
        return { step: { kind: 'group', index, group: this.groups[index], files }, unit: this.takeUnit() };
    }

    // End of the archive: a trailing complete group, if one is somehow still
    // pending, and any entries not yet handed back as a unit.
    finish(): AssemblerResult {
        if (this.meta && !this.monolithic && this.groupIdx < this.groups.length) {
            if (this.pending.size && this.pending.size === this.needed.length) {
                return this.completeGroup();
            }
        }
        return { step: null, unit: this.takeUnit() };
    }
}

export { SogstGroupAssembler };
export type { AssemblerResult, AssemblerStep };
