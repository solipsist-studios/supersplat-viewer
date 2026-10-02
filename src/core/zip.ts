// Minimal ZIP reader for .sogst containers. The whole-file loader
// (`load-sogst.ts`) and the incremental streamer (`stream-sogst.ts`) read
// the same records. The layout lives here so that the two parsers cannot
// disagree about the format.
//
// The layout constants are field offsets from PKWARE's APPNOTE 6.3.x,
// section 4.3. Each record starts with a fixed-size header. Variable-length
// name, extra and comment fields follow it, so `size` covers the header
// only.

// Record signatures, little-endian.
const ZIP_LOCAL_MAGIC = 0x04034b50; // "PK\x03\x04"
const ZIP_CDR_MAGIC = 0x02014b50; // "PK\x01\x02"
const ZIP_EOCD_MAGIC = 0x06054b50; // "PK\x05\x06"

// Local file header (APPNOTE 4.3.7).
const ZIP_LFH = {
    size: 30,
    flags: 6, // u16 general-purpose bit flag
    compressedSize: 18, // u32
    nameLength: 26, // u16
    extraLength: 28 // u16
} as const;

// Central directory record (APPNOTE 4.3.12).
const ZIP_CDR = {
    size: 46,
    compression: 10, // u16
    compressedSize: 20, // u32
    nameLength: 28, // u16
    extraLength: 30, // u16
    commentLength: 32, // u16
    localHeaderOffset: 42 // u32
} as const;

// End of central directory record (APPNOTE 4.3.16).
const ZIP_EOCD = {
    size: 22,
    entryCount: 8, // u16 entries in the central directory on this disk
    cdrOffset: 16 // u32 offset of the first central-directory record
} as const;

// The EOCD carries a trailing comment of up to 0xffff bytes. Its own length
// field sits inside the record, so a reader cannot seek to it. The reader
// must scan backwards over the widest comment the record can have.
const ZIP_MAX_COMMENT_BYTES = 0xffff;

// General-purpose bit 3. The writer sets it when it writes zero sizes in the
// local header and puts the real sizes in a data descriptor after the
// payload.
const ZIP_FLAG_DATA_DESCRIPTOR = 0x8;

// Compression methods this reader accepts.
const ZIP_METHOD_STORE = 0;
const ZIP_METHOD_DEFLATE = 8;

type ZipEntry = {
    filename: string;
    deflated: boolean;
    data: Uint8Array;
};

// Central-directory walk over a complete archive. Our encoder always writes
// stored entries. This reader also accepts deflate entries, because another
// tool can re-zip the archive.
const parseZipEntries = (buffer: ArrayBuffer): ZipEntry[] => {
    const view = new DataView(buffer);
    const u16 = (o: number) => view.getUint16(o, true);
    const u32 = (o: number) => view.getUint32(o, true);

    let eocd = -1;
    const scanEnd = Math.max(0, buffer.byteLength - ZIP_EOCD.size - ZIP_MAX_COMMENT_BYTES);
    for (let o = buffer.byteLength - ZIP_EOCD.size; o >= scanEnd; o--) {
        if (u32(o) === ZIP_EOCD_MAGIC) {
            eocd = o;
            break;
        }
    }
    if (eocd < 0) {
        throw new Error('sogst: invalid zip (no end-of-central-directory)');
    }

    const numFiles = u16(eocd + ZIP_EOCD.entryCount);
    let offset = u32(eocd + ZIP_EOCD.cdrOffset);
    const entries: ZipEntry[] = [];
    for (let i = 0; i < numFiles; i++) {
        if (u32(offset) !== ZIP_CDR_MAGIC) {
            throw new Error('sogst: invalid zip (bad central-directory record)');
        }
        const compression = u16(offset + ZIP_CDR.compression);
        const compressedSize = u32(offset + ZIP_CDR.compressedSize);
        const filenameLength = u16(offset + ZIP_CDR.nameLength);
        const extraLength = u16(offset + ZIP_CDR.extraLength);
        const commentLength = u16(offset + ZIP_CDR.commentLength);
        const lfhOffset = u32(offset + ZIP_CDR.localHeaderOffset);
        const filename = new TextDecoder().decode(new Uint8Array(buffer, offset + ZIP_CDR.size, filenameLength));

        if (u32(lfhOffset) !== ZIP_LOCAL_MAGIC) {
            throw new Error('sogst: invalid zip (bad local file header)');
        }
        // The central directory can record different name and extra lengths
        // from the local header, so read the payload offset from the local
        // header.
        const dataOffset =
            lfhOffset + ZIP_LFH.size + u16(lfhOffset + ZIP_LFH.nameLength) + u16(lfhOffset + ZIP_LFH.extraLength);

        if (compression !== ZIP_METHOD_STORE && compression !== ZIP_METHOD_DEFLATE) {
            throw new Error(`sogst: unsupported zip compression method ${compression}`);
        }
        entries.push({
            filename,
            deflated: compression === ZIP_METHOD_DEFLATE,
            data: new Uint8Array(buffer, dataOffset, compressedSize)
        });
        offset += ZIP_CDR.size + filenameLength + extraLength + commentLength;
    }
    return entries;
};

// One stored entry read off a stream, with its local header's offset in the
// archive. `data` is a view into the reader's buffer: copy it before the
// next append(), which may compact or reallocate that buffer.
type ZipStreamEntry = {
    name: string;
    data: Uint8Array;
    offset: number;
};

// Starting size of a ZipStreamReader's buffer. It grows by doubling.
const ZIP_STREAM_INITIAL_BYTES = 1 << 20;

// Incremental reader for a ZIP arriving in chunks. It walks entries by their
// local headers, so it reads each entry as soon as the entry's bytes are in,
// without the central directory.
//
// The buffer keeps only bytes not yet read: append() first drops everything
// next() has returned, so it grows only to fit the largest entry. Set
// `retainAll` before the next append() to keep the whole archive instead (a
// caller that must decode the complete file afterwards).
//
// The walk relies on local headers carrying real sizes, so it rejects
// entries that use data descriptors (general-purpose bit 3), whose local
// sizes are zero.
class ZipStreamReader {
    retainAll = false;

    private bytes = new Uint8Array(ZIP_STREAM_INITIAL_BYTES);

    // unread bytes are bytes[start, end); bytes[0] is archive offset `base`
    private start = 0;

    private end = 0;

    private base = 0;

    private done = false;

    // Total bytes appended so far.
    received = 0;

    append(chunk: Uint8Array) {
        if (!this.retainAll && this.start > 0) {
            this.bytes.copyWithin(0, this.start, this.end);
            this.base += this.start;
            this.end -= this.start;
            this.start = 0;
        }
        const needed = this.end + chunk.length;
        if (needed > this.bytes.length) {
            let size = this.bytes.length;
            while (size < needed) {
                size *= 2;
            }
            const grown = new Uint8Array(size);
            grown.set(this.bytes.subarray(0, this.end), 0);
            this.bytes = grown;
        }
        this.bytes.set(chunk, this.end);
        this.end += chunk.length;
        this.received += chunk.length;
    }

    // The next complete entry, or null until more bytes arrive (or after
    // the central directory).
    next(): ZipStreamEntry | null {
        if (this.done || this.end - this.start < ZIP_LFH.size) {
            return null;
        }
        const view = new DataView(this.bytes.buffer, this.start, ZIP_LFH.size);
        if (view.getUint32(0, true) !== ZIP_LOCAL_MAGIC) {
            // central directory reached — no more entries
            this.done = true;
            return null;
        }
        // A writer that emits data descriptors writes zero local sizes. The
        // entry would then end at its header, and the next read would start
        // inside the payload and treat it as an entry: garbage names and no
        // error. This is a malformed archive, not a stream underrun.
        if ((view.getUint16(ZIP_LFH.flags, true) & ZIP_FLAG_DATA_DESCRIPTOR) !== 0) {
            throw new Error('sogst: archive uses ZIP data descriptors, which the format forbids');
        }
        const compressedSize = view.getUint32(ZIP_LFH.compressedSize, true);
        const nameLength = view.getUint16(ZIP_LFH.nameLength, true);
        const extraLength = view.getUint16(ZIP_LFH.extraLength, true);
        const total = ZIP_LFH.size + nameLength + extraLength + compressedSize;
        if (this.end - this.start < total) {
            return null;
        }
        const nameStart = this.start + ZIP_LFH.size;
        const name = new TextDecoder().decode(this.bytes.subarray(nameStart, nameStart + nameLength));
        const data = this.bytes.subarray(nameStart + nameLength + extraLength, this.start + total);
        const offset = this.base + this.start;
        this.start += total;
        return { name, data, offset };
    }

    // The whole archive. Only meaningful with retainAll set from the start.
    archive(): ArrayBuffer {
        return this.bytes.slice(0, this.end).buffer;
    }
}

const inflateRaw = async (compressed: Uint8Array): Promise<Uint8Array> => {
    const stream = new Blob([compressed as unknown as ArrayBuffer])
        .stream()
        .pipeThrough(new DecompressionStream('deflate-raw'));
    return new Uint8Array(await new Response(stream).arrayBuffer());
};

export {
    inflateRaw,
    parseZipEntries,
    ZipStreamReader,
    ZIP_CDR,
    ZIP_EOCD,
    ZIP_FLAG_DATA_DESCRIPTOR,
    ZIP_LFH,
    ZIP_LOCAL_MAGIC,
    ZIP_CDR_MAGIC,
    ZIP_EOCD_MAGIC
};
export type { ZipEntry, ZipStreamEntry };
