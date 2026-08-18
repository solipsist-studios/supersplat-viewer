// Minimal ZIP reader for .sogst containers. Both the whole-file loader
// (`load-sogst.ts`) and the incremental streamer (`stream-sogst.ts`) walk
// the same records, so the record layout lives here rather than as two sets
// of bare byte offsets.
//
// Layout constants are field offsets from PKWARE's APPNOTE 6.3.x, section
// 4.3. Each record is a fixed-size header followed by variable-length
// name/extra/comment fields, so `size` is the fixed part only.

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

// The EOCD carries a trailing comment of up to 0xffff bytes, and its own
// length field sits inside the record — so the only way to find the record
// is to scan backwards over the widest comment it could have.
const ZIP_MAX_COMMENT_BYTES = 0xffff;

// General-purpose bit 3: sizes are zero in the local header and follow the
// payload in a data descriptor instead.
const ZIP_FLAG_DATA_DESCRIPTOR = 0x8;

// Compression methods this reader accepts.
const ZIP_METHOD_STORE = 0;
const ZIP_METHOD_DEFLATE = 8;

type ZipEntry = {
    filename: string;
    deflated: boolean;
    data: Uint8Array;
};

// Central-directory walk over a complete archive (stored + deflate
// entries). Our encoder always writes stored entries; deflate is handled
// for robustness against re-zipped files.
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
        // The central directory's name/extra lengths need not match the
        // local header's, so the payload offset comes from the local header.
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

const inflateRaw = async (compressed: Uint8Array): Promise<Uint8Array> => {
    const stream = new Blob([compressed as unknown as ArrayBuffer])
        .stream()
        .pipeThrough(new DecompressionStream('deflate-raw'));
    return new Uint8Array(await new Response(stream).arrayBuffer());
};

export {
    inflateRaw,
    parseZipEntries,
    ZIP_CDR,
    ZIP_EOCD,
    ZIP_FLAG_DATA_DESCRIPTOR,
    ZIP_LFH,
    ZIP_LOCAL_MAGIC,
    ZIP_CDR_MAGIC,
    ZIP_EOCD_MAGIC
};
export type { ZipEntry };
