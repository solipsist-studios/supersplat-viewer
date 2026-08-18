// Durable IndexedDB cache for SOGST payloads, shared by the byte-range chunk
// streamer and the whole-file loader. Everything lives in one database/store
// so the debug UI's "Clear SOGST Cache" wipes both.
//
// The cache splits a large payload across several entries. Structured-cloning
// one ArrayBuffer of several hundred MB into IndexedDB raises memory enough
// to crash the tab. We saw this on 300MB files.
//
// A manifest entry at the base key describes the pieces, which live at
// `<key>#<i>`. Each piece is written in its own transaction, so peak overhead
// stays near the size of one piece.

const SOGST_IDB_NAME = 'supersplat-sogst-chunks';
const SOGST_IDB_STORE = 'ranges';

// 32MB pieces. This size clones without memory pressure, and it keeps a
// 300MB file to about 10 transactions.
const PIECE_BYTES = 32 * 1024 * 1024;

type PieceManifest = {
    sogstPieces: number;
    totalBytes: number;
};

const isManifest = (value: unknown): value is PieceManifest => {
    return (
        !!value &&
        typeof value === 'object' &&
        typeof (value as PieceManifest).sogstPieces === 'number' &&
        typeof (value as PieceManifest).totalBytes === 'number'
    );
};

let sogstDbPromise: Promise<IDBDatabase | null> | null = null;

const openSogstDb = (): Promise<IDBDatabase | null> => {
    if (typeof indexedDB === 'undefined') {
        return Promise.resolve(null);
    }

    if (sogstDbPromise) {
        return sogstDbPromise;
    }

    sogstDbPromise = new Promise((resolve) => {
        const request = indexedDB.open(SOGST_IDB_NAME, 1);

        request.onupgradeneeded = () => {
            const db = request.result;
            if (!db.objectStoreNames.contains(SOGST_IDB_STORE)) {
                db.createObjectStore(SOGST_IDB_STORE);
            }
        };

        request.onsuccess = () => resolve(request.result);
        request.onerror = () => resolve(null);
    });

    return sogstDbPromise;
};

const idbGetValue = (db: IDBDatabase, key: string): Promise<unknown> => {
    return new Promise((resolve) => {
        const tx = db.transaction(SOGST_IDB_STORE, 'readonly');
        const store = tx.objectStore(SOGST_IDB_STORE);
        const request = store.get(key);

        request.onsuccess = () => resolve(request.result);
        request.onerror = () => resolve(null);
    });
};

const idbPutValue = (db: IDBDatabase, key: string, value: unknown): Promise<boolean> => {
    return new Promise((resolve) => {
        let ok = false;
        try {
            const tx = db.transaction(SOGST_IDB_STORE, 'readwrite');
            const store = tx.objectStore(SOGST_IDB_STORE);
            const request = store.put(value, key);
            request.onsuccess = () => {
                ok = true;
            };
            tx.oncomplete = () => resolve(ok);
            tx.onerror = () => resolve(false);
            tx.onabort = () => resolve(false);
        } catch {
            resolve(false);
        }
    });
};

const idbGetBuffer = async (key: string): Promise<ArrayBuffer | null> => {
    const db = await openSogstDb();
    if (!db) {
        return null;
    }

    const value = await idbGetValue(db, key);

    // Payloads below PIECE_BYTES are stored directly as ArrayBuffers, with
    // no manifest entry.
    if (value instanceof ArrayBuffer) {
        return value;
    }

    if (!isManifest(value)) {
        return null;
    }

    // Reassemble a pieced payload, one transaction per piece.
    const result = new Uint8Array(value.totalBytes);
    let offset = 0;
    for (let i = 0; i < value.sogstPieces; i++) {
        const piece = await idbGetValue(db, `${key}#${i}`);
        if (!(piece instanceof ArrayBuffer) || offset + piece.byteLength > value.totalBytes) {
            return null;
        }
        result.set(new Uint8Array(piece), offset);
        offset += piece.byteLength;
    }

    return offset === value.totalBytes ? result.buffer : null;
};

const idbSetBuffer = async (key: string, buffer: ArrayBuffer): Promise<void> => {
    const db = await openSogstDb();
    if (!db) {
        return;
    }

    if (buffer.byteLength <= PIECE_BYTES) {
        await idbPutValue(db, key, buffer);
        return;
    }

    // Write the pieces first, then write the manifest last. A reader then
    // never sees a manifest whose pieces are missing. Each slice is a
    // temporary copy of at most PIECE_BYTES.
    const pieces = Math.ceil(buffer.byteLength / PIECE_BYTES);
    for (let i = 0; i < pieces; i++) {
        const piece = buffer.slice(i * PIECE_BYTES, Math.min((i + 1) * PIECE_BYTES, buffer.byteLength));

        const ok = await idbPutValue(db, `${key}#${i}`, piece);
        if (!ok) {
            return;
        }
    }

    await idbPutValue(db, key, { sogstPieces: pieces, totalBytes: buffer.byteLength } satisfies PieceManifest);
};

// Delete every entry whose key starts with the prefix, except `keep` and its
// piece entries. This drops stale copies of a file when its validator
// changes.
const idbDeleteByPrefix = async (prefix: string, keep?: string): Promise<void> => {
    const db = await openSogstDb();
    if (!db) {
        return;
    }

    await new Promise<void>((resolve) => {
        const tx = db.transaction(SOGST_IDB_STORE, 'readwrite');
        const store = tx.objectStore(SOGST_IDB_STORE);
        const request = store.getAllKeys();

        request.onsuccess = () => {
            for (const key of request.result) {
                if (typeof key !== 'string' || !key.startsWith(prefix)) {
                    continue;
                }
                if (keep && (key === keep || key.startsWith(`${keep}#`))) {
                    continue;
                }
                store.delete(key);
            }
        };
        tx.oncomplete = () => resolve();
        tx.onerror = () => resolve();
        tx.onabort = () => resolve();
    });
};

export { idbGetBuffer, idbSetBuffer, idbDeleteByPrefix };
