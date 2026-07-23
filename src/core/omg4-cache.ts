// Durable IndexedDB cache for OMG4 payloads, shared by the v1 chunk streamer
// (byte-range entries) and the v2 full-file loader. Everything lives in one
// database/store so the debug UI's "Clear OMG4 Cache" wipes both.
//
// Large payloads are split across multiple entries: structured-cloning a
// single multi-hundred-MB ArrayBuffer into IndexedDB spikes memory hard
// enough to crash the tab (observed on 300MB files). A manifest entry at the
// base key describes the pieces, stored at `<key>#<i>`; each piece is written
// in its own transaction so peak overhead stays around one piece.

const OMG4_IDB_NAME = 'supersplat-omg4-chunks';
const OMG4_IDB_STORE = 'ranges';

// 32MB pieces: small enough to clone without memory pressure, large enough
// that a 300MB file is only ~10 transactions.
const PIECE_BYTES = 32 * 1024 * 1024;

type PieceManifest = {
    omg4Pieces: number;
    totalBytes: number;
};

const isManifest = (value: unknown): value is PieceManifest => {
    return !!value && typeof value === 'object' &&
        typeof (value as PieceManifest).omg4Pieces === 'number' &&
        typeof (value as PieceManifest).totalBytes === 'number';
};

let omg4DbPromise: Promise<IDBDatabase | null> | null = null;

const openOmg4Db = (): Promise<IDBDatabase | null> => {
    if (typeof indexedDB === 'undefined') {
        return Promise.resolve(null);
    }

    if (omg4DbPromise) {
        return omg4DbPromise;
    }

    omg4DbPromise = new Promise((resolve) => {
        const request = indexedDB.open(OMG4_IDB_NAME, 1);

        request.onupgradeneeded = () => {
            const db = request.result;
            if (!db.objectStoreNames.contains(OMG4_IDB_STORE)) {
                db.createObjectStore(OMG4_IDB_STORE);
            }
        };

        request.onsuccess = () => resolve(request.result);
        request.onerror = () => resolve(null);
    });

    return omg4DbPromise;
};

const idbGetValue = (db: IDBDatabase, key: string): Promise<unknown> => {
    return new Promise((resolve) => {
        const tx = db.transaction(OMG4_IDB_STORE, 'readonly');
        const store = tx.objectStore(OMG4_IDB_STORE);
        const request = store.get(key);

        request.onsuccess = () => resolve(request.result);
        request.onerror = () => resolve(null);
    });
};

const idbPutValue = (db: IDBDatabase, key: string, value: unknown): Promise<boolean> => {
    return new Promise((resolve) => {
        let ok = false;
        try {
            const tx = db.transaction(OMG4_IDB_STORE, 'readwrite');
            const store = tx.objectStore(OMG4_IDB_STORE);
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
    const db = await openOmg4Db();
    if (!db) {
        return null;
    }

    const value = await idbGetValue(db, key);

    // Small payloads (v1 range entries, pre-manifest v2 entries) are stored
    // directly as ArrayBuffers.
    if (value instanceof ArrayBuffer) {
        return value;
    }

    if (!isManifest(value)) {
        return null;
    }

    // Reassemble a pieced payload, one transaction per piece.
    const result = new Uint8Array(value.totalBytes);
    let offset = 0;
    for (let i = 0; i < value.omg4Pieces; i++) {
        // eslint-disable-next-line no-await-in-loop
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
    const db = await openOmg4Db();
    if (!db) {
        return;
    }

    if (buffer.byteLength <= PIECE_BYTES) {
        await idbPutValue(db, key, buffer);
        return;
    }

    // Write pieces first (each slice is a transient copy of at most
    // PIECE_BYTES), then the manifest last so readers never see a manifest
    // whose pieces are missing.
    const pieces = Math.ceil(buffer.byteLength / PIECE_BYTES);
    for (let i = 0; i < pieces; i++) {
        const piece = buffer.slice(i * PIECE_BYTES, Math.min((i + 1) * PIECE_BYTES, buffer.byteLength));
        // eslint-disable-next-line no-await-in-loop
        const ok = await idbPutValue(db, `${key}#${i}`, piece);
        if (!ok) {
            return;
        }
    }

    await idbPutValue(db, key, { omg4Pieces: pieces, totalBytes: buffer.byteLength } satisfies PieceManifest);
};

// Delete every entry whose key starts with the prefix except `keep` (and its
// piece entries) — used to drop stale copies of a file when its validator
// changes.
const idbDeleteByPrefix = async (prefix: string, keep?: string): Promise<void> => {
    const db = await openOmg4Db();
    if (!db) {
        return;
    }

    await new Promise<void>((resolve) => {
        const tx = db.transaction(OMG4_IDB_STORE, 'readwrite');
        const store = tx.objectStore(OMG4_IDB_STORE);
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
