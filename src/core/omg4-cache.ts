// Durable IndexedDB cache for OMG4 payloads, shared by the v1 chunk streamer
// (byte-range entries) and the v2 full-file loader. Everything lives in one
// database/store so the debug UI's "Clear OMG4 Cache" wipes both.

const OMG4_IDB_NAME = 'supersplat-omg4-chunks';
const OMG4_IDB_STORE = 'ranges';

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

const idbGetBuffer = async (key: string): Promise<ArrayBuffer | null> => {
    const db = await openOmg4Db();
    if (!db) {
        return null;
    }

    return new Promise((resolve) => {
        const tx = db.transaction(OMG4_IDB_STORE, 'readonly');
        const store = tx.objectStore(OMG4_IDB_STORE);
        const request = store.get(key);

        request.onsuccess = () => {
            const value = request.result;
            resolve(value instanceof ArrayBuffer ? value : null);
        };
        request.onerror = () => resolve(null);
    });
};

const idbSetBuffer = async (key: string, buffer: ArrayBuffer): Promise<void> => {
    const db = await openOmg4Db();
    if (!db) {
        return;
    }

    await new Promise<void>((resolve) => {
        const tx = db.transaction(OMG4_IDB_STORE, 'readwrite');
        const store = tx.objectStore(OMG4_IDB_STORE);
        store.put(buffer, key);
        tx.oncomplete = () => resolve();
        tx.onerror = () => resolve();
        tx.onabort = () => resolve();
    });
};

// Delete every entry whose key starts with the prefix except `keep` — used to
// drop stale copies of a file when its validator changes.
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
                if (typeof key === 'string' && key.startsWith(prefix) && key !== keep) {
                    store.delete(key);
                }
            }
        };
        tx.oncomplete = () => resolve();
        tx.onerror = () => resolve();
        tx.onabort = () => resolve();
    });
};

export { idbGetBuffer, idbSetBuffer, idbDeleteByPrefix };
