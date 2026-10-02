import { GSplatData, GSplatResource } from 'playcanvas';
import type { AppBase } from 'playcanvas';

import type { SogstMeta } from '../parsers/sogst';

import { uploadGsplatRows } from './gsplat-range-sync';
import { createSogstMotionTextures, uploadSogstMotionRows } from './sogst-motion';

// The GPU-side destination of a .sogst decode: one GSplatResource of a fixed
// splat capacity, plus the motion textures, created before any splat
// decodes.
//
// The decoder packs each decoded chunk straight into the resource's texture
// CPU copies at a destination index, then uploads the covering rows. Nothing
// holds the decoded attributes for the whole clip. A capacity of meta.count
// with destination == file index reproduces the old fully resident layout.

// Coefficients per colour channel for each SH band count. The engine's
// repack (updateSHData, and packSH in gsplat-range-sync.ts) reads the same
// table.
const SH_COEFFS_PER_CHANNEL: Record<number, number> = { 1: 3, 2: 8, 3: 15 };

// Number of f_rest_* properties for a band count, in the compact layout:
// channel c, coefficient k at f_rest_{c * coeffs + k}.
const shRestCount = (bands: number) => 3 * (SH_COEFFS_PER_CHANNEL[bands] ?? 0);

// Logit the placeholder data gives every slot before a splat decodes into
// it. Sigmoid of this is zero to float precision, so an empty slot draws
// nothing.
const EMPTY_OPACITY_LOGIT = -40;

// Rows per upload pass. A single multi-MB texSubImage2D into a texture the
// GPU is sampling can stall the driver for a frame or more, so a large range
// uploads in passes of about a megabyte, yielding between them.
const UPLOAD_CHUNK_SPLATS = 16384;

const yieldToUi = () =>
    new Promise((resolve) => {
        setTimeout(resolve, 0);
    });

const propertyNames = (meta: SogstMeta): string[] => {
    const names = [
        'x',
        'y',
        'z',
        'f_dc_0',
        'f_dc_1',
        'f_dc_2',
        'opacity',
        'scale_0',
        'scale_1',
        'scale_2',
        'rot_0',
        'rot_1',
        'rot_2',
        'rot_3'
    ];
    const restCount = meta.shN ? shRestCount(meta.shN.bands) : 0;
    for (let i = 0; i < restCount; i++) {
        names.push(`f_rest_${i}`);
    }
    return names;
};

// A GSplatData of `count` splats whose properties all share three arrays:
// zeros, ones for rot_0 (an identity rotation; all zeros would normalise to
// NaN), and EMPTY_OPACITY_LOGIT for opacity. With storage of length 0 it is
// a shell that only reports the count and the SH band count.
const placeholderData = (meta: SogstMeta, count: number, storageLength: number) => {
    const zeros = new Float32Array(storageLength);
    const ones = new Float32Array(storageLength).fill(1);
    const empty = new Float32Array(storageLength).fill(EMPTY_OPACITY_LOGIT);
    const storageFor = (name: string) => (name === 'rot_0' ? ones : name === 'opacity' ? empty : zeros);
    return new GSplatData([
        {
            name: 'vertex',
            count,
            properties: propertyNames(meta).map((name) => ({
                name,
                type: 'float' as const,
                byteSize: 4,
                storage: storageFor(name)
            }))
        }
    ]);
};

class SogstTarget {
    readonly resource: GSplatResource;

    readonly capacity: number;

    // Set when the app is destroyed. The decoder yields between slices, so a
    // destroy can land mid-decode; every write and upload re-checks this.
    destroyed = false;

    constructor(app: AppBase, meta: SogstMeta, capacity: number) {
        this.capacity = capacity;
        // The engine fills its textures from the placeholder (every slot
        // empty), then keeps a reference to it. After construction it reads
        // only numSplats and shBands from that reference, so swap in a shell
        // and let the full-length placeholder arrays go.
        this.resource = new GSplatResource(app.graphicsDevice, placeholderData(meta, capacity, capacity));
        this.resource.gsplatData = placeholderData(meta, capacity, 0);
        createSogstMotionTextures(this.resource, !!meta.accel);
        app.once('destroy', () => {
            this.destroyed = true;
        });
    }

    // Upload the rows covering [a, b) of every stream, in bounded passes.
    // shOnly uploads the SH streams alone, for deferred SH.
    async upload(a: number, b: number, shOnly = false) {
        for (let u = a; u < b; u += UPLOAD_CHUNK_SPLATS) {
            if (this.destroyed) {
                return;
            }
            const e = Math.min(b, u + UPLOAD_CHUNK_SPLATS);
            uploadGsplatRows(this.resource, u, e, shOnly);
            if (!shOnly) {
                uploadSogstMotionRows(this.resource, u, e);
            }
            await yieldToUi();
        }
    }

    // Release the GPU resource. Only for a load that failed before the
    // resource went on an entity: once it has, the entity owns it.
    destroy() {
        this.destroyed = true;
        try {
            this.resource.destroy();
        } catch {
            // the graphics device may already be torn down
        }
    }
}

export { SH_COEFFS_PER_CHANNEL, SogstTarget, shRestCount };
