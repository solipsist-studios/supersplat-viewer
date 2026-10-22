import type { GSplatResource, Texture } from 'playcanvas';

import { SH_TEXTURES, shTextureCount, shTextureWords } from './sh-textures';

type TypedArray = Uint8Array | Uint16Array | Uint32Array | Float32Array;

// Ranged uploads of a GSplatResource's stream textures. The engine's own
// updates re-upload whole textures, which is acceptable once but not for
// every decoded segment. The pack workers (sogst-pack-worker.ts) produce the
// texel words, the decoder copies them into the textures' persistent CPU
// copies, and these functions upload only the rows that cover them.

// The WebGPU members the row upload below reaches past the engine for. The
// engine's WebgpuTexture keeps its GPUTexture private and has no write(), and
// the device's GPUDevice is untyped in its published .d.ts.
type WebgpuTextureImpl = { gpuTexture?: object | null };
type WebgpuQueueDevice = {
    wgpu?: {
        queue: {
            writeTexture: (destination: object, data: ArrayBufferView, layout: object, size: object) => void;
        };
    };
};

// Upload the texture rows that cover splat texel indices [a, b). The caller
// has already updated the CPU level copy in place.
//
// WebGL has a partial write path (texture.write). The engine's WebGPU
// texture does not, and texture.upload() re-sends the whole level: on a
// segment-window resource that turns every installed segment into several
// full-texture uploads, which slowed installs on desktop WebGPU to ~2 s per
// segment. So WebGPU writes the rows through the queue directly. A texture
// not yet created on the GPU still takes texture.upload(), which then sends
// the whole level once, writes included.
const uploadTextureRows = (texture: Texture, elemsPerTexel: number, a: number, b: number) => {
    // Texture._levels has a union type that also covers image sources. A
    // typed array creates every gsplat stream texture, so narrow it to
    // that.
    const level = texture?._levels?.[0] as TypedArray | undefined;
    if (!level) {
        return;
    }
    const w = texture.width;
    const rowA = Math.floor(a / w);
    const rowB = Math.min(texture.height, Math.ceil(b / w));
    const rows = level.subarray(rowA * w * elemsPerTexel, rowB * w * elemsPerTexel);
    if (texture.impl?.write) {
        (texture.write(0, rowA, w, rowB - rowA, rows) as Promise<unknown>)?.catch(() => {
            /* upload races teardown; nothing to recover */
        });
        return;
    }
    const gpuTexture = (texture.impl as unknown as WebgpuTextureImpl | undefined)?.gpuTexture;
    const wgpu = (texture.device as unknown as WebgpuQueueDevice | undefined)?.wgpu;
    if (gpuTexture && wgpu && rowB > rowA) {
        wgpu.queue.writeTexture(
            { texture: gpuTexture, origin: { x: 0, y: rowA } },
            rows,
            { bytesPerRow: w * elemsPerTexel * level.BYTES_PER_ELEMENT, rowsPerImage: rowB - rowA },
            { width: w, height: rowB - rowA }
        );
        return;
    }
    texture.upload();
};

// Write a `width` x `height` block of RGBA8 texels at the origin of a
// texture with a typed-array CPU copy, for staging textures that are
// rewritten for every decode. The block goes to the GPU only, through the
// same two paths as uploadTextureRows. A texture not yet created on the GPU
// takes the block into its CPU copy and uploads that instead.
const writeTexelBlock = (texture: Texture, width: number, height: number, data: Uint8Array) => {
    if (texture.impl?.write) {
        (texture.write(0, 0, width, height, data) as Promise<unknown>)?.catch(() => {
            /* upload races teardown; nothing to recover */
        });
        return;
    }
    const gpuTexture = (texture.impl as unknown as WebgpuTextureImpl | undefined)?.gpuTexture;
    const wgpu = (texture.device as unknown as WebgpuQueueDevice | undefined)?.wgpu;
    if (gpuTexture && wgpu) {
        wgpu.queue.writeTexture(
            { texture: gpuTexture, origin: { x: 0, y: 0 } },
            data,
            { bytesPerRow: width * 4, rowsPerImage: height },
            { width, height }
        );
        return;
    }
    const level = texture._levels?.[0] as Uint8Array | undefined;
    if (level) {
        for (let row = 0; row < height; row++) {
            level.set(data.subarray(row * width * 4, (row + 1) * width * 4), row * texture.width * 4);
        }
        texture.upload();
    }
};

const uploadSHRows = (resource: GSplatResource, a: number, b: number) => {
    const bands = resource.shBands;
    for (let i = 0; i < shTextureCount(bands); i++) {
        uploadTextureRows(resource.streams.getTexture(SH_TEXTURES[i]), shTextureWords(bands, i), a, b);
    }
};

// Upload the texture rows covering [a, b) of the gsplat streams: all of them,
// or the SH streams alone.
const uploadGsplatRows = (resource: GSplatResource, a: number, b: number, shOnly = false) => {
    if (b <= a) {
        return;
    }
    if (!shOnly) {
        uploadTextureRows(resource.streams.getTexture('splatColor'), 4, a, b);
        uploadTextureRows(resource.streams.getTexture('transformA'), 4, a, b);
        uploadTextureRows(resource.streams.getTexture('transformB'), 4, a, b);
    }
    uploadSHRows(resource, a, b);
};

export { uploadGsplatRows, uploadTextureRows, writeTexelBlock };
