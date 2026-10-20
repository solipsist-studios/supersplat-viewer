import { PIXELFORMAT_RGBA8, Texture } from 'playcanvas';
import type { AppBase } from 'playcanvas';

// Main-thread WebP decoding for .sogst archives, through the app's graphics
// context. The pack workers (sogst-pack-worker.ts) normally decode WebP
// themselves; this is the fallback for a browser without WebGL2 in workers.
// It syncs every readback behind queued rendering work, so it costs the main
// thread far more than the workers do.

// Decode webp bytes into an engine texture. The decode must not premultiply.
// Several textures carry codebook indices next to an alpha channel that holds
// data, and sh0 stores opacity there. Premultiplication would corrupt the
// indices of low-alpha splats.
const decodeTexture = async (app: AppBase, bytes: Uint8Array, name: string): Promise<Texture> => {
    const bitmap = await createImageBitmap(new Blob([bytes as unknown as ArrayBuffer], { type: 'image/webp' }), {
        premultiplyAlpha: 'none',
        colorSpaceConversion: 'none'
    });
    const texture = new Texture(app.graphicsDevice, {
        name: `sogst-${name}`,
        width: bitmap.width,
        height: bitmap.height,
        format: PIXELFORMAT_RGBA8,
        mipmaps: false
    });
    // The engine's setSource typing predates ImageBitmap, but the runtime
    // upload path handles it (same as the engine's own image parser).
    texture.setSource(bitmap as unknown as HTMLCanvasElement);
    return texture;
};

// Read a texture's texels back as raw RGBA bytes.
const readTexels = async (texture: Texture): Promise<Uint8Array> => {
    const texels = await texture.read(0, 0, texture.width, texture.height, { mipLevel: 0, face: 0, immediate: true });
    return texels as Uint8Array;
};

export { decodeTexture, readTexels };
