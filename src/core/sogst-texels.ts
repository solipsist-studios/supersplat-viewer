import { PIXELFORMAT_RGBA8, Texture } from 'playcanvas';
import type { AppBase } from 'playcanvas';

// WebP payload decoding for .sogst archives: every attribute texture in the
// container is a lossless WebP, and the decoder needs its raw RGBA texels.

// Reads a decoded group payload's texel array. Both holders expose _levels[0]:
// TexelImage by construction, and Texture publicly — though declared there as a
// union wide enough to include image sources, while everything decoded here is
// filled from a typed array.
const texelsOf = (image: TexelImage | Texture | undefined): Uint8Array => image!._levels[0] as Uint8Array;

// Decode webp bytes into an engine texture. Decoding must not premultiply:
// several textures carry codebook indices next to a data-bearing alpha
// channel (sh0 stores opacity there), and premultiplication would corrupt
// the indices of low-alpha splats.
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

// The SOG iterator only reads texel arrays (_levels[0]) plus dimensions,
// so decode feeds it these plain holders instead of real GPU textures.
type TexelImage = {
    width: number;
    height: number;
    _levels: [Uint8Array];
    destroy: () => void;
};

// Decodes webp payloads to raw RGBA texels in a worker with its own
// OffscreenCanvas WebGL context. The main-thread alternative (upload +
// texture.read on the app's context) forces every readback to sync behind
// queued rendering work — profiled at ~40% of the main thread during
// streaming playback, and the single biggest source of first-pass stutter
// on weak devices. A 2D canvas cannot be used instead: getImageData
// premultiplies, corrupting codebook indices next to data-bearing alpha
// (sh0 stores opacity there). In the worker, readPixels blocks only the
// worker.
const TEXEL_WORKER_SRC = `
let canvas = null, gl = null, tex = null, fbo = null;
self.onmessage = async (e) => {
    const { id, bytes } = e.data;
    try {
        const bitmap = await createImageBitmap(new Blob([bytes], { type: 'image/webp' }), {
            premultiplyAlpha: 'none',
            colorSpaceConversion: 'none'
        });
        const w = bitmap.width, h = bitmap.height;
        if (!gl) {
            canvas = new OffscreenCanvas(1, 1);
            gl = canvas.getContext('webgl2', { antialias: false, depth: false });
            if (!gl) throw new Error('no webgl2 in worker');
            tex = gl.createTexture();
            fbo = gl.createFramebuffer();
        }
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
        gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.NONE);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, bitmap);
        bitmap.close();
        gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
        const data = new Uint8Array(w * h * 4);
        gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, data);
        self.postMessage({ id, width: w, height: h, data }, [data.buffer]);
    } catch (err) {
        self.postMessage({ id, error: String(err) });
    }
};
`;

class WebpTexelWorker {
    private worker: Worker | null = null;

    private pending = new Map<number, { resolve: (t: TexelImage) => void; reject: (e: Error) => void }>();

    private nextId = 0;

    decode(bytes: Uint8Array): Promise<TexelImage> {
        if (!this.worker) {
            const blob = new Blob([TEXEL_WORKER_SRC], { type: 'text/javascript' });
            this.worker = new Worker(URL.createObjectURL(blob));
            this.worker.onmessage = (e: MessageEvent) => {
                const { id, width, height, data, error } = e.data;
                const entry = this.pending.get(id);
                if (!entry) {
                    return;
                }
                this.pending.delete(id);
                if (error) {
                    entry.reject(new Error(error));
                } else {
                    entry.resolve({
                        width,
                        height,
                        _levels: [data],
                        destroy: () => {
                            /* plain holder: no GPU resource to free */
                        }
                    });
                }
            };
        }
        const id = this.nextId++;
        // exact-size copy so the underlying buffer can transfer without
        // detaching (or wholesale-cloning) the caller's archive buffer
        const copy = bytes.slice();
        return new Promise<TexelImage>((resolve, reject) => {
            this.pending.set(id, { resolve, reject });
            this.worker!.postMessage({ id, bytes: copy }, [copy.buffer]);
        });
    }

    destroy() {
        this.worker?.terminate();
        this.worker = null;
        const failure = new Error('texel worker destroyed');
        this.pending.forEach((entry) => entry.reject(failure));
        this.pending.clear();
    }
}

export { decodeTexture, readTexels, texelsOf, WebpTexelWorker };
export type { TexelImage };
