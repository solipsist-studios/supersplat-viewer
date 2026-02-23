#!/usr/bin/env python3
"""
omg4_to_4dgs.py  –  Convert an OMG4 compressed 4DGS model (.xz) to the
web-friendly .4dgs binary format consumed by the supersplat-viewer.

The .4dgs format stores pre-baked per-frame Gaussian attributes so that the
viewer can play back the animation without GPU-side MLP inference.

Requirements (the OMG4 training environment):
    torch  numpy  dahuffman  tinycudann  lzma  pickle

Usage:
    python omg4_to_4dgs.py \\
        --input  path/to/comp.xz \\
        --output path/to/scene.4dgs \\
        --frames 30 \\
        --fps    24 \\
        --time_min -0.5 \\
        --time_max  0.5

File format written (all values little-endian):
    Header (28 bytes):
        uint32  magic = 0x53474434  ("4DGS")
        uint32  version = 1
        uint32  numSplats
        uint32  numFrames
        float32 fps
        float32 timeDurationMin
        float32 timeDurationMax
    Per-frame record (repeated numFrames times):
        float32          timestamp
        float32[N * 14]  per-splat data, AoS layout:
            x  y  z  rot_0(w)  rot_1(x)  rot_2(y)  rot_3(z)
            scale_0  scale_1  scale_2   (log-space)
            opacity                      (logit-space)
            f_dc_0  f_dc_1  f_dc_2      (raw SH DC coefficients)
"""

import argparse
import lzma
import math
import os
import pickle
import struct
import sys

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Huffman decode (mirrors compress_utils.py in the OMG4 repository)
# ---------------------------------------------------------------------------

try:
    import dahuffman
    def huffman_decode(encoded_bytes, huffman_table):
        codec = dahuffman.HuffmanCodec(code_table=huffman_table)
        decoded = codec.decode(encoded_bytes)
        return np.array(decoded, dtype=np.uint16)
except ImportError:
    sys.exit("ERROR: 'dahuffman' package not found.  Install it with:  pip install dahuffman")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def vq_decode(codes, indices):
    """Decode a vector-quantised attribute layer.

    codes   : np.ndarray  [K, d] cluster centres (float16)
    indices : np.ndarray  [N]    Huffman-decoded uint16 cluster labels
    """
    return torch.tensor(codes[indices], dtype=torch.float32)


def decode_all_layers(code_list, index_list, htable_list):
    """Decode a full VQ attribute (possibly split into sub-vector slices)."""
    parts = []
    for codes, idx_bytes, htable in zip(code_list, index_list, htable_list):
        labels = huffman_decode(idx_bytes, htable)
        parts.append(vq_decode(codes, labels))
    return torch.cat(parts, dim=-1)


def frequency_encode(x: torch.Tensor, n_frequencies: int = 16) -> torch.Tensor:
    """Apply sinusoidal frequency (positional) encoding to each input dimension.

    For each input dimension d and frequency index k in [0, n_frequencies),
    two features are produced: sin(2^k * π * x_d) and cos(2^k * π * x_d).

    Args:
        x:            Input tensor of shape [N, D].
        n_frequencies: Number of frequency octaves per dimension.

    Returns:
        Encoded tensor of shape [N, D * n_frequencies * 2].
    """
    # x : [N, D]  →  output : [N, D * n_frequencies * 2]
    freqs = 2.0 ** torch.arange(n_frequencies, dtype=torch.float32, device=x.device)
    # [N, D, n_freq]
    angles = x.unsqueeze(-1) * (freqs * math.pi)
    encoded = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # [N, D, 2*n_freq]
    return encoded.reshape(x.shape[0], -1)                               # [N, D*2*n_freq]


def contract_to_unisphere(x: torch.Tensor) -> torch.Tensor:
    """Same scene contraction used during rendering in the OMG4 renderer."""
    aabb_min = torch.tensor([-1., -1., -1.], device=x.device)
    aabb_max = torch.tensor([ 1.,  1.,  1.], device=x.device)
    x = (x - aabb_min) / (aabb_max - aabb_min)
    x = x * 2 - 1
    mag = x.norm(dim=-1, keepdim=True)
    mask = mag.squeeze(-1) > 1
    x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
    x = x / 4 + 0.5
    return x


# ---------------------------------------------------------------------------
# Tiny CPU-side MLP forward pass (replaces tinycudann at export time)
# ---------------------------------------------------------------------------
# tinycudann stores FullyFusedMLP weights as a flat float16 array with the
# following layout for a 2-layer network (1 hidden layer):
#
#   Layer 0:  weight  [n_neurons, input_size_padded]   (row-major, float16)
#             bias    [n_neurons]                       (float16, appended)
#   Layer 1:  weight  [output_size_padded, n_neurons]
#             bias    [output_size_padded]
#
# Padding: all dimensions are rounded up to the nearest multiple of 16.
# For FullyFusedMLP the bias is NOT stored separately – FullyFusedMLP does
# not use biases.  Weights only.

def _pad16(n):
    return ((n + 15) // 16) * 16


def tcnn_mlp_forward(params_f16: np.ndarray,
                     x: torch.Tensor,
                     n_input: int,
                     n_hidden: int,
                     n_output: int,
                     activation: str = 'relu') -> torch.Tensor:
    """
    Evaluate a 1-hidden-layer FullyFusedMLP on CPU.

    params_f16 : flat float16 numpy array (the .params attribute)
    x          : [N, n_input] float32 input
    activation : 'relu' | 'leaky_relu'
    """
    inp_pad  = _pad16(n_input)
    hid_pad  = _pad16(n_hidden)
    out_pad  = _pad16(n_output)

    params = torch.tensor(params_f16, dtype=torch.float32)

    # Unpack weights (no biases in FullyFusedMLP)
    w0_size = inp_pad * hid_pad
    w1_size = hid_pad * out_pad
    assert params.numel() >= w0_size + w1_size, (
        f"params size mismatch: got {params.numel()}, need {w0_size + w1_size}"
    )

    W0 = params[:w0_size].reshape(hid_pad, inp_pad)[:n_hidden, :n_input]
    W1 = params[w0_size:w0_size + w1_size].reshape(out_pad, hid_pad)[:n_output, :n_hidden]

    h = x.float() @ W0.T
    if activation == 'relu':
        h = torch.relu(h)
    elif activation == 'leaky_relu':
        h = torch.nn.functional.leaky_relu(h, negative_slope=0.01)
    else:
        raise ValueError(f"Unknown activation: {activation}")

    return h @ W1.T


def tcnn_network_with_encoding_forward(params_f16: np.ndarray,
                                       xyz_norm: torch.Tensor,
                                       n_frequencies: int = 16,
                                       n_output: int = 13) -> torch.Tensor:
    """Forward pass for mlp_cont (NetworkWithInputEncoding)."""
    encoded = frequency_encode(xyz_norm, n_frequencies)          # [N, 4*32]
    n_input = encoded.shape[1]
    return tcnn_mlp_forward(params_f16, encoded, n_input=n_input,
                            n_hidden=64, n_output=n_output, activation='relu')


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert(xz_path: str, out_path: str,
            num_frames: int, fps: float,
            time_min: float, time_max: float) -> None:

    print(f"Loading {xz_path} …")
    with lzma.open(xz_path, "rb") as f:
        save_dict = pickle.load(f)

    # ── Decode geometry ──────────────────────────────────────────────────────
    xyz       = torch.tensor(save_dict['xyz'], dtype=torch.float32)          # [N, 3]
    t_center  = torch.tensor(save_dict['t'],   dtype=torch.float32)          # [N, 1]

    scaling   = decode_all_layers(save_dict['scale_code'],
                                  save_dict['scale_index'],
                                  save_dict['scale_htable'])                  # [N, 3]
    rotation  = decode_all_layers(save_dict['rotation_code'],
                                  save_dict['rotation_index'],
                                  save_dict['rotation_htable'])               # [N, 4]
    appearance = decode_all_layers(save_dict['app_code'],
                                   save_dict['app_index'],
                                   save_dict['app_htable'])                   # [N, 6]

    # 4D attributes
    scaling_t = decode_all_layers(save_dict['scaling_t_code'],
                                  save_dict['scaling_t_index'],
                                  save_dict['scaling_t_htable'])              # [N, 1]
    rotation_r = decode_all_layers(save_dict['rotation_r_code'],
                                   save_dict['rotation_r_index'],
                                   save_dict['rotation_r_htable'])            # [N, 4]

    # MLP weights (flat float16 numpy arrays)
    mlp_cont_params    = save_dict['MLP_cont']
    mlp_dc_params      = save_dict['MLP_dc']
    mlp_opacity_params = save_dict['MLP_opacity']

    N = xyz.shape[0]
    print(f"  {N:,} Gaussians | {num_frames} frames @ {fps} fps")

    # Static appearance features (first 3 dims = features_static,
    # next 3 dims = features_view).
    features_static = appearance[:, 0:3]   # [N, 3]

    # ── Write .4dgs file ─────────────────────────────────────────────────────
    FLOATS_PER_SPLAT = 14
    MAGIC   = 0x53474434
    VERSION = 1

    with open(out_path, 'wb') as fp:
        # Header
        fp.write(struct.pack('<IIIIFFF',
                             MAGIC, VERSION,
                             N, num_frames,
                             fps,
                             time_min, time_max))

        for fi in range(num_frames):
            timestamp = time_min + (time_max - time_min) * fi / max(num_frames - 1, 1)
            timestamp_norm = fi / max(num_frames - 1, 1)

            print(f"  Frame {fi + 1}/{num_frames}  t={timestamp:.4f}", end='\r')

            # ── Temporal masking ─────────────────────────────────────────────
            sigma_t  = torch.exp(scaling_t)                           # [N, 1]
            weight_t = torch.exp(-0.5 * (t_center - timestamp) ** 2
                                 / (sigma_t ** 2 + 1e-8))             # [N, 1]

            # ── Position (no 4D deformation for simplicity) ─────────────────
            pos = xyz.clone()                                          # [N, 3]

            # ── MLP inference ────────────────────────────────────────────────
            xyz_contracted = contract_to_unisphere(pos.clone())        # [N, 3]
            t_col = torch.full((N, 1), timestamp_norm)
            xyzt  = torch.cat([xyz_contracted, t_col], dim=1)         # [N, 4]

            cont_feat = tcnn_network_with_encoding_forward(
                mlp_cont_params, xyzt, n_frequencies=16, n_output=13) # [N, 13]

            space_feat = torch.cat([cont_feat, features_static], dim=-1)  # [N, 16]

            dc      = tcnn_mlp_forward(mlp_dc_params, space_feat,
                                       n_input=16, n_hidden=64, n_output=3,
                                       activation='leaky_relu')            # [N, 3]
            raw_opa = tcnn_mlp_forward(mlp_opacity_params, space_feat,
                                       n_input=16, n_hidden=64, n_output=1,
                                       activation='leaky_relu')            # [N, 1]

            # Combine temporal weight into opacity (logit-space adjustment)
            # opacity_final = sigmoid(raw_opa) * weight_t → store back in logit
            opa_sigmoid = torch.sigmoid(raw_opa.squeeze(-1))
            opa_final   = opa_sigmoid * weight_t.squeeze(-1)
            # Clamp to avoid log(0)
            opa_final   = opa_final.clamp(1e-6, 1 - 1e-6)
            opa_logit   = torch.log(opa_final / (1 - opa_final))          # [N]

            # Scales stay in log-space; normalise rotation quaternion
            rot_norm = torch.nn.functional.normalize(rotation, dim=-1)    # [N, 4]

            # ── Pack AoS ─────────────────────────────────────────────────────
            frame_data = np.empty((N, FLOATS_PER_SPLAT), dtype=np.float32)
            frame_data[:, 0 ] = pos[:, 0].numpy()            # x
            frame_data[:, 1 ] = pos[:, 1].numpy()            # y
            frame_data[:, 2 ] = pos[:, 2].numpy()            # z
            frame_data[:, 3 ] = rot_norm[:, 0].numpy()       # rot_0 (w)
            frame_data[:, 4 ] = rot_norm[:, 1].numpy()       # rot_1 (x)
            frame_data[:, 5 ] = rot_norm[:, 2].numpy()       # rot_2 (y)
            frame_data[:, 6 ] = rot_norm[:, 3].numpy()       # rot_3 (z)
            frame_data[:, 7 ] = scaling[:, 0].numpy()        # scale_0 (log)
            frame_data[:, 8 ] = scaling[:, 1].numpy()        # scale_1 (log)
            frame_data[:, 9 ] = scaling[:, 2].numpy()        # scale_2 (log)
            frame_data[:, 10] = opa_logit.detach().numpy()   # opacity (logit)
            frame_data[:, 11] = dc[:, 0].detach().numpy()    # f_dc_0
            frame_data[:, 12] = dc[:, 1].detach().numpy()    # f_dc_1
            frame_data[:, 13] = dc[:, 2].detach().numpy()    # f_dc_2

            fp.write(struct.pack('<f', timestamp))
            fp.write(frame_data.tobytes())

    print()
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"Wrote {out_path}  ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert OMG4 .xz checkpoint to supersplat-viewer .4dgs format')
    parser.add_argument('--input',    required=True, help='Path to comp.xz (OMG4 output)')
    parser.add_argument('--output',   required=True, help='Destination .4dgs file')
    parser.add_argument('--frames',   type=int,   default=30,   help='Number of output frames (default: 30)')
    parser.add_argument('--fps',      type=float, default=24.0, help='Frames per second (default: 24)')
    parser.add_argument('--time_min', type=float, default=-0.5, help='timeDuration min (default: -0.5)')
    parser.add_argument('--time_max', type=float, default= 0.5, help='timeDuration max (default:  0.5)')
    args = parser.parse_args()

    convert(
        xz_path   = args.input,
        out_path  = args.output,
        num_frames = args.frames,
        fps        = args.fps,
        time_min   = args.time_min,
        time_max   = args.time_max
    )
