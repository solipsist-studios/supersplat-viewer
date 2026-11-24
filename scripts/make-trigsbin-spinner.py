"""Browser check of the .trigsbin viewer. Makes spinner.trigsbin (three groups that move in ways a wrong port would show) and renders it at t = 0, 0.5, 1 with
cumuli_core + gsplat from the camera the viewer is given in settings.json. usage: PYTHONPATH=<cumuli-core>/src python scripts/make-trigsbin-spinner.py <out dir> <open4dgs-core checkout>

Then serve <out dir> together with public/ (index.html, index.js, index.css), open index.html?content=./spinner.trigsbin&settings=./settings.json
&noui and scrub to t (window.scrubTo(t)) only once the viewer has settled: wait for window.animationDuration > 0 and about 2.5 s
more, and about 0.5 s after the scrub before the screenshot. A scrub made while the page is still starting up can be lost, and the
picture is then the autoplay's, which looks like a motion bug and is not one. The viewer's picture should match expected_<t>.png (about 36 dB PSNR on WebGL: the floor is the two rasterisers'
own differences, which show at t = 0.5 too, where nothing moves)."""
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from cumuli_core import trigsbin

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
packer = Path(sys.argv[2]) / "deliver" / "make-trigs-timeline.py"
fps, frames = 30.0, 31
g = np.random.default_rng(3)


def sphere(n, radius, centre):
    v = g.standard_normal((n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return (v * radius + centre).astype(np.float32)


def group(means, rgb, omega, nu, anchor, c0t=(0, 0, 0), c1t=(0, 0, 0), c2t=(0, 0, 0), scale=0.02):
    n = len(means)
    z = lambda v: np.tile(np.asarray(v, np.float32), (n, 1))  # noqa: E731
    sh0 = ((np.asarray(rgb, np.float32) - 0.5) / 0.28209479177387814).reshape(-1, 1, 3) if np.ndim(rgb) == 2 else \
        np.tile(((np.asarray(rgb, np.float32) - 0.5) / 0.28209479177387814).reshape(1, 1, 3), (n, 1, 1))
    return {
        "means": means, "scales": np.full((n, 3), math.log(scale), np.float32), "quats": np.tile(np.array([0, 0, 0, 1], np.float32), (n, 1)),
        "opacities": np.full(n, 6.0, np.float32), "sh0": sh0.astype(np.float32), "shN": np.zeros((n, 0, 3), np.float32),
        "times": np.full(n, 0.5, np.float32), "log_durations": np.full(n, math.log(0.5), np.float32), "anchors": z(anchor),
        "zeta_base_r": z(omega), "zeta_base_t": z(nu), "zeta_c0_r": z((0, 0, 0)), "zeta_c0_t": z(c0t), "zeta_c1_r": z((0, 0, 0)),
        "zeta_c1_t": z(c1t), "zeta_c2_r": z((0, 0, 0)), "zeta_c2_t": z(c2t)}


a_pts = sphere(4000, 0.5, (0, 0, 0))
A = group(a_pts, 0.5 + a_pts, omega=(0, math.pi, 0), nu=(0, 0.4, 0), anchor=(0, 0, 0))             # spins and rises
b_pts = (np.array([-0.5, 0, 0], np.float32) + g.normal(scale=0.03, size=(500, 3))).astype(np.float32)
B = group(b_pts, (0.1, 1.0, 0.2), omega=(0, math.pi, 0), nu=(0, 0, 0), anchor=(-1.0, 0, 0), scale=0.03)   # swings about a pivot
c_pts = (np.array([1.0, 0.5, 0], np.float32) + g.normal(scale=0.05, size=(500, 3))).astype(np.float32)
C = group(c_pts, (1.0, 0.9, 0.1), omega=(0, 0, 0), nu=(0, 0, 0), anchor=(1.0, 0.5, 0),
          c0t=(0.8, 0, 0), c1t=(0, 0.8, 0), c2t=(-0.8, 0, 0), scale=0.03)                           # a bending path (Bezier)
params = {k: np.concatenate([A[k], B[k], C[k]]) for k in A}

duration = (frames - 1) / fps
with tempfile.TemporaryDirectory() as tmp:
    ckpt = Path(tmp) / "trigs_ckpt.pt"
    torch.save({"params": {k: torch.from_numpy(v) for k, v in params.items()}, "step": 1, "fps": fps, "n_frames": frames,
                "metrics": {"windows": [[0.0, duration]], "ownership_bounds": [0.0, duration]}}, ckpt)
    subprocess.run([sys.executable, str(packer), "from-checkpoint", "--checkpoint", str(ckpt), "--frame-rate", str(fps),
                    "--output", str(out / "spinner.trigsbin"), "--overwrite", "--tensor-dtype", "motion32"], check=True, capture_output=True)

# the viewer's camera, and the same one for gsplat (OpenCV: x right, y down, z forward)
pos, target, fov, size = np.array([0.0, 0.5, 3.2]), np.array([0.0, 0.0, 0.0]), 50.0, 640
(out / "settings.json").write_text(json.dumps({
    "version": 2, "tonemapping": "none", "highPrecisionRendering": False, "background": {"color": [0, 0, 0]},
    "postEffectSettings": {"sharpness": {"enabled": False, "amount": 0}, "bloom": {"enabled": False, "intensity": 1, "blurLevel": 2},
                           "grading": {"enabled": False, "brightness": 0, "contrast": 1, "saturation": 1, "tint": [1, 1, 1]},
                           "vignette": {"enabled": False, "intensity": 0.5, "inner": 0.3, "outer": 0.75, "curvature": 1},
                           "fringing": {"enabled": False, "intensity": 0.5}},
    "animTracks": [], "cameras": [{"initial": {"position": pos.tolist(), "target": target.tolist(), "fov": fov}}], "annotations": [],
    "startMode": "default"}))
fwd = (target - pos) / np.linalg.norm(target - pos)
right = np.cross(fwd, [0, 1, 0])
right /= np.linalg.norm(right)
down = np.cross(fwd, right)
rot = np.stack([right, down, fwd])             # world -> camera rows
w2c = np.eye(4)
w2c[:3, :3] = rot
w2c[:3, 3] = -rot @ pos
f = (size / 2) / math.tan(math.radians(fov) / 2)
K = np.array([[f, 0, size / 2], [0, f, size / 2], [0, 0, 1]])
vm = torch.tensor(w2c[None], dtype=torch.float32, device="cuda")
ks = torch.tensor(K[None], dtype=torch.float32, device="cuda")

from PIL import Image  # noqa: E402

from cumuli_core.render import render  # noqa: E402

model = trigsbin.load_model(out / "spinner.trigsbin", "cuda")
for t in (0.0, 0.5, 1.0):
    with torch.no_grad():
        means, quats, opac, sh, scales = model.slice(t)
        img, _, _ = render(means, quats, scales, opac, sh, (vm, ks), size, size, sh_degree=0)
    Image.fromarray((img[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)).save(out / f"expected_{t:.1f}.png")
print("wrote", out)
