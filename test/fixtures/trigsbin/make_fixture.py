"""Regenerates small.trigsbin and expected.json beside this file.

    PYTHONPATH=<cumuli-core>/src python make_fixture.py <open4dgs-core checkout>

A 64-splat checkpoint with random motion is packed by Open4DGS-Core's own packager (`from-checkpoint`, motion32), then
sliced at five times by cumuli_core.trigs in double precision from the very tensors the file holds. The viewer's test
(test/trigsbin.test.mjs) reads the file with the TypeScript reader and slices it with the TypeScript model, and the two must
agree. The file is a test input only: random numbers, no capture.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from cumuli_core import trigs, trigsbin

here = Path(__file__).resolve().parent
packer = Path(sys.argv[1]) / "deliver" / "make-trigs-timeline.py"
fps, frames, n = 30.0, 31, 64

g = np.random.default_rng(7)
f = lambda *s, sc=1.0: (g.standard_normal(s) * sc).astype(np.float32)  # noqa: E731
q = f(n, 4)
q /= np.linalg.norm(q, axis=1, keepdims=True)
params = {
    "means": f(n, 3), "scales": f(n, 3, sc=0.1) - 3.0, "quats": q, "opacities": f(n), "sh0": f(n, 1, 3, sc=0.2), "shN": f(n, 3, 3, sc=0.05),
    "times": g.uniform(0.0, 1.0, n).astype(np.float32), "log_durations": np.log(g.uniform(0.15, 0.6, n)).astype(np.float32), "anchors": f(n, 3),
    "zeta_base_r": f(n, 3, sc=0.5), "zeta_base_t": f(n, 3, sc=0.5), "zeta_c0_r": f(n, 3, sc=0.3), "zeta_c0_t": f(n, 3, sc=0.3),
    "zeta_c1_r": f(n, 3, sc=0.3), "zeta_c1_t": f(n, 3, sc=0.3), "zeta_c2_r": f(n, 3, sc=0.3), "zeta_c2_t": f(n, 3, sc=0.3)}
# two splats with no rotation at all, to cover the no-axis branch
for k in ("zeta_base_r", "zeta_c0_r", "zeta_c1_r", "zeta_c2_r"):
    params[k][:2] = 0.0

with tempfile.TemporaryDirectory() as tmp:
    ckpt = Path(tmp) / "trigs_ckpt.pt"
    duration = (frames - 1) / fps
    torch.save({"params": {k: torch.from_numpy(v) for k, v in params.items()}, "step": 1, "fps": fps, "n_frames": frames,
                "metrics": {"windows": [[0.0, duration]], "ownership_bounds": [0.0, duration]}}, ckpt)
    subprocess.run([sys.executable, str(packer), "from-checkpoint", "--checkpoint", str(ckpt), "--frame-rate", str(fps),
                    "--output", str(here / "small.trigsbin"), "--overwrite", "--tensor-dtype", "motion32"], check=True)

tf = trigsbin.read(here / "small.trigsbin")
clip = tf.clips[0]
d = {k: torch.from_numpy(v.astype(np.float64)) for k, v in clip.tensors.items()}
q = d["quats"]
tt = trigs.TrigsTensors(
    means=d["means"], quats_wxyz=torch.cat([q[:, 3:4], q[:, :3]], 1), log_scales=d["scales"], opacity_logit=d["opacities"], sh=d["sh0"],
    times=d["times"], log_sigma=d["log_durations"], anchors=d["anchors"], **{k: d[k] for k in trigs.MOTION_KEYS if k != "anchors"})
times = [0.0, 0.2, 0.5, 0.77, 1.0]
expected = {"count": int(tf.count), "fps": fps, "times": times, "slices": []}
for t in times:
    means, quats_wxyz, alpha = trigs.slice_at(tt, t)
    quats_xyzw = torch.cat([quats_wxyz[:, 1:], quats_wxyz[:, :1]], 1)
    expected["slices"].append({"t": t, "means": means.tolist(), "quats": quats_xyzw.tolist(), "alpha": alpha.tolist()})
(here / "expected.json").write_text(json.dumps(expected))
print("wrote", here / "small.trigsbin", (here / "small.trigsbin").stat().st_size, "bytes")
