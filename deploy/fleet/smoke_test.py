"""Container smoke test: the stack imports, CUDA works for torch and Warp, the repo is importable.

    python deploy/fleet/smoke_test.py
"""
import sys

import torch
import warp as wp

print("python", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.version.cuda)
print("torch.cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
x = torch.randn(512, 512, device="cuda")
print("torch matmul ok", float((x @ x).sum()) == float((x @ x).sum()))

wp.init()
print("warp", wp.__version__, "devices", [d.alias for d in wp.get_cuda_devices()])


@wp.kernel
def double(a: wp.array(dtype=float)):
    i = wp.tid()
    a[i] = float(i) * 2.0


a = wp.zeros(8, dtype=float, device="cuda:0")
wp.launch(double, dim=8, inputs=[a])
print("warp kernel", a.numpy().tolist())

import newton  # noqa: E402
import mujoco_warp  # noqa: E402
import isaaclab  # noqa: E402
import simtoolreal_newton  # noqa: E402

print("newton", getattr(newton, "__version__", "?"), "mujoco_warp", getattr(mujoco_warp, "__version__", "?"), "isaaclab", isaaclab.__version__)
print("repo", simtoolreal_newton.__file__)
print("SMOKE OK")
