# Training image for the tars/case servers: the exact Python stack of the
# desktop venv (deps/IsaacLab, uv.lock as it is on the desktop, Python 3.12,
# torch 2.11+cu128, warp 1.17, newton 1.6.0rc1, mujoco-warp 3.12), so a
# checkpoint trained here loads on the desktop and the physics is the same.
#
# Build context (scripts/tars_sync.sh docker sends it): this file, extras.txt,
# and IsaacLab/ = deps/IsaacLab without .venv/.git. Build on the server:
#   source ~/.docker_env.sh; docker build -t str_newton:v1 ~/docker_strnewton
#
# The repository itself is not copied: it is bind-mounted at
# /workspace/simtoolreal_newton at run time and reached through a .pth entry,
# so the mirror pushed by rsync is what runs. Warp compiles its kernels on
# first use; mount a persistent directory on /root/.cache/warp.
# Plain Ubuntu, not nvidia/cuda: the torch and warp wheels bundle every CUDA
# library they need, and the nvidia/cuda base images ship the CUDA
# forward-compatibility libcuda, which the container toolkit injects on a
# host whose driver is older than the image's CUDA (tars/case: driver 555 vs
# CUDA 12.8) and which GeForce cards refuse (cudaGetDeviceCount error 803).
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive TZ=Europe/Zurich
# What the NVIDIA container toolkit injects (the host driver's libcuda etc.).
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates build-essential \
        libgl1 libegl1 libglib2.0-0 libx11-6 libxext6 libxrender1 libsm6 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.11.8 /uv /uvx /usr/local/bin/
ENV UV_LINK_MODE=copy UV_PYTHON_INSTALL_DIR=/opt/uv-python UV_CACHE_DIR=/tmp/uv-cache
COPY IsaacLab /opt/IsaacLab
WORKDIR /opt/IsaacLab
RUN uv python install 3.12.13 \
    && uv sync --frozen --extra importers --extra rsl-rl \
    && rm -rf /tmp/uv-cache
COPY extras.txt /opt/extras.txt
RUN uv pip install --python /opt/IsaacLab/.venv/bin/python -r /opt/extras.txt \
    && rm -rf /tmp/uv-cache \
    && echo /workspace/simtoolreal_newton > /opt/IsaacLab/.venv/lib/python3.12/site-packages/simtoolreal_newton.pth
# The servers' driver (555, CUDA 12.5) is older than the image's CUDA 12.8
# label; the wheels run on it through CUDA minor-version compatibility, so
# the container toolkit's version requirement is switched off.
ENV NVIDIA_DISABLE_REQUIRE=1
ENV PATH=/opt/IsaacLab/.venv/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /workspace/simtoolreal_newton
