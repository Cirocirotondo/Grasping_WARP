#!/usr/bin/env bash
# One-shot setup: pinned Isaac Lab (Newton backend, kit-less), this package,
# and the URDF -> USD conversion. Re-runnable.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_REPO="https://github.com/isaac-sim/IsaacLab.git"
ISAACLAB_BRANCH="develop"
ISAACLAB_COMMIT="a8b4da3c29ae528b39d4b3c9444d782ce58d886d"   # develop, 2026-09-12 (Isaac Lab 3.0 / Newton 1.6.0rc1)
UV="${UV:-uv}"
if ! command -v "$UV" >/dev/null 2>&1; then
    echo "uv is required (https://docs.astral.sh/uv/): curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi
mkdir -p "$HERE/deps"
if [ ! -d "$HERE/deps/IsaacLab/.git" ]; then
    git clone --filter=blob:none -b "$ISAACLAB_BRANCH" "$ISAACLAB_REPO" "$HERE/deps/IsaacLab"
fi
git -C "$HERE/deps/IsaacLab" fetch --depth 1 origin "$ISAACLAB_COMMIT" 2>/dev/null || true
git -C "$HERE/deps/IsaacLab" checkout -q "$ISAACLAB_COMMIT" || echo "[WARN] could not pin Isaac Lab to $ISAACLAB_COMMIT; using the checked-out revision"
cd "$HERE/deps/IsaacLab"
# Newton + MuJoCo-Warp physics, the standalone URDF importer, RSL-RL for the
# Isaac Lab CLI path. Add `--extra isaacsim` for Isaac Sim / PhysX rendering.
"$UV" sync --extra importers --extra rsl-rl ${ISAACLAB_EXTRAS:-}
PY="$HERE/deps/IsaacLab/.venv/bin/python"
"$UV" pip install --python "$PY" -e "$HERE" --no-deps
"$PY" "$HERE/scripts/convert_urdf.py"
echo
echo "Done. Python interpreter: $PY"
echo "Smoke test: $PY scripts/test_headless_env.py --num-envs 16"
echo "Train:      $PY scripts/train.py --num-envs 4096"
