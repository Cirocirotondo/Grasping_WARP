#!/usr/bin/env bash
# Move code to a GPU server (tars by default) and results back.
#
#   scripts/tars_sync.sh push               # working tree + banks + USD -> server
#   scripts/tars_sync.sh docker             # Docker build context -> ~/docker_strnewton on the server
#   scripts/tars_sync.sh pull <run-dir>...  # server run(s) -> logs/simtoolreal/TARS_<run-dir>
#   scripts/tars_sync.sh runs               # list run directories on the server
#   scripts/tars_sync.sh stage <name>       # logs/staged/<name>/ -> server (seed checkpoints)
#   HOST=case.inf.ethz.ch scripts/tars_sync.sh ...   # same against case
#
# The server checkout (~/simtoolreal_newton) is a mirror of this working tree,
# uncommitted edits included, so a training there runs exactly the code
# sitting here. Everything .gitignore hides stays out except what the
# environment needs and takes long to rebuild: the transform banks and the
# USD converted from the URDF. Runs come back under a TARS_ / CASE_ prefix,
# so a run's origin is visible in its name.
set -eu

host="${HOST:-tars.inf.ethz.ch}"
remote_root="simtoolreal_newton"
docker_context="docker_strnewton"
local_root="/home/simone/simtoolreal_newton"
cd "${local_root}"

ssh_opts=(-o BatchMode=yes -o ConnectTimeout=15)
prefix="$(cut -d. -f1 <<< "${host}" | tr '[:lower:]' '[:upper:]')_"

case "${1:-}" in
  push)
    ssh "${ssh_opts[@]}" "${host}" "mkdir -p ${remote_root}/logs/queue ${remote_root}/logs/tars_launch ${remote_root}/.warp_cache"
    { git ls-files; git ls-files --others --exclude-standard; } | sort -u \
      | rsync -a --delete-missing-args --files-from=- ./ "${host}:${remote_root}/"
    rsync -a --exclude "*.bak*" --exclude ".pre_tensor_acceptance" banks/ "${host}:${remote_root}/banks/"
    rsync -a assets/usd/ "${host}:${remote_root}/assets/usd/"
    [ -f assets/ur5e_right_dg5f.urdf ] && rsync -a assets/ur5e_right_dg5f.urdf "${host}:${remote_root}/assets/"
    ssh "${ssh_opts[@]}" "${host}" "cd ${remote_root} && git status -sb | head -1 && git status -s | head -20"
    ;;
  docker)
    # Build context for deploy/fleet/Dockerfile: the Isaac Lab checkout as it
    # is here (its uv.lock included), minus the virtual environment and git.
    ssh "${ssh_opts[@]}" "${host}" "mkdir -p ${docker_context}"
    rsync -a --delete --exclude ".venv" --exclude ".git" --exclude "docs" --exclude "__pycache__" \
      deps/IsaacLab/ "${host}:${docker_context}/IsaacLab/"
    rsync -a deploy/fleet/Dockerfile deploy/fleet/extras.txt "${host}:${docker_context}/"
    echo "context in ${host}:~/${docker_context}; build with:"
    echo "  ssh ${host} 'source ~/.docker_env.sh; docker build -t str_newton:v1 ~/${docker_context}'"
    ;;
  pull)
    shift
    [ $# -ge 1 ] || { echo "pull needs at least one run directory name" >&2; exit 1; }
    for run in "$@"; do
      run="${run%/}"
      rsync -a --info=progress2 \
        "${host}:${remote_root}/logs/simtoolreal/${run}/" \
        "logs/simtoolreal/${prefix}${run}/"
      echo "pulled logs/simtoolreal/${prefix}${run}"
    done
    ;;
  stage)
    name="${2:?staged checkpoint directory name under logs/staged}"
    rsync -a --mkpath "logs/staged/${name}/" "${host}:${remote_root}/logs/staged/${name}/"
    echo "staged logs/staged/${name} on ${host}"
    ;;
  runs)
    ssh "${ssh_opts[@]}" "${host}" "ls -1t ${remote_root}/logs/simtoolreal 2>/dev/null"
    ;;
  *)
    sed -n '2,16p' "$0"
    exit 1
    ;;
esac
