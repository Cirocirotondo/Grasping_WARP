#!/usr/bin/env bash
# Launch one training on a GPU server (tars by default), detached, on a chosen GPU.
#
#   scripts/tars_train.sh <gpu 0|1> supervise --run-name N --target T \
#                         [--seed-checkpoint P --seed-iteration I] -- <train.py args>
#   scripts/tars_train.sh <gpu 0|1> <train.py arguments...>       # single attempt
#   scripts/tars_train.sh <gpu 0|1> exec <command...>              # e.g. a pose sweep
#   scripts/tars_train.sh status
#   scripts/tars_train.sh log <container-name> [tail-lines]
#   scripts/tars_train.sh stop <container-name>
#
#   HOST=case.inf.ethz.ch scripts/tars_train.sh ...               # same on case
#
# Everything runs inside the str_newton:v2 image (deploy/fleet/Dockerfile:
# the desktop venv's exact stack) with ~/simtoolreal_newton bind-mounted at
# /workspace/simtoolreal_newton, so the run directory lands in that mirror's
# logs/simtoolreal like a local run and comes back with scripts/tars_sync.sh
# pull. Push the code first: scripts/tars_sync.sh push. The Warp kernel cache
# is mounted from ~/simtoolreal_newton/.warp_cache so a container does not
# recompile every kernel at start.
#
# `supervise` wraps train.py in scripts/supervise_train.sh, the crash-restart
# loop every real run uses; its [sup] lines land in logs/queue/<run-name>.log
# in the mirror. Seed checkpoint paths are container paths = repo-relative
# paths (logs/staged/...).
#
# The container is named strn_gpu<N>_<run-name>, one per GPU: a second launch
# on a busy GPU is refused rather than sharing it. Video recording is on
# unless the arguments say otherwise. --gpus selects the physical card; inside
# the container it is always cuda:0.
set -eu

host="${HOST:-tars.inf.ethz.ch}"
remote_root="simtoolreal_newton"
image="str_newton:v2"
ssh_opts=(-o BatchMode=yes -o ConnectTimeout=15)

remote() { ssh "${ssh_opts[@]}" "${host}" "source ~/.docker_env.sh; $*"; }

docker_run() {
  # $1 = container name, $2 = gpu, $3... = the command, one argument per word.
  # The command crosses three shells (this one, ssh, the container's bash), so
  # it travels base64-encoded: JSON list values like '[69]' in a --set
  # override reach train.py with their quotes intact.
  local name="$1" gpu="$2"; shift 2
  local payload
  payload="$(printf '%q ' "$@"; printf '>> logs/tars_launch/%s.log 2>&1' "${name}")"
  payload="$(printf '%s' "${payload}" | base64 -w0)"
  remote "mkdir -p ${remote_root}/logs/tars_launch ${remote_root}/.warp_cache && docker rm ${name} >/dev/null 2>&1; \
    docker run -d --name ${name} \
      --runtime=nvidia --gpus device=${gpu} --ipc=host --ulimit stack=67108864 --ulimit core=0 --user root \
      -e TZ=Europe/Zurich -e NVIDIA_DISABLE_REQUIRE=1 -e FLEET_HOST=$(cut -d. -f1 <<< "${host}") \
      -v ~/${remote_root}:/workspace/simtoolreal_newton \
      -v ~/${remote_root}/.warp_cache:/root/.cache/warp \
      -v /usr/share/zoneinfo:/usr/share/zoneinfo:ro \
      -w /workspace/simtoolreal_newton \
      ${image} bash -c 'echo ${payload} | base64 -d | bash'"
}

refuse_busy_gpu() {
  # SHARE_GPU=1 lets a short exec (the +500 doom-check sweep, ~3 GB, a few
  # minutes) run beside a training on the same card; never for a training.
  if [ "${SHARE_GPU:-0}" = "1" ] && [ "${MODE_EXEC:-0}" = "1" ]; then return 0; fi
  if remote "docker ps --filter name=strn_gpu${1}_ --filter status=running -q" | grep -q .; then
    echo "GPU ${1} on ${host} already has a running strn container:" >&2
    remote "docker ps --filter name=strn_gpu${1}_ --format '{{.Names}} {{.Status}}'" >&2
    exit 1
  fi
}

case "${1:-}" in
  status)
    remote 'hostname; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; echo; docker ps -a --filter name=strn_ --format "table {{.Names}}\t{{.Status}}\t{{.RunningFor}}"'
    ;;
  log)
    name="${2:?container name}"
    remote "tail -n ${3:-40} ${remote_root}/logs/tars_launch/${name}.log"
    ;;
  stop)
    name="${2:?container name}"
    remote "docker stop ${name} && docker rm ${name}"
    ;;
  0|1)
    gpu="$1"; shift
    [ $# -ge 1 ] || { echo "no arguments given" >&2; exit 1; }
    if [ "$1" = "exec" ]; then
      shift
      name="strn_gpu${gpu}_exec_$(date +%Y%m%d_%H%M%S)_$$"
      MODE_EXEC=1 refuse_busy_gpu "${gpu}"
      docker_run "${name}" "${gpu}" "$@"
      echo "launched ${name} on ${host} GPU ${gpu}: $*"
      echo "  follow: HOST=${host} scripts/tars_train.sh log ${name}"
      exit 0
    fi
    mode=train
    if [ "$1" = "supervise" ]; then mode=supervise; shift; fi
    run_name="$(sed -n 's/.*--run-name[= ]\([^ ]*\).*/\1/p' <<< "$*")"
    [ -n "${run_name}" ] || { echo "the arguments must carry --run-name" >&2; exit 1; }
    case "$*" in *record-video*) ;; *) set -- "$@" --record-video ;; esac
    name="strn_gpu${gpu}_${run_name}"
    refuse_busy_gpu "${gpu}"
    if [ "${mode}" = supervise ]; then
      docker_run "${name}" "${gpu}" bash scripts/supervise_train.sh "$@"
    else
      docker_run "${name}" "${gpu}" python scripts/train.py "$@"
    fi
    echo "launched ${name} on ${host} GPU ${gpu}"
    echo "  follow: HOST=${host} scripts/tars_train.sh log ${name}"
    echo "  status: HOST=${host} scripts/tars_train.sh status"
    ;;
  *)
    sed -n '2,11p' "$0"
    exit 1
    ;;
esac
