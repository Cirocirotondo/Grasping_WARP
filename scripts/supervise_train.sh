#!/usr/bin/env bash
# Run one training to an absolute iteration target, restarting it from its
# newest checkpoint whenever the process dies before getting there.
#
#   scripts/supervise_train.sh --run-name NAME --target ITER \
#       [--seed-checkpoint PATH --seed-iteration N] [--python BIN] \
#       [--log-root DIR] [--max-attempts K] -- <train.py arguments>
#
# The same script drives a run on the desktop (pass --python
# deps/IsaacLab/.venv/bin/python) and inside the tars/case container (default
# python = the image's venv), so a run's supervision is identical wherever it
# lands.
#
# Why a loop at all: a long run can die for reasons that have nothing to do
# with the recipe (a CUDA fault, a solver blow-up the guard did not catch,
# the machine's own hiccups), and a plain restart from the last model_<N>.pt
# loses at most save_interval (100) iterations. Everything the run needs to
# know lives in its directory, so the loop never has to interpret the crash.
# The trainer's own divergence abort (abort_on_divergence) exits non-zero as
# well: after ABORT_LIMIT such exits in a row the loop gives up, because a
# recipe that diverges three times from the same checkpoint will not stop
# doing so.
#
# --iterations is relative to the checkpoint being resumed (train.py runs
# start + N), so each attempt passes TARGET - DONE, never the flat target.
#
# Files it leaves behind, for whoever is watching:
#   logs/queue/<NAME>.log      the [sup] state lines followed by the trainer output
#   logs/queue/<NAME>.rundir   the run directory, one line, written at start
#   <run dir>/SUPERVISOR_DONE  written when the loop ends, with the outcome
set -u

PY=python
LOG_ROOT=logs/simtoolreal
MAX_ATTEMPTS=15
ABORT_LIMIT=3
SEED_CKPT=""
SEED_ITER=0
RUN_NAME=""
TARGET=""
while [ $# -gt 0 ]; do
  case "$1" in
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --target) TARGET="$2"; shift 2 ;;
    --seed-checkpoint) SEED_CKPT="$2"; shift 2 ;;
    --seed-iteration) SEED_ITER="$2"; shift 2 ;;
    --python) PY="$2"; shift 2 ;;
    --log-root) LOG_ROOT="$2"; shift 2 ;;
    --max-attempts) MAX_ATTEMPTS="$2"; shift 2 ;;
    --) shift; break ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
[ -n "$RUN_NAME" ] && [ -n "$TARGET" ] || { sed -n '2,10p' "$0" >&2; exit 2; }
[ $# -gt 0 ] || { echo "no train.py arguments after --" >&2; exit 2; }

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
# The desktop venv must not see a stray PYTHONPATH (ROS leaves one behind).
unset PYTHONPATH
LAUNCH_TS=$(date +%Y-%m-%d_%H%M%S)
RUN_DIR="$REPO/$LOG_ROOT/${LAUNCH_TS}_${RUN_NAME}"
LOG="$REPO/logs/queue/${RUN_NAME}.log"
mkdir -p "$RUN_DIR" "$(dirname "$LOG")"
echo "$RUN_DIR" > "$REPO/logs/queue/${RUN_NAME}.rundir"

say() { echo "[sup] $(date +%H:%M:%S) $*" >> "$LOG"; }
say "run dir $RUN_DIR, target $TARGET, seed ${SEED_CKPT:-scratch} @ $SEED_ITER"

if [ -n "$SEED_CKPT" ] && [ ! -f "$SEED_CKPT" ]; then
  say "seed checkpoint missing: $SEED_CKPT"
  echo "status=failed reason=missing_seed" > "$RUN_DIR/SUPERVISOR_DONE"
  exit 1
fi

# Newest model_<N>.pt in the run directory, or nothing before the first save.
latest_ckpt() {
  ls "$RUN_DIR"/model_*.pt 2>/dev/null \
    | sed -n 's/.*model_\([0-9]*\)\.pt$/\1 &/p' | sort -n | tail -1 | cut -d' ' -f2
}

OUTCOME="status=failed reason=out_of_attempts"
ABORTS=0
PREV_DONE=-1
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  CKPT=$(latest_ckpt)
  if [ -n "$CKPT" ]; then
    DONE=$(basename "$CKPT" | sed 's/model_\([0-9]*\)\.pt/\1/')
  elif [ -n "$SEED_CKPT" ]; then
    CKPT="$SEED_CKPT"; DONE="$SEED_ITER"
  else
    CKPT=""; DONE=0
  fi
  if [ "$DONE" -ge "$TARGET" ]; then
    say "reached $DONE/$TARGET -- done"
    OUTCOME="status=done reached=$DONE target=$TARGET attempts=$((attempt - 1))"
    break
  fi
  # Three consecutive attempts that die without saving anything new are a
  # recipe problem (or a broken machine), not a hiccup.
  if [ "$DONE" -eq "$PREV_DONE" ]; then
    ABORTS=$((ABORTS + 1))
    if [ "$ABORTS" -ge "$ABORT_LIMIT" ]; then
      say "no progress in $ABORTS attempts from $DONE -- giving up"
      OUTCOME="status=failed reason=no_progress reached=$DONE target=$TARGET attempts=$((attempt - 1))"
      break
    fi
  else
    ABORTS=0
  fi
  PREV_DONE=$DONE
  REMAINING=$((TARGET - DONE))
  say "attempt $attempt: from ${CKPT:-scratch} @ $DONE, $REMAINING iterations to go"
  if [ -n "$CKPT" ]; then
    "$PY" scripts/train.py "$@" --run-name "$RUN_NAME" --log-dir "$RUN_DIR" \
        --iterations "$REMAINING" --resume "$CKPT" >> "$LOG" 2>&1
  else
    "$PY" scripts/train.py "$@" --run-name "$RUN_NAME" --log-dir "$RUN_DIR" \
        --iterations "$REMAINING" >> "$LOG" 2>&1
  fi
  CODE=$?
  say "attempt $attempt exited with code=$CODE"
  LAST=$(latest_ckpt); LAST_IT=0
  [ -n "$LAST" ] && LAST_IT=$(basename "$LAST" | sed 's/model_\([0-9]*\)\.pt/\1/')
  if [ "$CODE" -eq 0 ] || [ "$LAST_IT" -ge "$TARGET" ]; then
    say "finished at $LAST_IT/$TARGET"
    OUTCOME="status=done reached=$LAST_IT target=$TARGET attempts=$attempt"
    break
  fi
  sleep 30
done
say "supervisor finished: $OUTCOME"
echo "$OUTCOME" > "$RUN_DIR/SUPERVISOR_DONE"
