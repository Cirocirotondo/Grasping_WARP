#!/usr/bin/env bash
# Pull the light files of every fleet run into logs/simtoolreal/<PREFIX>_<run>/.
#
#     scripts/fleet_pull_light.sh            # every host
#     scripts/fleet_pull_light.sh tars       # one or more hosts by key
#
# "Light" means everything the training dashboard reads and nothing else:
# metrics.jsonl, config.json, SUPERVISOR_DONE, the recorded videos, the
# evaluation plots and the sweep/eval JSON verdicts. Checkpoints (*.pt,
# gigabytes per run) and TensorBoard events files are never transferred.
#
# One rsync per host, with filter rules, so an unreachable host costs one
# timeout and the others still land. Each host lands in its own staging tree
# under logs/fleet_pull/<host>/ (that is what rsync keeps in sync), and the
# runs are then published into logs/simtoolreal/ under the prefixed name the
# dashboard reads the host from. The second step hardlinks, so a pulled clip
# is stored once.

set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_ROOT="${REPO}/logs/simtoolreal"
STAGE_ROOT="${REPO}/logs/fleet_pull"

# key|remote|prefix
HOSTS=(
  "tars|scirelli@tars.inf.ethz.ch:simtoolreal_newton/logs/simtoolreal/|TARS_"
  "case|scirelli@case.inf.ethz.ch:simtoolreal_newton/logs/simtoolreal/|CASE_"
  "ur5|ur5:simtoolreal_newton/logs/simtoolreal/|UR5_"
)

wanted=("$@")

mkdir -p "${DEST_ROOT}" "${STAGE_ROOT}"
status=0

for entry in "${HOSTS[@]}"; do
  IFS='|' read -r key remote prefix <<<"${entry}"
  if [ ${#wanted[@]} -gt 0 ]; then
    match=0
    for name in "${wanted[@]}"; do
      [ "${name}" = "${key}" ] && match=1
    done
    [ "${match}" = "1" ] || continue
  fi

  stage="${STAGE_ROOT}/${key}"
  mkdir -p "${stage}"
  echo "== ${key}: ${remote}"
  # --prune-empty-dirs plus include/exclude rules: descend into every run
  # directory, take the light files, refuse everything else. The '*/' include
  # is what lets rsync walk the run directories at all; the trailing '*'
  # exclude is what drops checkpoints and anything new and heavy.
  rsync -a --prune-empty-dirs --partial \
    --timeout=180 \
    -e "ssh -o BatchMode=yes -o ConnectTimeout=20" \
    --include='*/' \
    --include='metrics.jsonl' \
    --include='config.json' \
    --include='SUPERVISOR_DONE' \
    --include='sweep_*.json' \
    --include='eval_*.json' \
    --include='videos/**' \
    --include='eval_videos/**' \
    --include='eval_plots/**' \
    --exclude='*.pt' \
    --exclude='events.out.tfevents.*' \
    --exclude='*' \
    "${remote}" "${stage}/"
  code=$?
  if [ "${code}" != "0" ]; then
    echo "   WARNING: ${key} unreachable or rsync failed (exit ${code}); keeping what is already local" >&2
    status=1
  fi

  # Publish each staged run under its prefixed name. --link-dest hardlinks the
  # unchanged files instead of copying them, and rsync preserves mtimes, so the
  # dashboard's "live" window still means what it says.
  published=0
  for run in "${stage}"/*/; do
    [ -d "${run}" ] || continue
    name="$(basename "${run}")"
    target="${DEST_ROOT}/${prefix}${name}"
    mkdir -p "${target}"
    rsync -a --link-dest="${run%/}" "${run}" "${target}/" && published=$((published + 1))
  done
  echo "   ${published} run(s) published as ${prefix}*"
done

exit "${status}"
