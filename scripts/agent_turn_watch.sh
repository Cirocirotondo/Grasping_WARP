#!/usr/bin/env bash
# Count tool calls per background agent transcript since the last check and
# flag any agent that is burning turns (a busy-wait loop). Cheap: one grep per
# file. State lives beside the transcripts.
#
#   scripts/agent_turn_watch.sh [limit-per-interval]
dir=/tmp/claude-1003/-home-simone-simtoolreal-newton/7ae121c3-3e46-46e8-8e10-89be9387f22a/tasks
limit=${1:-60}
state=$dir/.turn_watch
touch "$state"
for f in "$dir"/a*.output; do
  id=$(basename "$f" .output)
  now=$(grep -c '"command"' "$f" 2>/dev/null || echo 0)
  prev=$(grep "^$id " "$state" | cut -d' ' -f2); prev=${prev:-0}
  delta=$((now - prev))
  flag=""; [ "$delta" -gt "$limit" ] && flag="  <-- RUNAWAY (>$limit turns since last check)"
  echo "$id total=$now delta=$delta$flag"
  grep -v "^$id " "$state" > "$state.tmp" 2>/dev/null; echo "$id $now" >> "$state.tmp"; mv "$state.tmp" "$state"
done
