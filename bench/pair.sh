#!/usr/bin/env bash
# The standard bracket: ONE candidate arm against the build's baseline, with
# every gate and the verdict, and no hand-written chain (39차 idea 1: three of
# the 09-06 chain bugs were in hand-written glue).
#
#   bash bench/pair.sh <NAME> "<VLLM_...=1 VLLM_...=1>"
#   (normally through: fleet.sh pair <session> <NAME> "<knobs>" [est] [note])
#
# Steps: candidate boot + onepass (proof recorded) -> yield the fleet to a
# short probe if one waits -> a defaults boot ONLY when this build has no
# baseline or its noise floor needs a sample (fewer than 3), and only when
# nobody with a boot job is queued behind us (fleet.sh restore-needed) ->
# judge (delta vs floor, gates) -> verdict written. FLEET_REHEARSE=1 runs the
# whole thing without a GPU (ab-lever fabricates records; fleet.sh runs it in
# parallel, never holding the fleet).
set -uo pipefail
NAME=${1:?usage: pair.sh NAME "KNOBS"}
KNOBS=${2:-}
LOGD=${LOGD:-/home/choiceoh/glm53-logs}
REPO=${REPO:-/home/choiceoh/stkernel}
FLEET=${FLEET:-$LOGD/fleet.sh}
LEVER=${LEVER:-$LOGD/ab-lever2.sh}
S=${FLEET_SESSION:-pair}
PAIR_FLOOR_N=${PAIR_FLOOR_N:-3}
cd "$REPO" || exit 1
echo "== pair $NAME $(date +%T) knobs: ${KNOBS:-(none)} session=$S rehearse=${FLEET_REHEARSE:-0}"
python3 bench/baseline.py --brief ${KNOBS:+--knobs "$(echo $KNOBS | tr ' ' ',')"} 2>/dev/null | sed 's/^/   /'

echo "== $(date +%T) candidate arm $NAME"
bash "$LEVER" "$NAME" "$KNOBS" 2>&1 | tail -40

# a short probe waiting behind us can use the idle candidate serving; we keep
# our place and continue with the next boot after it
[ -x "$FLEET" ] && bash "$FLEET" yield "$S" 15 2>&1 | sed 's/^/   /'

# does this build need a defaults sample? (no baseline, or a thin floor)
need_base=0
bl=$(python3 bench/baseline.py --brief 2>/dev/null)
case "$bl" in *"NONE for build"*|*"none for build"*) need_base=1;; esac
nb=$(python3 - <<'PY' 2>/dev/null
import sys, os
sys.path.insert(0, "bench")
from baseline import load, for_build, is_baseline, deployed_build, deployed_git
rows = load(); b = deployed_build(); g = deployed_git()
print(sum(1 for r in for_build(rows, b, g) if is_baseline(r)[0] and not r.get("rehearsal")))
PY
)
[ "${nb:-0}" -lt "$PAIR_FLOOR_N" ] && need_base=1
if [ "$need_base" = 1 ]; then
  # the baseline SAMPLE is a measurement, not a restore: without it this build
  # has no verdict (FUS7 #3, 20:00: "no baseline on this build" after the sample
  # was skipped because a boot job followed). Only a bare restore is skippable.
  echo "== $(date +%T) defaults arm ${NAME}BASE (baseline sample ${nb:-0}/$PAIR_FLOOR_N on this build; the verdict needs it)"
  bash "$LEVER" "${NAME}BASE" "" 2>&1 | tail -30
else
  if bash "$FLEET" restore-needed "$S" >/dev/null 2>&1; then
    echo "== $(date +%T) restore boot (defaults, no leg: the build already has ${nb} baseline samples)"
    LEGS=none bash "$LEVER" "${NAME}RESTORE" "" 2>&1 | tail -12
  else
    echo "== $(date +%T) restore skipped: a boot job follows and replaces this serving"
  fi
fi

echo "== $(date +%T) judge"
python3 bench/judge.py "$NAME" --write ${FLEET_REHEARSE:+--allow-rehearsal}
echo "== pair $NAME done $(date +%T)"
