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
# baseline (one sample by default; PAIR_FLOOR_N=3 explicitly fills the floor) ->
# judge (delta vs floor, gates) -> verdict written. FLEET_REHEARSE=1 runs the
# whole thing without a GPU (ab-lever fabricates records; fleet.sh runs it in
# parallel, never holding the fleet).
set -uo pipefail
NAME=${1:?usage: pair.sh NAME "KNOBS"}
KNOBS=${2:-}
LOGD=${LOGD:-/home/choiceoh/glm53-logs}
REPO=${REPO:-/home/choiceoh/stkernel}
FLEET=${FLEET:-$LOGD/fleet.sh}
LEVER=$REPO/bench/ab-lever.sh
export LEGS=onepass
S=${FLEET_SESSION:-pair}
PAIR_FLOOR_N=${PAIR_FLOOR_N:-1}
[[ "$PAIR_FLOOR_N" =~ ^[1-9][0-9]*$ ]] || { echo 'PAIR_FLOOR_N must be positive'; exit 2; }
cd "$REPO" || exit 1
echo "== pair $NAME $(date +%T) knobs: ${KNOBS:-(none)} session=$S rehearse=${FLEET_REHEARSE:-0}"
python3 bench/baseline.py --brief ${KNOBS:+--knobs "$(echo $KNOBS | tr ' ' ',')"} 2>/dev/null | sed 's/^/   /'

echo "== $(date +%T) candidate arm $NAME"
bash "$LEVER" "$NAME" "$KNOBS" 2>&1 | tail -40 || exit $?
python3 bench/judge.py "$NAME" --write --fail-invalid ${FLEET_REHEARSE:+--allow-rehearsal} || exit $?

# a short probe waiting behind us can use the idle candidate serving; we keep
# our place and continue with the next boot after it
[ -x "$FLEET" ] && bash "$FLEET" yield "$S" 15 2>&1 | sed 's/^/   /'

# Reuse a valid sample by default. A thin noise floor is reported, not
# automatically replenished on each candidate.
need_base=0
nb=$(python3 bench/baseline.py --count-for "$NAME") || exit $?
[ "${nb:-0}" -lt "$PAIR_FLOOR_N" ] && need_base=1
if [ "$need_base" = 1 ]; then
  echo "== $(date +%T) defaults arm ${NAME}BASE (baseline sample ${nb:-0}/$PAIR_FLOOR_N on this build)"
  bash "$LEVER" "${NAME}BASE" "" 2>&1 | tail -30 || exit $?
  python3 bench/judge.py "$NAME" --write --fail-invalid ${FLEET_REHEARSE:+--allow-rehearsal} || exit $?
else
  echo "== $(date +%T) baseline reused; release immediately for the next job"
fi

echo "== $(date +%T) judge"
python3 bench/judge.py "$NAME" --write ${FLEET_REHEARSE:+--allow-rehearsal}
echo "== pair $NAME done $(date +%T)"
