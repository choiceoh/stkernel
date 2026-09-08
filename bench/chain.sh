#!/usr/bin/env bash
# N arms in ONE hold, no hand-written glue (39차, operator "chain 헬퍼 만들어"):
# the peers' chains (p1/p4/apc2/df3) were two or three ab-lever arms plus a
# custom check. This helper records proof per arm, yields to a short probe
# between arms, samples defaults only when the build's floor is thin, and
# writes verdicts before releasing to the next job.
#
#   bash bench/chain.sh NAME=KNOBS [NAME=KNOBS ...]
#   (normally: fleet.sh chain <session> [est] [note] -- NAME=KNOBS ...)
#
#   NAME=""                       a defaults arm (counts as the build's baseline sample)
#   NAME="VLLM_X=1 VLLM_Y=1"      a candidate arm
# Each arm runs exactly one canonical onepass. Extra GPU hooks are rejected
# before any arm boots; passive onepass metrics stay in the same workload.
# FLEET_REHEARSE=1 runs everything without a GPU (ab-lever fabricates records).
set -uo pipefail
LOGD=${LOGD:-/home/choiceoh/glm53-logs}
REPO=${REPO:-/home/choiceoh/stkernel}
FLEET=${FLEET:-$LOGD/fleet.sh}
LEVER=$REPO/bench/ab-lever.sh
S=${FLEET_SESSION:-chain}
CHAIN_FLOOR_N=${CHAIN_FLOOR_N:-1}
[[ "$CHAIN_FLOOR_N" =~ ^[1-9][0-9]*$ ]] || { echo 'CHAIN_FLOOR_N must be positive'; exit 2; }
cd "$REPO" || exit 1

declare -a NAMES=() KNOBS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --after|--legs) echo 'chain.sh: GPU work requires onepass; --after and --legs are disabled' >&2; exit 2;;
    *=*)     NAMES+=("${1%%=*}"); KNOBS+=("${1#*=}"); shift;;
    *) echo "chain.sh: arm must be NAME=KNOBS, got: $1" >&2; exit 2;;
  esac
done
[ ${#NAMES[@]} -gt 0 ] || { echo "usage: chain.sh NAME=KNOBS [...]" >&2; exit 2; }

echo "== chain ${NAMES[*]} $(date +%T) session=$S rehearse=${FLEET_REHEARSE:-0}"
python3 bench/baseline.py --brief 2>/dev/null | sed 's/^/   /'
had_defaults=0
for i in "${!NAMES[@]}"; do
  n=${NAMES[$i]}; k=${KNOBS[$i]}
  [ -z "$k" ] && had_defaults=1
  echo "== $(date +%T) arm $n: ${k:-(defaults)}"
  LEGS=onepass bash "$LEVER" "$n" "$k" 2>&1 | tail -40 || exit $?
  # Publish while this arm's evidence is fresh, before another boot/check.
  # A missing baseline remains incomplete until the final pass below.
  if [ -n "$k" ]; then
    python3 bench/judge.py "$n" --write --fail-invalid ${FLEET_REHEARSE:+--allow-rehearsal} || exit $?
  fi
  # between arms: a short queued probe may use this idle serving; we keep our place
  [ -x "$FLEET" ] && bash "$FLEET" yield "$S" 15 2>&1 | sed 's/^/   /'
done

# the build's baseline: a defaults arm above counts; else take one only while
# the floor is thin. Recovery belongs to the central idle owner.
last=${NAMES[$((${#NAMES[@]} - 1))]}
nb=$(python3 bench/baseline.py --count-for "$last") || exit $?
if [ "$had_defaults" = 1 ] && [ -z "${KNOBS[$((${#NAMES[@]} - 1))]}" ]; then
  echo "== $(date +%T) last measured arm remains available ($last = defaults)"
elif [ "${nb:-0}" -lt "$CHAIN_FLOOR_N" ]; then
  # The missing baseline is necessary measurement evidence even with a successor.
  echo "== $(date +%T) defaults arm ${last}BASE (baseline sample ${nb:-0}/$CHAIN_FLOOR_N on this build; the verdict needs it)"
  bash "$LEVER" "${last}BASE" "" 2>&1 | tail -30 || exit $?
else
  echo "== $(date +%T) baseline reused; release immediately for the next job"
fi

echo "== $(date +%T) judge"
for i in "${!NAMES[@]}"; do
  [ -n "${KNOBS[$i]}" ] || continue
  python3 bench/judge.py "${NAMES[$i]}" --write ${FLEET_REHEARSE:+--allow-rehearsal} | sed "s/^/   /"
done
echo "== chain done $(date +%T)"
