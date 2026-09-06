#!/usr/bin/env bash
# N arms in ONE hold, no hand-written glue (39차, operator "chain 헬퍼 만들어"):
# the peers' chains (p1/p4/apc2/df3) were two or three ab-lever arms plus a
# custom check and their own restore -- this is that shape with the fleet's
# rules built in: proof per arm, yield to a short probe between arms, a
# defaults sample only when the build's floor is thin, the restore boot only
# when nobody boots behind us, judge with the noise floor, verdicts written.
#
#   bash bench/chain.sh NAME=KNOBS [NAME=KNOBS ...] [--after NAME 'cmd'] [--legs NAME none]
#   (normally: fleet.sh chain <session> [est] [note] -- NAME=KNOBS ...)
#
#   NAME=""                       a defaults arm (counts as the build's baseline sample)
#   NAME="VLLM_X=1 VLLM_Y=1"      a candidate arm
#   --after NAME 'cmd'            run cmd while NAME's boot is up, after its leg
#                                 (the "mixed-batch SP admission check" kind of step)
#   --legs NAME none              boot NAME but run no leg (a config check, a restore)
# FLEET_REHEARSE=1 runs everything without a GPU (ab-lever fabricates records).
set -uo pipefail
LOGD=${LOGD:-/home/choiceoh/glm53-logs}
REPO=${REPO:-/home/choiceoh/stkernel}
FLEET=${FLEET:-$LOGD/fleet.sh}
LEVER=${LEVER:-$LOGD/ab-lever2.sh}
S=${FLEET_SESSION:-chain}
CHAIN_FLOOR_N=${CHAIN_FLOOR_N:-3}
cd "$REPO" || exit 1

declare -a NAMES=() KNOBS=()
declare -A AFTER=() ARMLEGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --after) AFTER[$2]=$3; shift 3;;
    --legs)  ARMLEGS[$2]=$3; shift 3;;
    *=*)     NAMES+=("${1%%=*}"); KNOBS+=("${1#*=}"); shift;;
    *) echo "chain.sh: arm must be NAME=KNOBS, got: $1" >&2; exit 2;;
  esac
done
[ ${#NAMES[@]} -gt 0 ] || { echo "usage: chain.sh NAME=KNOBS [...] [--after NAME cmd] [--legs NAME none]" >&2; exit 2; }

echo "== chain ${NAMES[*]} $(date +%T) session=$S rehearse=${FLEET_REHEARSE:-0}"
python3 bench/baseline.py --brief 2>/dev/null | sed 's/^/   /'
had_defaults=0
for i in "${!NAMES[@]}"; do
  n=${NAMES[$i]}; k=${KNOBS[$i]}
  [ -z "$k" ] && had_defaults=1
  echo "== $(date +%T) arm $n: ${k:-(defaults)}${ARMLEGS[$n]:+  legs=${ARMLEGS[$n]}}"
  LEGS="${ARMLEGS[$n]:-onepass}" bash "$LEVER" "$n" "$k" 2>&1 | tail -40
  if [ -n "${AFTER[$n]:-}" ]; then
    echo "== $(date +%T) after $n: ${AFTER[$n]}"
    bash -c "${AFTER[$n]}" 2>&1 | tail -20
  fi
  # between arms: a short queued probe may use this idle serving; we keep our place
  [ -x "$FLEET" ] && bash "$FLEET" yield "$S" 15 2>&1 | sed 's/^/   /'
done

# the build's baseline: a defaults arm above counts; else take one only while
# the floor is thin AND nobody boots behind us; else a bare restore if needed
nb=$(python3 - <<'PY' 2>/dev/null
import sys
sys.path.insert(0, "bench")
from baseline import load, for_build, is_baseline, deployed_build, deployed_git
rows = load(); b = deployed_build(); g = deployed_git()
print(sum(1 for r in for_build(rows, b, g) if is_baseline(r)[0] and not r.get("rehearsal")))
PY
)
last=${NAMES[$((${#NAMES[@]} - 1))]}
if [ "$had_defaults" = 1 ] && [ -z "${KNOBS[$((${#NAMES[@]} - 1))]}" ]; then
  echo "== $(date +%T) production stays on the last arm ($last = defaults)"
elif [ "${nb:-0}" -lt "$CHAIN_FLOOR_N" ]; then
  # the baseline SAMPLE is a measurement, not a restore: only a bare restore is
  # skippable when a boot job follows (FUS7 #3 lost its verdict to that confusion)
  echo "== $(date +%T) defaults arm ${last}BASE (baseline sample ${nb:-0}/$CHAIN_FLOOR_N on this build; the verdict needs it)"
  bash "$LEVER" "${last}BASE" "" 2>&1 | tail -30
elif bash "$FLEET" restore-needed "$S" >/dev/null 2>&1; then
  echo "== $(date +%T) restore boot (defaults, no leg; the build has ${nb:-0} baseline samples)"
  LEGS=none bash "$LEVER" "${last}RESTORE" "" 2>&1 | tail -12
else
  echo "== $(date +%T) restore skipped: a boot job follows and replaces this serving"
fi

echo "== $(date +%T) judge"
for i in "${!NAMES[@]}"; do
  [ -n "${KNOBS[$i]}" ] || continue
  python3 bench/judge.py "${NAMES[$i]}" --write ${FLEET_REHEARSE:+--allow-rehearsal} | sed "s/^/   /"
done
echo "== chain done $(date +%T)"
