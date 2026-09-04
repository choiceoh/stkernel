#!/usr/bin/env bash
# One arm of a GLM-5.3 A/B, measured the way the ledger judges: boot, then
# BOTH channels from the same boot.
#
# Decode and prefill are not separable here. A kernel change aimed at decode
# can regress prefill, and prefill is TTFT -- the number a user actually
# feels. Measuring decode alone hides that until the next boot, which has
# happened in this lane more than once. So this script runs both or neither:
# there is no flag to skip prefill.
#
# The judgment channel for decode is C=1 step/s, not tok/s. tok/s carries the
# acceptance draw: a base leg on 2026-09-04 read 30.9-43.0 tok/s (39% spread)
# while step/s held 14.5-15.3 (5.5%). Comparing tok/s across arms compares
# luck. bench/bracket.py records both and judges on step/s.
#
#   ab-glm53.sh base                 # megakernel off (profile default)
#   ab-glm53.sh cand                 # megakernel on
#   ab-glm53.sh cand "VLLM_GLM53_MK_KDA=1"   # extra knobs for this arm
set -uo pipefail
ARM=${1:?usage: ab-glm53.sh base|cand [extra env]}
EXTRA=${2:-}
NAME=${AB_NAME:-MKSERVE}
REPS=${AB_REPS:-6}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
HEAD=${HEAD_IP:-10.10.10.2}

case "$ARM" in
  base) ARM_ENV="VLLM_GLM53_MEGAKERNEL=0" ;;   # stock: the profile ships the set ON since 28차
  # MK_PDL rides with the segments: the launch form the kernels were tuned
  # for (17-19 pct per launch, bench probe), and the profile default since
  # 2026-09-04 -- named here so an older profile cannot drop it silently.
  cand) ARM_ENV="VLLM_GLM53_MEGAKERNEL=1 VLLM_GLM53_MK_MHC=1 VLLM_GLM53_MK_GEMM=1 VLLM_GLM53_MK_MLA=1 VLLM_GLM53_MK_PDL=1" ;;
  *) echo "ABORT: arm must be base or cand"; exit 1 ;;
esac

echo "== boot arm=$ARM =="
env $ARM_ENV $EXTRA bash "$REPO/launchers/start-glm53-nvfp4-tp4.sh" || {
  echo "ABORT: boot failed"; exit 1; }

echo "== wait for health =="
for i in $(seq 1 200); do
  [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' "http://$HEAD:8000/health")" = 200 ] \
    && { echo "up after $((i*15))s"; break; }
  sleep 15
done
[ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' "http://$HEAD:8000/health")" = 200 ] || {
  echo "ABORT: never came up"; exit 1; }

# What the boot decided about memory and arming, so the arm is auditable
# later without re-reading a log that the next boot truncates.
echo "== boot fingerprint =="
grep -hE "armed=|MK W4 packs|kda layout gate|GMU|KV pinned" \
  /home/choiceoh/glm53-logs/glm53.log 2>/dev/null | tail -6
free -g | sed -n 2p

echo "== decode (C=1 step/s is the judgment channel) =="
python3 "$REPO/bench/bracket.py" leg --name "$NAME" --tag \
  "$([ "$ARM" = cand ] && echo cand || echo base)" --reps "$REPS"

echo "== prefill (same boot -- never a separate one) =="
python3 "$REPO/probes/prefill_ladder.py" 2048 8192

echo "== arm=$ARM done =="
