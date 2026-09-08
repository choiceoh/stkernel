#!/usr/bin/env bash
# Verify actual capture before traffic, then run the standard onepass on that boot.
set -euo pipefail
name=${1:?}; knobs=${2:-}; out=${INPUT_REUSE_SERVING_OUT:?}
kind=baseline
[[ $knobs != *VLLM_GLM53_MK_INPUT_REUSE=1* ]] || kind=candidate
collect() {
  local prefix=$1 status=0 code
  curl -fsS --max-time 5 http://127.0.0.1:18000/health >/dev/null || return 1
  python3 "$REPO/probes/input_reuse_runtime_proof.py" "$kind" > "$out/$prefix-srv2.json" || status=1
  for node in 1 3 4; do
    ssh -o BatchMode=yes "choiceoh@10.10.10.$node" python3 - "$kind" \
      < "$REPO/probes/input_reuse_runtime_proof.py" > "$out/$prefix-srv$node.json" || status=1
  done
  code=$(curl -s --max-time 3 -o /dev/null -w '%{http_code}' http://10.10.10.2:18000/health || true)
  [[ $code == 000 ]] || status=1
  return "$status"
}
rc=0
LEGS=none bash "$REPO/bench/ab-lever.sh" "$name" "$knobs" > "$out/prepare-$name.log" 2>&1 || rc=$?
tail -n 24 "$out/prepare-$name.log"
collect "prepared-$name" || rc=1
if [[ $rc == 0 && ${LEGS:-onepass} != none ]]; then
  cold=0
  if grep -q 'first boot on build' "$out/prepare-$name.log"; then cold=1; fi
  SKIP_BOOT=1 MK_COLD_COMPILE=$cold bash "$REPO/bench/ab-lever.sh" "$name" "$knobs" || rc=$?
fi
collect "runtime-$name" || rc=1
if [[ -f /home/choiceoh/glm53-logs/boot-$name.log ]]; then
  cp "/home/choiceoh/glm53-logs/boot-$name.log" "$out/boot-$name.log"
fi
exit "$rc"
