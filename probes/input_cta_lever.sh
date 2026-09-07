#!/usr/bin/env bash
# Preserve the canonical onepass gates while retaining its separate SSE channels.
set -euo pipefail
name=${1:?}; knobs=${2:-}; out=${INPUT_CTA_SERVING_OUT:?}
mode=0
for pair in $knobs; do
  [[ $pair != VLLM_GLM53_MK_INPUT_CTA=* ]] || mode=${pair#*=}
done
[[ $mode =~ ^[0-3]$ ]] || exit 2
collect() {
  local prefix=$1 status=0
  curl -fsS --max-time 5 http://127.0.0.1:18000/health >/dev/null || return 1
  python3 "$REPO/probes/input_cta_runtime_proof.py" "$mode" > "$out/$prefix-srv2.json" || status=1
  for node in 1 3 4; do
    ssh -o BatchMode=yes "choiceoh@10.10.10.$node" python3 - "$mode" \
      < "$REPO/probes/input_cta_runtime_proof.py" > "$out/$prefix-srv$node.json" || status=1
  done
  [[ $(curl -s --max-time 3 -o /dev/null -w '%{http_code}' http://10.10.10.2:18000/health || true) == 000 ]] || status=1
  return "$status"
}
rc=0
LEGS=none bash "$REPO/bench/ab-lever.sh" "$name" "$knobs" > "$out/prepare-$name.log" 2>&1 || rc=$?
tail -n 16 "$out/prepare-$name.log"
collect "prepared-$name" || rc=1
if [[ $rc == 0 && ${LEGS:-onepass} != none ]]; then
  export SPEC_K=5 BENCH_MODEL=glm-5.3-flash
  export INPUT_REUSE_CHANNELS_OUT=$out/channels-$name.jsonl
  export MK_COLD_COMPILE=0
  if grep -q 'first boot on build' "$out/prepare-$name.log"; then export MK_COLD_COMPILE=1; fi
  timeout 900 python3 "$REPO/probes/input_reuse_channels.py" --name "$name" || rc=$?
fi
collect "runtime-$name" || rc=1
cp /home/choiceoh/glm53-logs/glm53.log "$out/boot-$name.log"
exit "$rc"
