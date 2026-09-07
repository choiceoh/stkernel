#!/usr/bin/env bash
# Capture all ranks before chain.sh's failure judge can start another boot.
set -euo pipefail
name=${1:?}; knobs=${2:-}; out=${INPUT_REUSE_SERVING_OUT:?}
kind=baseline
[[ $knobs != *VLLM_GLM53_MK_INPUT_REUSE=1* ]] || kind=candidate
rc=0
bash "$REPO/bench/ab-lever.sh" "$name" "$knobs" || rc=$?
if curl -fsS --max-time 5 http://127.0.0.1:18000/health >/dev/null; then
  python3 "$REPO/probes/input_reuse_runtime_proof.py" "$kind" > "$out/runtime-$name-srv2.json" || rc=1
  for node in 1 3 4; do
    ssh -o BatchMode=yes "choiceoh@10.10.10.$node" python3 - "$kind" \
      < "$REPO/probes/input_reuse_runtime_proof.py" > "$out/runtime-$name-srv$node.json" || rc=1
  done
  cp "/home/choiceoh/glm53-logs/boot-$name.log" "$out/boot-$name.log"
  code=$(curl -s --max-time 3 -o /dev/null -w '%{http_code}' http://10.10.10.2:18000/health || true)
  [[ $code == 000 ]] || rc=1
fi
exit "$rc"
