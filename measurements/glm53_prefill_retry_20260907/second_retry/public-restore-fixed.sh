#!/usr/bin/env bash
set -euo pipefail
export KV_TOKENS=2000000 MAX_LEN=1048576
export GLM53_API_PORT=8000 GLM53_API_HOST=0.0.0.0 HEAD=10.10.10.2
export LEGS=none PREFILL_WARMUP=0
trap 'rc=$?; printf "%s\n" "$rc" > /tmp/glm53-prefill-retry2.7gy5_ruh/public-restore-exit-code' EXIT
bash /home/choiceoh/stkernel-prefill-fused-serving/bench/ab-lever.sh SPFR20907PUBLIC ""
