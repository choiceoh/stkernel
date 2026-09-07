#!/usr/bin/env bash
# Execute only inside a canonical fleet.sh run --gpu hold, after probe gates.
set -euo pipefail
cd /home/choiceoh/stkernel-glm53-m8-next
export REPO=/home/choiceoh/stkernel-glm53-m8-next
export LEVER=$REPO/bench/ab-lever.sh
export FLEET=/home/choiceoh/stkernel/bench/fleet.sh
export IMAGE=sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211
export PREFILL_WARMUP=0 QUALITY_CTX=2000,32000,128000 MAX_JOBS=2
export ONEPASS_FIXED_DECODE_TOKENS=2048 ONEPASS_FIXED_DECODE_REPS=3 ONEPASS_REQUIRE_EXCLUSIVE=1
bash launchers/deploy-overlays.sh glm53 > /home/choiceoh/glm53-logs/M8NEXT/deploy.log 2>&1
bash bench/chain.sh \
  'M8NEXTB1=' \
  --after M8NEXTB1 'python3 bench/judge.py M8NEXTB1 --fail-invalid' \
  'M8NEXTA1=VLLM_GLM53_MK_FP8_PACK2=1 VLLM_GLM53_MK_GEMM_TRANSPOSE_M8=2 VLLM_GLM53_MK_M8_FASTPATH=1' \
  'M8NEXTA2=VLLM_GLM53_MK_FP8_PACK2=1 VLLM_GLM53_MK_GEMM_TRANSPOSE_M8=2 VLLM_GLM53_MK_M8_FASTPATH=1' \
  'M8NEXTB2=' \
  --after M8NEXTB2 'python3 bench/judge.py M8NEXTB2 --fail-invalid'
