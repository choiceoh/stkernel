#!/usr/bin/env bash
set -euo pipefail
cd /home/choiceoh/stkernel-glm53-m8-serving
export REPO=/home/choiceoh/stkernel-glm53-m8-serving
export LEVER=$REPO/bench/ab-lever.sh
export FLEET=/home/choiceoh/stkernel/bench/fleet.sh
export IMAGE=sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211
export PREFILL_WARMUP=0
export QUALITY_CTX=2000,32000,128000
export MAX_JOBS=2
bash launchers/deploy-overlays.sh glm53 > /home/choiceoh/glm53-logs/glm53-m8-serving-deploy.log 2>&1
bash bench/chain.sh \
  'M8SERVB1=' \
  --after M8SERVB1 'SKIP_BOOT=1 bash "$LEVER" M8SERVB2 ""' \
  'M8SERVA1=VLLM_GLM53_MK_FP8_PACK2=1 VLLM_GLM53_MK_GEMM_TRANSPOSE_M8=2' \
  --after M8SERVA1 'SKIP_BOOT=1 bash "$LEVER" M8SERVA2 "VLLM_GLM53_MK_FP8_PACK2=1 VLLM_GLM53_MK_GEMM_TRANSPOSE_M8=2" && python3 bench/judge.py M8SERVA2 --write --fail-invalid' \
  'M8SERVB3='
python3 bench/judge.py M8SERVA2 --write
