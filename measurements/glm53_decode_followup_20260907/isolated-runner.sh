#!/usr/bin/env bash
# Canonical fleet hold required. B is profile runtime + disabled disk FP8 cache.
# This cache override is recorded; B is NOT a reusable profile-default baseline.
set -euo pipefail
cd /home/choiceoh/stkernel-mhc-reuse
export REPO=/home/choiceoh/stkernel-mhc-reuse
export LEVER=$REPO/bench/ab-lever.sh
export FLEET=/home/choiceoh/stkernel/bench/fleet.sh
export IMAGE=sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211
export PREFILL_WARMUP=0 QUALITY_CTX=2000,32000,128000 MAX_JOBS=2
export ONEPASS_FIXED_DECODE_TOKENS=2048 ONEPASS_FIXED_DECODE_REPS=5 ONEPASS_REQUIRE_EXCLUSIVE=1
export VLLM_GLM53_FP8_CACHE=0
D=/home/choiceoh/glm53-logs/MHCBF16
export GLM53_API_PORT=18000 GLM53_API_HOST=127.0.0.1 HEAD=127.0.0.1
restore_public_endpoint() {
  local rc=$?
  trap - EXIT
  echo "== restore public endpoint after isolated bracket (rc=$rc)"
  GLM53_API_PORT=8000 GLM53_API_HOST=0.0.0.0 HEAD=10.10.10.2 LEGS=none \
    bash "$LEVER" MHCBF16PUBLIC "" > "$D/public-restore.log" 2>&1 || return 1
  return "$rc"
}
trap restore_public_endpoint EXIT
# Capacity recovery was completed by runner.sh; never repeat that archive.
bash launchers/deploy-overlays.sh glm53 > "$D/isolated-deploy.log" 2>&1
bash bench/chain.sh \
 'MHCBF16LB1=' \
 'MHCBF16LA1=VLLM_GLM53_MK_FP8_PACK2=1 VLLM_GLM53_MK_GEMM_TRANSPOSE_M8=2 VLLM_GLM53_MK_M8_FASTPATH=1 VLLM_GLM53_MK_MHC_BF16=1' \
 --after MHCBF16LA1 'python3 bench/judge.py MHCBF16LA1 --base MHCBF16LB1 --fail-invalid' \
 'MHCBF16LA2=VLLM_GLM53_MK_FP8_PACK2=1 VLLM_GLM53_MK_GEMM_TRANSPOSE_M8=2 VLLM_GLM53_MK_M8_FASTPATH=1 VLLM_GLM53_MK_MHC_BF16=1' \
 'MHCBF16LB2=' \
 --after MHCBF16LB2 'python3 bench/judge.py MHCBF16LA1 --base MHCBF16LB2 --fail-invalid; python3 bench/judge.py MHCBF16LA2 --base MHCBF16LB2 --fail-invalid'
