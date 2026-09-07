#!/usr/bin/env bash
# Resume after the valid B1; the failed A1 never changed the serving container.
set -euo pipefail
cd /home/choiceoh/stkernel-glm53-m8-next
export REPO=/home/choiceoh/stkernel-glm53-m8-next
export LEVER=$REPO/bench/ab-lever.sh
export FLEET=/home/choiceoh/stkernel/bench/fleet.sh
export IMAGE=sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211
export PREFILL_WARMUP=0 QUALITY_CTX=2000,32000,128000 MAX_JOBS=2
export ONEPASS_FIXED_DECODE_TOKENS=2048 ONEPASS_FIXED_DECODE_REPS=3 ONEPASS_REQUIRE_EXCLUSIVE=1
D=/home/choiceoh/glm53-logs/M8NEXT
test "$(cut -c1-12 /home/choiceoh/glm53-cache/.overlay-sha)" = 83bc6639a0a2
test "$(sed -n '1p' /home/choiceoh/overlays/glm53/manifest.tsv)" = '# source_commit=b13dc8ba3f56752e6391a8b45b96eed864656705'
python3 "$D/cache-archive.py" verify-backup "$D/cache-archive-manifest.json" "$D/cache-archive-srv1"
ssh choiceoh@10.10.10.1 sudo -n python3 /tmp/m8-cache-archive.py remove-archived > "$D/cache-removed.json"
ssh choiceoh@10.10.10.1 df -h /home/choiceoh > "$D/srv1-space-after.txt"
cat "$D/cache-removed.json" "$D/srv1-space-after.txt"
bash bench/chain.sh \
  'M8NEXTA1=VLLM_GLM53_MK_FP8_PACK2=1 VLLM_GLM53_MK_GEMM_TRANSPOSE_M8=2 VLLM_GLM53_MK_M8_FASTPATH=1' \
  'M8NEXTA2=VLLM_GLM53_MK_FP8_PACK2=1 VLLM_GLM53_MK_GEMM_TRANSPOSE_M8=2 VLLM_GLM53_MK_M8_FASTPATH=1' \
  'M8NEXTB2=' \
  --after M8NEXTB2 'python3 bench/judge.py M8NEXTB2 --fail-invalid'
