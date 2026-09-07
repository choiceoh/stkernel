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
mkdir -p "$D/cache-archive-srv1"
# No serving job can read/write these derived files during this fleet hold.
ssh choiceoh@10.10.10.1 sudo -n python3 /tmp/mhcbf16-cache-archive.py plan > "$D/cache-archive-manifest.json"
scp -q choiceoh@10.10.10.1:/tmp/mhcbf16-fp8-cache-files.list "$D/cache-files.list"
rsync -a --rsync-path="sudo -n rsync" --from0 --files-from="$D/cache-files.list" choiceoh@10.10.10.1:/home/choiceoh/glm53-cache/glm53-fp8/ "$D/cache-archive-srv1/"
python3 "$D/cache-archive.py" verify-backup "$D/cache-archive-manifest.json" "$D/cache-archive-srv1"
ssh choiceoh@10.10.10.1 sudo -n python3 /tmp/mhcbf16-cache-archive.py remove-archived > "$D/cache-removed.json"
ssh choiceoh@10.10.10.1 df -h /home/choiceoh > "$D/space-after.txt"
cat "$D/cache-removed.json" "$D/space-after.txt"
bash launchers/deploy-overlays.sh glm53 > "$D/deploy.log" 2>&1
bash bench/chain.sh \
 'MHCBF16B1=' \
 'MHCBF16A1=VLLM_GLM53_MK_FP8_PACK2=1 VLLM_GLM53_MK_GEMM_TRANSPOSE_M8=2 VLLM_GLM53_MK_M8_FASTPATH=1 VLLM_GLM53_MK_MHC_BF16=1' \
 --after MHCBF16A1 'python3 bench/judge.py MHCBF16A1 --base MHCBF16B1 --fail-invalid' \
 'MHCBF16A2=VLLM_GLM53_MK_FP8_PACK2=1 VLLM_GLM53_MK_GEMM_TRANSPOSE_M8=2 VLLM_GLM53_MK_M8_FASTPATH=1 VLLM_GLM53_MK_MHC_BF16=1' \
 'MHCBF16B2=' \
 --after MHCBF16B2 'python3 bench/judge.py MHCBF16A1 --base MHCBF16B2 --fail-invalid; python3 bench/judge.py MHCBF16A2 --base MHCBF16B2 --fail-invalid'
