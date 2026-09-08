#!/usr/bin/env bash
# Production restore is exclusive to the central five-minute idle controller.
set -euo pipefail
session=${FLEET_SESSION:?}
controller=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python3 "$controller/fleet_idle.py" authorize "${FLEET_DIR:?}" "$session" >/dev/null || exit 2
export FLEET_BOOT_INTENT=recovery
# Apply public-default isolation to deployment and the health check too, not
# only to the final boot. In particular IMAGE/PROFILE must not select an arm.
# Match every caller-precedence argument to the GLM ct_load_profile call,
# including set-but-empty CUSTOM_OPS_AXIS, then clear the launcher's additional
# environment controls. The approved profile supplies all serving defaults.
public_default_keys=(
  IMAGE MOE_BACKEND ENABLE_EP EAGER GRAPH_CAP MAX_SEQS MAX_BATCHED MAX_LEN
  GMU SPEC_K KV_DTYPE KV_BYTES DFLASH2 SPEC ASYNC_SCHED ATTN_BACKEND
  MODEL_HOST_PATH SERVED_NAME DRAFT_TP DRAFT_KV CUSTOM_OPS_AXIS COMPILE_CFG
  EXTRA_ENV LOAD_FORMAT DRAFT_SAMPLE REJECT_METHOD PREFIX_CACHE DECODE_FIRST
  CHAT_TEMPLATE REASONING_PARSER MM_LIMIT PREFILL_WARMUP PREFILL_WARMUP_LENS
  MAMBA_CACHE_DTYPE INDEX_CACHE_FREQ OVERLAY_DIR DRAFT_HOST_PATH
  AUDIT B12X_EP_TOPK CG_MEM_PROFILE CG_UTIL_DELTA DRY_RUN GLOO_IFNAME
  GRAPH_DEBUG KV_HYBRID_BLOCKS KV_TOKENS MM_ENCODER_ATTN MM_ENCODER_TP_MODE
  MOE_CUTOVER PIECEWISE PREBUILD SKIP_MM_PROFILING SKIP_PREFLIGHT SPEC_K_FORCE
  TORCH_CPP_LOG_LEVEL TORCH_DISTRIBUTED_DEBUG TORCH_LOGS NCCL_ASYNC_ERR
  FLEET_REHEARSE DEPLOY_PRESERVE_IDENTICAL HEAD_URL
)
unset "${public_default_keys[@]}"
while IFS= read -r key; do
  case "$key" in
    VLLM_*|ONEPASS_*|MK_*|STARTUP_CACHE_*|PROFILE|PROFILE_*|MODEL_HOST_PATH|IMAGE|SPEC_K|SPEC|LEVER|SKIP_BOOT|HEAD|GLM53_API_HOST|GLM53_API_PORT)
      unset "$key" ;;
  esac
done < <(compgen -e)
[[ -n ${FLEET_RECOVERY_RECEIPT:-} ]] || {
  echo 'prevalidated approved recovery receipt is missing; idle controller must prepare it'
  exit 2
}
recovery=$(python3 "${FLEET_RUNNER_REPO:?}/bench/fleet_validation.py" verify-recovery \
  --receipt "$FLEET_RECOVERY_RECEIPT" --format shell)
eval "$recovery"
repo=$FLEET_RECOVERY_REPO
export FLEET_DEPLOY_RECOVERY_RECEIPT=$FLEET_RECOVERY_RECEIPT
cd "$repo"
# A completed public defaults arm of this approved build needs no second boot.
if python3 "${FLEET_RUNNER_REPO:?}/bench/fleet_entry.py" production-current "$repo"; then
  echo 'approved public defaults already healthy; no restore boot'
  exit 0
fi
bash launchers/deploy-overlays.sh glm53
python3 - "$repo" "$session" <<'PY'
import os, subprocess, sys
env = {k:v for k,v in os.environ.items() if not k.startswith(('VLLM_', 'ONEPASS_', 'MK_', 'STARTUP_CACHE_'))}
for key in ('IMAGE', 'SPEC_K', 'SPEC', 'LEVER', 'SKIP_BOOT', 'HEAD', 'GLM53_API_HOST', 'GLM53_API_PORT'):
    env.pop(key, None)
env.update(REPO=sys.argv[1], GLM53_API_HOST='0.0.0.0', GLM53_API_PORT='8000', HEAD='10.10.10.2',
           PREFILL_WARMUP='1', LEGS='none', SKIP_BOOT='0')
raise SystemExit(subprocess.call(['bash', 'bench/ab-lever.sh', sys.argv[2]+'RESTORE', ''], env=env))
PY
