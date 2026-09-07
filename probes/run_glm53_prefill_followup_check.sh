#!/usr/bin/env bash
# One bounded fleet GPU probe turn; never changes serving or shared caches.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
eval "$(
  . "$REPO/profiles/glm53.env"
  printf 'PREFILL_IMAGE=%q\n' "$PROFILE_IMAGE"
)"
export PREFILL_PROBE_IMAGE_ID
PREFILL_PROBE_IMAGE_ID=$(docker image inspect "${PREFILL_PROBE_IMAGE_ID:-$PREFILL_IMAGE}" --format '{{.Id}}')
bash "$REPO/probes/run_glm53_prefill_transport_check.sh" --compile-only
bash "$REPO/probes/run_glm53_prefill_transport_check.sh" --skip-scale --fuse-mhc
# Raw codecs: prove direct packets plus fused MHC at every timed length.
bash "$REPO/probes/run_glm53_prefill_tp4_check.sh" fp8-v3 \
  --rows 128 129 2128 4095 4096 6143 6144 6912 8185 \
  --direct-nccl --fuse-mhc --timing
# Deliberately distinct gates, including TP padding across both boundaries.
bash "$REPO/probes/run_glm53_prefill_tp4_check.sh" fp8-v3 \
  --rows 4095 4096 6143 6144 --ag-min-tokens 6144 --rs-min-tokens 4096 \
  --direct-nccl --fuse-mhc
