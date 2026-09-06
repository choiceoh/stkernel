#!/usr/bin/env bash
# Small isolated probe: use fleet.sh run --gpu --probe before the GPU arm.
# Never deploys, stops serving, or modifies shared compilation caches.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
compile_only=0
with_tp4=0
skip_scale=0
while [[ $# != 0 ]]; do
  case $1 in
    --compile-only) compile_only=1;;
    --tp4) with_tp4=1;;
    --skip-scale) skip_scale=1;;
    *) echo 'usage: run_glm53_prefill_transport_check.sh [--compile-only|--tp4] [--skip-scale]' >&2; exit 2;;
  esac
  shift
done
PROBE_LOG_DIR=$(mktemp -d "${TMPDIR:-/tmp}/glm53-prefill-transport.XXXXXX")
echo "validation log: $PROBE_LOG_DIR/run.log"
exec > >(tee "$PROBE_LOG_DIR/run.log") 2>&1
date -u
eval "$(
  . "$REPO/profiles/glm53.env"
  printf 'IMAGE=%q\nTARGET_PREFIX=%q\n' "$PROFILE_IMAGE" "$TARGET_PREFIX"
)"
docker image inspect "$IMAGE" --format 'image={{.Id}}'
mounts=(-v "$REPO:/repo:ro")
for pair in \
  'glm53_runtime/glm53_prefill_collectives.py:vllm/distributed/device_communicators/glm53_prefill_collectives.py' \
  'glm53_model/glm53_nvfp4_scale.py:vllm/model_executor/layers/glm53_nvfp4_scale.py' \
  'glm53_model/glm53_fp8_dense.py:vllm/model_executor/layers/glm53_fp8_dense.py'; do
  source_path="$REPO/overlay/modules/${pair%%:*}"
  [[ -f "$source_path" ]] || { echo "missing $source_path" >&2; exit 1; }
  mounts+=(-v "$source_path:${TARGET_PREFIX%/}/${pair#*:}:ro")
done
if [[ $compile_only == 1 ]]; then
  exec docker run --rm --network none --cpus 2 --memory 4g --entrypoint python3 \
    -e VLLM_GLM53_PREFILL_SP=0 "${mounts[@]}" "$IMAGE" \
    /repo/probes/glm53_prefill_transport_check.py --compile-only
fi
docker run --rm --network none --gpus all --cpus 4 --memory 6g --entrypoint python3 \
  -e VLLM_GLM53_PREFILL_SP=0 "${mounts[@]}" "$IMAGE" \
  /repo/probes/glm53_prefill_transport_check.py
if [[ $skip_scale == 0 ]]; then
  docker run --rm --network none --gpus all --cpus 4 --memory 6g --entrypoint python3 \
    -e VLLM_GLM53_PREFILL_SP=0 "${mounts[@]}" "$IMAGE" \
    /repo/probes/glm53_nvfp4_scale_check.py
fi
if [[ $with_tp4 == 1 ]]; then
  bash "$REPO/probes/run_glm53_prefill_tp4_check.sh" fp8-v3
fi
