#!/usr/bin/env bash
# Run through fleet.sh run --gpu --probe. No deployment or service restart.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
python3 probes/glm53_probe_memory.py
eval "$(
  . profiles/glm53.env
  printf 'PROFILE_IMAGE=%q\nTARGET_PREFIX=%q\n' "$PROFILE_IMAGE" "$TARGET_PREFIX"
)"
IMAGE=$(docker image inspect "${IMAGE:-$PROFILE_IMAGE}" --format '{{.Id}}')
echo "revision=$(git rev-parse HEAD) image=$IMAGE"
probe_container="mla32-probe-${FLEET_SESSION:-manual}-$$"
trap 'docker rm -f "$probe_container" >/dev/null 2>&1 || true' EXIT
docker run --rm --name "$probe_container" --network none --gpus all --cpus 4 --memory 6g \
  --entrypoint /bin/bash -v "$REPO:/repo:ro" -e "MK_PKG_PATH=${TARGET_PREFIX%/}" \
  -e MAX_JOBS=1 -e VLLM_GLM53_MEGAKERNEL=1 -e VLLM_GLM53_MK_MLA=1 \
  "$IMAGE" -lc '
    set -euo pipefail
    python3 /repo/probes/mk_mla_prefill32_check.py
    compute-sanitizer --error-exitcode 99 --tool memcheck python3 /repo/probes/mk_mla_prefill32_check.py --sanitize
    compute-sanitizer --error-exitcode 99 --tool racecheck python3 /repo/probes/mk_mla_prefill32_check.py --sanitize
  '
