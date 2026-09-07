#!/usr/bin/env bash
# Run through fleet.sh run --gpu --probe. No deployment or service restart.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
sanitize_only=0
if [[ ${1:-} == --sanitize-only ]]; then sanitize_only=1; shift; fi
[[ $# == 0 ]] || { echo 'usage: run_mk_mla_prefill32_check.sh [--sanitize-only]'; exit 2; }
revision=${PREFILL32_SOURCE_REV:-$(git rev-parse HEAD)}
# The public head can have less spare UMA than a worker. Choose a node using
# the same unmodified admission guard, before starting any GPU process.
# All four nodes are covered by the fleet hold. This never changes serving.
if ! python3 probes/glm53_probe_memory.py; then
  if [[ ${PREFILL32_LOCAL_ONLY:-0} == 1 ]]; then
    exit 3
  fi
  selected=""
  for node in 10.10.10.1 10.10.10.3 10.10.10.4; do
    if ssh -o BatchMode=yes -o ConnectTimeout=5 "choiceoh@$node" python3 - \
        < probes/glm53_probe_memory.py; then
      selected=$node
      break
    fi
  done
  [[ -n $selected ]] || { echo 'No node has probe memory headroom; GPU gate not run'; exit 3; }
  remote_dir=$(ssh -o BatchMode=yes "choiceoh@$selected" mktemp -d /tmp/glm53-mla32-probe.XXXXXXXX)
  [[ $remote_dir =~ ^/tmp/glm53-mla32-probe\.[A-Za-z0-9]+$ ]] || exit 3
  tar -czf - probes/glm53_probe_memory.py probes/mk_mla_prefill32_check.py \
    probes/run_mk_mla_prefill32_check.sh profiles/glm53.env \
    overlay/modules/glm53_megakernel/glm53_megakernel.{py,cu} | \
    ssh -o BatchMode=yes "choiceoh@$selected" "tar -xzf - -C '$remote_dir'"
  echo "probe_node=$selected source=$revision remote_dir=$remote_dir"
  # Values are a git SHA, a Docker image ID and a mktemp path, never arbitrary
  # shell snippets. A pin is required when forwarding IMAGE to the worker.
  [[ $revision =~ ^[a-f0-9]{40}$ ]] || exit 3
  [[ ${IMAGE:-} =~ ^sha256:[a-f0-9]{64}$ ]] || exit 3
  extra_arg=""
  [[ $sanitize_only == 0 ]] || extra_arg=--sanitize-only
  exec ssh -o BatchMode=yes "choiceoh@$selected" \
    "cd '$remote_dir' && PREFILL32_LOCAL_ONLY=1 PREFILL32_SOURCE_REV='$revision' IMAGE='$IMAGE' bash probes/run_mk_mla_prefill32_check.sh $extra_arg"
fi
eval "$(
  . profiles/glm53.env
  printf 'PROFILE_IMAGE=%q\nTARGET_PREFIX=%q\n' "$PROFILE_IMAGE" "$TARGET_PREFIX"
)"
IMAGE=$(docker image inspect "${IMAGE:-$PROFILE_IMAGE}" --format '{{.Id}}')
echo "probe_node=$(hostname) revision=$revision image=$IMAGE"
tool_dir=/usr/local/cuda/compute-sanitizer
[[ -x $tool_dir/compute-sanitizer ]] || { echo 'Host compute-sanitizer missing'; exit 3; }
sha256sum "$tool_dir/compute-sanitizer"
docker run --rm --runtime runc --network none --memory 256m --cpus 1 \
  -v "$tool_dir:/opt/glm-probe-sanitizer:ro" \
  --entrypoint /opt/glm-probe-sanitizer/compute-sanitizer "$IMAGE" --version
probe_container="mla32-probe-${FLEET_SESSION:-manual}-$$"
trap 'docker rm -f "$probe_container" >/dev/null 2>&1 || true' EXIT
timeout --signal=TERM --kill-after=30s 12m docker run --rm --name "$probe_container" --network none --gpus all --cpus 4 --memory 6g \
  --entrypoint /bin/bash -v "$REPO:/repo:ro" -e "MK_PKG_PATH=${TARGET_PREFIX%/}" \
  -v "$tool_dir:/opt/glm-probe-sanitizer:ro" -e "PREFILL32_SANITIZE_ONLY=$sanitize_only" \
  -e MAX_JOBS=1 -e VLLM_GLM53_MEGAKERNEL=1 -e VLLM_GLM53_MK_MLA=1 \
  "$IMAGE" -lc '
    set -euo pipefail
    if [[ $PREFILL32_SANITIZE_ONLY != 1 ]]; then
      python3 /repo/probes/mk_mla_prefill32_check.py
    fi
    /opt/glm-probe-sanitizer/compute-sanitizer --error-exitcode 99 --tool memcheck python3 /repo/probes/mk_mla_prefill32_check.py --sanitize
    /opt/glm-probe-sanitizer/compute-sanitizer --error-exitcode 99 --tool racecheck python3 /repo/probes/mk_mla_prefill32_check.py --sanitize
  '
