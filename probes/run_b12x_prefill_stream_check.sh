#!/usr/bin/env bash
# fleet.sh run --gpu --probe only. Isolated sources/cache; no deployment.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
revision=${MOE_STREAM_SOURCE_REV:-$(git rev-parse HEAD)}
[[ $revision =~ ^[a-f0-9]{40}$ ]] || exit 3
[[ ${IMAGE:-} =~ ^sha256:[a-f0-9]{64}$ ]] || { echo 'immutable IMAGE required'; exit 3; }
if ! python3 probes/glm53_probe_memory.py; then
  [[ ${MOE_STREAM_LOCAL_ONLY:-0} != 1 ]] || exit 3
  selected=""
  for node in 10.10.10.1 10.10.10.3 10.10.10.4; do
    if ssh -o BatchMode=yes -o ConnectTimeout=5 "choiceoh@$node" python3 - \
        < probes/glm53_probe_memory.py; then
      selected=$node
      break
    fi
  done
  [[ -n $selected ]] || { echo 'No node has MoE probe memory headroom'; exit 3; }
  remote_dir=$(ssh -o BatchMode=yes "choiceoh@$selected" mktemp -d /tmp/glm53-moe-stream.XXXXXXXX)
  [[ $remote_dir =~ ^/tmp/glm53-moe-stream\.[A-Za-z0-9]+$ ]] || exit 3
  tar -czf - build/glm53 probes/glm53_probe_memory.py probes/b12x_static_probe.py \
    probes/b12x_prefill_stream_check.py probes/run_b12x_prefill_stream_check.sh | \
    ssh -o BatchMode=yes "choiceoh@$selected" "tar -xzf - -C '$remote_dir'"
  echo "probe_node=$selected source=$revision remote_dir=$remote_dir"
  exec ssh -o BatchMode=yes "choiceoh@$selected" \
    "cd '$remote_dir' && MOE_STREAM_LOCAL_ONLY=1 MOE_STREAM_SOURCE_REV='$revision' IMAGE='$IMAGE' bash probes/run_b12x_prefill_stream_check.sh"
fi
log_dir=$(mktemp -d /tmp/glm53-moe-stream-evidence.XXXXXXXX)
echo "probe_node=$(hostname) revision=$revision image=$IMAGE evidence=$log_dir"
mounts=(-v "$REPO:/repo:ro" -v "$log_dir:/evidence")
while IFS=$'\t' read -r source target rest; do
  if [[ $target == */flashinfer/* || $source == flashinfer_b12x_moe.py ]]; then
    [[ -f "$REPO/build/glm53/$source" ]] || exit 3
    mounts+=(-v "$REPO/build/glm53/$source:$target:ro")
  fi
done < build/glm53/manifest.tsv
probe_container="moe-stream-probe-${FLEET_SESSION:-manual}-$$"
trap 'docker rm -f "$probe_container" >/dev/null 2>&1 || true' EXIT
timeout --signal=TERM --kill-after=30s 12m docker run --rm --name "$probe_container" \
  --network none --gpus all --cpus 4 --memory 8g --entrypoint /bin/bash \
  "${mounts[@]}" -e MAX_JOBS=1 "$IMAGE" -lc '
    set -euo pipefail
    python3 /repo/probes/b12x_prefill_stream_check.py --output /evidence/numerics.json
    compute-sanitizer --error-exitcode 99 --tool memcheck python3 /repo/probes/b12x_prefill_stream_check.py --sanitize --output /evidence/memcheck.json
    compute-sanitizer --error-exitcode 99 --tool racecheck python3 /repo/probes/b12x_prefill_stream_check.py --sanitize --output /evidence/racecheck.json
    echo MOE_STREAM_ALL_GATES_PASS
  ' 2>&1 | tee "$log_dir/run.log"
