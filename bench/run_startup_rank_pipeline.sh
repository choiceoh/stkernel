#!/usr/bin/env bash
# HEAD_URL=http://127.0.0.1:8000 fleet.sh run --gpu SESSION 40 "rank restore B/A/A/B" -- bash bench/run_startup_rank_pipeline.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export REPO=$PWD
: "${FLEET_SESSION:?run through fleet.sh run --gpu}"
[[ ${FLEET_RESTORE_MANAGED:-0} == 1 ]] || { echo 'managed production recovery required'; exit 2; }
[[ ${HEAD_URL:-} == http://127.0.0.1:8000 ]] || {
  echo 'export HEAD_URL=http://127.0.0.1:8000 before fleet.sh run so its supervisor can observe loopback health'
  exit 2
}
[[ $(cut -d'|' -f1 /home/choiceoh/glm53-logs/fleet/holder) == "$FLEET_SESSION" ]] || exit 2
[[ -z $(git status --porcelain --untracked-files=normal) ]] || exit 2
export STARTUP_CACHE_EVIDENCE=${STARTUP_CACHE_EVIDENCE:-/home/choiceoh/glm53-logs/rank-pipeline-$(date +%Y%m%d-%H%M%S)}
[[ ! -e $STARTUP_CACHE_EVIDENCE/source-commit.txt ]] || { echo 'fresh evidence directory required'; exit 2; }
mkdir -p "$STARTUP_CACHE_EVIDENCE"
git fetch origin main
git merge-base --is-ancestor origin/main HEAD || { echo 'current main required before service changes'; exit 2; }
python3 "${FLEET_RUNNER_REPO:-$REPO}/bench/fleet_entry.py" idle "$STARTUP_CACHE_EVIDENCE/before-metrics.txt"
IMAGE=sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211
docker image inspect "$IMAGE" >/dev/null
# The supervisor restores approved production or hands it to an acknowledged successor.
sampler_pid=
cleanup() {
  rc=$?
  trap - EXIT
  docker stop -t 2 "rankpipe-$FLEET_SESSION" >/dev/null 2>&1 || true
  if [[ -n $sampler_pid ]]; then kill "$sampler_pid" 2>/dev/null || true; wait "$sampler_pid" 2>/dev/null || true; fi
  echo "$rc" > "$STARTUP_CACHE_EVIDENCE/runner-exit"
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
# Release the previous serving model before the isolated, bounded copy correctness gate.
if docker inspect glm53 >/dev/null 2>&1; then docker stop -t 30 glm53 >/dev/null; fi
pids=()
for node in 1 3 4; do
  ssh -n -o BatchMode=yes "choiceoh@10.10.10.$node" 'if docker inspect glm53-worker >/dev/null 2>&1; then docker stop -t 30 glm53-worker; else docker info >/dev/null; fi' > /dev/null &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done
# CPU and CUDA checks use the exact immutable serving image, with no network.
args=(run --rm --name "rankpipe-$FLEET_SESSION" --network=none --memory=4g --cpus=2
      -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1
      --mount "type=bind,src=$REPO,dst=/repo,readonly"
      --mount "type=bind,src=$STARTUP_CACHE_EVIDENCE,dst=/evidence"
      --workdir /repo --entrypoint python3)
timeout 120 docker "${args[@]}" --runtime=runc -e CUDA_VISIBLE_DEVICES= "$IMAGE" \
  /repo/bench/rank_pipeline_cpu_gate.py > "$STARTUP_CACHE_EVIDENCE/cpu-image.log" 2>&1
timeout 300 docker "${args[@]}" --gpus device=0 "$IMAGE" \
  /repo/probes/glm53_rank_pipeline_check.py > "$STARTUP_CACHE_EVIDENCE/gpu-exact.log" 2>&1
bash launchers/deploy-overlays.sh glm53 > "$STARTUP_CACHE_EVIDENCE/deploy.log" 2>&1
python3 bench/startup_deploy_receipts.py "$STARTUP_CACHE_EVIDENCE/deployed-sources.json"
# Loopback serving prevents external traffic from entering the timed brackets.
export STARTUP_CACHE_PREFIX=RANKPIPE STARTUP_CACHE_MODE=rank-pipeline
export GLM53_API_HOST=127.0.0.1 GLM53_API_PORT=8000 HEAD=127.0.0.1 GLM53_BASE=http://127.0.0.1:8000
export ONEPASS_REQUIRE_EXCLUSIVE=1
python3 bench/startup_host_memory.py "$STARTUP_CACHE_EVIDENCE" > "$STARTUP_CACHE_EVIDENCE/memwatch.out" 2>&1 &
sampler_pid=$!
bash bench/startup_cache_boots.sh
