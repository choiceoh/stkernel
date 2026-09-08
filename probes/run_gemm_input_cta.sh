#!/usr/bin/env bash
# Exact eight-slice CTA probe; same-build controls and supervised production handoff.
set -euo pipefail
cd "${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
export REPO=$PWD
IMAGE=sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211
out=${INPUT_CTA_OUT:-/home/choiceoh/glm53-logs/INPUTCTA0907}
session=${FLEET_SESSION:?}
IFS='|' read -r held _pid _host _start _est _note kind < /home/choiceoh/glm53-logs/fleet/holder
[[ $held == "$session" && $kind == boot ]] || exit 2
[[ -z $(git status --porcelain) ]] || exit 2
git fetch origin
python3 bench/fleet_source.py require-base origin/main || { echo 'ABORT before stopping service: candidate needs current main'; exit 2; }
[[ ! -e $out/source.commit ]] || { echo 'ABORT: fresh evidence required'; exit 2; }
mkdir -p "$out/build"
git rev-parse HEAD > "$out/source.commit"
python3 "${FLEET_RUNNER_REPO:-$REPO}/bench/fleet_entry.py" idle "$out/before-metrics.txt"

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  docker stop -t 2 "inputcta-$session" >/dev/null 2>&1 || true
  echo "$rc" > "$out/runner.exit"
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
docker image inspect "$IMAGE" >/dev/null
if docker inspect glm53 >/dev/null 2>&1; then
  [[ $(docker inspect glm53 --format '{{.Image}}') == "$IMAGE" ]] || exit 2
fi
if docker inspect glm53 >/dev/null 2>&1; then docker stop -t 30 glm53 >/dev/null; fi
pids=()
for node in 1 3 4; do
  ssh -o BatchMode=yes "choiceoh@10.10.10.$node" 'if docker inspect glm53-worker >/dev/null 2>&1; then docker stop -t 30 glm53-worker; else docker info >/dev/null; fi' >/dev/null &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done
available_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
(( available_kib >= 16 * 1024 * 1024 )) || { echo 'ABORT: less than 16 GiB available'; exit 1; }
args=(run --rm --name "inputcta-$session" --gpus device=0 --network=none
      --cpuset-cpus=14-17 --memory=10g --shm-size=1g
      --mount "type=bind,src=$REPO,dst=/repo,readonly"
      --mount "type=bind,src=$out,dst=/evidence"
      --mount "type=bind,src=$out/build,dst=/build"
      --mount 'type=bind,src=/usr/local/cuda/compute-sanitizer,dst=/san,readonly'
      --workdir /repo)
timeout 420 docker "${args[@]}" --entrypoint python3 "$IMAGE" \
  /repo/probes/gemm_input_cta.py > "$out/probe.log" 2>&1
for tool in racecheck memcheck; do
  timeout 240 docker "${args[@]}" --entrypoint /san/compute-sanitizer "$IMAGE" \
    --tool "$tool" --target-processes application-only --error-exitcode 77 \
    --kernel-name kns=mk_gemm_input_cta_kernel \
    python3 /repo/probes/gemm_input_cta.py --check-only \
    --out "/evidence/$tool.json" > "$out/$tool.log" 2>&1
done
