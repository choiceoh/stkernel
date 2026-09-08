#!/usr/bin/env bash
# Channel-resolved quality follow-up; canonical onepass and judge remain active.
set -euo pipefail
# REPO selects the committed candidate checkout supplied to fleet preflight.
cd "${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
export REPO=$PWD
CANONICAL=/home/choiceoh/stkernel
export FLEET=$CANONICAL/bench/fleet.sh LEVER=$REPO/probes/input_reuse_channel_lever.sh
export IMAGE=sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211
export INPUT_REUSE_SERVING_OUT=${INPUT_REUSE_SERVING_OUT:-/home/choiceoh/glm53-logs/INPUTCHANNELS0907}
out=$INPUT_REUSE_SERVING_OUT
session=${FLEET_SESSION:?}
IFS='|' read -r held _pid _host _start _est _note kind < /home/choiceoh/glm53-logs/fleet/holder
[[ $held == "$session" && $kind == boot ]] || exit 2
[[ -z $(git status --porcelain) ]] || exit 2
git fetch origin
python3 bench/fleet_source.py require-base origin/main || { echo 'ABORT before stopping service: candidate needs current main'; exit 2; }
[[ ! -s $out/source.commit ]] || { echo 'ABORT: fresh evidence required'; exit 2; }
mkdir -p "$out/build"
git rev-parse HEAD > "$out/source.commit"
python3 "${FLEET_RUNNER_REPO:-$REPO}/bench/fleet_entry.py" idle "$out/before-metrics.txt"

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  docker stop -t 2 "inputserve-$session" >/dev/null 2>&1 || true
  echo "$rc" > "$out/runner.exit"
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
cp /home/choiceoh/glm53-logs/glm53.log "$out/before-head.log" 2>/dev/null || true
if [[ -n ${INPUT_REUSE_GPU_EVIDENCE:-} ]]; then
  python3 probes/reuse_input_gpu_evidence.py "$INPUT_REUSE_GPU_EVIDENCE" "$out"
else
if docker inspect glm53 >/dev/null 2>&1; then docker stop -t 30 glm53 >/dev/null; fi
pids=()
for node in 1 3 4; do
  ssh -o BatchMode=yes "choiceoh@10.10.10.$node" 'if docker inspect glm53-worker >/dev/null 2>&1; then docker stop -t 30 glm53-worker; else docker info >/dev/null; fi' >/dev/null &
  pids+=($!)
done
stop_failed=0
for pid in "${pids[@]}"; do wait "$pid" || stop_failed=1; done
[[ $stop_failed == 0 ]] || exit 1
available_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
(( available_kib >= 16 * 1024 * 1024 )) || { echo 'ABORT: less than 16 GiB available'; exit 1; }
args=(run --rm --name "inputserve-$session" --gpus device=0 --network=none
      --cpuset-cpus=14-17 --memory=10g --shm-size=1g
      --mount "type=bind,src=$REPO,dst=/repo,readonly"
      --mount "type=bind,src=$out,dst=/evidence"
      --mount "type=bind,src=$out/build,dst=/build"
      --mount 'type=bind,src=/usr/local/cuda/compute-sanitizer,dst=/san,readonly'
      --workdir /repo)
timeout 360 docker "${args[@]}" --entrypoint python3 "$IMAGE" \
  /repo/probes/gemm_input_serving_gate.py > "$out/production-gate.log" 2>&1
for tool in racecheck memcheck; do
  timeout 240 docker "${args[@]}" --entrypoint /san/compute-sanitizer "$IMAGE" \
    --tool "$tool" --target-processes application-only --error-exitcode 77 \
    --kernel-name kns=mk_input_pack_kernel --kernel-name kns=mk_gemm_input_kernel \
    python3 /repo/probes/gemm_input_serving_gate.py --check-only \
    --out "/evidence/$tool.json" > "$out/$tool.log" 2>&1
done
fi
bash launchers/deploy-overlays.sh glm53 > "$out/deploy.log" 2>&1
export GLM53_API_HOST=127.0.0.1 GLM53_API_PORT=18000 HEAD=127.0.0.1
export PREFILL_WARMUP=0 QUALITY_CTX=2000,32000,128000 MAX_JOBS=2
export ONEPASS_FIXED_DECODE_TOKENS=2048 ONEPASS_FIXED_DECODE_REPS=5 ONEPASS_REQUIRE_EXCLUSIVE=1
export ONEPASS_JSONL=$out/records.raw.jsonl ONEPASS_VERDICTS=$out/verdicts.jsonl
bash bench/chain.sh \
  'IRCHANB1=' \
  'IRCHANA1=VLLM_GLM53_MK_INPUT_REUSE=1' \
  'IRCHANB2='
