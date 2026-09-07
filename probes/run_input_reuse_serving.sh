#!/usr/bin/env bash
# Actual-source GPU gates, then matched B/A/A/B with a guaranteed public restore.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export REPO=$PWD
CANONICAL=/home/choiceoh/stkernel
RESTORE_REPO=/home/choiceoh/stkernel-input-reuse-restore-20260907
export FLEET=$CANONICAL/bench/fleet.sh LEVER=$REPO/probes/input_reuse_lever.sh
export IMAGE=sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211
export INPUT_REUSE_SERVING_OUT=${INPUT_REUSE_SERVING_OUT:-/home/choiceoh/glm53-logs/INPUTSERVE0907}
out=$INPUT_REUSE_SERVING_OUT
session=${FLEET_SESSION:?}
IFS='|' read -r held _pid _host _start _est _note kind < /home/choiceoh/glm53-logs/fleet/holder
[[ $held == "$session" && $kind == boot ]] || exit 2
[[ -z $(git status --porcelain) && -z $(git -C "$RESTORE_REPO" status --porcelain) ]] || exit 2
[[ ! -s $out/source.commit ]] || { echo 'ABORT: fresh evidence required'; exit 2; }
mkdir -p "$out/build"
git rev-parse HEAD > "$out/source.commit"
requests=$(curl -fsS --max-time 5 http://10.10.10.2:8000/metrics | awk '/^vllm:num_requests_(running|waiting)/ {s+=$2} END {print s+0}')
[[ $requests == 0 ]] || { echo 'ABORT: active requests'; exit 2; }
touched=0
cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  docker stop -t 2 "inputserve-$session" >/dev/null 2>&1 || true
  if [[ $touched == 1 ]]; then
    if (
      cd "$RESTORE_REPO"
      bash launchers/deploy-overlays.sh glm53 || exit 1
      env -u ONEPASS_JSONL -u ONEPASS_VERDICTS REPO="$RESTORE_REPO" \
        GLM53_API_HOST=0.0.0.0 GLM53_API_PORT=8000 HEAD=10.10.10.2 \
        PREFILL_WARMUP=1 LEGS=none bash bench/ab-lever.sh "${session}RESTORE" ''
    ) > "$out/restore.log" 2>&1; then
      echo restored > "$out/restore.status"
    else
      echo FAILED > "$out/restore.status"; rc=1
    fi
  fi
  echo "$rc" > "$out/runner.exit"
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
touched=1
cp /home/choiceoh/glm53-logs/glm53.log "$out/before-head.log"
docker stop -t 30 glm53 >/dev/null
pids=()
for node in 1 3 4; do
  ssh -o BatchMode=yes "choiceoh@10.10.10.$node" docker stop -t 30 glm53-worker >/dev/null &
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
bash launchers/deploy-overlays.sh glm53 > "$out/deploy.log" 2>&1
export GLM53_API_HOST=127.0.0.1 GLM53_API_PORT=18000 HEAD=127.0.0.1
export PREFILL_WARMUP=0 QUALITY_CTX=2000,32000,128000 MAX_JOBS=2
export ONEPASS_FIXED_DECODE_TOKENS=2048 ONEPASS_FIXED_DECODE_REPS=5 ONEPASS_REQUIRE_EXCLUSIVE=1
export ONEPASS_JSONL=$out/records.raw.jsonl ONEPASS_VERDICTS=$out/verdicts.jsonl
bash bench/chain.sh \
  'IREUSEB1=' \
  'IREUSEA1=VLLM_GLM53_MK_INPUT_REUSE=1' \
  'IREUSEA2=VLLM_GLM53_MK_INPUT_REUSE=1' \
  'IREUSEB2='
