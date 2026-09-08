#!/usr/bin/env bash
# Supervised maintenance: correctness first, then one same-build B/A/B bracket.
set -euo pipefail
export REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
session=${FLEET_SESSION:?}
[[ ${FLEET_RESTORE_MANAGED:-0} == 1 ]] || { echo 'supervised boot hold required'; exit 2; }
IFS='|' read -r held _pid _host _start _est _note kind < /home/choiceoh/glm53-logs/fleet/holder
[[ $held == "$session" && $kind == boot && -z $(git status --porcelain) ]] || exit 2
git fetch origin main
git merge-base --is-ancestor origin/main HEAD || { echo 'ABORT: candidate needs current main'; exit 2; }
export AR_CONSUMER_OUT=${AR_CONSUMER_OUT:-/home/choiceoh/glm53-logs/ARCONSUMER-$session}
[[ ! -e $AR_CONSUMER_OUT ]] || { echo 'fresh evidence required'; exit 2; }
mkdir -p "$AR_CONSUMER_OUT"
git rev-parse HEAD > "$AR_CONSUMER_OUT/source.commit"
python3 "${FLEET_RUNNER_REPO:-$REPO}/bench/fleet_entry.py" idle "$AR_CONSUMER_OUT/before-metrics.txt"
# The fleet supervisor owns recovery, including early exits and queue handoff.
docker stop -t 30 glm53 >/dev/null
pids=()
for node in 1 3 4; do
  ssh -o BatchMode=yes "choiceoh@10.10.10.$node" docker stop -t 30 glm53-worker >/dev/null &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done
python3 probes/run_ar_consumer_gpu.py --out "$AR_CONSUMER_OUT/gpu"
bash launchers/deploy-overlays.sh glm53 > "$AR_CONSUMER_OUT/deploy.log" 2>&1
export FLEET=/home/choiceoh/stkernel/bench/fleet.sh LEVER=$REPO/probes/ar_consumer_lever.sh
export GLM53_API_HOST=127.0.0.1 GLM53_API_PORT=18000 HEAD=127.0.0.1
export PREFILL_WARMUP=0 QUALITY_CTX=2000,32000,128000 MAX_JOBS=2
export ONEPASS_FIXED_DECODE_TOKENS=2048 ONEPASS_FIXED_DECODE_REPS=3 ONEPASS_REQUIRE_EXCLUSIVE=1
export ONEPASS_JSONL=$AR_CONSUMER_OUT/records.raw.jsonl ONEPASS_VERDICTS=$AR_CONSUMER_OUT/verdicts.jsonl
bash bench/chain.sh "${session}B1=" \
  "${session}A1=VLLM_GLM53_AR_CONSUMER_PDL=1" "${session}B2="
