#!/usr/bin/env bash
# Fleet maintenance lane; restore latest approved main even if a probe fails.
set -euo pipefail
cd "$(dirname "$0")/.."
export REPO=$PWD
RESTORE_REPO=/home/choiceoh/stkernel-moe-reform-restore-20260908
out=${MOE_ONEPASS_OUT:-/home/choiceoh/glm53-logs/MOEREFORM0908}
session=${FLEET_SESSION:?}
IFS='|' read -r held _pid _host _start _est _note kind < /home/choiceoh/glm53-logs/fleet/holder
[[ $held == "$session" && $kind == boot ]] || exit 2
[[ -z $(git status --porcelain) && -z $(git -C "$RESTORE_REPO" status --porcelain) ]] || exit 2
git fetch origin
git merge-base --is-ancestor origin/main HEAD || { echo 'ABORT: candidate needs current main'; exit 2; }
[[ ! -e $out ]] || { echo 'ABORT: fresh evidence required'; exit 2; }
mkdir -p "$out"
curl -fsS --max-time 5 http://127.0.0.1:8000/metrics > "$out/before-metrics.txt"
python3 - "$out/before-metrics.txt" <<'PY'
from pathlib import Path
import sys
lines=Path(sys.argv[1]).read_text().splitlines()
for key in ('num_requests_running','num_requests_waiting'):
    values=[float(line.rsplit(' ',1)[1]) for line in lines if line.startswith('vllm:'+key+'{')]
    assert values and sum(values)==0,(key,values)
PY
touched=0
cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  docker stop -t 2 "moereform-$session" >/dev/null 2>&1 || true
  if [[ $touched == 1 ]]; then
    if (
      cd "$RESTORE_REPO"
      git fetch origin || exit 1
      git merge --ff-only origin/main || exit 1
      bash launchers/deploy-overlays.sh glm53 || exit 1
      env -u ONEPASS_JSONL -u ONEPASS_VERDICTS REPO="$RESTORE_REPO" \
        GLM53_API_HOST=0.0.0.0 GLM53_API_PORT=8000 HEAD=10.10.10.2 \
        PREFILL_WARMUP=1 LEGS=none bash bench/ab-lever.sh "${session}RESTORE" ''
    ) > "$out/restore.log" 2>&1; then
      curl -fsS --max-time 5 http://127.0.0.1:8000/health > "$out/restore.health"
      docker inspect glm53 --format '{{.Id}} {{.Image}} {{.State.StartedAt}}' > "$out/restore.boot"
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
[[ $(docker inspect glm53 --format '{{.Image}}') == sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211 ]] || exit 2
touched=1
docker stop -t 30 glm53 >/dev/null
pids=()
for node in 1 3 4; do
  ssh -o BatchMode=yes "choiceoh@10.10.10.$node" docker stop -t 30 glm53-worker >/dev/null &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done
python3 probes/run_moe_reform_guarded.py --maintenance --out "$out/numerics"
bash launchers/deploy-overlays.sh glm53 > "$out/deploy.log" 2>&1
export MOE_ONEPASS_OUT=$out
export FLEET=/home/choiceoh/stkernel/bench/fleet.sh LEVER=$REPO/probes/moe_reform_lever.sh
export GLM53_API_HOST=127.0.0.1 GLM53_API_PORT=18000 HEAD=127.0.0.1
export PREFILL_WARMUP=0 QUALITY_CTX=2000,32000,128000 MAX_JOBS=2
export ONEPASS_FIXED_DECODE_TOKENS=2048 ONEPASS_FIXED_DECODE_REPS=3 ONEPASS_REQUIRE_EXCLUSIVE=1
export ONEPASS_JSONL=$out/records.raw.jsonl ONEPASS_VERDICTS=$out/verdicts.jsonl
bash bench/chain.sh 'MOERFA1=VLLM_GLM53_B12X_STATIC_V2=t,r' 'MOERFB1='

