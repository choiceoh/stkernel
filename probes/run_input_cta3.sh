#!/usr/bin/env bash
# Fleet maintenance lane; recovery is owned by the central idle controller.
set -euo pipefail
cd /home/choiceoh/stkernel-input-cta3-20260908
export REPO=$PWD
out=${INPUT_CTA3_OUT:-/home/choiceoh/glm53-logs/INPUTCTA30908}
session=${FLEET_SESSION:?}
IFS='|' read -r held _pid _host _start _est _note kind < /home/choiceoh/glm53-logs/fleet/holder
[[ $held == "$session" && $kind == boot ]] || exit 2
[[ -z $(git status --porcelain) ]] || exit 2
git fetch origin
python3 bench/fleet_source.py require-base origin/main || { echo 'ABORT: candidate needs current main'; exit 2; }
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
cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  docker stop -t 2 "inputnext-$session" >/dev/null 2>&1 || true
  echo "$rc" > "$out/runner.exit"
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
[[ $(docker inspect glm53 --format '{{.Image}}') == sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211 ]] || exit 2
docker stop -t 30 glm53 >/dev/null
pids=()
for node in 1 3 4; do
  ssh -o BatchMode=yes "choiceoh@10.10.10.$node" docker stop -t 30 glm53-worker >/dev/null &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done
python3 probes/run_input_cta_guarded.py --three-slice --maintenance --out "$out"
