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
stop_serving() {
  python3 - <<'PY'
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os,shlex,subprocess
held=Path('/home/choiceoh/glm53-logs/fleet/holder').read_text().split('|')
assert held[0]==os.environ['FLEET_SESSION'] and held[-1].strip()=='boot'
code='import subprocess,sys; name=sys.argv[1]; names=subprocess.check_output(["docker","ps","--format","{{.Names}}"],text=True).splitlines(); subprocess.run(["docker","stop","-t","30",name],check=True,timeout=40) if name in names else None'
def stop(node):
    cmd=['python3','-c',code,'glm53' if node==2 else 'glm53-worker']
    if node!=2:cmd=['ssh','-o','BatchMode=yes','choiceoh@10.10.10.'+str(node),shlex.join(cmd)]
    subprocess.run(cmd,check=True,timeout=50)
with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(stop,[2,1,3,4]))
PY
}
touched=0
cleanup() {
  local rc=$?
  trap - EXIT
  # A live loopback-only server looks "booting" to public-port admission.
  # Release our serving processes before the supervisor transfers the hold;
  # its restore/handoff policy retains responsibility for the public service.
  if [[ $touched == 1 && ${FLEET_RESTORE_MANAGED:-0} == 1 ]]; then
    stop_serving > "$AR_CONSUMER_OUT/stop-experiment.log" 2>&1 || rc=1
  fi
  exit "$rc"
}
trap cleanup EXIT
# A donor's last request/metrics update can still be draining at handoff.
# Keep the same strict idle gate, but give it a bounded interval to pass;
# a missing or nonzero counter never authorizes stopping that serving.
idle_ready=0
for attempt in {1..60}; do
  if python3 "${FLEET_RUNNER_REPO:-$REPO}/bench/fleet_entry.py" idle \
       "$AR_CONSUMER_OUT/before-metrics.txt" \
       > "$AR_CONSUMER_OUT/idle-$attempt.log" 2>&1; then
    idle_ready=1
    break
  fi
  echo "Waiting for idle serving ($attempt/60): $(tail -1 "$AR_CONSUMER_OUT/idle-$attempt.log")"
  sleep 5
done
[[ $idle_ready == 1 ]] || { echo 'ABORT: serving did not become idle'; exit 2; }
# The fleet supervisor owns recovery, including early exits and queue handoff.
touched=1
stop_serving > "$AR_CONSUMER_OUT/stop-before-probe.log" 2>&1
python3 probes/run_ar_consumer_gpu.py --out "$AR_CONSUMER_OUT/gpu"
bash launchers/deploy-overlays.sh glm53 > "$AR_CONSUMER_OUT/deploy.log" 2>&1
export FLEET=/home/choiceoh/stkernel/bench/fleet.sh LEVER=$REPO/probes/ar_consumer_lever.sh
export GLM53_API_HOST=127.0.0.1 GLM53_API_PORT=18000 HEAD=127.0.0.1
export PREFILL_WARMUP=0 QUALITY_CTX=2000,32000,128000 MAX_JOBS=2
export ONEPASS_FIXED_DECODE_TOKENS=2048 ONEPASS_FIXED_DECODE_REPS=3 ONEPASS_REQUIRE_EXCLUSIVE=1
export ONEPASS_JSONL=$AR_CONSUMER_OUT/records.raw.jsonl ONEPASS_VERDICTS=$AR_CONSUMER_OUT/verdicts.jsonl
bash bench/chain.sh "${session}B1=" \
  "${session}A1=VLLM_GLM53_AR_CONSUMER_PDL=1" "${session}B2="
