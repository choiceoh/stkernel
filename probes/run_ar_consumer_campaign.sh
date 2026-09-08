#!/usr/bin/env bash
# One canonical onepass candidate, with a matched default baseline only if missing.
set -euo pipefail
export REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
gpu_evidence=
baseline_only=0
while (( $# )); do
  case $1 in
    --gpu-evidence)
      [[ $# -ge 2 && -n $2 && $2 != --* && -z $gpu_evidence ]] || { echo 'one GPU evidence directory required'; exit 2; }
      gpu_evidence=$2; shift 2;;
    --baseline-only)
      [[ $baseline_only == 0 ]] || { echo 'duplicate --baseline-only'; exit 2; }
      baseline_only=1; shift;;
    *) echo 'usage: run_ar_consumer_campaign.sh [--gpu-evidence DIR] [--baseline-only]'; exit 2;;
  esac
done
session=${FLEET_SESSION:?}
export LOGD=${LOGD:-/home/choiceoh/glm53-logs}
export FLEET_DIR=${FLEET_DIR:-$LOGD/fleet}
[[ ${FLEET_RESTORE_MANAGED:-0} == 1 ]] || { echo 'supervised boot hold required'; exit 2; }
IFS='|' read -r held _pid _host _start _est _note kind < "$FLEET_DIR/holder"
[[ $held == "$session" && $kind == boot && -z $(git status --porcelain) ]] || exit 2
git fetch origin main
git merge-base --is-ancestor origin/main HEAD || { echo 'ABORT: candidate needs current main'; exit 2; }
# The profile may already include this optimization. Measure its opposite as
# the candidate so the standard profile-default baseline remains meaningful.
profile_mode=$(sed -nE 's/^VLLM_GLM53_AR_CONSUMER_PDL=([01])$/\1/p' profiles/glm53.env | tail -1)
[[ $profile_mode == 0 || $profile_mode == 1 ]] || { echo 'explicit profile AR consumer mode required'; exit 2; }
candidate_mode=$((1 - profile_mode))
export AR_CONSUMER_OUT=${AR_CONSUMER_OUT:-$LOGD/ARCONSUMER-$session}
[[ ! -e $AR_CONSUMER_OUT ]] || { echo 'fresh evidence required'; exit 2; }
mkdir -p "$AR_CONSUMER_OUT"
git rev-parse HEAD > "$AR_CONSUMER_OUT/source.commit"
if [[ -n $gpu_evidence ]]; then
  echo '--gpu-evidence is obsolete: the canonical onepass is the only GPU workload'
fi
stop_serving() {
  python3 - <<'PY'
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os,shlex,subprocess
held=(Path(os.environ['FLEET_DIR'])/'holder').read_text().split('|')
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
  # the central idle controller retains responsibility for public recovery.
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
# The central idle controller owns recovery after this turn is released.
touched=1
stop_serving > "$AR_CONSUMER_OUT/stop-before-onepass.log" 2>&1
bash launchers/deploy-overlays.sh glm53 > "$AR_CONSUMER_OUT/deploy.log" 2>&1
export FLEET=$REPO/bench/fleet.sh LEVER=$REPO/bench/ab-lever.sh
export GLM53_API_HOST=127.0.0.1 GLM53_API_PORT=18000 HEAD=127.0.0.1
export PREFILL_WARMUP=0 ONEPASS_REQUIRE_EXCLUSIVE=1
# A fresh private ledger would force another baseline for every reservation.
# Keep using the shared standard ledger; onepass retains the session identity.
export ONEPASS_JSONL=${ONEPASS_JSONL:-$LOGD/bracket-onepass.jsonl}
export ONEPASS_VERDICTS=${ONEPASS_VERDICTS:-$AR_CONSUMER_OUT/verdicts.jsonl}
printf '%s\n' "$ONEPASS_JSONL" > "$AR_CONSUMER_OUT/onepass-ledger.path"
if [[ $baseline_only == 1 ]]; then
  # Compatibility for a pending request explicitly asking for one default arm.
  bash bench/ab-lever.sh "${session}BASE" ""
else
  echo "onepass candidate AR_CONSUMER_PDL=$candidate_mode; baseline is profile mode $profile_mode"
  bash bench/pair.sh "${session}AR${candidate_mode}" "VLLM_GLM53_AR_CONSUMER_PDL=$candidate_mode"
fi
