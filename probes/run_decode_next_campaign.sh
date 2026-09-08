#!/usr/bin/env bash
# PR498: correctness gates, then exactly one all-three A and one default B.
set -euo pipefail
export REPO=$(cd "$(dirname "$0")/.." && pwd)
export IMAGE=sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211
cd "$REPO"
for control in SKIP_BOOT FLEET_REHEARSE DRY_RUN SKIP_PREFLIGHT; do
  [[ ${!control:-0} == 0 ]] || { echo "real boot and normal preflight required: $control"; exit 2; }
done
session=${FLEET_SESSION:?}
[[ ${FLEET_RESTORE_MANAGED:-0} == 1 ]] || { echo 'supervised boot hold required'; exit 2; }
IFS='|' read -r held _pid _host _start _est _note kind < /home/choiceoh/glm53-logs/fleet/holder
[[ $held == "$session" && $kind == boot && -z $(git status --porcelain) ]] || exit 2
git fetch origin main
python3 bench/fleet_source.py require-base origin/main
export DECODE_NEXT_OUT=${DECODE_NEXT_OUT:-/home/choiceoh/glm53-logs/DECODE-NEXT-$session}
[[ ! -e $DECODE_NEXT_OUT ]] || { echo 'fresh evidence required'; exit 2; }
mkdir -p "$DECODE_NEXT_OUT"
git rev-parse HEAD > "$DECODE_NEXT_OUT/source.commit"
python3 - <<'PY'
from pathlib import Path
expected={'VLLM_GLM53_AR_COMPACT_CTA':'0','VLLM_GLM53_AR_PROXY_INLINE':'0',
          'VLLM_GLM53_B12X_STATIC_V2':'t,r','VLLM_GLM53_AR_CONSUMER_PDL':'1',
          'VLLM_GLM53_MK_PDL':'1','SPEC_K':'5'}
values={}
for line in Path('profiles/glm53.env').read_text().splitlines():
    key,sep,value=line.partition('=')
    if sep:values[key]=value.split('#',1)[0].strip()
assert all(values.get(key)==value for key,value in expected.items()), 'frozen default profile changed'
PY
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
    if node!=2:cmd=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5','choiceoh@10.10.10.'+str(node),shlex.join(cmd)]
    subprocess.run(cmd,check=True,timeout=50)
with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(stop,[2,1,3,4]))
PY
}
touched=0
cleanup() {
  local rc=$?
  trap - EXIT
  # Stop this experiment's loopback serving; the central idle owner recovers.
  if [[ $touched == 1 && ${FLEET_RESTORE_MANAGED:-0} == 1 ]]; then
    stop_serving > "$DECODE_NEXT_OUT/stop-experiment.log" 2>&1 || rc=1
  fi
  printf '%s\n' "$rc" > "$DECODE_NEXT_OUT/campaign.exit"
  exit "$rc"
}
trap cleanup EXIT
idle_ready=0
for attempt in {1..60}; do
  if python3 "${FLEET_RUNNER_REPO:-$REPO}/bench/fleet_entry.py" idle \
    "$DECODE_NEXT_OUT/before-metrics.txt" > "$DECODE_NEXT_OUT/idle-$attempt.log" 2>&1; then
    idle_ready=1; break
  fi
  echo "Waiting for idle serving ($attempt/60): $(tail -1 "$DECODE_NEXT_OUT/idle-$attempt.log")"
  sleep 5
done
[[ $idle_ready == 1 ]] || { echo 'ABORT: serving did not become idle'; exit 2; }
touched=1
stop_serving > "$DECODE_NEXT_OUT/stop-before-probe.log" 2>&1
python3 probes/run_decode_transport_gpu.py --out "$DECODE_NEXT_OUT/transport-gpu" --stage probe
python3 probes/run_decode_sf6_gpu.py --out "$DECODE_NEXT_OUT/sf6-gpu"
python3 - "$DECODE_NEXT_OUT" <<'PY'
from pathlib import Path
import sys
sys.path.insert(0,'probes')
from run_decode_transport_gpu import verify_admission as transport
from run_decode_sf6_gpu import verify_admission as sf6
root=Path(sys.argv[1]); transport(root/'transport-gpu'); sf6(root/'sf6-gpu')
PY
bash launchers/deploy-overlays.sh glm53 > "$DECODE_NEXT_OUT/deploy.log" 2>&1
export FLEET=/home/choiceoh/stkernel/bench/fleet.sh LEVER=$REPO/probes/decode_next_lever.sh
export GLM53_API_HOST=127.0.0.1 GLM53_API_PORT=18000 HEAD=127.0.0.1
export PREFILL_WARMUP=0 QUALITY_CTX=2000,32000,128000 MAX_JOBS=2
export ONEPASS_FIXED_DECODE_TOKENS=2048 ONEPASS_FIXED_DECODE_REPS=3 ONEPASS_REQUIRE_EXCLUSIVE=1
export ONEPASS_COMBINE_MIN_CTX=32000
export FLEET_WORKLOAD='{"ctx":[2000,32000,128000],"seed":7,"max_tokens":400,"combine_min_ctx":32000,"fixed_decode_tokens":2048,"fixed_decode_reps":3,"require_exclusive":true}'
export FLEET_OBJECTIVE='{"metric":"decode_steps"}'
export ONEPASS_JSONL=$DECODE_NEXT_OUT/records.raw.jsonl ONEPASS_VERDICTS=$DECODE_NEXT_OUT/verdicts.jsonl
# An empty final arm is recognized by chain as the baseline: no third boot.
rc=0
bash bench/chain.sh "${session}A=VLLM_GLM53_AR_COMPACT_CTA=1 VLLM_GLM53_AR_PROXY_INLINE=1 VLLM_GLM53_B12X_STATIC_V2=t,r,sf6" \
  "${session}B=" || rc=$?
python3 probes/analyze_decode_next_onepass.py "$DECODE_NEXT_OUT" \
  --candidate "${session}A" --baseline "${session}B" > "$DECODE_NEXT_OUT/comparison.json" || rc=1
cat "$DECODE_NEXT_OUT/comparison.json"
exit "$rc"
