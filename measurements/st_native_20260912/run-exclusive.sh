#!/usr/bin/env bash
# Run on srv2. A bounded exclusive candidate run; restore the pinned serving
# release on success, failure, or interruption. Never modify its env file.
set -euo pipefail
ROOT=/home/choiceoh/st-native-f4d7-20260912
CANDIDATE=/home/choiceoh/st-releases/perf-f4d7-20260912a
BASELINE=/home/choiceoh/st-releases/5734b29fde84
cd "$ROOT"
node_sh() {
  local ip=$1; shift
  if [ "$ip" = 10.10.10.2 ]; then bash -c "$*"; else ssh -o BatchMode=yes -o ConnectTimeout=10 "$ip" "$@"; fi
}
collect() {
  local phase=$1 r=0 ip
  mkdir -p "$ROOT/$phase"
  for ip in 10.10.10.2 10.10.10.1 10.10.10.3 10.10.10.4; do
    node_sh "$ip" 'docker logs st-glm53' > "$ROOT/$phase/rank$r.log" 2>&1 || true
    node_sh "$ip" 'docker inspect st-glm53' > "$ROOT/$phase/container-rank$r.json" 2>&1 || true
    node_sh "$ip" "cat /home/choiceoh/glm53-logs/st-native-f4d7-dumps/memory-rank$r.json" > "$ROOT/$phase/memory-rank$r.json" 2>/dev/null || true
    r=$((r+1))
  done
}
wait_door() {
  local port=$1 i
  for i in $(seq 1 120); do
    if curl -fsS --max-time 4 "http://127.0.0.1:$port/v1/models" > /dev/null; then return 0; fi
    [ "$(docker inspect --format '{{.State.Running}}' st-glm53 2>/dev/null)" = true ] || return 1
    sleep 10
  done
  return 1
}
changed=0
restore() {
  local rc=$?
  trap - EXIT INT TERM
  set +e
  if [ "$changed" = 1 ]; then
    collect candidate
    bash "$CANDIDATE/launchers/start-st-glm53.sh" stop > restore.log 2>&1
    unset STK_execution STK_moe_static ST_TIER_DIR ST_DUMP_DIR
    set -a
    source /home/choiceoh/.config/st-glm53.env
    set +a
    for ip in 10.10.10.2 10.10.10.1 10.10.10.3 10.10.10.4; do
      node_sh "$ip" "PYTHONPATH=/home/choiceoh/st-perf-f4d7 python3 /home/choiceoh/st-perf-f4d7/engine_reclaim.py" >> restore.log 2>&1
    done
    bash "$BASELINE/launchers/start-st-glm53.sh" >> restore.log 2>&1
    wait_door 8000 >> restore.log 2>&1
    healthy=$?
    [ "$healthy" = 0 ] || rc=1
    curl -fsS --max-time 5 http://127.0.0.1:8000/ > restored-status.json
  fi
  systemctl --user start st-glm53.service
  systemctl --user is-active st-glm53.service > restored-service.txt
  printf '%s\n' "$rc" > exclusive-exit-code.txt
  exit "$rc"
}
systemctl --user is-active --quiet st-glm53.service
docker inspect st-glm53 > baseline-container.json
trap restore EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
systemctl --user stop st-glm53.service
python3 - <<'PY'
import json,time,urllib.request
for _ in range(60):
    with urllib.request.urlopen('http://127.0.0.1:8000/',timeout=5) as r: state=json.load(r)
    if not any(state.get(k) for k in ('running','waiting','queued','parking','resuming')): break
    time.sleep(1)
else: raise SystemExit('production has active requests; exclusive run not started')
print('Production drained; candidate run begins',flush=True)
PY
changed=1
bash "$BASELINE/launchers/start-st-glm53.sh" stop
export ST_ENGINE_DIR="$CANDIDATE" ST_IMAGE=st-engine:perf-f4d7-a ST_PRODUCTION=0 PORT=8001
export CKPT="$BASELINE/st-glm53-meta"
export STK_execution=parity STK_moe_static=t,r,sf6,q0
export ST_TIER_DIR=/home/choiceoh/glm53-logs/st-native-f4d7-tier
export ST_DUMP_DIR=/home/choiceoh/glm53-logs/st-native-f4d7-dumps
bash "$CANDIDATE/launchers/start-st-glm53.sh"
wait_door 8001
curl -fsS http://127.0.0.1:8001/metrics > metrics-before.txt
set +e
GLM53_API_PORT=8001 BENCH_MODEL=glm-5.3-flash SPEC_K=5 timeout --signal=TERM --kill-after=15 1200 \
  python3 -u run_onepass_st.py --name ST-NATIVE-F4D7-20260912A \
  --ctx 2000,32000,128000 --max-tokens 400 --num-spec 5 --seed 7 \
  --require-exclusive --out "$ROOT/result.jsonl"
rc=$?
set -e
curl -fsS http://127.0.0.1:8001/metrics > metrics-after.txt || true
printf '%s\n' "$rc" > onepass-exit-code.txt
exit "$rc"
