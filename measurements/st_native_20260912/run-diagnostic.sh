#!/usr/bin/env bash
# Run on srv2. A bounded exclusive candidate run; restore the pinned serving
# release on success, failure, or interruption. Never modify its env file.
set -euo pipefail
ROOT=/home/choiceoh/st-native-diag-f4d7-20260912
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
    node_sh "$ip" "cat /home/choiceoh/glm53-logs/st-native-diag-dumps/memory-rank$r.json" > "$ROOT/$phase/memory-rank$r.json" 2>/dev/null || true
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
    node_sh 10.10.10.4 "PYTHONPATH=/home/choiceoh/st-perf-f4d7 python3 /home/choiceoh/st-perf-f4d7/st_restore_reclaim.py" > restore-reclaim.log 2>&1 &
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
owner="$(whoami)@$(hostname) st-glm53 prefill-diag pid=$$"
(set -C; printf '%s\n' "$owner" > /home/choiceoh/st-fleet.lock)
pids=()
r=0
for ip in 10.10.10.2 10.10.10.1 10.10.10.3 10.10.10.4; do
  node_sh "$ip" "timeout 1800 docker run --name st-glm53 --gpus all --network host --ipc host --device /dev/infiniband --ulimit memlock=-1 -v /home/choiceoh/st-perf-f4d7:/repo -v $BASELINE/st-glm53-meta:/meta:ro -v /home/choiceoh/models:/home/choiceoh/models:ro -v /home/choiceoh/glm53-cache:/cache -v /home/choiceoh/glm53-logs:/home/choiceoh/glm53-logs -e PYTHONPATH=/repo -e MAX_JOBS=2 -e ST_ONESHOT_BUILD_ROOT=/cache/st-oneshot -e ST_DENSE_BUILD_ROOT=/cache/st-dense -e ST_MLA_BUILD_ROOT=/cache/mla -e FLASHINFER_WORKSPACE_BASE=/cache -e TRITON_CACHE_DIR=/cache/triton -e TILELANG_CACHE_DIR=/cache/tilelang -e DG_JIT_CACHE_DIR=/cache/deep_gemm -e RANK=$r -e WORLD_SIZE=4 -e MASTER_PORT=29678 -w /repo --entrypoint python3 st-engine:perf-f4d7-a -u /repo/engine_native_prefill_check.py --ranks /home/choiceoh/models/st-glm53-9391-up-gate-full --ckpt-meta /meta --drafter-dir /home/choiceoh/models/GLM-5.3-Flash-DFlash2 --prompt /repo/prompt32k.txt --requests /repo/diagnostic-requests.json --hold-dir /home/choiceoh/glm53-logs/native-control-v5" > "$ROOT/diag-rank$r.log" 2>&1 &
  pids+=($!)
  r=$((r+1))
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
exit "$failed"
