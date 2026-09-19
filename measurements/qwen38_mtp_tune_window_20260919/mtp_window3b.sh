#!/usr/bin/env bash
# The MTP campaign's third window, second half (2026-09-19). The first training (91 steps from the checkpoint's head,
# peak 5e-5) took the held-out windows from 2.608 to 3.093 tokens a step and was still rising as its rate reached zero;
# the operator asked for more ("좀 더 학습해도 좋을것 같은데"). mtp_window3.sh releases after its three arms, so
# takeover.sh started this script and killed that one (-9: no EXIT trap, the lease kept) during its last arm's
# requests. This script waits for them, trains again from the first run's head (two more epochs, a fresh schedule),
# exports, serves that head once -- then ALWAYS stops and releases.
set -uo pipefail
OWNER="session/q38mtp-tune-0919"
TREE="$HOME/st-worktrees/q38mtp-win"
LOCK=/home/choiceoh/glm53-logs/st-fleet.lock
OUT=/home/choiceoh/glm53-logs/q38mtp-window-0919
WORK=/home/choiceoh/q38mtp-train-0919
TUNED2=/home/choiceoh/models/st-qwen38-mtp-tuned2
CKPT=/home/choiceoh/models/qwen38-flash-next-nvfp4
RANKS=/home/choiceoh/models/st-qwen38-tep4
DUMPS=/home/choiceoh/glm53-logs/st-qwen38-dumps
IMAGE=st-engine:qwen38
PENDING_ARM=${PENDING_ARM:-tuned-cut}
STEPS2=${STEPS2:-182}
EVERY2=${EVERY2:-26}
LR2=${LR2:-3e-5}
NODES=(10.10.10.2 10.10.10.1 10.10.10.3 10.10.10.4)
URL=http://10.10.10.2:8000
T_DOWN=$(date -d "2026-09-19 15:33:21" +%s)
cd "$TREE"
lease() { python3 engine/base/fleet_lease.py "$@" --path "$LOCK"; }
SELF_IPS=" $(hostname -I 2>/dev/null) "
node_sh() { local ip=$1; shift; if [[ "$SELF_IPS" == *" $ip "* ]]; then bash -c "$*"; else ssh -o BatchMode=yes -o ConnectTimeout=5 "choiceoh@$ip" "$@"; fi; }
stamp() { echo "[$(date +%H:%M:%S)] $*"; }
released=0
BEAT=""
stop_fleet() { ST_LEASE_OWNER="$OWNER" bash launchers/start-st-qwen38.sh stop 2>&1 | tail -1; }
stop_tune() { local ip; for ip in "${NODES[@]}"; do node_sh "$ip" "docker rm -f q38tune >/dev/null 2>&1" || true; done; }
finish() {
  [ "$released" = 1 ] && return
  [ -n "$BEAT" ] && kill "$BEAT" 2>/dev/null
  stamp "stop + release"
  stop_tune
  stop_fleet
  lease release --owner "$OWNER" || true
  released=1
  stamp "lease: $(lease read)"
  stamp "production down since handover: $(( $(date +%s) - T_DOWN )) s so far"
}
trap finish EXIT
( while sleep 120; do lease renew --owner "$OWNER" >/dev/null 2>&1; done ) &
BEAT=$!
lease renew --owner "$OWNER" >/dev/null 2>&1 || true

wait_lane() {
  local i n
  for i in $(seq 1 300); do
    n=$(node_sh 10.10.10.4 "docker ps --format '{{.Names}}' | grep -c '^st-probe-' || true")
    [ "$n" = 0 ] && return 0
    [ $((i % 20)) = 1 ] && stamp "waiting for srv4's probe to finish ($n up)"
    sleep 6
  done
  stamp "srv4's probe did not finish in 30 min"; return 1
}
logs() {
  local r
  for r in 0 1 2 3; do node_sh "${NODES[$r]}" "docker logs st-qwen38 2>&1" > "$OUT/$1-rank$r.log" 2>/dev/null; done
  grep -E "ready in|lanes qualified|drafter:|STALL|Traceback|Error|illegal|mtp inputs" "$OUT/$1-rank0.log" | tail -10 | cut -c1-300
}
boot() {
  local label=$1; shift
  wait_lane || return 1
  lease renew --owner "$OWNER" >/dev/null 2>&1 || true
  stamp "== boot $label ($*)"
  env ST_LEASE_OWNER="$OWNER" ST_SPEC_K=3 "$@" bash launchers/start-st-qwen38.sh > "$OUT/$label-launch.log" 2>&1
  local rc=$?
  tail -2 "$OUT/$label-launch.log"
  [ $rc = 0 ] || { stamp "launcher rc=$rc"; return 1; }
}
evalrun() {
  local label=$1; shift
  if ! boot "$label" "$@"; then logs "$label"; stop_fleet; return 1; fi
  python3 "$OUT/mtp_requests.py" eval "$URL" "$label" "$OUT/prompts.jsonl" > "$OUT/$label-requests.jsonl"
  local rq=$?
  tail -1 "$OUT/$label-requests.jsonl" | cut -c1-300
  logs "$label"
  stop_fleet
  return $rq
}
tune() {
  docker run --rm --gpus all --ipc host --shm-size 16g --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "$TREE":/repo:ro -v "$CKPT":/fullckpt:ro -v "$WORK/base":/ckpt:ro -v "$RANKS":/ranks:ro -v "$DUMPS":/dumps:ro \
    -v "$WORK":/work -v "$WORK/cache":/cache \
    --entrypoint /bin/bash "$IMAGE" -lc "cd /repo && PYTHONPATH=/repo python3 -u $*"
}

stamp "== 3b: took the window over during $PENDING_ARM's requests (the operator asked for a second training)"
while pgrep -f "mtp_requests.py eval $URL $PENDING_ARM" >/dev/null; do sleep 5; done
tail -1 "$OUT/$PENDING_ARM-requests.jsonl" | cut -c1-300
logs "$PENDING_ARM"
stop_fleet

# -- the second training: from the first run's head, the same runs ----------------------------------------------------
trained2=0
if [ -f "$WORK/run/head.safetensors" ]; then
  rm -rf "$WORK/run1" && mv "$WORK/run" "$WORK/run1"
  stamp "== second training: $STEPS2 steps from the first run's head (step $(python3 -c "
import json, sys
from safetensors import safe_open
with safe_open('$WORK/run1/head.safetensors', 'np') as f: print(json.loads(f.metadata()['meta'])['step'])
" 2>/dev/null || echo '?')), peak $LR2, eval every $EVERY2"
  ok=1
  for r in 0 1 2 3; do
    ip=${NODES[$r]}
    node_sh "$ip" "rm -rf $WORK/run $WORK/init && mkdir -p $WORK/init" || ok=0
    if [[ "$SELF_IPS" == *" $ip "* ]]; then
      cp "$WORK/run1/head.safetensors" "$WORK/init/head.safetensors" && cp "$OUT/train_rank2.sh" "$WORK/train_rank2.sh" \
        && cp "$TREE/engine/profiles/qwen38/mtp_tune.py" /home/choiceoh/st-engine-qwen38/engine/profiles/qwen38/mtp_tune.py || ok=0
    else
      scp -q "$WORK/run1/head.safetensors" "choiceoh@$ip:$WORK/init/head.safetensors" \
        && scp -q "$OUT/train_rank2.sh" "choiceoh@$ip:$WORK/train_rank2.sh" \
        && scp -q "$TREE/engine/profiles/qwen38/mtp_tune.py" "choiceoh@$ip:/home/choiceoh/st-engine-qwen38/engine/profiles/qwen38/mtp_tune.py" || ok=0
    fi
    node_sh "$ip" "grep -c 'def learning_rate' /home/choiceoh/st-engine-qwen38/engine/profiles/qwen38/mtp_tune.py >/dev/null" || ok=0
  done
  if [ "$ok" = 1 ]; then
    for r in 3 2 1 0; do node_sh "${NODES[$r]}" "bash $WORK/train_rank2.sh $r $STEPS2 $EVERY2 $LR2" || stamp "rank $r did not start"; done
    deadline=$(( $(date +%s) + 40 * 60 ))
    while [ "$(date +%s)" -lt "$deadline" ]; do
      up=0
      for ip in "${NODES[@]}"; do
        [ "$(node_sh "$ip" "docker inspect -f '{{.State.Running}}' q38tune 2>/dev/null")" = true ] && up=$((up + 1))
      done
      [ "$up" = 0 ] && break
      lease renew --owner "$OWNER" >/dev/null 2>&1 || true
      sleep 20
    done
    for r in 0 1 2 3; do
      node_sh "${NODES[$r]}" "docker logs q38tune 2>&1" > "$OUT/tune2-train-rank$r.log" 2>/dev/null
      stamp "rank $r exit $(node_sh "${NODES[$r]}" "docker inspect -f '{{.State.ExitCode}}' q38tune 2>/dev/null")"
    done
    stop_tune
    grep -E '"event": "(eval|saved|end)"' "$OUT/tune2-train-rank0.log" | cut -c1-300
    if [ -f "$WORK/run/head.safetensors" ]; then
      stamp "== export (second head)"
      rm -rf "$WORK/tuned2"
      tune -m engine.profiles.qwen38.mtp_tune export --tuned /work/run/head.safetensors --ckpt /fullckpt --ckpt-meta /ranks \
           --out /work/tuned2 > "$OUT/tune2-export.log" 2>&1
      tail -1 "$OUT/tune2-export.log" | cut -c1-300
      spread=1
      for r in 0 1 2 3; do
        ip=${NODES[$r]}
        f="$WORK/tuned2/mtp-tuned-r${r}of4.safetensors"
        [ -f "$f" ] || { spread=0; break; }
        node_sh "$ip" "mkdir -p $TUNED2" || spread=0
        if [[ "$SELF_IPS" == *" $ip "* ]]; then cp "$f" "$TUNED2/"; else scp -q "$f" "choiceoh@$ip:$TUNED2/" || spread=0; fi
      done
      [ "$spread" = 1 ] && trained2=1
      stamp "spread to four nodes: $spread"
    else
      stamp "the second training did not beat the first head on the held-out windows: nothing new to serve"
    fi
  else
    stamp "the second training's files did not reach every node: skipped"
  fi
else
  stamp "no first head at $WORK/run/head.safetensors: the second training has nothing to start from"
fi

if [ "$trained2" = 1 ]; then
  evalrun tuned2-off ST_DRAFT_THRESHOLD=off ST_TAP_MTP_INPUTS=0 ST_MTP_TUNED="$TUNED2" || stamp "tuned2-off did not serve"
fi

finish
for i in $(seq 1 120); do
  if curl -fsS -m 3 "$URL/v1/models" 2>/dev/null | grep -q '"id"'; then stamp "production door answers"; break; fi
  sleep 5
done
stamp "lease after: $(lease read | cut -c1-160)"
stamp "production downtime: $(( $(date +%s) - T_DOWN )) s"
