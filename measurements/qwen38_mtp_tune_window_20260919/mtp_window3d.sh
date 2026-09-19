#!/usr/bin/env bash
# The MTP campaign's third window, last part (2026-09-19): the door held idle for a peer session's C=1 decode-step
# profile (its operator instruction: GLM's 09-14 tool, POST /v1/engine/profile + /metrics) so production does not go
# down again for it, then stop and release. takeover3.sh started this and killed mtp_window3c.sh (-9, the lease kept)
# during tuned3-off's requests -- or, when the third training served nothing, right after 3c said so (then this boots
# the second head for the hold).
set -uo pipefail
OWNER="session/q38mtp-tune-0919"
TREE="$HOME/st-worktrees/q38mtp-win"
LOCK=/home/choiceoh/glm53-logs/st-fleet.lock
OUT=/home/choiceoh/glm53-logs/q38mtp-window-0919
WORK=/home/choiceoh/q38mtp-train-0919
TUNED2=/home/choiceoh/models/st-qwen38-mtp-tuned2
TUNED3=/home/choiceoh/models/st-qwen38-mtp-tuned3
CKPT=/home/choiceoh/models/qwen38-flash-next-nvfp4
RANKS=/home/choiceoh/models/st-qwen38-tep4
DUMPS=/home/choiceoh/glm53-logs/st-qwen38-dumps
IMAGE=st-engine:qwen38
PENDING_ARM=${PENDING_ARM-tuned3-off}
HOLD_MIN=${HOLD_MIN:-10}
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

stamp "== 3d: took the window over (${PENDING_ARM:-no arm pending}): the door held for a peer's C=1 profile, then the release"
if [ -n "$PENDING_ARM" ]; then
  while pgrep -f "mtp_requests.py eval $URL $PENDING_ARM" >/dev/null; do sleep 5; done
  tail -1 "$OUT/$PENDING_ARM-requests.jsonl" | cut -c1-300
  logs "$PENDING_ARM"
  held="$PENDING_ARM"
else
  stop_tune
  held=profile
  boot profile ST_DRAFT_THRESHOLD=off ST_TAP_MTP_INPUTS=0 ST_MTP_TUNED="$TUNED2" || stamp "the profile boot failed"
  for i in $(seq 1 60); do curl -fsS -m 3 "$URL/v1/models" 2>/dev/null | grep -q '"id"' && break; sleep 5; done
fi
rm -f "$OUT/profile.done"
echo "$held" > "$OUT/profile.open"
stamp "profile window open on the $held boot: the door is idle for up to $HOLD_MIN min -- touch $OUT/profile.done to end it"
for i in $(seq 1 $((HOLD_MIN * 12))); do
  [ -f "$OUT/profile.done" ] && break
  [ $((i % 24)) = 0 ] && lease renew --owner "$OWNER" >/dev/null 2>&1
  sleep 5
done
stamp "profile window closed ($( [ -f "$OUT/profile.done" ] && echo "done" || echo "timed out"))"
rm -f "$OUT/profile.open"
logs "$held-after-profile"
stop_fleet

# -- every head's held-out rates at the served sampler (mtp_tune eval --sampler 1.0,20,0.95), one head a node ---------
estimate() {
  local r ip h names=(orig run1 run2 run) pids=()
  mkdir -p "$WORK/heads"
  for h in run1 run2 run; do [ -f "$WORK/$h/head.safetensors" ] && cp "$WORK/$h/head.safetensors" "$WORK/heads/$h.safetensors"; done
  cp "$TREE/engine/profiles/qwen38/mtp_tune.py" /home/choiceoh/st-engine-qwen38/engine/profiles/qwen38/mtp_tune.py
  for r in 1 2 3; do
    ip=${NODES[$r]}
    node_sh "$ip" "mkdir -p $WORK/heads"
    scp -q "$TREE/engine/profiles/qwen38/mtp_tune.py" "choiceoh@$ip:/home/choiceoh/st-engine-qwen38/engine/profiles/qwen38/mtp_tune.py"
    h=${names[$r]}
    [ -f "$WORK/heads/$h.safetensors" ] && scp -q "$WORK/heads/$h.safetensors" "choiceoh@$ip:$WORK/heads/$h.safetensors"
  done
  for r in 0 1 2 3; do
    ip=${NODES[$r]}; h=${names[$r]}
    tuned=""; [ "$h" != orig ] && tuned="--tuned /heads/$h.safetensors"
    [ "$h" != orig ] && [ ! -f "$WORK/heads/$h.safetensors" ] && continue
    ( node_sh "$ip" "docker run --rm --gpus all --ipc host --shm-size 16g --user \$(id -u):\$(id -g) -e HOME=/tmp -v /home/choiceoh/st-engine-qwen38:/repo:ro -v $WORK/base:/ckpt:ro -v $WORK/data:/data:ro -v $WORK/heads:/heads:ro -v $WORK/cache:/cache --entrypoint /bin/bash $IMAGE -lc 'cd /repo && PYTHONPATH=/repo python3 -u -m engine.profiles.qwen38.mtp_tune eval --data /data --ckpt /ckpt --eval-windows 64 --sampler 1.0,20,0.95 $tuned'" > "$OUT/estimate-$h.log" 2>&1 ) &
    pids+=($!)
  done
  [ ${#pids[@]} -gt 0 ] && wait "${pids[@]}"
  for h in "${names[@]}"; do [ -f "$OUT/estimate-$h.log" ] && stamp "estimate $h: $(tail -1 "$OUT/estimate-$h.log" | cut -c1-700)"; done
}
stamp "== estimates: the four heads' held-out rates at T=1, top-k 20, top-p 0.95 (exact match, drawn drafts, top-2)"
estimate

finish
for i in $(seq 1 120); do
  if curl -fsS -m 3 "$URL/v1/models" 2>/dev/null | grep -q '"id"'; then stamp "production door answers"; break; fi
  sleep 5
done
stamp "lease after: $(lease read | cut -c1-160)"
stamp "production downtime: $(( $(date +%s) - T_DOWN )) s"
