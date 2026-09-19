#!/usr/bin/env bash
# Window 5's takeover at the fourth training (2026-09-19). The runs gave 3,353,624 positions and 200 steps (4 ranks x 2
# windows of up to 1,024 positions a step) would see about a third of them; the operator, 18:53: "600스텝 해버려".
# Waits for 5c to reach its training with the data spread, kills 5c with -9 (its EXIT trap would release the lease),
# holds the lease with its own heartbeat, restarts the training at 600 steps (eval every 100, an 85-minute deadline),
# then the export, the arms and the release exactly as 5c would. 5c's variables and functions are taken whole.
set -uo pipefail
OWNER="session/q38mtp-tune5-0919"
TREE="$HOME/st-worktrees/q38mtp-w4"
LOCK=/home/choiceoh/glm53-logs/st-fleet.lock
OUT=/home/choiceoh/glm53-logs/q38mtp-window4-0919
WORK=/home/choiceoh/q38mtp-train-0919
TUNED3=/home/choiceoh/models/st-qwen38-mtp-tuned3
TUNED4=/home/choiceoh/models/st-qwen38-mtp-tuned4
CKPT=/home/choiceoh/models/qwen38-flash-next-nvfp4
RANKS=/home/choiceoh/models/st-qwen38-tep4
DUMPS=/home/choiceoh/glm53-logs/st-qwen38-dumps
IMAGE=st-engine:qwen38
STEPS4=${STEPS4:-200}
EVERY4=${EVERY4:-40}
LR4=${LR4:-3e-5}
EVAL_WINDOWS=${EVAL_WINDOWS:-96}
NODES=(10.10.10.2 10.10.10.1 10.10.10.3 10.10.10.4)
URL=http://10.10.10.2:8000
mkdir -p "$OUT" "$WORK/cache"
cd "$TREE"
lease() { python3 engine/base/fleet_lease.py "$@" --path "$LOCK"; }
SELF_IPS=" $(hostname -I 2>/dev/null) "
node_sh() { local ip=$1; shift; if [[ "$SELF_IPS" == *" $ip "* ]]; then bash -c "$*"; else ssh -o BatchMode=yes -o ConnectTimeout=5 "choiceoh@$ip" "$@"; fi; }
stamp() { echo "[$(date +%H:%M:%S)] $*"; }
released=0
BEAT=""
T_DOWN=""
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
  [ -n "$T_DOWN" ] && stamp "production down since handover: $(( $(date +%s) - T_DOWN )) s so far"
}
trap finish EXIT

mine() { case "$(lease read)" in "session $OWNER "*) return 0 ;; esac; return 1; }
wait_lane() { return 0; }                     # the operator: run beside the lane (contention is fine for this window)
wait_lane_unused() {
  local i n
  for i in $(seq 1 300); do
    n=$(node_sh 10.10.10.4 "docker ps --format '{{.Names}}' | grep -c '^st-probe-' || true")
    [ "$n" = 0 ] && return 0
    [ $((i % 20)) = 1 ] && stamp "waiting for srv4's probe to finish ($n up)"
    sleep 6
  done
  stamp "srv4's probe did not finish in 30 min"; return 1
}
serving_quiet() {
  curl -fsS -m 3 "$URL/v1/models" >/dev/null 2>&1 || return 1
  case "$(lease read)" in "production "*"running=0"*"waiting=0"*) return 0 ;; esac
  return 1
}
logs() {
  local r
  for r in 0 1 2 3; do node_sh "${NODES[$r]}" "docker logs st-qwen38 2>&1" > "$OUT/$1-rank$r.log" 2>/dev/null; done
  grep -E "ready in|lanes qualified|drafter:|STALL|Traceback|Error|illegal|mtp inputs" "$OUT/$1-rank0.log" | tail -10 | cut -c1-300
}
boot() {   # boot <label> [ENV=VALUE ...]
  local label=$1; shift
  wait_lane || return 1
  lease renew --owner "$OWNER" >/dev/null 2>&1 || true
  stamp "== boot $label ($*)"
  env ST_LEASE_OWNER="$OWNER" ST_SPEC_K=3 "$@" bash launchers/start-st-qwen38.sh > "$OUT/$label-launch.log" 2>&1
  local rc=$?
  tail -2 "$OUT/$label-launch.log"
  [ $rc = 0 ] || { stamp "launcher rc=$rc"; return 1; }
}
evalrun() {   # evalrun <label> <sampler T,K,P | greedy> [ENV=VALUE ...]
  local label=$1 sampler=$2; shift 2
  if ! boot "$label" "$@"; then logs "$label"; stop_fleet; return 1; fi
  if [ "$sampler" = greedy ]; then
    python3 "$OUT/mtp_requests2.py" eval "$URL" "$label" "$OUT/prompts.jsonl" > "$OUT/$label-requests.jsonl"
  else
    python3 "$OUT/mtp_requests2.py" eval "$URL" "$label" "$OUT/prompts.jsonl" "$sampler" > "$OUT/$label-requests.jsonl"
  fi
  local rq=$?
  tail -1 "$OUT/$label-requests.jsonl" | cut -c1-300
  logs "$label"
  stop_fleet
  return $rq
}
databoot() {   # databoot <label> <conversations.jsonl> [raw.jsonl]
  local label=$1 convos=$2 raw=${3:-}
  if boot "$label" ST_DRAFT_THRESHOLD=off ST_DRAFT_CANDIDATES=0 ST_MTP_TUNED="$TUNED3"; then   # #1266 off: the data first
    python3 "$OUT/mtp_requests2.py" prefill2 "$URL" "$label" "$convos" 4 all $raw > "$OUT/$label-requests.jsonl"
    tail -1 "$OUT/$label-requests.jsonl" | cut -c1-300
    grep -c '"error"' "$OUT/$label-requests.jsonl" | sed 's/^/  request errors: /'
  else
    stamp "the $label boot failed"
  fi
  logs "$label"
  stop_fleet
}
tune() {
  docker run --rm --gpus all --ipc host --shm-size 16g --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "$TREE":/repo:ro -v "$CKPT":/fullckpt:ro -v "$WORK/base":/ckpt:ro -v "$RANKS":/ranks:ro -v "$DUMPS":/dumps:ro \
    -v "$WORK":/work -v "$WORK/cache":/cache \
    --entrypoint /bin/bash "$IMAGE" -lc "cd /repo && PYTHONPATH=/repo python3 -u $*"
}


# == take600: window 5c's fourth training at 600 steps ===================================================================
T_DOWN=$(date -d "2026-09-19 18:31:33" +%s)            # 5c took the free fleet then
for i in $(seq 1 360); do grep -q "== fourth training" "$OUT/window5.log" && break; sleep 5; done
line=$(grep "== fourth training" "$OUT/window5.log" | tail -1)
case "$line" in
  *"spread ok=1"*) ;;
  *) stamp "take600: 5c did not reach a spread training (${line:-no training line}): leaving 5c alone"; released=1; exit 1 ;;
esac
pkill -9 -f "bash mtp_window5c.sh"                     # -9: 5c's EXIT trap would release the lease
sleep 10                                               # a rank 5c was starting over ssh lands before the stop below
mine || { stamp "take600: the lease is not ours: $(lease read)"; released=1; exit 1; }
( while sleep 120; do lease renew --owner "$OWNER" >/dev/null 2>&1; done ) &
BEAT=$!
stamp "take600: 5c stopped at its fourth training (operator: \"600스텝 해버려\"), the lease held here"
stop_tune
sleep 3
for ip in "${NODES[@]}"; do
  node_sh "$ip" "docker ps --format '{{.Names}}' | grep -q '^q38tune\$' && docker rm -f q38tune >/dev/null 2>&1; rm -rf $WORK/run && mkdir -p $WORK/run" \
    || stamp "take600: $ip did not clear"
done
STEPS4=600
EVERY4=100
trained4=0
stamp "== fourth training: $STEPS4 steps from the third head, peak $LR4, eval every $EVERY4 on $EVAL_WINDOWS windows"
for r in 3 2 1 0; do node_sh "${NODES[$r]}" "bash $WORK/train_rank4.sh $r $STEPS4 $EVERY4 $LR4 $EVAL_WINDOWS" || stamp "rank $r did not start"; done
deadline=$(( $(date +%s) + 85 * 60 ))
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
  node_sh "${NODES[$r]}" "docker logs q38tune 2>&1" > "$OUT/tune4-train-rank$r.log" 2>/dev/null
  stamp "rank $r exit $(node_sh "${NODES[$r]}" "docker inspect -f '{{.State.ExitCode}}' q38tune 2>/dev/null")"
done
stop_tune
grep -E '"event": "(eval|saved|end)"' "$OUT/tune4-train-rank0.log" | cut -c1-400
if [ -f "$WORK/run/head.safetensors" ]; then
  stamp "== export (fourth head)"
  rm -rf "$WORK/tuned4"
  tune -m engine.profiles.qwen38.mtp_tune export --tuned /work/run/head.safetensors --ckpt /fullckpt --ckpt-meta /ranks \
       --out /work/tuned4 > "$OUT/tune4-export.log" 2>&1
  tail -1 "$OUT/tune4-export.log" | cut -c1-300
  spread=1
  for r in 0 1 2 3; do
    ip=${NODES[$r]}
    f="$WORK/tuned4/mtp-tuned-r${r}of4.safetensors"
    [ -f "$f" ] || { spread=0; break; }
    node_sh "$ip" "mkdir -p $TUNED4" || spread=0
    if [[ "$SELF_IPS" == *" $ip "* ]]; then cp "$f" "$TUNED4/"; else scp -q "$f" "choiceoh@$ip:$TUNED4/" || spread=0; fi
  done
  [ "$spread" = 1 ] && trained4=1
  stamp "spread to four nodes: $spread"
else
  stamp "the fourth training did not beat the third head on the held-out windows: nothing new to serve"
fi

# -- C: the arms -------------------------------------------------------------------------------------------------------------
BEST="$TUNED3"; [ "$trained4" = 1 ] && BEST="$TUNED4"
# the greedy arms on the argmax drafts (candidates 0): #1266's first GPU run is the last arm alone
evalrun tuned3-off greedy ST_DRAFT_THRESHOLD=off ST_TAP_MTP_INPUTS=0 ST_DRAFT_CANDIDATES=0 ST_MTP_TUNED="$TUNED3" \
  || stamp "tuned3-off did not serve"
[ "$trained4" = 1 ] && { evalrun tuned4-off greedy ST_DRAFT_THRESHOLD=off ST_TAP_MTP_INPUTS=0 ST_DRAFT_CANDIDATES=0 \
  ST_MTP_TUNED="$TUNED4" || stamp "tuned4-off did not serve"; }
evalrun best-t1-exact 1.0,20,0.95 ST_DRAFT_THRESHOLD=off ST_TAP_MTP_INPUTS=0 ST_MTP_TUNED="$BEST" ST_DRAFT_CANDIDATES=0 \
  || stamp "best-t1-exact did not serve"
evalrun best-t1-block 1.0,20,0.95 ST_DRAFT_THRESHOLD=off ST_TAP_MTP_INPUTS=0 ST_MTP_TUNED="$BEST" ST_DRAFT_CANDIDATES=20 \
  || stamp "best-t1-block did not serve"

finish
for i in $(seq 1 120); do
  if curl -fsS -m 3 "$URL/v1/models" 2>/dev/null | grep -q '"id"'; then stamp "production door answers"; break; fi
  sleep 5
done
stamp "lease after: $(lease read | cut -c1-160)"
stamp "production downtime: $(( $(date +%s) - T_DOWN )) s"
