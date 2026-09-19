#!/usr/bin/env bash
# Window 5's second takeover (2026-09-19): the original head's greedy arm on the 180 prompts. The pair tuned4 - tuned3
# came out +0.5% +- 0.6% over all 180 and -1.0% +- 2.6% on Deneb's unseen conversations, and window 3 measured the
# tuned heads against the original on the v1 templates only. 5c's variables and functions are taken whole.
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


# == take-orig: one more greedy arm, the original head on the 180 prompts ====================================================
# The greedy pair tuned4 - tuned3 was +0.5% +- 0.6% over 180 prompts and -1.0% +- 2.6% on Deneb's unseen conversations:
# whether a tuned head should be the default at all needs the original head on the same prompts. Kills the takeover
# script (-9, the lease kept by this script's heartbeat) while an arm's requests run, lets those requests finish, then
# runs the arms still to come and orig-off, and releases.
T_DOWN=$(date -d "2026-09-19 18:31:33" +%s)
TUNED4=/home/choiceoh/models/st-qwen38-mtp-tuned4
running=""
for i in $(seq 1 360); do
  running=$(pgrep -af "mtp_requests2.py eval $URL " | grep -v pgrep | sed -E 's/.* eval [^ ]+ ([^ ]+) .*/\1/' | head -1)
  [ -n "$running" ] && break
  sleep 5
done
[ -n "$running" ] || { stamp "take-orig: no arm's requests running: leaving the takeover alone"; released=1; exit 1; }
pkill -9 -f "bash mtp_window5_take600.sh"
sleep 2
mine || { stamp "take-orig: the lease is not ours: $(lease read)"; released=1; exit 1; }
( while sleep 120; do lease renew --owner "$OWNER" >/dev/null 2>&1; done ) &
BEAT=$!
stamp "take-orig: took the window during $running's requests (the original head's greedy arm added)"
while pgrep -f "mtp_requests2.py eval $URL $running " >/dev/null; do sleep 5; done
tail -1 "$OUT/$running-requests.jsonl" | cut -c1-300
logs "$running"
stop_fleet
if [ "$running" = best-t1-exact ]; then
  evalrun best-t1-block 1.0,20,0.95 ST_DRAFT_THRESHOLD=off ST_TAP_MTP_INPUTS=0 ST_MTP_TUNED="$TUNED4" ST_DRAFT_CANDIDATES=20 \
    || stamp "best-t1-block did not serve"
fi
evalrun orig-off greedy ST_DRAFT_THRESHOLD=off ST_TAP_MTP_INPUTS=0 ST_DRAFT_CANDIDATES=0 || stamp "orig-off did not serve"

finish
for i in $(seq 1 120); do
  if curl -fsS -m 3 "$URL/v1/models" 2>/dev/null | grep -q '"id"'; then stamp "production door answers"; break; fi
  sleep 5
done
stamp "lease after: $(lease read | cut -c1-160)"
stamp "production downtime: $(( $(date +%s) - T_DOWN )) s"
