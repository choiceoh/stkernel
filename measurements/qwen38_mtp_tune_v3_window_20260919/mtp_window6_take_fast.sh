#!/usr/bin/env bash
# Window 6's takeover at its training (2026-09-19): the operator asked the trainer to go faster ("500스텝 학습도 병렬로
# 더 빠르게 할수 있나" -> "고쳐봐"); the four GPUs already train in data parallel, so the step itself shrinks -- the routed
# experts in two batched matmuls and the target's vocabulary rows once a window (branch qwen38-mtp-tune-fast). Waits
# for the window's training line, kills the window script (-9, the lease kept), A/Bs the tree's trainer and the fast
# one for 15 steps on srv2's GPU alone, trains on the fast one only if it is >10% faster and its step-5 loss matches,
# then the export, the T=1 arms and the release as the window would. The window script's functions taken whole.
set -uo pipefail
OWNER="session/q38mtp-tune6-0919"
TREE="$HOME/st-worktrees/q38mtp-w6"
LOCK=/home/choiceoh/glm53-logs/st-fleet.lock
OUT=/home/choiceoh/glm53-logs/q38mtp-window6-0919
WORK=/home/choiceoh/q38mtp-train-0919
TUNED3=/home/choiceoh/models/st-qwen38-mtp-tuned3
TUNED5=/home/choiceoh/models/st-qwen38-mtp-tuned5
CKPT=/home/choiceoh/models/qwen38-flash-next-nvfp4
RANKS=/home/choiceoh/models/st-qwen38-tep4
DUMPS=/home/choiceoh/glm53-logs/st-qwen38-dumps
IMAGE=st-engine:qwen38
STEPS5=${STEPS5:-500}
EVERY5=${EVERY5:-100}
LR5=${LR5:-5e-5}
TAP_CAP=${TAP_CAP:-140}
PROMPTS=$OUT/prompts80.jsonl
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
  case "$(lease read)" in *"asked to yield"*) return 1 ;; esac   # another session is next: a yield would take its turn
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
    python3 "$OUT/mtp_requests2.py" eval "$URL" "$label" "$PROMPTS" > "$OUT/$label-requests.jsonl"
  else
    python3 "$OUT/mtp_requests2.py" eval "$URL" "$label" "$PROMPTS" "$sampler" > "$OUT/$label-requests.jsonl"
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



# == take-fast: the fifth training on the faster trainer, if an A/B on srv2's GPU says so ===============================
released=1                                                 # until this holds the window, the EXIT trap stops nothing
T_DOWN=$(date -d "2026-09-19 22:00:32" +%s)            # the window took the fleet then
for i in $(seq 1 900); do grep -q "== fifth training" "$OUT/window6.log" && break; sleep 5; done
line=$(grep "== fifth training" "$OUT/window6.log" | tail -1)
case "$line" in
  *"spread ok=1"*) ;;
  *) stamp "take-fast: the window did not reach a spread training (${line:-no training line}): leaving it alone"; exit 1 ;;
esac
pids=$(pgrep -f "[b]ash mtp_window6.sh" | tr '\n' ' ')
[ -n "$pids" ] && kill -9 $pids                            # -9: its EXIT trap would release the lease
sleep 10
mine || { stamp "take-fast: the lease is not ours: $(lease read)"; exit 1; }
released=0
( while sleep 120; do lease renew --owner "$OWNER" >/dev/null 2>&1; done ) &
BEAT=$!
stamp "take-fast: the window taken at its training (operator: \"고쳐봐\") -- the batched-experts trainer tried against the tree's"
stop_tune
sleep 3
stop_tune                                                  # a rank the window's ssh was still starting

ab() {   # ab <label> [extra docker args]: 15 steps on srv2's GPU alone -> {"s_a_step", "loss_1_at_5"} or {"failed"}
  local label=$1; shift
  rm -rf "$WORK/ab-$label" && mkdir -p "$WORK/ab-$label"
  docker run --rm --name "q38tune-ab-$label" --gpus all --ipc host --shm-size 16g --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "$TREE":/repo:ro "$@" -v "$WORK/base":/ckpt:ro -v "$WORK/data6":/data:ro -v "$WORK/ab-$label":/out \
    -v "$WORK/cache":/cache --entrypoint /bin/bash "$IMAGE" -lc \
    "cd /repo && PYTHONPATH=/repo timeout 900 python3 -u -m engine.profiles.qwen38.mtp_tune train --data /data --ckpt /ckpt \
     --out /out --steps 15 --accumulate 2 --eval-every 1000 --eval-windows 4 --log-every 5 --lr $LR5" > "$OUT/ab-$label.log" 2>&1
  python3 - "$OUT/ab-$label.log" <<'PY'
import json, sys
rows = {}
for line in open(sys.argv[1], errors="replace"):
    if line.startswith('{"event": "train"'):
        r = json.loads(line)
        rows[r["step"]] = r
if 5 in rows and 15 in rows:
    print(json.dumps({"s_a_step": round((rows[15]["t"] - rows[5]["t"]) / 10, 3), "loss_1_at_5": rows[5]["loss_1"],
                      "loss_1_at_15": rows[15]["loss_1"]}))
else:
    print(json.dumps({"failed": True}))
PY
}
old=$(ab tree)
new=$(ab fast -v "$OUT/mtp_tune_fast.py":/repo/engine/profiles/qwen38/mtp_tune.py:ro)
stamp "trainer A/B (15 steps, srv2 alone): tree $old; fast $new"
choice=$(python3 - "$old" "$new" <<'PY'
import json, sys
o, n = (json.loads(a) for a in sys.argv[1:3])
fast = (not o.get("failed") and not n.get("failed") and n["s_a_step"] < 0.9 * o["s_a_step"]
        and abs(n["loss_1_at_5"] - o["loss_1_at_5"]) <= 0.03 * abs(o["loss_1_at_5"]) + 1e-3)
print("fast" if fast else "tree")
PY
)
if [ "$choice" = fast ]; then
  for ip in "${NODES[@]}"; do
    if [[ "$SELF_IPS" == *" $ip "* ]]; then cp "$OUT/mtp_tune_fast.py" "$HOME/st-engine-qwen38/engine/profiles/qwen38/mtp_tune.py"
    else scp -q "$OUT/mtp_tune_fast.py" "choiceoh@$ip:st-engine-qwen38/engine/profiles/qwen38/mtp_tune.py"; fi \
      || { stamp "$ip: the fast trainer not copied -- the tree's trains"; choice=tree; }
  done
fi
if [ "$choice" = tree ]; then                             # the tree's trainer everywhere (a copy may have landed)
  for ip in "${NODES[@]}"; do
    if [[ "$SELF_IPS" == *" $ip "* ]]; then cp "$TREE/engine/profiles/qwen38/mtp_tune.py" "$HOME/st-engine-qwen38/engine/profiles/qwen38/mtp_tune.py"
    else scp -q "$TREE/engine/profiles/qwen38/mtp_tune.py" "choiceoh@$ip:st-engine-qwen38/engine/profiles/qwen38/mtp_tune.py"; fi
  done
fi
stamp "== fifth training on the $choice trainer: $STEPS5 steps from the original head, peak $LR5, eval every $EVERY5 on $EVAL_WINDOWS windows"
trained5=0
for ip in "${NODES[@]}"; do node_sh "$ip" "rm -rf $WORK/run6 && mkdir -p $WORK/run6"; done
for r in 3 2 1 0; do node_sh "${NODES[$r]}" "bash $WORK/train_rank6.sh $r $STEPS5 $EVERY5 $LR5 $EVAL_WINDOWS" || stamp "rank $r did not start"; done
deadline=$(( $(date +%s) + 100 * 60 ))
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
  node_sh "${NODES[$r]}" "docker logs q38tune 2>&1" > "$OUT/tune5-train-rank$r.log" 2>/dev/null
  stamp "rank $r exit $(node_sh "${NODES[$r]}" "docker inspect -f '{{.State.ExitCode}}' q38tune 2>/dev/null")"
done
stop_tune
grep -E '"event": "(eval|saved|end)"' "$OUT/tune5-train-rank0.log" | cut -c1-400
if [ -f "$WORK/run6/head.safetensors" ]; then
  stamp "== export (fifth head)"
  rm -rf "$WORK/tuned5"
  tune -m engine.profiles.qwen38.mtp_tune export --tuned /work/run6/head.safetensors --ckpt /fullckpt --ckpt-meta /ranks \
       --out /work/tuned5 > "$OUT/tune5-export.log" 2>&1
  tail -1 "$OUT/tune5-export.log" | cut -c1-300
  spread=1
  for r in 0 1 2 3; do
    ip=${NODES[$r]}
    f="$WORK/tuned5/mtp-tuned-r${r}of4.safetensors"
    [ -f "$f" ] || { spread=0; break; }
    node_sh "$ip" "mkdir -p $TUNED5" || spread=0
    if [[ "$SELF_IPS" == *" $ip "* ]]; then cp "$f" "$TUNED5/"; else scp -q "$f" "choiceoh@$ip:$TUNED5/" || spread=0; fi
  done
  [ "$spread" = 1 ] && trained5=1
  stamp "spread to four nodes: $spread"
else
  stamp "the fifth training did not beat the original head on the held-out boot: nothing new to serve"
fi

# -- the arms: T=1 with block verification, the original head against the fifth, on the 80 prompts ------------------------
evalrun orig-t1 1.0,20,0.95 ST_DRAFT_THRESHOLD=off ST_TAP_MTP_INPUTS=0 ST_DRAFT_CANDIDATES=20 || stamp "orig-t1 did not serve"
if [ "$trained5" = 1 ]; then
  evalrun tuned5-t1 1.0,20,0.95 ST_DRAFT_THRESHOLD=off ST_TAP_MTP_INPUTS=0 ST_DRAFT_CANDIDATES=20 ST_MTP_TUNED="$TUNED5" \
    || stamp "tuned5-t1 did not serve"
fi

finish
for i in $(seq 1 120); do
  if curl -fsS -m 3 "$URL/v1/models" 2>/dev/null | grep -q '"id"'; then stamp "production door answers"; break; fi
  sleep 5
done
stamp "lease after: $(lease read | cut -c1-160)"
stamp "production downtime: $(( $(date +%s) - T_DOWN )) s"
