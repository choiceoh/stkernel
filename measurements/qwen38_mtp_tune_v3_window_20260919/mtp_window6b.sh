#!/usr/bin/env bash
# Window 6, part b (2026-09-19): the training-text boot died in the NVMe tier's restore (engine/base/runner.py: seq 1
# already owns resident resources, phase settling transfers) and the window built its runs from the held-out boot
# alone. This holds the window's lease, prefills the held-out and the training texts again with every boot's tier off
# (ST_TIER_DIR=off), builds the answer-only runs, drops the raw taps only when both splits are there, A/Bs the
# trainers on srv2 alone, trains 500 steps, exports, runs the two T=1 arms and releases.
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



# == 6b: window 6 again from its data boots, every boot with the NVMe tier off =============================================
released=1                                                 # until the lease is confirmed ours, the EXIT trap stops nothing
T_DOWN=$(date -d "2026-09-19 22:00:32" +%s)            # the window took the fleet then
root_sh() { docker run --rm -v "$DUMPS":/d --entrypoint /bin/sh "$IMAGE" -c "$*"; }
prefixes() { ls "$DUMPS/mtp-inputs" 2>/dev/null | sed -nE 's/^mtp-inputs-([0-9]{8}-[0-9]{6})-[0-9]+\.npz$/\1/p' | sort -u; }
free_gb() { df -BG --output=avail /home/choiceoh | tail -1 | tr -dc 0-9; }
textboot() {   # textboot <label> <texts.jsonl>: tap on (raised cap), the tier off, the texts prefilled as raw text
  local label=$1 texts=$2
  if boot "$label" ST_TIER_DIR=off ST_TAP_MTP_INPUTS=1 ST_TAP_MTP_CAP_GIB=$TAP_CAP ST_DRAFT_THRESHOLD=off ST_DRAFT_CANDIDATES=0; then
    python3 "$OUT/prefill_text.py" "$URL" "$label" "$texts" 6 > "$OUT/$label-requests.jsonl"
    tail -1 "$OUT/$label-requests.jsonl" | cut -c1-300
    grep -c '"error"' "$OUT/$label-requests.jsonl" | sed 's/^/  request errors: /'
  else
    stamp "the $label boot failed"
  fi
  logs "$label"
  stop_fleet
}
done_count() { python3 -c "import json,sys; print(json.loads(open(sys.argv[1]).read().splitlines()[-1]).get('final', {}).get('done', 0))" "$1" 2>/dev/null || echo 0; }

mine || { stamp "6b: the lease is not ours: $(lease read)"; exit 1; }
released=0
pkill -f "[w]hile sleep 90; do python3 engine/base/fleet_lease.py renew" || true     # the stopgap heartbeat, replaced
( while sleep 120; do lease renew --owner "$OWNER" >/dev/null 2>&1; done ) &
BEAT=$!
stamp "6b: the training-text boot died in the NVMe tier's restore at 22:14 (runner: seq 1 already owns resident resources, phase settling transfers) -- both data boots again, every boot with ST_TIER_DIR=off"
stop_tune
stop_fleet
root_sh "rm -f /d/mtp-inputs/mtp-inputs-*.npz; true"       # a dead boot's leftovers; the held-out runs of 22:00 go too
stamp "tap directory: $(ls $DUMPS/mtp-inputs | wc -l) shards; free $(free_gb) GB"

textboot data6h2 "$OUT/heldout_text.jsonl"
heldout=$(prefixes | tail -1)
stamp "held-out boot: ${heldout:-none} ($(done_count "$OUT/data6h2-requests.jsonl") of $(wc -l < "$OUT/heldout_text.jsonl") prefilled)"
textboot data6t2 "$OUT/train_text.jsonl"
trained_texts=$(done_count "$OUT/data6t2-requests.jsonl")
stamp "tap: $(ls $DUMPS/mtp-inputs | wc -l) shards, $(du -sh $DUMPS/mtp-inputs | cut -f1), boots $(prefixes | tr '\n' ' '); training texts prefilled $trained_texts; free $(free_gb) GB"

trained5=0
if [ -n "$heldout" ] && [ "$(prefixes | wc -l)" -ge 2 ] && [ "${trained_texts:-0}" -ge 2000 ]; then
  stamp "== runs: the answers only (--answer-after), the held-out boot $heldout as the evaluation"
  rm -rf "$WORK/data6"
  tune -m engine.profiles.qwen38.mtp_tune data --taps /dumps/mtp-inputs --out /work/data6 --min-length 16 \
       --answer-after 248045,74455,198 --eval-boots "$heldout" > "$OUT/tune-data6.log" 2>&1
  tail -1 "$OUT/tune-data6.log" | cut -c1-300
  split=$(python3 -c "import json; i = json.load(open('$WORK/data6/runs.json')); print(i['train'], i['eval'])" 2>/dev/null || echo "0 0")
  set -- $split
  if [ "${1:-0}" -gt 500000 ] && [ "${2:-0}" -gt 50000 ]; then
    root_sh "rm -f /d/mtp-inputs/mtp-inputs-*.npz"
    stamp "raw v3 taps dropped (train $1, eval $2 answer positions in the runs): free $(free_gb) GB"
    ok=1
    spreads=()
    for r in 0 1 2 3; do
      ip=${NODES[$r]}
      node_sh "$ip" "rm -rf $WORK/run6 && mkdir -p $WORK/run6 $WORK/cache" || ok=0
      if [[ "$SELF_IPS" == *" $ip "* ]]; then
        cp "$OUT/train_rank6.sh" "$WORK/train_rank6.sh" || ok=0
      else
        ( rsync -aW --delete "$WORK/data6/" "choiceoh@$ip:$WORK/data6/" \
            && scp -q "$OUT/train_rank6.sh" "choiceoh@$ip:$WORK/train_rank6.sh" && echo "$ip ready" ) &
        spreads+=($!)
      fi
    done
    for pid in "${spreads[@]}"; do wait "$pid" || ok=0; done
    stamp "spread ok=$ok"
    [ "$ok" = 1 ] && run_training=1
  else
    stamp "the runs hold train ${1:-0} / eval ${2:-0} positions: no training (the raw taps kept)"
  fi
else
  stamp "no held-out boot, no training boot, or under 2,000 training texts prefilled: no training"
fi

if [ "${run_training:-0}" = 1 ]; then
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
fi

# -- the arms: T=1 with block verification, the original head against the fifth, on the 80 prompts ------------------------
evalrun orig-t1 1.0,20,0.95 ST_TIER_DIR=off ST_DRAFT_THRESHOLD=off ST_TAP_MTP_INPUTS=0 ST_DRAFT_CANDIDATES=20 || stamp "orig-t1 did not serve"
if [ "$trained5" = 1 ]; then
  evalrun tuned5-t1 1.0,20,0.95 ST_TIER_DIR=off ST_DRAFT_THRESHOLD=off ST_TAP_MTP_INPUTS=0 ST_DRAFT_CANDIDATES=20 ST_MTP_TUNED="$TUNED5" \
    || stamp "tuned5-t1 did not serve"
fi

finish
for i in $(seq 1 120); do
  if curl -fsS -m 3 "$URL/v1/models" 2>/dev/null | grep -q '"id"'; then stamp "production door answers"; break; fi
  sleep 5
done
stamp "lease after: $(lease read | cut -c1-160)"
stamp "production downtime: $(( $(date +%s) - T_DOWN )) s"
