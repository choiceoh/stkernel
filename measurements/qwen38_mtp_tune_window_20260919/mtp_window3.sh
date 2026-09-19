#!/usr/bin/env bash
# The MTP campaign's window, third form (2026-09-19): production hands over only when its door answers and nothing is
# in flight (the second form's yield met a rebooting production, which died mid-boot and dropped the request), and
# every Qwen boot waits out a single-lane probe on srv4 (the first form's boots all met one). Then: the tuner's GPU
# smoke; one data boot -- a prefill of OpenRouter's text (samples.jsonl) if it is there, else the train prompts
# decoded for DATA_S; the runs cut and spread; the head trained on the four nodes (data parallel); exported and
# spread; the original and the tuned head paired (drafts uncut, and the tuned head cut); then ALWAYS stop and release.
set -uo pipefail
OWNER="session/q38mtp-tune-0919"
TREE="$HOME/st-worktrees/q38mtp-win"
LOCK=/home/choiceoh/glm53-logs/st-fleet.lock
OUT=/home/choiceoh/glm53-logs/q38mtp-window-0919
WORK=/home/choiceoh/q38mtp-train-0919
TUNED=/home/choiceoh/models/st-qwen38-mtp-tuned
CKPT=/home/choiceoh/models/qwen38-flash-next-nvfp4
RANKS=/home/choiceoh/models/st-qwen38-tep4
DUMPS=/home/choiceoh/glm53-logs/st-qwen38-dumps
IMAGE=st-engine:qwen38
SAMPLES=${SAMPLES:-$OUT/samples.jsonl}
DATA_S=${DATA_S:-1500}
TRAIN_MIN=${TRAIN_MIN:-12}
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
wait_lane() {   # a single-lane probe on srv4 finishes first: the launcher refuses beside one
  local i n
  for i in $(seq 1 300); do
    n=$(node_sh 10.10.10.4 "docker ps --format '{{.Names}}' | grep -c '^st-probe-' || true")
    [ "$n" = 0 ] && return 0
    [ $((i % 20)) = 1 ] && stamp "waiting for srv4's probe to finish ($n up)"
    sleep 6
  done
  stamp "srv4's probe did not finish in 30 min"; return 1
}
serving_quiet() {   # production's door answers and nothing is in flight
  curl -fsS -m 3 "$URL/v1/models" >/dev/null 2>&1 || return 1
  case "$(lease read)" in "production "*"running=0"*"waiting=0"*) return 0 ;; esac
  return 1
}

stamp "tree $(git log --oneline -1 | cut -c1-100)"
stamp "lease before: $(lease read)"
stamp "samples: $( [ -s "$SAMPLES" ] && wc -l < "$SAMPLES" || echo none) -- $( [ -s "$SAMPLES" ] && echo prefill || echo "decode for $DATA_S s")"
for i in $(seq 1 200); do serving_quiet && break; [ $((i % 20)) = 1 ] && stamp "waiting for production to be up and quiet"; sleep 6; done
serving_quiet || { stamp "production not up and quiet in 20 min: $(lease read)"; released=1; exit 1; }
wait_lane || { released=1; exit 1; }
NOTE="operator window: MTP head fine-tune -- data, four-node training, original and tuned head paired"
lease yield --requester "$OWNER" --kind session --pid $$ --host "$(hostname -s)" --est-minutes 90 --note "$NOTE" >/dev/null
stamp "yield asked"
for i in $(seq 1 200); do mine && break; sleep 3; done
if mine; then stamp "held: $(lease read | cut -c1-160)"; else
  stamp "production did not hand over in 10 min: $(lease read)"; lease withdraw-yield --requester "$OWNER"; released=1; exit 1
fi
T_DOWN=$(date +%s)
for i in $(seq 1 60); do                        # production's containers leave before anything starts
  busy=""
  for ip in "${NODES[@]}"; do
    n=$(node_sh "$ip" "docker ps --format '{{.Names}}' | grep -E '^(glm53|q38|vllm|st-)' | tr '\n' ' '" 2>/dev/null)
    [ -n "$n" ] && busy="$busy $ip:$n"
  done
  [ -z "$busy" ] && break
  sleep 3
done
[ -z "$busy" ] || { stamp "containers still up:$busy"; exit 1; }
( while sleep 120; do lease renew --owner "$OWNER" >/dev/null 2>&1; done ) &
BEAT=$!

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
evalrun() {   # evalrun <label> [ENV=VALUE ...]
  local label=$1; shift
  if ! boot "$label" "$@"; then logs "$label"; stop_fleet; return 1; fi
  python3 "$OUT/mtp_requests.py" eval "$URL" "$label" "$OUT/prompts.jsonl" > "$OUT/$label-requests.jsonl"
  local rq=$?
  tail -1 "$OUT/$label-requests.jsonl" | cut -c1-300
  logs "$label"
  stop_fleet
  return $rq
}
tune() {   # a container on this node over the tuner (torch's own backward launches Triton kernels: /cache)
  docker run --rm --gpus all --ipc host --shm-size 16g --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "$TREE":/repo:ro -v "$CKPT":/fullckpt:ro -v "$WORK/base":/ckpt:ro -v "$RANKS":/ranks:ro -v "$DUMPS":/dumps:ro \
    -v "$WORK":/work -v "$WORK/cache":/cache \
    --entrypoint /bin/bash "$IMAGE" -lc "cd /repo && PYTHONPATH=/repo python3 -u $*"
}

# -- 0: the tuner's first GPU run ---------------------------------------------------------------------------------------
stamp "== tuner smoke on the GPU (synthetic streams)"
tune /work/smoke_gpu.py > "$OUT/tune-smoke-gpu.log" 2>&1
tail -4 "$OUT/tune-smoke-gpu.log" | cut -c1-300
per_window=$(python3 - "$OUT/tune-smoke-gpu.log" <<'EOF'
import json, sys
s = [json.loads(l)["s"] for l in open(sys.argv[1]) if l.startswith('{"window"')]
print(s[-1] if s else 0)
EOF
)
stamp "smoke: $per_window s a window"

# -- A: data ------------------------------------------------------------------------------------------------------------
if boot data ST_DRAFT_THRESHOLD=off; then
  if [ -s "$SAMPLES" ]; then
    python3 "$OUT/mtp_requests.py" prefill "$URL" data "$SAMPLES" 4 > "$OUT/data-requests.jsonl"
  else
    python3 "$OUT/mtp_requests.py" data "$URL" data "$OUT/prompts.jsonl" "$DATA_S" 4 > "$OUT/data-requests.jsonl"
  fi
  tail -1 "$OUT/data-requests.jsonl" | cut -c1-300
else
  stamp "the data boot failed"
fi
logs data
stop_fleet
stamp "tap: $(ls $DUMPS/mtp-inputs 2>/dev/null | wc -l) shards, $(du -sh $DUMPS/mtp-inputs 2>/dev/null | cut -f1)"

# -- B: the runs, the four-node training, the export ---------------------------------------------------------------------
trained=0
if [ -n "$(ls $DUMPS/mtp-inputs 2>/dev/null)" ] && python3 -c "import sys; sys.exit(0 if float('$per_window') > 0 else 1)"; then
  stamp "== runs"
  rm -rf "$WORK/data" "$WORK/run"
  tune -m engine.profiles.qwen38.mtp_tune data --taps /dumps/mtp-inputs --out /work/data > "$OUT/tune-data.log" 2>&1
  tail -1 "$OUT/tune-data.log" | cut -c1-300
  stamp "== spread the runs and the base"
  for r in 1 2 3; do
    ip=${NODES[$r]}
    ( node_sh "$ip" "mkdir -p $WORK/cache && rm -rf $WORK/data $WORK/run"
      rsync -a "$WORK/data/" "choiceoh@$ip:$WORK/data/" && rsync -a "$WORK/base/" "choiceoh@$ip:$WORK/base/" \
        && scp -q "$WORK/train_rank.sh" "choiceoh@$ip:$WORK/train_rank.sh" && echo "$ip ready" ) &
  done
  wait
  steps=$(python3 -c "print(max(20, int($TRAIN_MIN * 60 * 0.8 / (2 * $per_window + 0.2))))")
  every=$(( steps / 4 > 0 ? steps / 4 : 1 ))
  stamp "== train on four nodes: $steps steps of 2 windows a rank"
  for r in 3 2 1 0; do node_sh "${NODES[$r]}" "bash $WORK/train_rank.sh $r $steps $every" || stamp "rank $r did not start"; done
  deadline=$(( $(date +%s) + TRAIN_MIN * 60 * 2 + 600 ))
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
    node_sh "${NODES[$r]}" "docker logs q38tune 2>&1" > "$OUT/tune-train-rank$r.log" 2>/dev/null
    stamp "rank $r exit $(node_sh "${NODES[$r]}" "docker inspect -f '{{.State.ExitCode}}' q38tune 2>/dev/null")"
  done
  stop_tune
  grep -E '"event": "(eval|saved|end)"' "$OUT/tune-train-rank0.log" | cut -c1-300
  if [ -f "$WORK/run/head.safetensors" ]; then
    stamp "== export"
    rm -rf "$WORK/tuned" "$WORK/tuned.incomplete"
    tune -m engine.profiles.qwen38.mtp_tune export --tuned /work/run/head.safetensors --ckpt /fullckpt --ckpt-meta /ranks \
         --out /work/tuned > "$OUT/tune-export.log" 2>&1
    tail -1 "$OUT/tune-export.log" | cut -c1-300
    spread=1
    for r in 0 1 2 3; do
      ip=${NODES[$r]}
      f="$WORK/tuned/mtp-tuned-r${r}of4.safetensors"
      [ -f "$f" ] || { spread=0; break; }
      node_sh "$ip" "mkdir -p $TUNED" || spread=0
      if [[ "$SELF_IPS" == *" $ip "* ]]; then cp "$f" "$TUNED/"; else scp -q "$f" "choiceoh@$ip:$TUNED/" || spread=0; fi
    done
    [ "$spread" = 1 ] && trained=1
    stamp "spread to four nodes: $spread"
  else
    stamp "no tuned head beat the original on the held-out windows: nothing to serve"
  fi
fi

# -- C: the original and the tuned head, back to back -------------------------------------------------------------------
evalrun orig-off ST_DRAFT_THRESHOLD=off ST_TAP_MTP_INPUTS=0 || stamp "orig-off did not serve"
if [ "$trained" = 1 ]; then
  evalrun tuned-off ST_DRAFT_THRESHOLD=off ST_TAP_MTP_INPUTS=0 ST_MTP_TUNED="$TUNED" || stamp "tuned-off did not serve"
  evalrun tuned-cut ST_TAP_MTP_INPUTS=0 ST_MTP_TUNED="$TUNED" || stamp "tuned-cut did not serve"
else
  evalrun orig-cut ST_TAP_MTP_INPUTS=0 || stamp "orig-cut did not serve"
fi

finish
for i in $(seq 1 120); do
  if curl -fsS -m 3 "$URL/v1/models" 2>/dev/null | grep -q '"id"'; then stamp "production door answers"; break; fi
  sleep 5
done
stamp "lease after: $(lease read | cut -c1-160)"
stamp "production downtime: $(( $(date +%s) - T_DOWN )) s"
