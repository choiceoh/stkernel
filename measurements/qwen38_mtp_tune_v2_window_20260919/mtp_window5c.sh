#!/usr/bin/env bash
# The MTP campaign's fifth window (2026-09-19): window 4 again from its second data boot. Window 4 (18:07-18:19) prefilled
# data3a and stopped when srv4's single-GPU lane took a 120-minute batch (sm121-batchB-0919b) its launcher would not boot
# beside. The operator: "너는 경합일어나도 상관없으니까 그 세션에 물어보고 같이하던가" -- so this window runs beside the
# lane: no lane waits, and the measurement tree's launcher leaves st-probe-* out of its busy check (a local edit, not on
# main). The arms' ms/step may carry the lane's load; their tokens/step do not. What follows is window 4's own notes.
#
# (window 4) The operator asked for ~1.5M positions of better text ("150만개로 하고
# 템플릿 주제를 넓히지그래", "프롬프트 생성기나 학습데이터 품질이 아주 중요해"), mixed rather than served-only ("적절히
# 섞어야해"), greedy-heavy ("t=0 학습데이터가 더 좋을수도"): datagen2.py's conversations (system prompts, multi-turn,
# tools, long documents, plain chat) and documents as raw text. This window: production hands over only when quiet and
# no single-lane probe runs; two data boots with the tap on -- the greedy conversations and the documents (data3a), the
# sampled ones (data3b), so each has its own boot in the runs and in the evaluation's by-boot rates; the runs over every
# tap; the fourth training from the third head (held-out windows taken from every boot in turn); the export; four arms
# (greedy: tuned3, tuned4; T=1: tuned4 with the exact match and with block verification, #1266); ALWAYS stop and release.
# Data boot c is Deneb's own text (operator: "데네브에 통과시켜서 원문 일부 뽑는건?" -- transcripts, wiki, code, files
# approved; on the fleet for prefill only). Names in the synthetic text are the operator's thirteen (names2.py).
# Tree: main a8e3c4de (#1263, #1266 and #1271 merged).
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

stamp "tree $(git log --oneline -1 | cut -c1-100)"
stamp "lease before: $(lease read)"
stamp "boots: a $(wc -l < "$OUT/boot_a.jsonl")+$(wc -l < "$OUT/raw_a.jsonl") raw, b $(wc -l < "$OUT/boot_b.jsonl"), c $(wc -l < "$OUT/boot_c.jsonl")+$(wc -l < "$OUT/raw_c.jsonl") raw"
for f in "$TUNED3/mtp-tuned-r0of4.safetensors" "$WORK/run/head.safetensors"; do [ -f "$f" ] || { stamp "missing $f"; released=1; exit 1; }; done
NOTE="operator window: MTP head -- the rest of window 4 (data3b, data3c with Deneb's text, a fourth training, T=1 arms), beside srv4's lane"
if [ "$(lease read)" = free ]; then
  # production cannot come back while srv4's lane runs a probe (st-supervisor, 18:19:25: "fleet taken
  # (10.10.10.4:st-probe-srv2-2334272): waiting -- no launch attempt"): the fleet is idle and free, so the window takes it
  lease acquire --owner "$OWNER" --kind session --pid $$ --est-minutes 70 --note "$NOTE" >/dev/null \
    || { stamp "acquire refused: $(lease read)"; released=1; exit 1; }
  stamp "acquired the free fleet: $(lease read | cut -c1-160)"
else
  for i in $(seq 1 200); do serving_quiet && break; [ $((i % 20)) = 1 ] && stamp "waiting for production to be up and quiet"; sleep 6; done
  serving_quiet || { stamp "production not up and quiet in 20 min: $(lease read)"; released=1; exit 1; }
  lease yield --requester "$OWNER" --kind session --pid $$ --host "$(hostname -s)" --est-minutes 75 --note "$NOTE" >/dev/null
  stamp "yield asked"
  for i in $(seq 1 200); do mine && break; sleep 3; done
  if mine; then stamp "held: $(lease read | cut -c1-160)"; else
    stamp "production did not hand over in 10 min: $(lease read)"; lease withdraw-yield --requester "$OWNER"; released=1; exit 1
  fi
fi
T_DOWN=$(date +%s)
for i in $(seq 1 60); do
  busy=""
  for ip in "${NODES[@]}"; do
    n=$(node_sh "$ip" "docker ps --format '{{.Names}}' | grep -E '^(glm53|q38|vllm|st-)' | grep -v '^st-probe-' | tr '\n' ' '" 2>/dev/null)   # the lane's probes stay (operator)
    [ -n "$n" ] && busy="$busy $ip:$n"
  done
  [ -z "$busy" ] && break
  sleep 3
done
[ -z "$busy" ] || { stamp "containers still up:$busy"; exit 1; }
( while sleep 120; do lease renew --owner "$OWNER" >/dev/null 2>&1; done ) &
BEAT=$!

# -- A: the v2 text through the fleet, the tap on --------------------------------------------------------------------------
before=$(ls "$DUMPS/mtp-inputs" | wc -l)
# data3a ran in window 4 (18:07-18:17): its shards are in the tap already
databoot data3b "$OUT/boot_b.jsonl"                             # the synthetic T=1 conversations
databoot data3c "$OUT/boot_c.jsonl" "$OUT/raw_c.jsonl"         # Deneb's transcripts, and its wiki, code and documents
stamp "tap: $before -> $(ls $DUMPS/mtp-inputs | wc -l) shards, $(du -sh $DUMPS/mtp-inputs | cut -f1)"

# -- B: the runs, the fourth training, the export ----------------------------------------------------------------------------
trained4=0
if [ "$(ls $DUMPS/mtp-inputs | wc -l)" -gt "$before" ]; then
  rm -rf "$WORK/run3" && mv "$WORK/run" "$WORK/run3"
  stamp "== runs (every tap: the prefilled v1 text, the decoded, the v2 greedy and the v2 sampled)"
  rm -rf "$WORK/data"
  tune -m engine.profiles.qwen38.mtp_tune data --taps /dumps/mtp-inputs --out /work/data > "$OUT/tune-data4.log" 2>&1
  tail -1 "$OUT/tune-data4.log" | cut -c1-300
  ok=1
  spreads=()
  for r in 0 1 2 3; do
    ip=${NODES[$r]}
    node_sh "$ip" "rm -rf $WORK/run $WORK/init && mkdir -p $WORK/init" || ok=0
    if [[ "$SELF_IPS" == *" $ip "* ]]; then
      cp "$WORK/run3/head.safetensors" "$WORK/init/head.safetensors" && cp "$OUT/train_rank4.sh" "$WORK/train_rank4.sh" || ok=0
    else
      ( rsync -aW --delete "$WORK/data/" "choiceoh@$ip:$WORK/data/" \
          && scp -q "$WORK/run3/head.safetensors" "choiceoh@$ip:$WORK/init/head.safetensors" \
          && scp -q "$OUT/train_rank4.sh" "choiceoh@$ip:$WORK/train_rank4.sh" && echo "$ip ready" ) &
      spreads+=($!)
    fi
  done
  for pid in "${spreads[@]}"; do wait "$pid" || ok=0; done       # not a bare `wait`: $BEAT never ends
  stamp "== fourth training: $STEPS4 steps from the third head, peak $LR4, eval every $EVERY4 on $EVAL_WINDOWS windows (spread ok=$ok)"
  if [ "$ok" = 1 ]; then
    for r in 3 2 1 0; do node_sh "${NODES[$r]}" "bash $WORK/train_rank4.sh $r $STEPS4 $EVERY4 $LR4 $EVAL_WINDOWS" || stamp "rank $r did not start"; done
    deadline=$(( $(date +%s) + 50 * 60 ))
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
  fi
else
  stamp "no new shards: no fourth training"
fi

# -- C: the arms -------------------------------------------------------------------------------------------------------------
BEST="$TUNED3"; [ "$trained4" = 1 ] && BEST="$TUNED4"
# the greedy arms and the data boots on the argmax drafts (candidates 0): #1266's first GPU run is the last arm alone
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
