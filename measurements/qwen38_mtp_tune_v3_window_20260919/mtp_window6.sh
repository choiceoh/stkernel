#!/usr/bin/env bash
# The MTP campaign's sixth window (2026-09-19/20): self-distribution data. Window 5 showed the head drafts only inside
# the served model's own answers, and its data was mostly other text; the operator: "2번 작업으로 빨리 넘어가는게
# 나을것 같기도", "a" (Deneb's prompts through OpenRouter's qwen3.8-flash), "메일분석은?", "진행". The served model's
# own answers -- to Deneb's prompts (986), Deneb's mail (600, all thinking, half under its email-analysis skill) and
# synthetic conversations (1,074) -- rendered by the door's own template (render_all.py) and prefilled as raw text with
# the tap on: the held-out split (by session) in a boot of its own, then the training texts. Runs keep the answers only
# (mtp_tune data --answer-after, --eval-boots, #1301); the fifth training starts from the original head; the arms are
# the original, the third and the fifth head greedy, and the fifth at T=1 with block verification, on 80 prompts.
# Tree: PR #1301's branch (bf74dbc5) + the local lane-probe edit of the launcher. ALWAYS stops and releases.
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


released=1                                                 # until the fleet is this window's, the EXIT trap stops nothing

textboot() {   # textboot <label> <texts.jsonl>: a boot with the tap on (raised cap) and the texts prefilled as raw text
  local label=$1 texts=$2
  if boot "$label" ST_TAP_MTP_INPUTS=1 ST_TAP_MTP_CAP_GIB=$TAP_CAP ST_DRAFT_THRESHOLD=off ST_DRAFT_CANDIDATES=0; then
    python3 "$OUT/prefill_text.py" "$URL" "$label" "$texts" 6 > "$OUT/$label-requests.jsonl"
    tail -1 "$OUT/$label-requests.jsonl" | cut -c1-300
    grep -c '"error"' "$OUT/$label-requests.jsonl" | sed 's/^/  request errors: /'
  else
    stamp "the $label boot failed"
  fi
  logs "$label"
  stop_fleet
}
root_sh() {   # a root shell over the dumps directory (the fleet writes it as root)
  docker run --rm -v "$DUMPS":/d --entrypoint /bin/sh "$IMAGE" -c "$*"
}
prefixes() { ls "$DUMPS/mtp-inputs" 2>/dev/null | sed -nE 's/^mtp-inputs-([0-9]{8}-[0-9]{6})-[0-9]+\.npz$/\1/p' | sort -u; }
free_gb() { df -BG --output=avail /home/choiceoh | tail -1 | tr -dc 0-9; }

stamp "tree $(git log --oneline -1 | cut -c1-100) (+ the local lane-probe edit of the launcher)"
stamp "lease before: $(lease read)"
stamp "texts: train $(wc -l < "$OUT/train_text.jsonl"), held-out $(wc -l < "$OUT/heldout_text.jsonl"); eval $(wc -l < "$PROMPTS") prompts"
for f in "$OUT/prefill_text.py" "$OUT/mtp_requests2.py" "$OUT/train_rank6.sh" "$PROMPTS" "$TUNED3/mtp-tuned-r0of4.safetensors"; do
  [ -f "$f" ] || { stamp "missing $f"; released=1; exit 1; }
done
# -- room: the archived Deneb v2 prefill (window 5's data3c) is other models' text -- no use to a head that drafts the
# served model's own answers -- and the most sensitive shards; and the tap directory must hold this window's boots only
# (another session's Qwen boots since 21:22 -- a GPTQ/RTN repack changes the target, so its taps are not this head's
# data -- go to their own directory, dropped if srv2 needs the room)
tidy_taps() {
  root_sh "rm -f /d/mtp-inputs-archive-20260919/mtp-inputs-20260919-093746-*.npz; mkdir -p /d/mtp-inputs-other-0919; \
           for f in /d/mtp-inputs/mtp-inputs-*.npz; do [ -e \"\$f\" ] && mv -n \"\$f\" /d/mtp-inputs-other-0919/; done; true"
  if [ "$(free_gb)" -lt 200 ]; then root_sh "rm -rf /d/mtp-inputs-other-0919"; stamp "other boots' taps dropped for room"; fi
  stamp "tap directory: $(ls $DUMPS/mtp-inputs | wc -l) shards; archive $(du -sh $DUMPS/mtp-inputs-archive-20260919 | cut -f1); free $(free_gb) GB"
}
stamp "srv2 free $(free_gb) GB (the tap directory is tidied once the fleet is this window's)"

NOTE="operator window: MTP head -- self-distribution data (the served model's own answers to Deneb's prompts, mail analyses, synthetic), answer-only training, 80-prompt live eval"
if [ "$(lease read)" = free ]; then
  lease acquire --owner "$OWNER" --kind session --pid $$ --est-minutes 180 --note "$NOTE" >/dev/null \
    || { stamp "acquire refused: $(lease read)"; released=1; exit 1; }
  stamp "acquired the free fleet: $(lease read | cut -c1-160)"
else
  # another session may hold the fleet (21:39: session/q38gptq-0919): wait -- session holders are never asked
  for i in $(seq 1 3000); do serving_quiet && break; [ $((i % 100)) = 1 ] && stamp "waiting for production to be up and quiet: $(lease read | cut -c1-120)"; sleep 6; done
  serving_quiet || { stamp "production not up and quiet in 5 h: $(lease read)"; released=1; exit 1; }
  lease yield --requester "$OWNER" --kind session --pid $$ --host "$(hostname -s)" --est-minutes 180 --note "$NOTE" >/dev/null
  stamp "yield asked"
  for i in $(seq 1 200); do mine && break; sleep 3; done
  if mine; then stamp "held: $(lease read | cut -c1-160)"; else
    stamp "production did not hand over in 10 min: $(lease read)"; lease withdraw-yield --requester "$OWNER"; released=1; exit 1
  fi
fi
T_DOWN=$(date +%s)
released=0                                                 # the fleet is this window's from here: the trap stops and releases
tidy_taps                                                  # only now: the other session's boots are over
[ "$(free_gb)" -ge 200 ] || { stamp "under 200 GB free on srv2: releasing"; exit 1; }
for i in $(seq 1 60); do
  busy=""
  for ip in "${NODES[@]}"; do
    n=$(node_sh "$ip" "docker ps --format '{{.Names}}' | grep -E '^(glm53|q38|vllm|st-)' | grep -v '^st-probe-' | tr '\n' ' '" 2>/dev/null)
    [ -n "$n" ] && busy="$busy $ip:$n"
  done
  [ -z "$busy" ] && break
  sleep 3
done
[ -z "$busy" ] || { stamp "containers still up:$busy"; exit 1; }
( while sleep 120; do lease renew --owner "$OWNER" >/dev/null 2>&1; done ) &
BEAT=$!

# -- A: the held-out texts in a boot of their own, then the training texts ---------------------------------------------
textboot data6h "$OUT/heldout_text.jsonl"
heldout=$(prefixes | tail -1)
stamp "held-out boot: ${heldout:-none}"
textboot data6t "$OUT/train_text.jsonl"
stamp "tap: $(ls $DUMPS/mtp-inputs | wc -l) shards, $(du -sh $DUMPS/mtp-inputs | cut -f1), boots $(prefixes | tr '\n' ' '); free $(free_gb) GB"

# -- B: answer-only runs, the raw taps dropped, the fifth training from the original head, the export ----------------------
trained5=0
if [ -n "$heldout" ] && [ "$(prefixes | wc -l)" -ge 2 ]; then
  stamp "== runs: the answers only (--answer-after), the held-out boot $heldout as the evaluation"
  rm -rf "$WORK/data6"
  tune -m engine.profiles.qwen38.mtp_tune data --taps /dumps/mtp-inputs --out /work/data6 --min-length 16 \
       --answer-after 248045,74455,198 --eval-boots "$heldout" > "$OUT/tune-data6.log" 2>&1
  tail -1 "$OUT/tune-data6.log" | cut -c1-300
  positions=$(python3 -c "import json; print(json.load(open('$WORK/data6/runs.json'))['positions'])" 2>/dev/null || echo 0)
  if [ "${positions:-0}" -gt 100000 ]; then
    root_sh "rm -f /d/mtp-inputs/mtp-inputs-*.npz"
    stamp "raw v3 taps dropped (the runs hold the answers; the texts stay on srv2/srv4): free $(free_gb) GB"
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
    stamp "== fifth training: $STEPS5 steps from the original head, peak $LR5, eval every $EVERY5 on $EVAL_WINDOWS windows (spread ok=$ok)"
    if [ "$ok" = 1 ]; then
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
  else
    stamp "the runs hold ${positions:-0} positions: no training (the raw taps kept)"
  fi
else
  stamp "no held-out boot or no training boot in the tap: no training"
fi

# -- C: the arms on the 80 prompts -------------------------------------------------------------------------------------------
# production samples at T=1 with block verification (#1266, +6.3% on the fourth head in window 5): the default (D11) is
# judged on that setting alone -- the original head against the fifth (operator: "5번이나 평가해야해? 그냥 같은 t=1만")
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
