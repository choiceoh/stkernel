#!/usr/bin/env bash
# The MTP campaign's third window, third part (2026-09-19). The second training (from the first head) was still rising
# (3.093 -> 3.13 held-out) and the live arms were greedy: the tuned head gained +4.9% on the target's own greedy text
# against +18.6% on OpenRouter's -- the head is weak where the text is not the target's. The operator: "데이터 추가로
# 모아서 3번째 학습까지도 할만한데". takeover2.sh started this and killed mtp_window3b.sh (-9, the lease kept) during
# tuned2-off's requests. This script waits for them, boots the second head with the tap on and decodes the train
# prompts (their own temperatures, later passes at 0.8) for DATA_S, cuts the runs again (the old ones keep their split),
# trains a third time from the second head over both, exports, serves it once -- then ALWAYS stops and releases.
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
PENDING_ARM=${PENDING_ARM:-tuned2-off}
DATA_S=${DATA_S:-900}
STEPS3=${STEPS3:-130}
EVERY3=${EVERY3:-26}
LR3=${LR3:-3e-5}
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

stamp "== 3c: took the window over during $PENDING_ARM's requests (more data from the target's own decoding, a third training)"
while pgrep -f "mtp_requests.py eval $URL $PENDING_ARM" >/dev/null; do sleep 5; done
tail -1 "$OUT/$PENDING_ARM-requests.jsonl" | cut -c1-300
logs "$PENDING_ARM"
stop_fleet

HEAD="$TUNED2"; [ -f "$TUNED2/mtp-tuned-r0of4.safetensors" ] || HEAD=/home/choiceoh/models/st-qwen38-mtp-tuned
before=$(ls "$DUMPS/mtp-inputs" | wc -l)
# -- the target's own text: the train prompts decoded with the tap on ------------------------------------------------
if boot data2 ST_DRAFT_THRESHOLD=off ST_MTP_TUNED="$HEAD"; then
  python3 "$OUT/mtp_requests.py" data "$URL" data2 "$OUT/prompts.jsonl" "$DATA_S" 4 > "$OUT/data2-requests.jsonl"
  tail -1 "$OUT/data2-requests.jsonl" | cut -c1-300
else
  stamp "the data boot failed"
fi
logs data2
# (review, PR #1275) the tap writes on 4,096 rows or its 30 s timer and a stopped container flushes nothing: this
# immediate stop may have dropped the last < 4,096 positions of the boot -- #1285 writes them at a stop
stop_fleet
stamp "tap: $before -> $(ls $DUMPS/mtp-inputs | wc -l) shards, $(du -sh $DUMPS/mtp-inputs | cut -f1)"

# -- the runs again, the third training from the second head ----------------------------------------------------------
trained3=0
if [ -f "$WORK/run/head.safetensors" ] && [ "$(ls $DUMPS/mtp-inputs | wc -l)" -gt "$before" ]; then
  rm -rf "$WORK/run2" && mv "$WORK/run" "$WORK/run2"
  stamp "== runs (the prefilled text and the decoded)"
  rm -rf "$WORK/data"
  tune -m engine.profiles.qwen38.mtp_tune data --taps /dumps/mtp-inputs --out /work/data > "$OUT/tune-data3.log" 2>&1
  tail -1 "$OUT/tune-data3.log" | cut -c1-300
  ok=1
  for r in 0 1 2 3; do
    ip=${NODES[$r]}
    node_sh "$ip" "rm -rf $WORK/run $WORK/init && mkdir -p $WORK/init" || ok=0
    if [[ "$SELF_IPS" == *" $ip "* ]]; then
      cp "$WORK/run2/head.safetensors" "$WORK/init/head.safetensors" || ok=0
    else
      rsync -aW --delete "$WORK/data/" "choiceoh@$ip:$WORK/data/" \
        && scp -q "$WORK/run2/head.safetensors" "choiceoh@$ip:$WORK/init/head.safetensors" || ok=0
    fi
  done
  stamp "== third training: $STEPS3 steps from the second head, peak $LR3, eval every $EVERY3 (spread ok=$ok)"
  if [ "$ok" = 1 ]; then
    for r in 3 2 1 0; do node_sh "${NODES[$r]}" "bash $WORK/train_rank2.sh $r $STEPS3 $EVERY3 $LR3" || stamp "rank $r did not start"; done
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
      node_sh "${NODES[$r]}" "docker logs q38tune 2>&1" > "$OUT/tune3-train-rank$r.log" 2>/dev/null
      stamp "rank $r exit $(node_sh "${NODES[$r]}" "docker inspect -f '{{.State.ExitCode}}' q38tune 2>/dev/null")"
    done
    stop_tune
    grep -E '"event": "(eval|saved|end)"' "$OUT/tune3-train-rank0.log" | cut -c1-300
    if [ -f "$WORK/run/head.safetensors" ]; then
      stamp "== export (third head)"
      rm -rf "$WORK/tuned3"
      tune -m engine.profiles.qwen38.mtp_tune export --tuned /work/run/head.safetensors --ckpt /fullckpt --ckpt-meta /ranks \
           --out /work/tuned3 > "$OUT/tune3-export.log" 2>&1
      tail -1 "$OUT/tune3-export.log" | cut -c1-300
      spread=1
      for r in 0 1 2 3; do
        ip=${NODES[$r]}
        f="$WORK/tuned3/mtp-tuned-r${r}of4.safetensors"
        [ -f "$f" ] || { spread=0; break; }
        node_sh "$ip" "mkdir -p $TUNED3" || spread=0
        if [[ "$SELF_IPS" == *" $ip "* ]]; then cp "$f" "$TUNED3/"; else scp -q "$f" "choiceoh@$ip:$TUNED3/" || spread=0; fi
      done
      [ "$spread" = 1 ] && trained3=1
      stamp "spread to four nodes: $spread"
    else
      stamp "the third training did not beat the second head on the held-out windows: nothing new to serve"
    fi
  fi
else
  stamp "no second head or no new shards: no third training"
fi

if [ "$trained3" = 1 ]; then
  evalrun tuned3-off ST_DRAFT_THRESHOLD=off ST_TAP_MTP_INPUTS=0 ST_MTP_TUNED="$TUNED3" || stamp "tuned3-off did not serve"
fi

finish
for i in $(seq 1 120); do
  if curl -fsS -m 3 "$URL/v1/models" 2>/dev/null | grep -q '"id"'; then stamp "production door answers"; break; fi
  sleep 5
done
stamp "lease after: $(lease read | cut -c1-160)"
stamp "production downtime: $(( $(date +%s) - T_DOWN )) s"
