#!/bin/bash
# dsv4 0731 TP=4 supervisor — boot-start + crash recovery (4 nodes via CRS812 fabric).
# Modeled on ~/dsv4-hybrid-supervisor.sh (the proven TP2 one), adapted for the 4-node
# hy4 stack: srv2 head(rank0) + srv3(1)/srv1(2)/srv4(3) workers, launched by
# ~/start-hy4-tp4.sh. Rollback to the 2-node prod stack:
#   systemctl --user disable --now dsv4-tp4 && systemctl --user enable --now dsv4-hybrid
set -uo pipefail
LAUNCHER=/home/choiceoh/start-hy4-tp4.sh
BASE=http://127.0.0.1:8000
WORKERS="10.10.10.3 10.10.10.1 10.10.10.4"
CHAT_TIMEOUT=60
FAILS_NEEDED=3
BOOT_GRACE=1500          # kernel recompile on a cold cache can take ~10min; be generous
fails=0
log(){ echo "$(date '+%F %T') $*"; }

# --- wait until every worker node is actually reachable AND its docker is up ---
wait_for_fleet(){
  local tries=0
  while :; do
    local all=1
    for w in $WORKERS; do
      ssh -n -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=no "choiceoh@$w" \
        'docker info >/dev/null 2>&1' >/dev/null 2>&1 || all=0
    done
    [ "$all" = 1 ] && { log "fleet ready"; return 0; }
    tries=$((tries+1))
    [ $((tries % 12)) -eq 1 ] && log "waiting for workers ($WORKERS)..."
    sleep 10
  done
}

api_up(){ curl -fsS --max-time 5 "$BASE/v1/models" 2>/dev/null | grep -q deepseek-v4-flash; }

# Real generation probe — /v1/models stays 200 even when the engine is a corpse
# (2026-07 "위장 건강" lesson). Only a completed chat proves the TP ring is alive.
chat_ok(){
  curl -fsS --max-time "$CHAT_TIMEOUT" "$BASE/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"ping"}],"max_tokens":4}' \
    2>/dev/null | grep -q '"choices"'
}

forensics(){
  local d=/home/choiceoh/dsv4-tp4-forensics/$(date +%Y%m%d-%H%M%S)
  mkdir -p "$d" 2>/dev/null || return 0
  docker exec hy4 tail -400 /tmp/hy4.log > "$d/head.log" 2>&1 || true
  docker logs --tail=200 hy4 > "$d/head-ctr.log" 2>&1 || true
  for w in $WORKERS; do
    ssh -n -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=no "choiceoh@$w" \
      "docker exec hy4-worker tail -400 /tmp/hy4.log" > "$d/worker-$w.log" 2>&1 || true
  done
  free -m > "$d/free.txt" 2>&1; nvidia-smi -q > "$d/nvidia-smi.txt" 2>&1 || true
  ls -dt /home/choiceoh/dsv4-tp4-forensics/*/ 2>/dev/null | tail -n +11 | xargs -r rm -rf
  log "forensics: $d"
}

launch(){
  log "launching TP4 stack"
  forensics
  fails=0
  bash "$LAUNCHER" >>/home/choiceoh/dsv4-tp4-launch.log 2>&1 || { log "launcher returned nonzero"; return 1; }
  local waited=0
  while [ $waited -lt $BOOT_GRACE ]; do
    api_up && { log "API up after ${waited}s"; return 0; }
    if [ -z "$(docker ps -q --filter name=hy4)" ]; then log "head container died during boot"; return 1; fi
    sleep 15; waited=$((waited+15))
  done
  log "boot grace exceeded (${BOOT_GRACE}s)"; return 1
}

log "=== dsv4-tp4 supervisor start ==="
wait_for_fleet
# Adopt an already-healthy stack instead of stomping it (manual launches, restarts).
if api_up && chat_ok; then log "existing stack healthy — adopting"; else launch || log "initial launch failed; will retry via health loop"; fi

while :; do
  sleep 30
  if api_up && chat_ok; then
    [ "$fails" -gt 0 ] && log "recovered (fails reset)"
    fails=0
    continue
  fi
  fails=$((fails+1))
  log "health check failed ($fails/$FAILS_NEEDED)"
  if [ "$fails" -ge "$FAILS_NEEDED" ]; then
    wait_for_fleet
    launch || log "relaunch failed; retrying next cycle"
  fi
done
