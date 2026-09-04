#!/bin/bash
# dsv4 0731 TP=4 supervisor — boot-start + crash recovery (4 nodes via CRS812 fabric).
# Modeled on ~/dsv4-hybrid-supervisor.sh (the proven TP2 one), adapted for the 4-node
# hy4 stack: srv2 head(rank0) + srv3(1)/srv1(2)/srv4(3) workers, launched by
# ~/start-hy4-tp4.sh. Rollback to the 2-node prod stack:
#   systemctl --user disable --now dsv4-tp4 && systemctl --user enable --now dsv4-hybrid
set -uo pipefail
LAUNCHER=/home/choiceoh/start-hy4-tp4.sh
BASE=http://127.0.0.1:8000
# Served name this supervisor watches for. Default keeps the dsv4 behaviour
# byte-identical; the override exists so the file matches start-hy4-tp4.sh,
# which already takes SERVED_NAME. A supervisor that greps for a name the
# launcher was told to change would report a healthy engine as dead.
SERVED_NAME="${SERVED_NAME:-deepseek-v4-flash}"
WORKERS="10.10.10.3 10.10.10.1 10.10.10.4"
# Long-context ingests block NEW interactive requests until the prefill ends
# (NVIDIA forum #378890, 2x Spark: 145-159s blocked at 262K; our 430K at
# ~2.5k tok/s prefill => ~170s+ worst case). The probe must outlast that or
# the supervisor relaunches a HEALTHY stack mid-ingest. Cost: a hung-but-
# listening corpse now takes up to ~FAILS_NEEDED*(CHAT_TIMEOUT+30)s (~16min)
# to detect — dead ports still fail fast (connection refused).
CHAT_TIMEOUT=300
FAILS_NEEDED=3
BOOT_GRACE=3600          # cold/invalidated kernel+AOT recompile (e.g. after an
                         # overlay-source change) can far exceed 25min; the poll
                         # below exits early the moment the API comes up, so a
                         # long grace only delays declaring a truly hung boot.
fails=0

# Relaunch pacing. A launcher that failed three times running will fail the
# fourth, and an attempt is not free: bash "$LAUNCHER" runs drop_caches on all
# four nodes and creates/destroys containers. So a fixed ~2 min retry turns one
# broken boot into fleet-wide churn -- on srv4, which hosts unrelated tenants
# (nemotron, solarflow, the SolarFlow DB), it evicts their page cache every two
# minutes, indefinitely. Observed 2026-09-04: three relaunches in five minutes
# against a launcher bug (#289), and nothing here would ever have stopped it.
#
# Backoff is a NEXT-ALLOWED-TIME rather than a sleep, so the health probe keeps
# running every 30 s and a stack fixed by hand is still adopted immediately.
# After LAUNCH_HOLD_AFTER consecutive failures it stops relaunching entirely and
# says so once: at that point the launcher needs a human, not another attempt.
launch_fails=0
next_launch_at=0
held_logged=0
LAUNCH_BACKOFF_BASE=60
LAUNCH_BACKOFF_MAX=1800
LAUNCH_HOLD_AFTER=5
log(){ echo "$(date '+%F %T') $*"; }

# --- wait until local docker AND every worker node's docker are up ---
# (user units cannot order against the system docker.service — the unit's
#  After=docker.service is a no-op — so poll the local daemon here too)
wait_for_fleet(){
  local tries=0
  while :; do
    local all=1
    docker info >/dev/null 2>&1 || all=0
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

api_up(){ curl -fsS --max-time 5 "$BASE/v1/models" 2>/dev/null | grep -q "$SERVED_NAME"; }

# Real generation probe — /v1/models stays 200 even when the engine is a corpse
# (2026-07 "위장 건강" lesson). Only a completed chat proves the TP ring is alive.
chat_ok(){
  curl -fsS --max-time "$CHAT_TIMEOUT" "$BASE/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":4}" \
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

# EVERY launcher invocation goes through here. The hold counts ATTEMPTS, not
# launch() returns: launch() returns 0 on api_up alone and /v1/models answers
# 200 from a corpse (see chat_ok), so crediting its return reset the counter
# every round and the hold was unreachable in the one failure mode this pacing
# exists for. Only a confirmed generation (the health branch) clears it. Three
# callers -- the boot, the health loop, the wedge watchdog -- and an uncounted
# one buys a destructive relaunch beyond LAUNCH_HOLD_AFTER.
attempt_launch(){
  launch || true
  launch_fails=$((launch_fails+1))
  _backoff=$(( LAUNCH_BACKOFF_BASE * (1 << (launch_fails - 1)) ))
  [ "$_backoff" -gt "$LAUNCH_BACKOFF_MAX" ] && _backoff=$LAUNCH_BACKOFF_MAX
  next_launch_at=$(( $(date +%s) + _backoff ))
  log "launch attempt $launch_fails/$LAUNCH_HOLD_AFTER done; no another for ${_backoff}s unless it goes healthy"
}

log "=== dsv4-tp4 supervisor start ==="

# --- engine-wedge watchdog + daily cache hygiene (MEASUREMENTS.md 08-09) ---
# Morning incident: one engine degraded monotonically (2,528 -> 2,152 over ~40
# min) and a restart fully recovered it. Prefix-cache churn did NOT reproduce
# at the same bench counts, so the standing suspect is a wedged request (e.g.
# a probe orphaned by a supervisor restart) that continuous batching keeps
# scheduling. Discriminator: requests running but generation_tokens STATIC --
# a legit long generation always advances the counter.
WEDGE_CYCLES="${WEDGE_CYCLES:-30}"          # 30 x 30s = 15 min
CACHE_RESET_HOUR="${CACHE_RESET_HOUR:-04}"  # local hour; empty disables
_gen_last="-1"; _wedge=0; _reset_day=""

wedge_check(){
  local m running gen
  m=$(curl -s -m 5 "$BASE/metrics" 2>/dev/null) || { _wedge=0; return 0; }
  running=$(printf "%s" "$m" | awk '/^vllm:num_requests_running/ {s+=$2} END {printf "%d", s+0}')
  gen=$(printf "%s" "$m" | awk '/^vllm:generation_tokens_total/ {s+=$2} END {printf "%.0f", s+0}')
  if [ "$running" -gt 0 ] && [ "$gen" = "$_gen_last" ]; then
    _wedge=$((_wedge+1))
    if [ "$_wedge" -ge "$WEDGE_CYCLES" ]; then
      log "WEDGE: $running request(s) running, generation static for $((WEDGE_CYCLES*30))s — recycling"
      forensics
      attempt_launch
      _wedge=0; _gen_last="-1"
      return 0
    fi
  else
    _wedge=0
  fi
  _gen_last="$gen"
}

maybe_cache_reset(){
  [ -z "$CACHE_RESET_HOUR" ] && return 0
  local d h
  d=$(date +%Y%m%d); h=$(date +%H)
  [ "$h" = "$CACHE_RESET_HOUR" ] || return 0
  [ "$_reset_day" = "$d" ] && return 0
  # idle = nothing running or waiting, checked twice 20s apart
  _idle(){ curl -s -m 5 "$BASE/metrics" 2>/dev/null | awk '/^vllm:num_requests_running|^vllm:num_requests_waiting/ {s+=$2} END {exit (s>0)}'; }
  _idle || return 0
  sleep 20
  _idle || return 0
  if curl -s -m 15 -X POST "$BASE/reset_prefix_cache" -o /dev/null; then
    _reset_day="$d"
    log "daily idle prefix-cache reset done"
  fi
}

wait_for_fleet
# Adopt an already-healthy stack instead of stomping it (manual launches, restarts).
if api_up && chat_ok; then
  log "existing stack healthy — adopting"
else
  # counted like any relaunch, or a failed boot buys one past the cap; the
  # health loop clears it 30 s later if this one really came up.
  attempt_launch
fi

while :; do
  sleep 30
  if api_up && chat_ok; then
    [ "$fails" -gt 0 ] && log "recovered (fails reset)"
    fails=0
    if [ "$launch_fails" -gt 0 ]; then
      log "stack healthy again -- clearing $launch_fails launch attempt(s)"
      launch_fails=0; next_launch_at=0; held_logged=0
    fi
    wedge_check
    maybe_cache_reset
    continue
  fi
  fails=$((fails+1))
  log "health check failed ($fails/$FAILS_NEEDED)"
  if [ "$fails" -ge "$FAILS_NEEDED" ]; then
    if [ "$launch_fails" -ge "$LAUNCH_HOLD_AFTER" ]; then
      if [ "$held_logged" = 0 ]; then
        log "HELD after $launch_fails relaunches with no healthy stack -- not relaunching."
        log "  The launcher needs a human. This loop keeps probing and will adopt"
        log "  a healthy stack the moment one exists."
        held_logged=1
      fi
      continue
    fi
    _now=$(date +%s)
    [ "$_now" -lt "$next_launch_at" ] && continue
    wait_for_fleet
    attempt_launch   # counted; only the health branch above clears it
  fi
done
