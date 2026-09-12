#!/usr/bin/env bash
# ST engine (GLM-5.3, TP=4) supervisor: boot-start + crash recovery, modelled on dsv4-tp4-supervisor.sh.
# Runs on the head node (srv2 = rank 0). Health = a real 4-token chat completion every 30 s (a listening
# door is not health: the TP ring can be dead behind a live socket); FAILS_NEEDED misses in a row ->
# forensics (docker logs of all four ranks, free, nvidia-smi) -> stop -> start. Relaunch pacing is a
# next-allowed-time with exponential backoff and a hard hold after LAUNCH_HOLD_AFTER attempts: a boot that
# fails five times needs a person, not a sixth attempt. The fleet lease (glm53-logs/st-fleet.lock, the
# ONE file every caller reads) is this loop's while it runs, as kind `production`: `start` takes it,
# `stop` releases it, and a lease of any other kind -- a ticket the queue granted, a session's boot --
# is a window this loop waits out without consuming a launch attempt. The queue asks this holder to
# hand over only through the quiet gate, and hands the fleet back by releasing when no ticket waits.
#
#   systemctl --user enable --now st-glm53      # launchers/st-glm53.service (this script)
#   ST_SUPERVISOR_ONCE=1 bash st-glm53-supervisor.sh   # one probe cycle, no launching: what would it do?
set -u
REPO=${ST_REPO:-/home/choiceoh/st-engine}                  # the rsynced tree on the head (start-st-glm53.sh's ENGINE_DIR)
LAUNCHER=${ST_LAUNCHER:-$REPO/launchers/start-st-glm53.sh}
BASE=${ST_BASE:-http://127.0.0.1:8000}
MODEL=${ST_MODEL:-glm-5.3-flash}
NODES=(10.10.10.2 10.10.10.1 10.10.10.3 10.10.10.4)
NAME=st-glm53
LOCK=${FLEET_LEASE_PATH:-/home/choiceoh/glm53-logs/st-fleet.lock}   # the one lease file (launchers/lib/fleet-lease.sh)
LEGACY_LOCK=/home/choiceoh/st-fleet.lock                            # older launchers wrote here
PROD_OWNER=production/$(hostname -s)/$$                             # this loop's own boots; production by KIND across restarts
CHAT_TIMEOUT=${CHAT_TIMEOUT:-300}       # a long ingest blocks new requests until its prefill ends: outlast it
FAILS_NEEDED=${FAILS_NEEDED:-3}
BOOT_GRACE=${BOOT_GRACE:-1800}          # cold JIT (triton/tilelang/DeepGEMM/MLA/CuTe-DSL) on four nodes
LAUNCH_BACKOFF_BASE=60
LAUNCH_BACKOFF_MAX=1800
LAUNCH_HOLD_AFTER=5
FORENSICS=${ST_FORENSICS:-/home/choiceoh/glm53-logs/st-forensics}
log(){ echo "$(date '+%F %T') $*"; }
SELF_IPS=" $(hostname -I 2>/dev/null) "                 # this loop runs on rank 0's node, which cannot ssh to itself
node_sh(){ local ip=$1; shift
  case "$SELF_IPS" in *" $ip "*) bash -c "$*" </dev/null; return ;; esac
  ssh -n -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new "choiceoh@$ip" "$@"; }

door_up(){ curl -fsS --max-time 5 "$BASE/v1/models" 2>/dev/null | grep -q "\"$MODEL\""; }
# A handover is not a failure. The engine refuses new work ON PURPOSE while it finishes the rows
# it holds, parks them where the next boot finds them (D16) and lets the lease go; `base/serve`
# answers /v1/models with an empty catalog and 503 "draining" for exactly this reader. Without
# this the drain looks like three dead health checks, and the relaunch that follows races the
# next holder -- in the window where the lease is released but not yet taken, `fleet_taken`
# cannot see it either. No -f: the body of a 503 is the whole point.
handing_over(){ curl -sS --max-time 5 "$BASE/v1/models" 2>/dev/null | grep -q '"status": *"draining"'; }
chat_ok(){
  curl -fsS --max-time "$CHAT_TIMEOUT" "$BASE/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":4,\"chat_template_kwargs\":{\"thinking\":false}}" \
    2>/dev/null | grep -q '"choices"'
}
containers_up(){
  local ip n
  for ip in "${NODES[@]}"; do
    n=$(node_sh "$ip" "docker ps -q --filter name=^$NAME\$ | wc -l" 2>/dev/null || echo 0)
    [ "$n" = 1 ] || return 1
  done
}
# The lease module runs on the head, where the file and its evidence (docker) are: locally when
# this loop is the head (it cannot ssh to itself), piped over ssh otherwise.
lease_at(){
  local path=$1; shift
  case "$SELF_IPS" in
    *" ${NODES[0]} "*) python3 "$REPO/engine/base/fleet_lease.py" "$@" --path "$path" ;;
    *) local quoted; printf -v quoted '%q ' "$@"
       ssh -o BatchMode=yes -o ConnectTimeout=8 "choiceoh@${NODES[0]}" "python3 - $quoted --path $path" < "$REPO/engine/base/fleet_lease.py" ;;
  esac
}
lease_head(){ lease_at "$LOCK" "$@"; }
fleet_taken(){   # someone else's fleet: a ticket's or a session's lease, an older lock, another stack -- never fight it
  local ip busy held who rc
  # an older launcher's lock at the older path: judged as it always was, by the container it names
  held=$(node_sh "${NODES[0]}" "cat $LEGACY_LOCK 2>/dev/null || true") || { echo "head unreachable"; return 0; }
  if [ -n "$held" ]; then
    lease_at "$LEGACY_LOCK" owner --container "$NAME" >/dev/null 2>&1 || { echo "lock: $held"; return 0; }
  fi
  # The ONE lease file, judged by KIND. Production's own lease is not "taken" whichever supervisor
  # pid took it (this loop restarts and adopts; a lease from before kinds naming st-glm53 is
  # production's too); a live lease of any other kind is a window the queue granted or a session's
  # boot; a stale one is nobody's. Reading only the legacy path here left every ticket's boot
  # invisible to this loop, whose crash recovery then evicted it (2026-09-12). An answer this loop
  # could not get is not free (D3).
  who=$(lease_head taken --kind production 2>/dev/null); rc=$?
  case "$rc" in
    0) echo "lease: $who"; return 0 ;;
    1) ;;
    *) echo "lease unreadable (rc=$rc)"; return 0 ;;
  esac
  for ip in "${NODES[@]}"; do
    busy=$(node_sh "$ip" "docker ps --format '{{.Names}}'" 2>/dev/null) || { echo "$ip unreachable"; return 0; }
    busy=$(printf '%s\n' "$busy" | grep -E '^(glm53|q38|vllm|st-)' | grep -vx "$NAME" || true)
    [ -z "$busy" ] || { echo "$ip:$busy"; return 0; }
  done
  return 1
}
forensics(){
  local d=$FORENSICS/$(date +%Y%m%d-%H%M%S) ip r
  mkdir -p "$d" 2>/dev/null || return 0
  for r in 0 1 2 3; do ip=${NODES[$r]}; node_sh "$ip" "docker logs --tail=400 $NAME" > "$d/rank$r-$ip.log" 2>&1 || true; done
  free -m > "$d/free.txt" 2>&1; nvidia-smi -q > "$d/nvidia-smi.txt" 2>&1 || true
  curl -s -m 5 "$BASE/metrics" > "$d/metrics.txt" 2>/dev/null || true
  ls -dt "$FORENSICS"/*/ 2>/dev/null | tail -n +11 | xargs -r rm -rf
  log "forensics: $d"
}
launch(){
  local taken waited=0
  if taken=$(fleet_taken); then log "fleet taken ($taken): not launching"; return 1; fi
  log "launching the ST fleet (production lease as $PROD_OWNER)"
  ST_LEASE_KIND=production LEASE_OWNER_PRODUCTION=$PROD_OWNER bash "$LAUNCHER" stop >>"$FORENSICS/launch.log" 2>&1 \
    || { log "stop refused; preserving fleet owner"; return 1; }
  ST_LEASE_KIND=production LEASE_OWNER_PRODUCTION=$PROD_OWNER bash "$LAUNCHER" >>"$FORENSICS/launch.log" 2>&1 \
    || { log "launcher returned nonzero (see $FORENSICS/launch.log)"; return 1; }
  while [ $waited -lt $BOOT_GRACE ]; do
    door_up && { log "door up after ${waited}s"; return 0; }
    containers_up || { log "a rank died during boot"; forensics; return 1; }
    sleep 15; waited=$((waited+15))
  done
  log "boot grace exceeded (${BOOT_GRACE}s)"; forensics; return 1
}
launch_fails=0; next_launch_at=0; held_logged=0; fails=0
attempt_launch(){
  local taken
  if taken=$(fleet_taken); then log "fleet taken ($taken): waiting without consuming a launch attempt"; return; fi
  if handing_over; then log "engine is handing the fleet over: waiting without consuming a launch attempt"; return; fi
  launch || true
  launch_fails=$((launch_fails+1))
  local backoff=$(( LAUNCH_BACKOFF_BASE * (1 << (launch_fails - 1)) ))
  [ "$backoff" -gt "$LAUNCH_BACKOFF_MAX" ] && backoff=$LAUNCH_BACKOFF_MAX
  next_launch_at=$(( $(date +%s) + backoff ))
  log "launch attempt $launch_fails/$LAUNCH_HOLD_AFTER done; none before ${backoff}s unless it goes healthy"
}

mkdir -p "$FORENSICS"
if [ "${ST_SUPERVISOR_ONCE:-0}" = 1 ]; then
  if taken=$(fleet_taken); then echo "fleet taken: $taken"
  elif handing_over; then echo "handing over: waiting"
  elif containers_up && door_up && chat_ok; then echo "healthy"
  else echo "would launch (containers_up=$(containers_up && echo yes || echo no) door_up=$(door_up && echo yes || echo no))"; fi
  exit 0
fi
log "=== st-glm53 supervisor start ==="
if containers_up && door_up && chat_ok; then
  log "existing ST fleet healthy -- adopting"
else
  attempt_launch
fi
while :; do
  sleep 30
  if containers_up && door_up && chat_ok; then
    [ "$fails" -gt 0 ] && log "recovered (fails reset)"
    fails=0
    if [ "$launch_fails" -gt 0 ]; then log "healthy again -- clearing $launch_fails launch attempt(s)"; launch_fails=0; next_launch_at=0; held_logged=0; fi
    continue
  fi
  if handing_over; then
    log "engine is handing the fleet over: waiting, this is not a failed health check"
    continue
  fi
  fails=$((fails+1))
  log "health check failed ($fails/$FAILS_NEEDED)"
  [ "$fails" -ge "$FAILS_NEEDED" ] || continue
  if [ "$launch_fails" -ge "$LAUNCH_HOLD_AFTER" ]; then
    [ "$held_logged" = 1 ] || { log "HELD after $launch_fails relaunches with no healthy fleet -- a person is needed; still probing, will adopt a healthy fleet"; held_logged=1; }
    continue
  fi
  [ "$(date +%s)" -lt "$next_launch_at" ] && continue
  if taken=$(fleet_taken); then
    # Someone else's fleet -- a ticket's window, a session's boot -- is not a failure of ours: no
    # forensics (each dump evicts one of the ten kept, and on 2026-09-13 02:49-02:55 every kept
    # dump was a "fleet taken" snapshot, the last real failure's evidence gone), no launch
    # attempt, one line per holder.
    taken_key=${taken%% since *}
    [ "$taken_key" = "$taken_logged" ] || { log "fleet taken ($taken): waiting -- no forensics, no launch attempt"; taken_logged=$taken_key; }
    continue
  fi
  taken_logged=
  forensics
  attempt_launch
done
