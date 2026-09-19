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
#
# Which MODEL production serves is launchers/st_production.py's selection (glm53 while nothing chose):
# the loop reads it every cycle, and when it moves -- the Deneb app, `st_production.py select` -- it
# lets the door go quiet (at most ST_SWITCH_QUIET_S), stops the fleet it runs and boots the chosen
# profile under the same production lease, publishing each step to st-production-state.json. A chosen
# model that cannot boot LAUNCH_HOLD_AFTER times in a row hands production back to glm53 rather than
# holding the fleet down: a model production cannot serve is not production.
#
#   systemctl --user enable --now st-glm53      # launchers/st-glm53.service (this script)
#   ST_SUPERVISOR_ONCE=1 bash st-glm53-supervisor.sh   # one probe cycle, no launching: what would it do?
set -u
REPO=${ST_REPO:-/home/choiceoh/st-engine}                  # the rsynced tree on the head (start-st-glm53.sh's ENGINE_DIR)
BASE=${ST_BASE:-http://127.0.0.1:8000}
NODES=(10.10.10.2 10.10.10.1 10.10.10.3 10.10.10.4)
production(){ python3 "$REPO/launchers/st_production.py" "$@"; }
# use_profile sets the three names the whole loop reads: the container its census and forensics look for,
# the model its door and its chat must answer to, and the launcher it boots. ST_LAUNCHER and ST_MODEL keep
# meaning glm53's (hand runs, the tests); ST_LAUNCHER_<PROFILE> overrides another profile's launcher. A
# profile st_production.py cannot describe is glm53: the default is what every box served before a choice.
use_profile(){
  local p=$1 name model launcher var
  if ! { name=$(production field "$p" container 2>/dev/null) && model=$(production field "$p" model 2>/dev/null) \
         && launcher=$(production field "$p" launcher 2>/dev/null); }; then
    p=glm53; name=st-glm53; model=glm-5.3-flash; launcher=start-st-glm53.sh
  fi
  PROFILE=$p; NAME=$name; MODEL=$model; LAUNCHER=$REPO/launchers/$launcher
  if [ "$p" = glm53 ]; then
    MODEL=${ST_MODEL:-$MODEL}; LAUNCHER=${ST_LAUNCHER:-$LAUNCHER}
  else
    var=ST_LAUNCHER_${p^^}; LAUNCHER=${!var:-$LAUNCHER}
  fi
}
wanted_profile(){ production selected 2>/dev/null || echo glm53; }
use_profile "$(wanted_profile)"
SERVING=                 # the profile this loop last saw answer a chat; empty while nothing of ours serves
PUBLISHED=
publish(){  # <phase> [detail]: st-production-state.json, written when it changes -- not every 30 s
  local key="$SERVING|$PROFILE|$*"
  [ "$key" = "$PUBLISHED" ] && return 0
  PUBLISHED=$key
  production state "${SERVING:-none}" "$PROFILE" "$@" >/dev/null 2>&1 || true
}
SWITCH_QUIET_S=${ST_SWITCH_QUIET_S:-120}   # a switch lets the rows being served finish, for this long at most
LOCK=${FLEET_LEASE_PATH:-/home/choiceoh/glm53-logs/st-fleet.lock}   # the one lease file (launchers/lib/fleet-lease.sh)
LEGACY_LOCK=/home/choiceoh/st-fleet.lock                            # older launchers wrote here
PROD_OWNER=production/$(hostname -s)/$$                             # this loop's own boots; production by KIND across restarts
CHAT_TIMEOUT=${CHAT_TIMEOUT:-300}       # a long ingest blocks new requests until its prefill ends: outlast it
FAILS_NEEDED=${FAILS_NEEDED:-3}
BOOT_GRACE=${BOOT_GRACE:-1800}          # cold JIT (triton/tilelang/DeepGEMM/MLA/CuTe-DSL) on four nodes
LAUNCH_BACKOFF_BASE=${ST_LAUNCH_BACKOFF_BASE:-60}
LAUNCH_BACKOFF_MAX=1800
LAUNCH_HOLD_AFTER=${ST_LAUNCH_HOLD_AFTER:-5}
FORENSICS=${ST_FORENSICS:-/home/choiceoh/glm53-logs/st-forensics}
FLEET_DIR=${FLEET_DIR:-/home/choiceoh/glm53-logs/fleet}   # the queue's files; its activity clock lives here (bench/fleet_idle.py)
RESTORE_GRACE=${ST_RESTORE_GRACE_S:-}      # set: a constant grace. Unset: the queue's own pace (restore-grace.json + window.json), floor 300
LOOP_SLEEP=${ST_SUPERVISOR_SLEEP:-30}; BOOT_POLL=${ST_BOOT_POLL:-15}; MAX_LOOPS=${ST_SUPERVISOR_LOOPS:-0}   # tests shorten and bound the loop
# A boot starts with an empty prefix cache, and nothing has ever filled it: `/v1/prefix/warm` and
# probes/st_prefix_warm.py have existed all along with no caller. Every relaunch -- three on
# 2026-09-16 -- made the first conversation prefill the prompt every conversation shares.
# One JSONL line per request body ({"messages": [...]}, {"prompt": ...} or {"ids": [...]}).
# Pinned, so `_victim` keeps them behind everything else until /v1/prefix/unpin. Missing file: skip.
WARM_FILE=${ST_WARM_FILE-/home/choiceoh/glm53-logs/st-warm.jsonl}   # `:-` would read ST_WARM_FILE= as unset; empty is off
WARM_TIMEOUT=${ST_WARM_TIMEOUT:-600}
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
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":4,\"retain\":false,\"chat_template_kwargs\":{\"thinking\":false,\"enable_thinking\":false}}" \
    2>/dev/null | grep -q '"choices"'
}
containers_up(){
  local ip n
  for ip in "${NODES[@]}"; do
    n=$(node_sh "$ip" "docker ps -q --filter name=^$NAME\$ | wc -l" 2>/dev/null || echo 0)
    [ "$n" = 1 ] || return 1
  done
}
HEALTH_DETAIL=
health(){  # 0 when a real chat answers -- the file's first line: a listening door is not health, and a
  # container census is not either (a node whose ssh stalls is not a dead ring). What each probe
  # said is kept for the log: "health check failed (n/3)" alone told nothing on 2026-09-13.
  local c d ch=no
  containers_up && c=yes || c=no
  door_up && d=yes || d=no
  [ "$d" = yes ] && chat_ok && ch=yes
  HEALTH_DETAIL="containers=$c door=$d chat=$ch"
  [ "$ch" = yes ]
}
head_age(){  # seconds since the head's container started; 1 when there is none
  local started; started=$(docker inspect -f '{{.State.StartedAt}}' "$NAME" 2>/dev/null) && [ -n "$started" ] || return 1
  echo $(( $(date +%s) - $(date -d "$started" +%s 2>/dev/null || echo 0) ))
}
booting_fleet(){  # every rank up, no door yet, the head's container younger than a boot takes: a boot in progress
  # -- ours, started by deploy-watch a moment before this loop, or the last launch's. On 2026-09-13
  # 03:54 this loop's first act after a deploy was to stop that boot and start another; and at
  # 03:59, 04:04 and 04:11 it relaunched a fleet whose door was up because the first chat after
  # a cold boot had not answered yet. Twenty minutes of flapping, four boots for one.
  containers_up || return 1
  door_up && return 1
  local age; age=$(head_age) || return 1
  [ "$age" -lt "$BOOT_GRACE" ]
}
queue_active_ago(){  # seconds since the queue last enqueued, granted or released (its activity clock); empty when it has none
  python3 - "$FLEET_DIR/idle-recovery.json" <<'PY'
import json, sys, time
try:
    t = json.load(open(sys.argv[1])).get("updated_at")
    print(int(time.time() - float(t)) if t else "")
except Exception:
    print("")
PY
}
queue_grace(){  # how long a free fleet waits before production returns: the queue's pace (bench/fleet_pace.py), a window, or the constant
  [ -z "$RESTORE_GRACE" ] || { echo "$RESTORE_GRACE"; return; }
  python3 - "$FLEET_DIR" <<'PY'
import json, sys, time
from pathlib import Path
d, now, grace, window = Path(sys.argv[1]), time.time(), 300, 0
try:
    grace = int(json.loads((d / "restore-grace.json").read_text()).get("seconds", 300))
except Exception:
    pass
try:
    w = json.loads((d / "window.json").read_text()); window = max(0, int(w.get("until", 0) - now))
except Exception:
    pass
print(max(grace, window, 300))
PY
}
warm_cache(){  # <what>: fill the prefix cache the boot started empty, after health and never before it
  [ -n "$WARM_FILE" ] && [ -f "$WARM_FILE" ] || return 0
  local what=$1 out rc=0
  # Never a boot failure. The fleet is already healthy by the time this runs; a cache that did not
  # warm is slower, not broken, and a warm that hangs must not hold the supervisor's loop.
  out=$(timeout "$WARM_TIMEOUT" python3 "$REPO/probes/st_prefix_warm.py" "$WARM_FILE" --url "$BASE" --pin --timeout "$WARM_TIMEOUT" 2>&1) || rc=$?
  out=$(echo "$out" | tail -1 | sed 's/^ *//')
  if [ "$rc" = 0 ]; then
    log "$what: prefix cache warmed -- $out"
  else
    log "$what: prefix warm did not finish (rc=$rc): $out"
  fi
  return 0
}

wait_for_health(){  # <what>: the door, then a real chat -- a listening door is not health (the file's first line)
  local what=$1 waited=0 door_seen=0
  while [ "$waited" -lt "$BOOT_GRACE" ]; do
    if door_up; then
      [ "$door_seen" = 1 ] || { door_seen=1; log "$what: door up after ${waited}s"; }
      if chat_ok; then
        log "$what: healthy after ${waited}s (a chat answered)"; fails=0; SERVING=$PROFILE; publish serving
        warm_cache "$what"; return 0
      fi
    fi
    containers_up || { log "$what: a rank died during boot"; forensics; return 1; }
    sleep "$BOOT_POLL"; waited=$((waited+BOOT_POLL))
  done
  log "$what: boot grace exceeded (${BOOT_GRACE}s)"; forensics; return 1
}
adopt_boot(){ log "a fleet is booting (head container $(head_age || echo '?')s old, no door yet): adopting it, not relaunching it"; wait_for_health adoption; }
wait_reason(){  # key<TAB>text: why not to launch right now; 1 when there is no reason
  local taken ago
  if taken=$(fleet_taken); then printf 'taken:%s\t%s\n' "${taken%% since *}" "fleet taken ($taken)"; return 0; fi
  if handing_over; then printf 'handover\tengine is handing the fleet over\n'; return 0; fi
  if booting_fleet; then printf 'booting\ta fleet is booting\n'; return 0; fi
  ago=$(queue_active_ago); local grace; grace=$(queue_grace)
  if [ -n "$ago" ] && [ "$ago" -lt "$grace" ]; then
    printf 'grace\tfleet free, but the queue was active %ss ago: %ss of quiet queue before production is restored (its next ticket takes the fleet as it is)\n' "$ago" "$grace"; return 0
  fi
  return 1
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
# The profile's launcher, under the production lease and the profile's own environment: systemd hands this
# loop st-glm53.env, and its RANKS_DIR must not reach another model's launch (st_production.py env).
run_launcher(){
  ( eval "$(production env "$PROFILE" 2>/dev/null)"
    ST_LEASE_KIND=production LEASE_OWNER_PRODUCTION=$PROD_OWNER exec bash "$LAUNCHER" "$@" )
}
launch(){
  local taken
  if taken=$(fleet_taken); then log "fleet taken ($taken): not launching"; return 1; fi
  log "launching the ST fleet (production lease as $PROD_OWNER): $PROFILE, $MODEL"
  publish launching
  run_launcher stop >>"$FORENSICS/launch.log" 2>&1 \
    || { log "stop refused; preserving fleet owner"; return 1; }
  run_launcher >>"$FORENSICS/launch.log" 2>&1 \
    || { log "launcher returned nonzero (see $FORENSICS/launch.log)"; return 1; }
  wait_for_health launch
}
launch_fails=0; next_launch_at=0; held_logged=0; fails=0; wait_logged=
attempt_launch(){
  local reason key text
  if reason=$(wait_reason); then
    IFS=$'\t' read -r key text <<< "$reason"
    case "$key" in
      booting) adopt_boot ;;
      *) log "$text: waiting without consuming a launch attempt"; publish waiting "$text" ;;
    esac
    return
  fi
  launch_counted
}
launch_counted(){  # launch, and count a failure that was ours
  local taken
  if launch; then launch_fails=0; next_launch_at=0; held_logged=0; return; fi
  if taken=$(fleet_taken); then
    # The launcher was refused because the fleet changed hands while we were launching -- a ticket
    # took the lease the launcher's `stop` let go (05:30 and 06:25 on 2026-09-13 counted as attempts
    # 1 and 5 and HELD production). That is a window, not a failed boot.
    log "the fleet was taken while launching ($taken): not a failed attempt"; return
  fi
  launch_fails=$((launch_fails+1))
  local backoff=$(( LAUNCH_BACKOFF_BASE * (1 << (launch_fails - 1)) ))
  [ "$backoff" -gt "$LAUNCH_BACKOFF_MAX" ] && backoff=$LAUNCH_BACKOFF_MAX
  next_launch_at=$(( $(date +%s) + backoff ))
  log "launch attempt $launch_fails/$LAUNCH_HOLD_AFTER failed; none before ${backoff}s unless it goes healthy"
}

# ---- which model production serves (launchers/st_production.py) ----
running_profile(){  # the profile whose container runs on the head: the fleet a restart of this loop inherits
  local p c
  for p in $(production profiles 2>/dev/null); do
    c=$(production field "$p" container 2>/dev/null) || continue
    [ -n "$(docker ps -q --filter "name=^$c\$" 2>/dev/null)" ] && { echo "$p"; return 0; }
  done
  return 1
}
wait_quiet(){  # a switch lets the rows being served finish -- the operator asked for it, so not forever
  local waited=0 load
  while [ "$waited" -lt "$SWITCH_QUIET_S" ]; do
    load=$(python3 "$REPO/engine/base/fleet_lease.py" load --metrics-url "$BASE/metrics" 2>/dev/null || echo unknown)
    case "$load" in 0|unknown) return 0 ;; esac   # quiet, or a door that cannot say: nothing to wait for
    sleep 5; waited=$((waited+5))
  done
  log "switch: the door did not go quiet in ${SWITCH_QUIET_S}s; what it serves now is cut"
}
switch_to(){  # <profile>: production moves to another model
  local want=$1 from=$PROFILE taken
  log "production model: $from -> $want (chosen: $(production show 2>/dev/null | python3 -c 'import json,sys
s = json.load(sys.stdin).get("selection") or {}
print((s.get("by") or "?") + (" -- " + s["note"] if s.get("note") else ""))' 2>/dev/null || echo '?'))"
  if taken=$(fleet_taken); then
    # Someone else's window: nothing of ours runs to stop. Production's next boot is simply the new model.
    use_profile "$want"; SERVING=; launch_fails=0; next_launch_at=0; held_logged=0; fails=0
    log "the fleet is taken ($taken): production boots $PROFILE when it comes back"
    publish waiting "fleet taken (${taken%% since *})"
    return 0
  fi
  publish switching "$from -> $want"
  if containers_up || door_up; then
    wait_quiet
    if ! run_launcher stop >>"$FORENSICS/launch.log" 2>&1; then
      log "switch: $from's stop was refused; production stays on $from and the next cycle asks again"
      publish switching "$from's stop was refused"
      return 1
    fi
  fi
  use_profile "$want"; SERVING=; launch_fails=0; next_launch_at=0; held_logged=0; fails=0; wait_logged=
  launch_counted     # now: the queue's restore grace is for a fleet a window let go, and this one was ours
}

mkdir -p "$FORENSICS"
if [ "${ST_SUPERVISOR_ONCE:-0}" = 1 ]; then
  if taken=$(fleet_taken); then echo "fleet taken: $taken"
  elif handing_over; then echo "handing over: waiting"
  elif health; then echo "healthy"
  elif booting_fleet; then echo "booting: would adopt (head container $(head_age || echo '?')s old, no door yet)"
  elif ago=$(queue_active_ago) && [ -n "$ago" ] && [ "$ago" -lt "$(queue_grace)" ]; then echo "fleet free, queue active ${ago}s ago: would wait (grace $(queue_grace)s)"
  else echo "would launch (containers_up=$(containers_up && echo yes || echo no) door_up=$(door_up && echo yes || echo no))"; fi
  exit 0
fi
log "=== st-glm53 supervisor start (production model: $PROFILE) ==="
if running=$(running_profile) && [ "$running" != "$PROFILE" ]; then
  # Inherit what runs; the loop's first cycle then switches to the selection the proper way.
  log "the head runs $running while $PROFILE is selected: adopting $running first"
  use_profile "$running"
fi
if health; then
  log "existing ST fleet healthy -- adopting"
  # An adopted fleet's prefix cache is as empty as a freshly booted one's -- nothing has
  # ever filled it. warm_cache lives inside wait_for_health, which a start that is already
  # healthy never enters, so the one path that skipped the warm was the common one. health
  # just passed, so this returns on the first poll; its return is the loop's business, not
  # a launch decision.
  wait_for_health adopted
else
  attempt_launch
fi
loops=0
while :; do
  sleep "$LOOP_SLEEP"
  if [ "$MAX_LOOPS" -gt 0 ]; then loops=$((loops+1)); [ "$loops" -le "$MAX_LOOPS" ] || { log "loop bound reached ($MAX_LOOPS)"; exit 0; }; fi
  want=$(wanted_profile)
  if [ "$want" != "$PROFILE" ]; then switch_to "$want"; continue; fi
  if health; then
    [ "$fails" -gt 0 ] && log "recovered (fails reset)"
    fails=0; wait_logged=; SERVING=$PROFILE; publish serving
    if [ "$launch_fails" -gt 0 ]; then log "healthy again -- clearing $launch_fails launch attempt(s)"; launch_fails=0; next_launch_at=0; held_logged=0; fi
    continue
  fi
  if reason=$(wait_reason); then
    # Someone else's fleet, a handover, our own boot in progress, a queue that was active a moment
    # ago: none is a failure of ours -- no forensics (each dump evicts one of the ten kept, and on
    # 2026-09-13 02:49-02:55 every kept dump was a "fleet taken" snapshot), no launch attempt, one
    # line per reason. A boot in progress is adopted and waited for.
    IFS=$'\t' read -r key text <<< "$reason"
    case "$key" in
      booting) adopt_boot ;;
      *) [ "$key" = "$wait_logged" ] || { log "$text: waiting -- no forensics, no launch attempt"; wait_logged=$key; }
         SERVING=; publish waiting "$text" ;;
    esac
    case "$key" in
      taken:*) # Someone else's window: the boots that failed before it are not this window's. The
               # hold is for a boot that fails five times in a ROW; a foreign window ends the row.
               [ "$launch_fails" = 0 ] || { log "a foreign window resets the launch count ($launch_fails attempt(s) before it)"; launch_fails=0; next_launch_at=0; held_logged=0; } ;;
    esac
    continue
  fi
  wait_logged=
  fails=$((fails+1))
  log "health check failed ($fails/$FAILS_NEEDED): $HEALTH_DETAIL"
  [ "$fails" -ge "$FAILS_NEEDED" ] || continue
  SERVING=
  if [ "$launch_fails" -ge "$LAUNCH_HOLD_AFTER" ]; then
    if [ "$PROFILE" != glm53 ]; then
      # A chosen model that cannot boot hands production back rather than holding the fleet down; the
      # next cycle reads the selection and switches. The chooser reads why in the state file.
      why="$PROFILE did not boot in $launch_fails attempts in a row"
      log "$why: production returns to glm53"
      production select glm53 --by supervisor --note "$why" >/dev/null 2>&1 || true
      publish reverted "$why"
      continue
    fi
    [ "$held_logged" = 1 ] || { log "HELD after $launch_fails relaunches with no healthy fleet -- a person is needed; still probing, will adopt a healthy fleet"; held_logged=1; }
    publish held "$launch_fails relaunches with no healthy fleet"
    continue
  fi
  [ "$(date +%s)" -lt "$next_launch_at" ] && continue
  forensics
  attempt_launch
done
