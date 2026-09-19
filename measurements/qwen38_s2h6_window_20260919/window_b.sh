#!/usr/bin/env bash
# The second half of the 2026-09-19 14:29 window, after its first script was stopped (kill -9, so its trap did not
# release the lease): its first boot ("one") was refused -- a lane probe (q38head-0919e) went in the handover's gap --
# and its second ("off") is serving the fixed requests. This waits for that driver, keeps its logs, then boots the
# default ("one") and the FP8 mixers (ST_HC_FP8=1), runs carry S2's lane probe, and ALWAYS stops and releases.
set -uo pipefail
OWNER="session/qwen38-s2h6-0919"
TREE="$HOME/stkernel/.claude/worktrees/win-s2h6-0919"
LOCK=/home/choiceoh/glm53-logs/st-fleet.lock
OUT=/home/choiceoh/glm53-logs/s2h6-window-0919
exec >>"$OUT/window.log" 2>&1
cd "$TREE" || exit 9
lease() { python3 engine/base/fleet_lease.py "$@" --path "$LOCK"; }
SELF_IPS=" $(hostname -I 2>/dev/null) "
node_sh() { local ip=$1; shift; if [[ "$SELF_IPS" == *" $ip "* ]]; then bash -c "$*"; else ssh -o BatchMode=yes -o ConnectTimeout=5 "choiceoh@$ip" "$@" < /dev/null; fi; }
stamp() { echo "[$(date +%H:%M:%S)] $*"; }
probes() { node_sh 10.10.10.4 "docker ps --format '{{.Names}}' | grep -c '^st-probe-' || true" 2>/dev/null; }
released=0
finish() {
  [ "$released" = 1 ] && return
  stamp "stop + release"
  ST_LEASE_OWNER="$OWNER" bash launchers/start-st-qwen38.sh stop 2>&1 | tail -4
  lease release --owner "$OWNER" || true
  released=1
  stamp "lease: $(lease read)"
}
trap finish EXIT
trap 'stamp "signal"; exit 1' HUP INT TERM

held=$(lease read)
case "$held" in
  "session $OWNER "*) stamp "== second script: the lease is still ours ($held)" ;;
  *) stamp "== second script: the lease is not ours ($held) -- not touching the fleet"; released=1; exit 1 ;;
esac
for i in $(seq 1 120); do                       # the stopped script's request driver finishes the "off" boot
  pgrep -f "overlap_requests.py http://10.10.10.2:8000 off" >/dev/null || break
  sleep 5
done
stamp "the off boot's requests: $(wc -l < "$OUT/off-requests.jsonl" 2>/dev/null) lines"
for r in 0 1 2 3; do
  ip=$(echo 10.10.10.2 10.10.10.1 10.10.10.3 10.10.10.4 | cut -d' ' -f$((r + 1)))
  node_sh "$ip" "docker logs st-qwen38 2>&1" > "$OUT/off-rank$r.log" 2>/dev/null
done
grep -E "ready in|STALL|Traceback|illegal|shared expert" "$OUT/off-rank0.log" | tail -6 | cut -c1-240
ST_LEASE_OWNER="$OWNER" bash launchers/start-st-qwen38.sh stop 2>&1 | tail -2

boot() {   # boot <label> [ENV=VALUE ...]
  local label=$1; shift
  stamp "== boot $label ($*)"
  local t0=$(date +%s)
  for i in $(seq 1 60); do                      # a lane ticket that went in a gap: the launcher refuses beside it
    n=$(probes); [ "${n:-0}" = 0 ] && break
    [ $((i % 6)) = 1 ] && stamp "a probe container is up on srv4 ($n) before boot $label; waiting (max 5 min)"
    sleep 5
  done
  env ST_LEASE_OWNER="$OWNER" "$@" bash launchers/start-st-qwen38.sh > "$OUT/$label-launch.log" 2>&1
  local rc=$?
  tail -3 "$OUT/$label-launch.log"
  [ $rc = 0 ] || { stamp "launcher rc=$rc"; return 1; }
  python3 "$OUT/overlap_requests.py" http://10.10.10.2:8000 "$label" "${DOOR_S:-600}" > "$OUT/$label-requests.jsonl"
  local rq=$?
  stamp "boot $label: launcher to last answer $(( $(date +%s) - t0 )) s, requests rc=$rq"
  for r in 0 1 2 3; do
    ip=$(echo 10.10.10.2 10.10.10.1 10.10.10.3 10.10.10.4 | cut -d' ' -f$((r + 1)))
    node_sh "$ip" "docker logs st-qwen38 2>&1" > "$OUT/$label-rank$r.log" 2>/dev/null
  done
  grep -E "ready in|lanes qualified|STALL|Traceback|Error|illegal|qualify|shared expert" "$OUT/$label-rank0.log" | tail -12 | cut -c1-300
  return $rq
}

boot one || stamp "the default boot did not serve"
ST_LEASE_OWNER="$OWNER" bash launchers/start-st-qwen38.sh stop 2>&1 | tail -2
boot fp8 ST_HC_FP8=1 || stamp "the FP8-mixer boot did not serve"
ST_LEASE_OWNER="$OWNER" bash launchers/start-st-qwen38.sh stop 2>&1 | tail -2
stamp "Qwen3.8 stopped"

stamp "== S2 lane ticket"
(cd "$HOME/stkernel/.claude/worktrees/s2-input-reuse-0919" && bash bench/fleet.sh run --gpu --detach qwen38-input-reuse-0919b 15 \
  "Qwen3.8 W4 input reuse at its decode projections (carry S2, PR #1241), inside the operator window" \
  -- bash probes/run_engine_probe.sh probes/engine_kernel_check.py --lanes qwen38_input_reuse) > "$OUT/s2-submit.json" 2>&1
LANELOG=$(python3 - "$OUT/s2-submit.json" <<'PY' 2>/dev/null
import json, sys
for line in open(sys.argv[1]):
    line = line.strip()
    if line.startswith("{") and "log_path" in line:
        print(json.loads(line)["log_path"]); break
PY
)
stamp "S2 submitted: log $LANELOG"
went=0
for i in $(seq 1 18); do
  grep -q "^GO qwen38-input-reuse-0919b" "$LANELOG" 2>/dev/null && { went=1; break; }
  sleep 5
done
if [ "$went" = 1 ]; then
  stamp "S2 went"
  for i in $(seq 1 168); do                     # at most 14 min (a dense compile first)
    grep -q '"event": "summary"\|Traceback' "$LANELOG" 2>/dev/null && break
    sleep 5
  done
  sleep 10
  grep -q '"event": "summary"' "$LANELOG" && stamp "S2 done" || { stamp "S2 not done: cancelling"; bash bench/fleet.sh cancel qwen38-input-reuse-0919b 2>&1 | tail -2; }
else
  stamp "S2 did not GO in 90 s: cancelling"; bash bench/fleet.sh cancel qwen38-input-reuse-0919b 2>&1 | tail -2
fi
cp "$LANELOG" "$OUT/s2-lane.log" 2>/dev/null || true
for i in $(seq 1 24); do n=$(probes); [ "${n:-0}" = 0 ] && break; sleep 5; done
stamp "probe containers on srv4: $(probes)"

finish
stamp "waiting for production's door"
for i in $(seq 1 120); do
  if curl -fsS -m 3 http://10.10.10.2:8000/v1/models 2>/dev/null | grep -q '"id"'; then
    stamp "production door answers: $(curl -fsS -m 3 http://10.10.10.2:8000/v1/models | head -c 160)"; break
  fi
  sleep 5
done
stamp "lease after: $(lease read)"
stamp "WINDOW CLOSED"
