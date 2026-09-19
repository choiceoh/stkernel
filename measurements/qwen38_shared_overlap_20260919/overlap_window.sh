#!/usr/bin/env bash
# The operator's fleet window for carry M5 (2026-09-19, "플릿 잡고 해"), on srv2 from a worktree at origin/main:
# prebuild while production serves, ask it to hand over, boot the launcher's default serving configuration (K=1,
# one-shot, four rows) twice -- the shared expert unforked, then forked (ST_SHARED_OVERLAP=all) -- send the same
# requests to each, then run carry H3's single-GPU probe while srv4 has room, and ALWAYS stop and release so the
# supervisor brings production back. The procedure is measurements/qwen38_serve_window_20260919/window_run.sh's.
set -uo pipefail
OWNER="session/qwen38-overlap-0919"
TREE="$HOME/stkernel/.claude/worktrees/win-overlap-0919"
LOCK=/home/choiceoh/glm53-logs/st-fleet.lock
OUT=/home/choiceoh/glm53-logs/overlap-window-0919
mkdir -p "$OUT"
exec >>"$OUT/window.log" 2>&1
cd "$TREE" || exit 9
lease() { python3 engine/base/fleet_lease.py "$@" --path "$LOCK"; }
SELF_IPS=" $(hostname -I 2>/dev/null) "
node_sh() { local ip=$1; shift; if [[ "$SELF_IPS" == *" $ip "* ]]; then bash -c "$*"; else ssh -o BatchMode=yes -o ConnectTimeout=5 "choiceoh@$ip" "$@"; fi; }
stamp() { echo "[$(date +%H:%M:%S)] $*"; }
probes() { node_sh 10.10.10.4 "docker ps --format '{{.Names}}' | grep -c '^st-probe-' || true" 2>/dev/null; }
taken=0
released=0
finish() {
  [ "$released" = 1 ] && return
  if [ "$taken" = 1 ]; then
    stamp "stop + release"
    ST_LEASE_OWNER="$OWNER" bash launchers/start-st-qwen38.sh stop 2>&1 | tail -4
    lease release --owner "$OWNER" || true
    stamp "lease: $(lease read)"
  else
    lease withdraw-yield --requester "$OWNER" >/dev/null 2>&1 || true
    stamp "nothing was taken; any yield request of ours is withdrawn"
  fi
  released=1
}
trap finish EXIT
trap 'stamp "signal"; exit 1' HUP INT TERM

stamp "tree $(git log --oneline -1 | cut -c1-80)"
stamp "lane queue: $(bash bench/fleet.sh status 2>&1 | sed -n '/^queue (/,/^log:/p' | grep -cE '^  [0-9]+\.' ) ticket(s) waiting"
stamp "lease before: $(lease read)"
stamp "prebuild (production still serves)"
timeout 900 bash launchers/start-st-qwen38.sh prebuild > "$OUT/prebuild.log" 2>&1; stamp "prebuild rc=$? ($(tail -1 "$OUT/prebuild.log" | cut -c1-160))"
for i in $(seq 1 60); do                        # the launcher refuses beside a probe container: let a running one end
  n=$(probes); [ "${n:-0}" = 0 ] && break
  [ $((i % 6)) = 1 ] && stamp "a probe container is up on srv4 ($n); waiting"
  sleep 10
done
[ "$(probes)" = 0 ] || { stamp "srv4 still runs a probe after 10 min: not taking the window"; exit 1; }

lease yield --requester "$OWNER" --kind session --pid $$ --host "$(hostname -s)" --est-minutes 25 \
      --note "operator window: carry M5 A/B (shared expert overlap off vs all, launcher defaults) + H3 lane probe" >/dev/null
stamp "yield asked"
for i in $(seq 1 100); do                       # the quiet gate hands over in seconds when nothing is outstanding
  held=$(lease read)
  case "$held" in *"$OWNER"*) break ;; esac
  sleep 3
done
case "$held" in
  *"$OWNER"*) taken=1; stamp "handed over: $held" ;;
  *) stamp "production did not hand over in 5 min: $held"; exit 1 ;;
esac
T_DOWN=$(date +%s)
for i in $(seq 1 60); do                        # production's containers leave before the launcher will start
  busy=""
  for ip in 10.10.10.2 10.10.10.1 10.10.10.3 10.10.10.4; do
    n=$(node_sh "$ip" "docker ps --format '{{.Names}}' | grep -E '^(glm53|q38|vllm|st-)' | tr '\n' ' '" 2>/dev/null)
    [ -n "$n" ] && busy="$busy $ip:$n"
  done
  [ -z "$busy" ] && break
  sleep 3
done
[ -z "$busy" ] || { stamp "containers still up:$busy"; exit 1; }

boot() {   # boot <label> [ENV=VALUE ...]
  local label=$1; shift
  stamp "== boot $label ($*)"
  local t0=$(date +%s)
  for i in $(seq 1 60); do                      # a lane ticket that went in the gap: the launcher refuses beside it
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
  grep -E "ready in|lanes qualified|STALL|Traceback|Error|illegal|qualify|shared" "$OUT/$label-rank0.log" | tail -12 | cut -c1-300
  return $rq
}

boot off || stamp "the unforked boot did not serve"
ST_LEASE_OWNER="$OWNER" bash launchers/start-st-qwen38.sh stop 2>&1 | tail -2
boot all ST_SHARED_OVERLAP=all || stamp "the forked boot did not serve"
ST_LEASE_OWNER="$OWNER" bash launchers/start-st-qwen38.sh stop 2>&1 | tail -2
stamp "Qwen3.8 stopped; production down $(( $(date +%s) - T_DOWN )) s so far"

# carry H3 while srv4 has room: the lane's single-GPU probe (a ticket of the queue; it GOes when there is room)
stamp "== H3 lane ticket"
bash bench/fleet.sh run --gpu --detach qwen38-mix-tiles-0919c 15 "Qwen3.8 mix_mean hidden tiles on a GB10 (carry H3), inside the M5 window" \
  -- bash probes/run_engine_probe.sh probes/engine_kernel_check.py --lanes qwen38_mix_tiles > "$OUT/h3-submit.json" 2>&1
H3LOG=$(python3 - "$OUT/h3-submit.json" <<'PY' 2>/dev/null
import json, sys
for line in open(sys.argv[1]):
    line = line.strip()
    if line.startswith("{") and "log_path" in line:
        print(json.loads(line)["log_path"]); break
PY
)
stamp "H3 submitted: log $H3LOG"
went=0
for i in $(seq 1 18); do                        # 90 s to GO; if the lane will not start here, the window does not wait
  grep -q "^GO qwen38-mix-tiles-0919c" "$H3LOG" 2>/dev/null && { went=1; break; }
  sleep 5
done
if [ "$went" = 1 ]; then
  stamp "H3 went"
  for i in $(seq 1 120); do                     # at most 10 min
    grep -q '"event": "summary"\|Traceback' "$H3LOG" 2>/dev/null && break
    sleep 5
  done
  sleep 10
  grep -q '"event": "summary"' "$H3LOG" && stamp "H3 done" || { stamp "H3 not done in 10 min: cancelling"; bash bench/fleet.sh cancel qwen38-mix-tiles-0919c 2>&1 | tail -2; }
else
  stamp "H3 did not GO in 90 s: cancelling"; bash bench/fleet.sh cancel qwen38-mix-tiles-0919c 2>&1 | tail -2
fi
cp "$H3LOG" "$OUT/h3-lane.log" 2>/dev/null || true
for i in $(seq 1 24); do n=$(probes); [ "${n:-0}" = 0 ] && break; sleep 5; done
stamp "probe containers on srv4: $(probes)"

finish
stamp "production down since handover: $(( $(date +%s) - T_DOWN )) s so far; waiting for its door"
for i in $(seq 1 120); do
  if curl -fsS -m 3 http://10.10.10.2:8000/v1/models 2>/dev/null | grep -q '"id"'; then
    stamp "production door answers: $(curl -fsS -m 3 http://10.10.10.2:8000/v1/models | head -c 160)"; break
  fi
  sleep 5
done
stamp "lease after: $(lease read)"
stamp "production downtime: $(( $(date +%s) - T_DOWN )) s"
stamp "WINDOW CLOSED"
