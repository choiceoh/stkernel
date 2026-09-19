#!/usr/bin/env bash
# The operator's Qwen3.8 fleet window (2026-09-19, "해봐" / "프로덕션은 내려"), on srv2 from the prepared worktree:
# ask production to hand over, boot the tree in the launcher's default serving configuration, send the fixed requests,
# boot once more at the operator's K=3, then ALWAYS stop and release so the supervisor brings production back.
set -uo pipefail
OWNER="session/p3-serve-0919"
TREE="$HOME/stkernel/.claude/worktrees/p3-window-0919"
LOCK=/home/choiceoh/glm53-logs/st-fleet.lock
OUT=/home/choiceoh/glm53-logs/p3-window-0919
mkdir -p "$OUT"
cd "$TREE"
lease() { python3 engine/base/fleet_lease.py "$@" --path "$LOCK"; }
SELF_IPS=" $(hostname -I 2>/dev/null) "
node_sh() { local ip=$1; shift; if [[ "$SELF_IPS" == *" $ip "* ]]; then bash -c "$*"; else ssh -o BatchMode=yes -o ConnectTimeout=5 "choiceoh@$ip" "$@"; fi; }
stamp() { echo "[$(date +%H:%M:%S)] $*"; }
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

stamp "tree $(git log --oneline -1 | cut -c1-80)"
stamp "lease before: $(lease read)"
stamp "probe containers on srv4: $(node_sh 10.10.10.4 "docker ps --format '{{.Names}}' | grep -c '^st-probe-' || true")"
lease yield --requester "$OWNER" --kind session --pid $$ --host "$(hostname -s)" --est-minutes 30 \
      --note "operator window: serve latest origin (Qwen3.8, main + #1225), default config then K=3" >/dev/null
stamp "yield asked"
for i in $(seq 1 100); do                       # the quiet gate hands over in seconds when nothing is outstanding
  held=$(lease read)
  case "$held" in *"$OWNER"*) break ;; esac
  sleep 3
done
case "$held" in
  *"$OWNER"*) stamp "handed over: $held" ;;
  *) stamp "production did not hand over in 5 min: $held"; lease withdraw-yield --requester "$OWNER"; released=1; exit 1 ;;
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
  env ST_LEASE_OWNER="$OWNER" "$@" bash launchers/start-st-qwen38.sh > "$OUT/$label-launch.log" 2>&1
  local rc=$?
  tail -3 "$OUT/$label-launch.log"
  [ $rc = 0 ] || { stamp "launcher rc=$rc"; return 1; }
  python3 "$OUT/window_requests.py" http://10.10.10.2:8000 "$label" "${DOOR_S:-600}" | tee "$OUT/$label-requests.jsonl"
  local rq=${PIPESTATUS[0]}
  stamp "boot $label: launcher to last answer $(( $(date +%s) - t0 )) s, requests rc=$rq"
  for r in 0 1 2 3; do
    ip=$(echo 10.10.10.2 10.10.10.1 10.10.10.3 10.10.10.4 | cut -d' ' -f$((r + 1)))
    node_sh "$ip" "docker logs st-qwen38 2>&1" > "$OUT/$label-rank$r.log" 2>/dev/null
  done
  grep -E "ready in|lanes qualified|STALL|Traceback|Error|illegal|qualify" "$OUT/$label-rank0.log" | tail -12 | cut -c1-300
  return $rq
}

boot default || stamp "default boot did not serve"
ST_LEASE_OWNER="$OWNER" bash launchers/start-st-qwen38.sh stop 2>&1 | tail -2
DOOR_S=420 boot k3 ST_SPEC_K=3 || stamp "K=3 boot did not serve"
finish
stamp "production down since handover: $(( $(date +%s) - T_DOWN )) s so far; waiting for its door"
for i in $(seq 1 120); do
  if curl -fsS -m 3 http://10.10.10.2:8000/v1/models 2>/dev/null | grep -q '"id"'; then
    stamp "production door answers: $(curl -fsS -m 3 http://10.10.10.2:8000/v1/models | head -c 200)"; break
  fi
  sleep 5
done
stamp "lease after: $(lease read)"
stamp "production downtime: $(( $(date +%s) - T_DOWN )) s"
