#!/usr/bin/env bash
# fleet.sh -- turn-taking for the TP4 fleet among sessions (39차, operator:
# "멀티세션간 플릿 테스트 대기 및 순번 같은거 프로그램 만들어").
#
# One FIFO queue and one hold, as files under $FLEET_DIR on srv2 (every
# session's chains run there). A session REQUESTS a turn, WAITS until it is
# at the head of the queue AND nobody holds the fleet AND no legacy bench /
# boot process is running (peers that never adopted this tool are still
# respected), then HOLDS it, runs, RELEASES. Nothing here ever kills a live
# holder: "바로 해" from the operator is `front` (jump the queue) and, for a
# holder whose process is gone, `kick`.
#
#   fleet.sh request <session> [est_min] [note]      enqueue (idempotent), print position
#   fleet.sh wait    <session> [timeout_min]         block until GO, then hold (pid = caller)
#   fleet.sh release <session>                       drop the hold (+ legacy FLEET-free-for-*.done)
#   fleet.sh run     <session> [est_min] [note] -- <cmd...>
#                                                    request + wait + run + release (trap)
#   fleet.sh status                                  holder, queue, liveness, legacy busy
#   fleet.sh adopt   <session> <pid> [est_min] [note] a job already running becomes the holder
#   fleet.sh front   <session>                       move to the head of the queue
#   fleet.sh cancel  <session>                       leave the queue
#   fleet.sh kick    [--force]                       drop a DEAD holder (--force: any holder;
#                                                    the operator's call, logged as such)
#   fleet.sh busy                                    "<bench procs> <running+waiting requests>"
#
# Files: queue (ticket|session|epoch|est_min|note), holder (session|pid|host|
# epoch|est_min|note), log. All edits under flock on $FLEET_DIR/.lock.
# Lives in the repo as bench/fleet.sh; srv2 runs ~/glm53-logs/fleet.sh.
set -uo pipefail
FLEET_DIR=${FLEET_DIR:-/home/choiceoh/glm53-logs/fleet}
LOGD=${LOGD:-/home/choiceoh/glm53-logs}
HEAD_URL=${HEAD_URL:-http://10.10.10.2:8000}
mkdir -p "$FLEET_DIR"
Q=$FLEET_DIR/queue; H=$FLEET_DIR/holder; L=$FLEET_DIR/log; LK=$FLEET_DIR/.lock
touch "$Q" "$L"
now() { date +%s; }
ts() { date +%F_%T; }
logit() { echo "$(ts) $*" >> "$L"; }
me() { hostname -s; }

# ---- legacy awareness: a peer that did not adopt this tool is still busy when
# its chain / boot / bench process runs or the engine has requests in flight.
# The patterns live here, in a file, so no caller's command line matches itself.
busy_procs() {
  ps -eo args | grep -cE "^(bash [a-zA-Z0-9_./-]*(lever-chain|ab-lever|onepass-after|chain|orchestrate)[a-zA-Z0-9_.-]*\.sh|bash /home/choiceoh/glm53-logs/ab-lever2\.sh|bash launchers/start-glm53|python3 (bench/onepass\.py|bench/bracket\.py|probes/))"
}
busy_reqs() {
  curl -s -m 3 "$HEAD_URL/metrics" 2>/dev/null | awk '/^vllm:num_requests_(running|waiting)/ {s+=$2} END {print s+0}'
}
booting() {  # a head container younger than 12 min is still booting (health not yet)
  docker ps --format '{{.Names}} {{.Status}}' 2>/dev/null | grep -E '^glm53 ' | grep -qE 'Up ([0-9]+ seconds|Less than a|[0-9] minutes|1[01] minutes)' \
    && [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' "$HEAD_URL/health")" != 200 ]
}
legacy_busy() { [ "$(busy_procs)" != 0 ] || [ "$(busy_reqs)" != 0 ] || booting; }

# ---- holder liveness: pid on this host checked directly; elsewhere, trust it
# until 3x its estimate has passed.
holder_alive() {
  [ -s "$H" ] || return 1
  IFS='|' read -r s pid host t0 est note < "$H"
  if [ "$host" = "$(me)" ] && [ -n "$pid" ]; then kill -0 "$pid" 2>/dev/null && return 0; return 1; fi
  [ $(( $(now) - t0 )) -lt $(( ${est:-30} * 60 * 3 )) ]
}
holder_line() { [ -s "$H" ] && IFS='|' read -r s pid host t0 est note < "$H" && echo "$s (pid $pid@$host, since $(date -d @$t0 +%H:%M), est ${est}m, $note)"; }

with_lock() { ( flock -x 9; "$@" ) 9>"$LK"; }

_enqueue() {  # session est note
  grep -q "^[0-9]*|$1|" "$Q" && return 0
  echo "$(now)$$|$1|$(now)|${2:-30}|${3:-}" >> "$Q"; logit "request $1 est=${2:-30}m $3"
}
_dequeue() { grep -v "^[0-9]*|$1|" "$Q" > "$Q.tmp"; mv "$Q.tmp" "$Q"; }
_position() { grep -n "^[0-9]*|$1|" "$Q" | head -1 | cut -d: -f1; }
_front() { { grep "^[0-9]*|$1|" "$Q"; grep -v "^[0-9]*|$1|" "$Q"; } > "$Q.tmp"; mv "$Q.tmp" "$Q"; logit "front $1"; }

_try_hold() {  # session pid est note -> 0 when held
  local s=$1 pid=$2 est=$3 note=$4
  if [ -s "$H" ]; then
    if holder_alive; then return 1; fi
    logit "auto-kick dead holder: $(holder_line)"; rm -f "$H"
  fi
  [ "$(head -1 "$Q" | cut -d'|' -f2)" = "$s" ] || return 1
  legacy_busy && return 1
  echo "$s|$pid|$(me)|$(now)|$est|$note" > "$H"; _dequeue "$s"
  rm -f "$LOGD"/FLEET-free-for-*.done 2>/dev/null; touch "$LOGD/FLEET-held-by-$s.done"
  logit "GO $s (pid $pid)"; return 0
}
_release() {  # session
  if [ -s "$H" ] && [ "$(cut -d'|' -f1 "$H")" = "$1" ]; then
    rm -f "$H" "$LOGD/FLEET-held-by-$1.done"; logit "release $1"
    # legacy markers for chains that still poll them
    for p in fusion mkg3 b12x glmfix; do touch "$LOGD/FLEET-free-for-$p.done"; done
    return 0
  fi
  echo "not the holder: $(holder_line 2>/dev/null || echo none)" >&2; return 1
}

cmd=${1:-status}; shift || true
case "$cmd" in
  request)
    s=${1:?session}; with_lock _enqueue "$s" "${2:-30}" "${3:-}"; echo "queued: $s at position $(_position "$s") of $(grep -c . "$Q")";;
  wait)
    s=${1:?session}; tmo=${2:-720}; pid=${FLEET_PID:-$PPID}
    est=$(grep "^[0-9]*|$s|" "$Q" | head -1 | cut -d'|' -f4); note=$(grep "^[0-9]*|$s|" "$Q" | head -1 | cut -d'|' -f5)
    [ -n "$est" ] || { with_lock _enqueue "$s" 30 ""; est=30; note=""; }
    t_end=$(( $(now) + tmo * 60 )); last=""
    while [ "$(now)" -lt "$t_end" ]; do
      if with_lock _try_hold "$s" "$pid" "$est" "$note"; then echo "GO $s $(ts)"; exit 0; fi
      why="pos $(_position "$s")/$(grep -c . "$Q")"; [ -s "$H" ] && why="$why, held by $(holder_line)"; legacy_busy && why="$why, legacy busy ($(busy_procs) procs, $(busy_reqs) reqs$(booting && echo ', booting'))"
      [ "$why" = "$last" ] || { echo "waiting: $why $(ts)"; last=$why; }
      sleep 15
    done
    echo "TIMEOUT $s after ${tmo}m" >&2; exit 1;;
  release) with_lock _release "${1:?session}";;
  run)
    s=${1:?session}; shift; est=30; note=""
    [ "${1:-}" != "--" ] && { est=$1; shift; }
    [ "${1:-}" != "--" ] && { note=$1; shift; }
    [ "${1:-}" = "--" ] && shift
    [ $# -gt 0 ] || { echo "usage: fleet.sh run <session> [est_min] [note] -- cmd..." >&2; exit 2; }
    with_lock _enqueue "$s" "$est" "$note"
    FLEET_PID=$$ bash "$0" wait "$s" "${FLEET_TIMEOUT_MIN:-720}" || exit 1
    trap 'with_lock _release "$s"' EXIT
    "$@"; rc=$?; exit $rc;;
  status)
    echo "fleet: $( [ -s "$H" ] && { holder_alive && echo "HELD by $(holder_line)" || echo "held by DEAD $(holder_line)"; } || echo FREE )"
    echo "legacy: $(busy_procs) bench/boot procs, $(busy_reqs) requests in flight$(booting && echo ', head booting')"
    echo "queue ($(grep -c . "$Q")):"; n=0; while IFS='|' read -r t s at est note; do n=$((n+1)); echo "  $n. $s (since $(date -d @$at +%H:%M), est ${est}m) $note"; done < "$Q"
    ls -t "$LOGD"/FLEET-*.done 2>/dev/null | head -4 | while read -r f; do echo "  marker $(stat -c %y "$f" | cut -c12-16) $(basename "$f")"; done
    echo "log:"; tail -4 "$L" | sed 's/^/  /';;
  adopt)  # a job that is ALREADY running (started before this tool, or by hand) becomes the holder
    s=${1:?session}; pid=${2:?pid}; est=${3:-30}; note=${4:-}
    if [ -s "$H" ] && holder_alive; then echo "fleet already held: $(holder_line)" >&2; exit 1; fi
    kill -0 "$pid" 2>/dev/null || { echo "pid $pid is not alive on $(me)" >&2; exit 1; }
    with_lock sh -c "echo '$s|$pid|$(me)|$(now)|$est|$note' > '$H'; rm -f '$LOGD'/FLEET-free-for-*.done; touch '$LOGD/FLEET-held-by-$s.done'"
    logit "adopt $s (pid $pid) est=${est}m $note"; echo "held by $s (pid $pid)";;
  front) with_lock _front "${1:?session}"; echo "$1 -> position $(_position "$1")";;
  cancel) with_lock _dequeue "${1:?session}"; logit "cancel $1"; echo "cancelled $1";;
  kick)
    if [ ! -s "$H" ]; then echo "nothing held"; exit 0; fi
    if holder_alive && [ "${1:-}" != "--force" ]; then echo "holder is ALIVE: $(holder_line) -- use --force only on the operator's word" >&2; exit 1; fi
    logit "kick${1:+ $1} of $(holder_line)"; rm -f "$H"; touch "$LOGD"/FLEET-free-for-{fusion,mkg3,b12x,glmfix}.done; echo "kicked";;
  busy) echo "$(busy_procs) $(busy_reqs)";;
  *) sed -n 2,26p "$0"; exit 2;;
esac
