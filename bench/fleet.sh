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
#   fleet.sh request [--probe] <session> [est] [note] enqueue (idempotent), print position
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
#   fleet.sh preflight [--probe] <session> [-- <cmd...>]
#                                                    checks BEFORE a boot is spent: srv2 copies
#                                                    (ab-lever2.sh, fleet.sh) == repo, chain syntax,
#                                                    every VLLM_* knob the chain sets is declared in
#                                                    the profile (the launcher forwards only those),
#                                                    baseline / duplicate-arm notice. `run` refuses
#                                                    a FAILed preflight (FLEET_PREFLIGHT=skip overrides)
#   fleet.sh restore-needed <session>                "yes" when nobody with a BOOT is queued behind you
#                                                    (the last holder restores production; a holder
#                                                    with a boot job behind it may skip the restore)
#   fleet.sh run --probe <session> [est] [note] -- <cmd...>
#                                                    a GPU probe (no boot): needs an IDLE serving
#                                                    (health 200, 0 requests) or no serving at all,
#                                                    never a boot in progress; est <= 15 may slip
#                                                    ahead of queued boot jobs (logged)
#   fleet.sh ledger [days]                           per-session holds, minutes, boots, records and
#                                                    boots that produced no measurement
# Records: every release appends to $FLEET_DIR/ledger.tsv; `status` derives ETAs
# from a session's last actual holds and flags a holder that is SILENT (no
# heartbeat for 10 min) or OVERDUE (2x its estimate) -- flags only, never a kill.
#
# Files: queue (ticket|session|epoch|est_min|note), holder (session|pid|host|
# epoch|est_min|note), log. All edits under flock on $FLEET_DIR/.lock.
# Lives in the repo as bench/fleet.sh; srv2 runs ~/glm53-logs/fleet.sh.
set -uo pipefail
FLEET_DIR=${FLEET_DIR:-/home/choiceoh/glm53-logs/fleet}
LOGD=${LOGD:-/home/choiceoh/glm53-logs}
REPO=${REPO:-/home/choiceoh/stkernel}
# "이 빌드의 기준점이 될 측정이 이미 있으면 알려주는 장치" (operator, 39차): before a
# session spends a boot on a defaults arm, say whether the deployed build already
# has one. bench/baseline.py reads the onepass records (overlay stamp + knobs).
baseline_line() {
  [ -f "$REPO/bench/baseline.py" ] || return 0
  (cd "$REPO" 2>/dev/null && timeout 20 python3 bench/baseline.py --brief 2>/dev/null) || true
}
HEAD_URL=${HEAD_URL:-http://10.10.10.2:8000}
mkdir -p "$FLEET_DIR"
Q=$FLEET_DIR/queue; H=$FLEET_DIR/holder; L=$FLEET_DIR/log; LK=$FLEET_DIR/.lock
touch "$Q" "$L"
now() { date +%s; }
ts() { date +%F_%T; }
logit() { echo "$(ts) $*" >> "$L"; }
me() { hostname -s; }
LEDGER=$FLEET_DIR/ledger.tsv; JSONL=$LOGD/bracket-onepass.jsonl
hb_file() { echo "$FLEET_DIR/hb.$1"; }
kind_of() { [ "${1:-}" = probe ] && echo probe || echo boot; }
serving_up() { docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^glm53$'; }
serving_idle() {  # a probe may run beside this: healthy, nothing in flight, not booting
  ! serving_up && return 0
  booting && return 1
  [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' "$HEAD_URL/health")" = 200 ] && [ "$(busy_reqs)" = 0 ]
}
# expected minutes for a session: median of its last 5 actual holds, else the estimate
expected_min() {  # session est
  local m; m=$(awk -F'\t' -v s="$1" '$2==s {v[++n]=$5} END {if (n) {asort(v); print v[int((n+1)/2)]}}' "$LEDGER" 2>/dev/null)
  echo "${m:-$2}"
}
# ---- preflight: the traps that cost a boot on 09-06, checked before the boot
preflight() {  # [--probe] session [-- cmd...] -> 0 PASS, 1 FAIL
  local ok=1 knobs="" chain="" kind=boot
  [ "${1:-}" = "--probe" ] && { kind=probe; shift; }
  echo "preflight $1 [$kind]:"
  for pair in "ab-lever2.sh:bench/ab-lever.sh" "fleet.sh:bench/fleet.sh"; do
    local copy=$LOGD/${pair%%:*} src=$REPO/${pair#*:}
    [ -f "$copy" ] || continue
    if [ "$(md5sum < "$copy")" = "$(md5sum < "$src")" ]; then echo "  PASS $copy == repo"
    else echo "  FAIL $copy differs from $src (sync: cp $src $copy.new && mv $copy.new $copy)"; ok=0; fi
  done
  shift
  [ "${1:-}" = "--" ] && shift
  if [ "${1:-}" = bash ] && [ -f "${2:-}" ]; then chain=$2; fi
  if [ -n "$chain" ]; then
    if bash -n "$chain" 2>/dev/null; then echo "  PASS syntax $chain"; else echo "  FAIL syntax $chain"; ok=0; fi
    knobs=$(grep -oE "VLLM_[A-Z0-9_]+=[^ \"'\\]*" "$chain" | sort -u)
  fi
  [ $# -gt 0 ] && knobs="$knobs $(printf '%s ' "$@" | grep -oE "VLLM_[A-Z0-9_]+=[^ \"']*" | sort -u)"
  # The declared-knob rule is about the LAUNCHER: it forwards only the keys
  # profiles/glm53.env declares, so a boot chain that sets an undeclared knob
  # silently measures the default and costs a boot. A probe has no launcher --
  # run_mk_probe.sh builds its own container and passes its own env -- so the
  # VLLM_* names inside a probe runner are not knobs at all. Checking them
  # FAILed a probe turn on 09-06 over run_mk_probe.sh's own PROBE_CACHE lines
  # (VLLM_CACHE_ROOT, VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR).
  if [ "$kind" = probe ]; then
    echo "  SKIP declared-knob check (probe: no launcher in the path)"
  else
    local k undeclared=""
    for k in $knobs; do
      grep -qE "^${k%%=*}=" "$REPO/profiles/glm53.env" || undeclared="$undeclared ${k%%=*}"
    done
    if [ -n "$undeclared" ]; then echo "  FAIL undeclared in profiles/glm53.env (the launcher forwards only declared keys):$undeclared"; ok=0
    elif [ -n "$knobs" ]; then echo "  PASS knobs declared: $(echo $knobs | tr ' ' ',')"; fi
    if [ -f "$REPO/bench/baseline.py" ]; then
      local kv; kv=$(echo $knobs | tr ' ' ',')
      (cd "$REPO" && timeout 20 python3 bench/baseline.py --brief ${kv:+--knobs "$kv"} 2>/dev/null | sed 's/^/  /') || true
    fi
  fi
  [ $ok = 1 ] && { echo "  -> PASS"; return 0; }
  echo "  -> FAIL (FLEET_PREFLIGHT=skip to override, logged)"; return 1
}
restore_needed() {  # session -> yes|no
  local next; next=$(grep -v "^[0-9]*|$1|" "$Q" | head -1)
  if [ -z "$next" ]; then echo "yes (nobody queued: the last holder restores production)"; return 0; fi
  local ns nk; ns=$(echo "$next" | cut -d'|' -f2); nk=$(kind_of "$(echo "$next" | cut -d'|' -f6)")
  if [ "$nk" = boot ]; then echo "no ($ns boots next and replaces whatever is up)"; return 1; fi
  echo "yes ($ns runs a probe next, not a boot)"; return 0
}

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
  IFS='|' read -r s pid host t0 est note kind < "$H"
  if [ "$host" = "$(me)" ] && [ -n "$pid" ]; then kill -0 "$pid" 2>/dev/null && return 0; return 1; fi
  [ $(( $(now) - t0 )) -lt $(( ${est:-30} * 60 * 3 )) ]
}
holder_line() { [ -s "$H" ] && IFS='|' read -r s pid host t0 est note kind < "$H" && echo "$s${kind:+ [$kind]} (pid $pid@$host, since $(date -d @$t0 +%H:%M), est ${est}m, $note)"; }

with_lock() { ( flock -x 9; "$@" ) 9>"$LK"; }

_enqueue() {  # session est note [kind] -- idempotent per session; a repeat refreshes est/note/kind in place
  local kind; kind=$(kind_of "${4:-}")
  if grep -q "^[0-9]*|$1|" "$Q"; then
    awk -F'|' -v OFS='|' -v s="$1" -v est="${2:-30}" -v note="${3:-}" -v kind="$kind" '$2==s {$4=est; $5=note; $6=kind} {print}' "$Q" > "$Q.tmp" && mv "$Q.tmp" "$Q"
    return 0
  fi
  echo "$(now)$$|$1|$(now)|${2:-30}|${3:-}|$kind" >> "$Q"; logit "request $1 est=${2:-30}m $3${4:+ [$4]}"
}
_dequeue() { grep -v "^[0-9]*|$1|" "$Q" > "$Q.tmp"; mv "$Q.tmp" "$Q"; }
_position() { grep -n "^[0-9]*|$1|" "$Q" | head -1 | cut -d: -f1; }
_front() { { grep "^[0-9]*|$1|" "$Q"; grep -v "^[0-9]*|$1|" "$Q"; } > "$Q.tmp"; mv "$Q.tmp" "$Q"; logit "front $1"; }

_try_hold() {  # session pid est note [kind] -> 0 when held
  local s=$1 pid=$2 est=$3 note=$4 kind; kind=$(kind_of "${5:-}")
  if [ -s "$H" ]; then
    if holder_alive; then return 1; fi
    logit "auto-kick dead holder: $(holder_line)"; rm -f "$H"
  fi
  if [ "$(head -1 "$Q" | cut -d'|' -f2)" != "$s" ]; then
    # a short probe may slip ahead of queued BOOT jobs while serving sits idle:
    # it costs them at most its bounded runtime and no boot (logged as slip)
    if [ "$kind" = probe ] && [ "${est:-30}" -le 15 ] && serving_idle; then logit "slip $s (probe, est ${est}m) ahead of $(head -1 "$Q" | cut -d'|' -f2)"; else return 1; fi
  fi
  [ "$kind" = probe ] && ! serving_idle && return 1
  # never hand the fleet to a dead job (an orphaned waiter whose run process
  # was killed took a turn for pid 3710362 on 09-06 and was auto-kicked 2 s
  # later, dropping the live request with the same session name)
  [ -z "$pid" ] || kill -0 "$pid" 2>/dev/null || return 1
  legacy_busy && return 1
  echo "$s|$pid|$(me)|$(now)|$est|$note|$kind" > "$H"; _dequeue "$s"
  rm -f "$LOGD"/FLEET-free-for-*.done 2>/dev/null; touch "$LOGD/FLEET-held-by-$s.done"
  logit "GO $s (pid $pid)"; return 0
}
_ledger_row() {  # session -- from the holder file, before it is removed
  IFS='|' read -r s pid host t0 est note kind < "$H"
  local held boots recs; held=$(( ($(now) - t0 + 30) / 60 ))
  boots=$(find "$LOGD" -maxdepth 1 -name 'boot-*.log' -newermt "@$t0" 2>/dev/null | wc -l)
  [ "$boots" = 0 ] && [ -f "$LOGD/glm53.log" ] && [ "$(stat -c %Y "$LOGD/glm53.log")" -ge "$t0" ] && boots=1
  recs=$(python3 - "$JSONL" "$t0" <<'PY' 2>/dev/null || echo 0
import json, sys, time
n = 0
try:
    for l in open(sys.argv[1]):
        if not l.strip(): continue
        r = json.loads(l)
        if time.mktime(time.strptime(r.get("t", "1970-01-01 00:00:00"), "%Y-%m-%d %H:%M:%S")) >= int(sys.argv[2]): n += 1
except Exception: pass
print(n)
PY
)
  local wasted=$(( boots - recs )); [ $wasted -lt 0 ] && wasted=0
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(ts)" "$s" "${kind:-boot}" "$note" "$held" "$boots" "$recs" "$wasted" >> "$LEDGER"
  logit "ledger $s held=${held}m boots=$boots records=$recs wasted=$wasted"
  ls -t "$LOGD"/boot-*.log 2>/dev/null | head -4 | while read -r f; do [ "$(stat -c %Y "$f")" -ge "$t0" ] && logit "  kept $f"; done
}
_release() {  # session
  if [ -s "$H" ] && [ "$(cut -d'|' -f1 "$H")" = "$1" ]; then
    _ledger_row "$1"
    rm -f "$H" "$LOGD/FLEET-held-by-$1.done" "$(hb_file "$1")"; logit "release $1"
    # legacy markers for chains that still poll them
    for p in fusion mkg3 b12x glmfix; do touch "$LOGD/FLEET-free-for-$p.done"; done
    return 0
  fi
  echo "not the holder: $(holder_line 2>/dev/null || echo none)" >&2; return 1
}

cmd=${1:-status}; shift || true
case "$cmd" in
  request)
    kind=boot; [ "${1:-}" = "--probe" ] && { kind=probe; shift; }
    s=${1:?session}; with_lock _enqueue "$s" "${2:-30}" "${3:-}" "$kind"; echo "queued: $s [$kind] at position $(_position "$s") of $(grep -c . "$Q")"; baseline_line;;
  wait)
    s=${1:?session}; tmo=${2:-720}; pid=${FLEET_PID:-$PPID}
    est=$(grep "^[0-9]*|$s|" "$Q" | head -1 | cut -d'|' -f4); note=$(grep "^[0-9]*|$s|" "$Q" | head -1 | cut -d'|' -f5)
    kind=$(kind_of "$(grep "^[0-9]*|$s|" "$Q" | head -1 | cut -d'|' -f6)")
    [ -n "$est" ] || { with_lock _enqueue "$s" 30 "" "$kind"; est=30; note=""; }
    t_end=$(( $(now) + tmo * 60 )); last=""
    while [ "$(now)" -lt "$t_end" ]; do
      # an orphaned waiter (its run process gone) must not keep polling for a
      # dead pid; a request that vanished (a stale sibling took it, or a
      # cancel) is re-queued at the back instead of waiting forever at "pos /0"
      kill -0 "$pid" 2>/dev/null || { echo "parent $pid is gone; giving up $(ts)" >&2; with_lock _dequeue "$s"; exit 1; }
      [ -n "$(_position "$s")" ] || { with_lock _enqueue "$s" "$est" "$note" "$kind"; echo "re-queued: $s (entry was gone) $(ts)"; }
      if with_lock _try_hold "$s" "$pid" "$est" "$note" "$kind"; then echo "GO $s $(ts)"; exit 0; fi
      why="pos $(_position "$s")/$(grep -c . "$Q")"; [ -s "$H" ] && why="$why, held by $(holder_line)"; legacy_busy && why="$why, legacy busy ($(busy_procs) procs, $(busy_reqs) reqs$(booting && echo ', booting'))"
      [ "$why" = "$last" ] || { echo "waiting: $why $(ts)"; last=$why; }
      sleep 15
    done
    echo "TIMEOUT $s after ${tmo}m" >&2; exit 1;;
  release) with_lock _release "${1:?session}";;
  run)
    kind=boot; [ "${1:-}" = "--probe" ] && { kind=probe; shift; }
    s=${1:?session}; shift; est=30; note=""
    [ "${1:-}" != "--" ] && { est=$1; shift; }
    [ "${1:-}" != "--" ] && { note=$1; shift; }
    [ "${1:-}" = "--" ] && shift
    [ $# -gt 0 ] || { echo "usage: fleet.sh run [--probe] <session> [est_min] [note] -- cmd..." >&2; exit 2; }
    pf=(); [ "$kind" = probe ] && pf=(--probe)
    if ! preflight ${pf[@]+"${pf[@]}"} "$s" -- "$@"; then
      if [ "${FLEET_PREFLIGHT:-}" = skip ]; then logit "preflight FAIL overridden by $s"; else logit "preflight FAIL $s (not queued)"; exit 3; fi
    fi
    with_lock _enqueue "$s" "$est" "$note" "$kind"
    FLEET_PID=$$ bash "$0" wait "$s" "${FLEET_TIMEOUT_MIN:-720}" || exit 1
    # heartbeat: a holder that goes SILENT (hung chain, wedged node) shows in status
    ( while kill -0 $$ 2>/dev/null; do touch "$(hb_file "$s")"; sleep 30; done ) & hb=$!
    trap 'kill $hb 2>/dev/null; with_lock _release "$s"' EXIT
    "$@"; rc=$?; exit $rc;;
  status)
    echo "fleet: $( [ -s "$H" ] && { holder_alive && echo "HELD by $(holder_line)" || echo "held by DEAD $(holder_line)"; } || echo FREE )"
    remaining=0
    if [ -s "$H" ]; then
      IFS='|' read -r hs hpid hhost ht0 hest hnote hkind < "$H"; held=$(( ($(now) - ht0) / 60 ))
      remaining=$(( hest - held )); [ $remaining -lt 0 ] && remaining=0
      [ $held -gt $(( hest * 2 )) ] && echo "  OVERDUE: held ${held}m against est ${hest}m (a flag, not a kill: operator's kick --force)"
      hbf=$(hb_file "$hs"); [ -f "$hbf" ] && [ $(( $(now) - $(stat -c %Y "$hbf") )) -gt 600 ] && echo "  SILENT: no heartbeat for $(( ($(now) - $(stat -c %Y "$hbf")) / 60 ))m"
    fi
    echo "legacy: $(busy_procs) bench/boot procs, $(busy_reqs) requests in flight$(booting && echo ', head booting')"
    echo "queue ($(grep -c . "$Q")):"; n=0; eta=$remaining; while IFS='|' read -r t s at est note kind; do n=$((n+1)); exp=$(expected_min "$s" "$est"); echo "  $n. $s${kind:+ [$kind]} (since $(date -d @$at +%H:%M), est ${est}m, expect ~${exp}m, ETA ~$(date -d "@$(( $(now) + eta * 60 ))" +%H:%M)) $note"; eta=$(( eta + exp )); done < "$Q"
    ls -t "$LOGD"/FLEET-*.done 2>/dev/null | head -4 | while read -r f; do echo "  marker $(stat -c %y "$f" | cut -c12-16) $(basename "$f")"; done
    echo "log:"; tail -4 "$L" | sed 's/^/  /'
    baseline_line;;
  adopt)  # a job that is ALREADY running (started before this tool, or by hand) becomes the holder
    s=${1:?session}; pid=${2:?pid}; est=${3:-30}; note=${4:-}
    if [ -s "$H" ] && holder_alive; then echo "fleet already held: $(holder_line)" >&2; exit 1; fi
    kill -0 "$pid" 2>/dev/null || { echo "pid $pid is not alive on $(me)" >&2; exit 1; }
    with_lock sh -c "echo '$s|$pid|$(me)|$(now)|$est|$note|boot' > '$H'; rm -f '$LOGD'/FLEET-free-for-*.done; touch '$LOGD/FLEET-held-by-$s.done'"
    logit "adopt $s (pid $pid) est=${est}m $note"; echo "held by $s (pid $pid)";;
  front) with_lock _front "${1:?session}"; echo "$1 -> position $(_position "$1")";;
  cancel) with_lock _dequeue "${1:?session}"; logit "cancel $1"; echo "cancelled $1";;
  kick)
    if [ ! -s "$H" ]; then echo "nothing held"; exit 0; fi
    if holder_alive && [ "${1:-}" != "--force" ]; then echo "holder is ALIVE: $(holder_line) -- use --force only on the operator's word" >&2; exit 1; fi
    logit "kick${1:+ $1} of $(holder_line)"; rm -f "$H"; touch "$LOGD"/FLEET-free-for-{fusion,mkg3,b12x,glmfix}.done; echo "kicked";;
  busy) echo "$(busy_procs) $(busy_reqs)";;
  preflight)
    [ $# -ge 1 ] || { echo "usage: fleet.sh preflight [--probe] <session> [-- cmd...]" >&2; exit 2; }
    preflight "$@";;
  restore-needed) restore_needed "${1:?session}";;
  ledger)
    days=${1:-1}; since=$(date -d "-${days} days" +%F)
    echo "ledger since $since ($LEDGER):"
    printf '  %-12s %5s %7s %5s %7s %6s\n' session holds min boots records wasted
    awk -F'\t' -v since="$since" 'substr($1,1,10) >= since {h[$2]++; m[$2]+=$5; b[$2]+=$6; r[$2]+=$7; w[$2]+=$8}
      END {for (s in h) printf "  %-12s %5d %7d %5d %7d %6d\n", s, h[s], m[s], b[s], r[s], w[s]}' "$LEDGER" 2>/dev/null | sort
    echo "  (wasted = boots that produced no onepass record during the hold)";;
  *) sed -n 2,26p "$0"; exit 2;;
esac
