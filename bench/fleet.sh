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
#   fleet.sh board [n]                               every session's last n verdicts + today's boots
#   fleet.sh ledger [days]                           per-session holds, minutes, boots, records and
#                                                    boots that produced no measurement
#   fleet.sh pair <session> <NAME> "<knobs>" [est] [note]
#                                                    the standard bracket (bench/pair.sh): candidate
#                                                    boot + onepass + proof, yield to a short probe,
#                                                    defaults boot only when the build lacks a
#                                                    baseline / a 3-sample floor and nobody boots
#                                                    behind you, judge with the noise floor, verdict
#   fleet.sh deploy <session> <rev>                  the holder deploys <rev> (git + overlays) and the
#                                                    build registry gets stamp <-> sha <- session
#   fleet.sh yield <session> [max_est]               holder lets a short queued probe run beside its
#                                                    idle serving, keeps its place, resumes after
#   fleet.sh nodes                                   the four nodes: ssh, GPU, stray containers, RAM,
#                                                    model path -- run at GO (warn; FLEET_NODES=strict refuses)
#   fleet.sh notify <session> "<cmd>"                hook run on GO / release / preflight-fail / yield
#                                                    with args <event> <session> <note>; every event
#                                                    also lands in $FLEET_DIR/events.log
#   fleet.sh run --gpu|--cpu <session> ...           SAY which it is (operator): --gpu takes a turn in
#                                                    the queue; --cpu runs NOW in parallel under nice,
#                                                    never holding. A --cpu job whose script or command
#                                                    shows GPU use (ab-lever, a boot, a probe container,
#                                                    torch.cuda...) is REFUSED (--cpu --force overrides,
#                                                    logged). Without a flag the classifier decides:
#                                                    GPU evidence -> queue; CPU evidence (rehearsal,
#                                                    CPU probes, tests, compile checks) -> parallel;
#                                                    no evidence -> queue, and it says so
# `status` also prints what production is serving (defaults, or which knobs)
# and the deployed build; a release with an empty queue and production not on
# the defaults prints the restore command (FLEET_AUTO_RESTORE=1 runs it).
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
# ---- events + notification hook (idea 7)
_event() {  # event session note
  echo "$(ts) $1 $2 $3" >> "$FLEET_DIR/events.log"
  local hook="$FLEET_DIR/notify.$2"
  [ -f "$hook" ] && ( timeout 20 bash -c "$(cat "$hook")" _ "$1" "$2" "$3" >/dev/null 2>&1 & )
  return 0
}
# ---- GPU / no-GPU classification (operator: "gpu 없이 할수 있는 작업 같으면 병렬로")
# Evidence for GPU wins over evidence for CPU; no evidence at all is treated as
# GPU (queued) and says so. A rehearsal never needs the GPU.
classify_cmd() {  # cmd... -> gpu|nogpu|unknown
  [ "${FLEET_REHEARSE:-0}" = 1 ] && { echo nogpu; return; }
  local text="$*" f
  for f in "$@"; do
    [ -f "$f" ] || continue
    case "$f" in
      *.py) text="$text $(grep -vE '^\s*#' "$f" 2>/dev/null | grep -oE 'torch\.cuda|\.cuda\(|device=.cuda|--gpus|docker run' | head -3)";;   # code, not docstrings
      *)    text="$text $(grep -vE '^\s*#' "$f" 2>/dev/null)";;
    esac
  done
  local gpu='ab-lever|start-glm53|deploy-overlays|run_mk_probe|run_megakernel_bench|docker run|--gpus|onepass\.py|bracket\.py|bench-dec|torch\.cuda|nvidia-smi|\.cu\b|cuda_'
  local cpu='MK_PROBE_NO_GPU=1|head_pack_accuracy_cpu|baseline\.py|judge\.py|test_logic\.py|b12x_static_compile_check|compile\.sh|nvcc |bash -n|^git |md5sum|proof\.py'
  if echo "$text" | grep -qE "$gpu"; then echo gpu
  elif echo "$text" | grep -qE "$cpu"; then echo nogpu
  else echo unknown; fi
}
production_line() {  # what is serving, judged from the container's env (idea 9)
  serving_up || { echo "production: no serving container"; return 0; }
  local k; k=$( (cd "$REPO" 2>/dev/null && timeout 20 python3 - <<'PY'
import sys; sys.path.insert(0, "bench")
try:
    from onepass import _served_build
    b = _served_build(".") or {}
    kn = {k: v for k, v in (b.get("knobs") or {}).items() if v not in ("0", "", "off")}
    print(("NOT defaults: " + ",".join(f"{k}={v}" for k, v in sorted(kn.items()))) if kn else "defaults")
except Exception as e:
    print(f"unknown ({e.__class__.__name__})")
PY
) 2>/dev/null)
  local h; h=$(curl -s -m 3 -o /dev/null -w '%{http_code}' "$HEAD_URL/health" 2>/dev/null)
  echo "production: ${k:-unknown} (health ${h:-000})"
}
deployed_line() {  # the build registry (idea 3): stamp <-> sha, and the checkout vs deployed
  local stamp sha head; stamp=$(cut -c1-12 "${MK_OVERLAY_STAMP:-$HOME/glm53-cache/.overlay-sha}" 2>/dev/null)
  [ -n "$stamp" ] || { echo "deployed: unknown (no overlay stamp)"; return 0; }
  sha=$(awk -F'\t' -v st="$stamp" 'index($2, st)==1 {sha=$3; who=$4; at=$1} END {if (sha) print sha " (by " who ", " substr(at,12,5) ")"}' "$FLEET_DIR/builds.tsv" 2>/dev/null)
  head=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null)
  echo "deployed: $stamp${sha:+ = $sha}"
  [ -n "$sha" ] && [ -n "$head" ] && [ "${sha%% *}" != "$head" ] && echo "  NOTE: the checkout ($head) is not the deployed build (${sha%% *}) -- a bench without a deploy runs the deployed one"
  return 0
}
nodes_check() {  # idea 8: the four nodes before a boot; 0 = all fine
  local ok=0 ip out
  for ip in ${FLEET_NODES_IPS:-10.10.10.1 10.10.10.2 10.10.10.3 10.10.10.4}; do
    out=$(timeout 12 ssh -o BatchMode=yes -o ConnectTimeout=5 "choiceoh@$ip" "nvidia-smi -L >/dev/null 2>&1 && echo gpu=ok || echo gpu=FAIL; echo stray=\$(docker ps --format '{{.Names}}' 2>/dev/null | grep -vc '^glm53\$'); echo ram=\$(free -g | awk 'NR==2{print \$7}')G; for p in ${FLEET_NODE_PATHS:-/home/choiceoh/models/glm53-redhat-nvfp4}; do [ -e \"\$p\" ] && echo path=ok || echo path=MISSING:\$p; done" 2>/dev/null | tr '\n' ' ')
    [ -n "$out" ] || { out="ssh=FAIL"; }
    echo "  node $ip: $out"
    echo "$out" | grep -qE "FAIL|MISSING|stray=[1-9]" && ok=1
  done
  return $ok
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

_enqueue() {  # session est note [kind] [pid] -- idempotent per session; a repeat refreshes est/note/kind in place
  local kind pid; kind=$(kind_of "${4:-}"); pid=${5:-}
  if grep -q "^[0-9]*|$1|" "$Q"; then
    # two live processes under one session name would merge into one ticket
    # and take one turn between them (09-06: `run fusion` twice); refuse
    local qpid; qpid=$(grep "^[0-9]*|$1|" "$Q" | head -1 | cut -d'|' -f7)
    if [ -n "$qpid" ] && [ -n "$pid" ] && [ "$qpid" != "$pid" ] && kill -0 "$qpid" 2>/dev/null && [ "${FLEET_SAME_SESSION:-0}" != 1 ]; then
      echo "session '$1' is already queued by a live process (pid $qpid): use another name (e.g. $1-2), or FLEET_SAME_SESSION=1 to share the ticket" >&2
      logit "refused duplicate session $1 (pid $pid vs queued $qpid)"; return 2
    fi
    awk -F'|' -v OFS='|' -v s="$1" -v est="${2:-30}" -v note="${3:-}" -v kind="$kind" -v pid="$pid" '$2==s {$4=est; $5=note; $6=kind; if (pid!="") $7=pid} {print}' "$Q" > "$Q.tmp" && mv "$Q.tmp" "$Q"
    return 0
  fi
  echo "$(now)$$|$1|$(now)|${2:-30}|${3:-}|$kind|$pid" >> "$Q"; logit "request $1 est=${2:-30}m $3${4:+ [$4]}"
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
  logit "GO $s (pid $pid)"; _event GO "$s" "$note"; return 0
}
_ledger_row() {  # session -- from the holder file, before it is removed
  IFS='|' read -r s pid host t0 est note kind < "$H"
  local held boots recs; held=$(( ($(now) - t0 + 30) / 60 ))
  boots=$(find "$LOGD" -maxdepth 1 -name 'boot-*.log' -newermt "@$t0" 2>/dev/null | wc -l)
  [ "$boots" = 0 ] && [ "${kind:-boot}" = boot ] && [ -f "$LOGD/glm53.log" ] && [ "$(stat -c %Y "$LOGD/glm53.log")" -ge "$t0" ] && boots=1
  [ "${kind:-boot}" = probe ] && boots=0
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
_yield_requeue() { _enqueue "$1" "$2" "$3" boot "${FLEET_PID:-$PPID}"; _front "$1"; }   # the yielding holder keeps its place: head of the queue
_release() {  # session
  if [ -s "$H" ] && [ "$(cut -d'|' -f1 "$H")" = "$1" ]; then
    _ledger_row "$1"
    rm -f "$H" "$LOGD/FLEET-held-by-$1.done" "$(hb_file "$1")"; logit "release $1"; _event release "$1" ""
    if [ ! -s "$Q" ] && [ "${FLEET_NO_RESTORE_CHECK:-0}" != 1 ]; then
      local pl; pl=$(production_line)
      case "$pl" in *"NOT defaults"*|*"no serving"*|*"health 000"*|*"health 5"*)
        if [ "${FLEET_AUTO_RESTORE:-0}" = 1 ]; then
          logit "auto-restore after $1: $pl"; ( LEGS=none PREFILL_WARMUP=1 nohup bash "$LOGD/ab-lever2.sh" PRODRESTORE "" > "$LOGD/auto-restore.log" 2>&1 & )
        else
          echo "NOTE: queue empty and $pl -- restore with: LEGS=none bash $LOGD/ab-lever2.sh PRODRESTORE \"\" (FLEET_AUTO_RESTORE=1 does it)"
        fi;;
      esac
    fi
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
    s=${1:?session}; with_lock _enqueue "$s" "${2:-30}" "${3:-}" "$kind" "$PPID" || exit 6; echo "queued: $s [$kind] at position $(_position "$s") of $(grep -c . "$Q")"; baseline_line;;
  wait)
    s=${1:?session}; tmo=${2:-720}; pid=${FLEET_PID:-$PPID}
    est=$(grep "^[0-9]*|$s|" "$Q" | head -1 | cut -d'|' -f4); note=$(grep "^[0-9]*|$s|" "$Q" | head -1 | cut -d'|' -f5)
    kind=$(kind_of "$(grep "^[0-9]*|$s|" "$Q" | head -1 | cut -d'|' -f6)")
    [ -n "$est" ] || { with_lock _enqueue "$s" 30 "" "$kind" "$pid" || exit 6; est=30; note=""; }
    t_end=$(( $(now) + tmo * 60 )); last=""
    while [ "$(now)" -lt "$t_end" ]; do
      # an orphaned waiter (its run process gone) must not keep polling for a
      # dead pid; a request that vanished (a stale sibling took it, or a
      # cancel) is re-queued at the back instead of waiting forever at "pos /0"
      kill -0 "$pid" 2>/dev/null || { echo "parent $pid is gone; giving up $(ts)" >&2; with_lock _dequeue "$s"; exit 1; }
      [ -n "$(_position "$s")" ] || { with_lock _enqueue "$s" "$est" "$note" "$kind" "$pid" || exit 6; echo "re-queued: $s (entry was gone) $(ts)"; }
      if with_lock _try_hold "$s" "$pid" "$est" "$note" "$kind"; then echo "GO $s $(ts)"; exit 0; fi
      why="pos $(_position "$s")/$(grep -c . "$Q")"; [ -s "$H" ] && why="$why, held by $(holder_line)"; legacy_busy && why="$why, legacy busy ($(busy_procs) procs, $(busy_reqs) reqs$(booting && echo ', booting'))"
      [ "$why" = "$last" ] || { echo "waiting: $why $(ts)"; last=$why; }
      sleep 15
    done
    echo "TIMEOUT $s after ${tmo}m" >&2; exit 1;;
  release) with_lock _release "${1:?session}";;
  run)
    kind=boot; force=""
    forced=""
    while :; do case "${1:-}" in --probe) kind=probe; shift;; --cpu|--nogpu) force=nogpu; shift;; --gpu) force=gpu; shift;; --force) forced=1; shift;; *) break;; esac; done
    s=${1:?session}; shift; est=30; note=""
    [ "${1:-}" != "--" ] && { est=$1; shift; }
    [ "${1:-}" != "--" ] && { note=$1; shift; }
    [ "${1:-}" = "--" ] && shift
    [ $# -gt 0 ] || { echo "usage: fleet.sh run --gpu|--cpu [--probe] <session> [est_min] [note] -- cmd..." >&2; exit 2; }
    export FLEET_SESSION=$s
    auto=$(classify_cmd "$@"); cls=${force:-$auto}
    if [ "$force" = nogpu ] && [ "$auto" = gpu ] && [ -z "$forced" ]; then
      echo "REFUSED: you said --cpu but the job shows GPU use (a boot, ab-lever, a probe container, torch.cuda); --cpu --force overrides" >&2
      logit "refused --cpu $s: classifier saw GPU use"; exit 5
    fi
    [ -z "$force" ] && echo "no --gpu/--cpu given: classified as $auto"
    if [ "$cls" = nogpu ]; then
      # no GPU needed: run now, in parallel, under nice; no hold, no queue
      logit "nogpu-start $s $note"; _event nogpu-start "$s" "$note"; t0=$(now)
      nice -n 19 "$@"; rc=$?
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(ts)" "$s" nogpu "$note" $(( ($(now) - t0 + 30) / 60 )) 0 0 0 >> "$LEDGER"
      logit "nogpu-done $s rc=$rc"; _event nogpu-done "$s" "$note"; exit $rc
    fi
    [ "$cls" = unknown ] && echo "no evidence either way -> queued as GPU (say --cpu to run in parallel)"
    pf=(); [ "$kind" = probe ] && pf=(--probe)
    if ! preflight ${pf[@]+"${pf[@]}"} "$s" -- "$@"; then
      if [ "${FLEET_PREFLIGHT:-}" = skip ]; then logit "preflight FAIL overridden by $s"; else logit "preflight FAIL $s (not queued)"; _event preflight-fail "$s" "$note"; exit 3; fi
    fi
    with_lock _enqueue "$s" "$est" "$note" "$kind" "$$" || exit 6
    FLEET_PID=$$ bash "$0" wait "$s" "${FLEET_TIMEOUT_MIN:-720}" || exit 1
    if [ "$kind" = boot ]; then
      echo "nodes:"; if ! nodes_check; then
        if [ "${FLEET_NODES:-warn}" = strict ]; then logit "nodes FAIL $s -> released"; with_lock _release "$s"; exit 4; fi
        echo "  (warnings only; FLEET_NODES=strict refuses)"
      fi
    fi
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
    echo "queue ($(grep -c . "$Q")):"; n=0; eta=$remaining; while IFS='|' read -r t s at est note kind qpid; do n=$((n+1)); exp=$(expected_min "$s" "$est"); echo "  $n. $s${kind:+ [$kind]} (since $(date -d @$at +%H:%M), est ${est}m, expect ~${exp}m, ETA ~$(date -d "@$(( $(now) + eta * 60 ))" +%H:%M)) $note"; eta=$(( eta + exp )); done < "$Q"
    ls -t "$LOGD"/FLEET-*.done 2>/dev/null | head -4 | while read -r f; do echo "  marker $(stat -c %y "$f" | cut -c12-16) $(basename "$f")"; done
    echo "log:"; tail -4 "$L" | sed 's/^/  /'
    production_line; deployed_line; baseline_line;;
  adopt)  # a job that is ALREADY running (started before this tool, or by hand) becomes the holder
    s=${1:?session}; pid=${2:?pid}; est=${3:-30}; note=${4:-}
    if [ -s "$H" ] && holder_alive; then echo "fleet already held: $(holder_line)" >&2; exit 1; fi
    kill -0 "$pid" 2>/dev/null || { echo "pid $pid is not alive on $(me)" >&2; exit 1; }
    with_lock sh -c "echo '$s|$pid|$(me)|$(now)|$est|$note|boot' > '$H'; rm -f '$LOGD'/FLEET-free-for-*.done; touch '$LOGD/FLEET-held-by-$s.done'"
    logit "adopt $s (pid $pid) est=${est}m $note"; echo "held by $s (pid $pid)";;
  front) with_lock _front "${1:?session}"; echo "$1 -> position $(_position "$1")";;
  cancel)
    s=${1:?session}; qpid=$(grep "^[0-9]*|$s|" "$Q" | head -1 | cut -d'|' -f7)
    # a live waiter re-queues a vanished entry within 15 s (its wait loop), so
    # the waiter goes first -- it is this tool's own process, recorded at request
    if [ -n "$qpid" ] && kill -0 "$qpid" 2>/dev/null && grep -q "fleet.sh" "/proc/$qpid/cmdline" 2>/dev/null; then kill "$qpid" 2>/dev/null; sleep 1; echo "stopped waiter pid $qpid"; fi
    with_lock _dequeue "$s"; logit "cancel $s"; echo "cancelled $s";;
  kick)
    if [ ! -s "$H" ]; then echo "nothing held"; exit 0; fi
    if holder_alive && [ "${1:-}" != "--force" ]; then echo "holder is ALIVE: $(holder_line) -- use --force only on the operator's word" >&2; exit 1; fi
    logit "kick${1:+ $1} of $(holder_line)"; rm -f "$H"; touch "$LOGD"/FLEET-free-for-{fusion,mkg3,b12x,glmfix}.done; echo "kicked";;
  busy) echo "$(busy_procs) $(busy_reqs)";;
  preflight)
    [ $# -ge 1 ] || { echo "usage: fleet.sh preflight [--probe] <session> [-- cmd...]" >&2; exit 2; }
    preflight "$@";;
  pair)
    s=${1:?session}; name=${2:?NAME}; knobs=${3:-}; est=${4:-25}; note=${5:-pair $name}
    [ "${FLEET_REHEARSE:-0}" = 1 ] && lane=--cpu || lane=--gpu
    exec bash "$0" run $lane "$s" "$est" "$note" -- bash "$REPO/bench/pair.sh" "$name" "$knobs";;
  deploy)
    s=${1:?session}; rev=${2:?rev}
    [ -s "$H" ] && [ "$(cut -d'|' -f1 "$H")" = "$s" ] || { echo "deploy needs the fleet: $s is not the holder ($(holder_line 2>/dev/null || echo none))" >&2; exit 1; }
    ( cd "$REPO" && git fetch -q origin "$rev" && git checkout -q -B ab FETCH_HEAD && git log --oneline -1 && bash launchers/deploy-overlays.sh glm53 2>&1 | tail -3 ) || { logit "deploy FAILED $s $rev"; exit 1; }
    stamp=$(cut -c1-12 "${MK_OVERLAY_STAMP:-$HOME/glm53-cache/.overlay-sha}" 2>/dev/null); sha=$(git -C "$REPO" rev-parse --short HEAD)
    printf '%s\t%s\t%s\t%s\t%s\n' "$(ts)" "$stamp" "$sha" "$s" "$rev" >> "$FLEET_DIR/builds.tsv"
    logit "deploy $s $rev -> build $stamp = $sha"; echo "deployed build $stamp = $sha (registry: $FLEET_DIR/builds.tsv)";;
  yield)
    s=${1:?session}; max=${2:-15}
    [ -s "$H" ] && [ "$(cut -d'|' -f1 "$H")" = "$s" ] || { echo "not the holder"; exit 0; }
    cand=$(awk -F'|' -v m="$max" '$6=="probe" && $4+0<=m {print $2; exit}' "$Q")
    [ -n "$cand" ] || { echo "nothing to yield to"; exit 0; }
    serving_idle || { echo "serving not idle; not yielding"; exit 0; }
    IFS='|' read -r hs hpid hhost ht0 hest hnote hkind < "$H"
    logit "yield $s -> $cand"; _event yield "$s" "$cand"
    with_lock _yield_requeue "$s" "$hest" "$hnote"
    FLEET_NO_RESTORE_CHECK=1 with_lock _release "$s"
    echo "yielded to $cand; waiting to resume"
    # give the probe its head start: its waiter polls every 15 s, ours would win the race otherwise
    for i in $(seq 1 15); do [ -s "$H" ] && [ "$(cut -d'|' -f1 "$H")" = "$cand" ] && break; sleep 3; done
    FLEET_PID=${FLEET_PID:-$PPID} bash "$0" wait "$s" "${FLEET_TIMEOUT_MIN:-720}";;
  nodes) nodes_check;;
  notify) s=${1:?session}; shift; [ $# -gt 0 ] && { echo "$*" > "$FLEET_DIR/notify.$s"; echo "hook for $s: $*"; } || { rm -f "$FLEET_DIR/notify.$s"; echo "hook for $s removed"; };;
  events) tail -"${1:-20}" "$FLEET_DIR/events.log" 2>/dev/null;;
  board) (cd "$REPO" && FLEET_DIR="$FLEET_DIR" LOGD="$LOGD" python3 bench/board.py --n "${1:-12}");;
  classify) shift 0; classify_cmd "$@";;
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
