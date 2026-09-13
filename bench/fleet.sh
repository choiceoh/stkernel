#!/usr/bin/env bash
# fleet.sh -- one queue for canonical onepass GPU experiments.
#
# QUICKSTART
#   fleet.sh onepass s NAME [est] [note]                 reuse idle serving, onepass only
#   fleet.sh pair s NAME "VLLM_X=1" [est] [note]          candidate + missing matched baseline
#   fleet.sh chain s [est] [note] -- A="VLLM_X=1" B=""    onepass once per requested arm
#   fleet.sh submit agent spec.json                      async CPU / canonical GPU pair, deduplicated
#   fleet.sh result ID | inbox agent --after CURSOR       shared evidence without holding GPUs
#   fleet.sh run --cpu s [est] [note] -- <cmd>             CPU work runs now in parallel
#   fleet.sh run --gpu [--detach] s [est] [note] -- <canonical argv>
#   fleet.sh run --gpu --prepared MANIFEST s [est] [note] -- <canonical argv>
#   fleet.sh prepare s --spec prep.json -- <cmd>          prepare inputs without queueing
#   fleet.sh retry agent ID --reason "fixed"              reuse valid evidence
#   fleet.sh edit s --expect-revision N -- <canonical argv> revise before GO, retain ticket
#   fleet.sh pause s --reason "revise inputs" | resume s  retain ticket and age
#   fleet.sh show [s] | logs s | history s               exact command, retained output
#   fleet.sh status | board | events | ledger [days]      queue, timings and results
#   fleet.sh classify --explain <cmd>                    CPU/GPU classification evidence
#   fleet.sh prune [--days N] [--apply]                  the queue's own debris, dry by default
#   fleet.sh run --gpu s [est] [note] -- bash probes/run_engine_check.sh --layers 0-4
#   fleet.sh run --gpu s [est] [note] -- bash probes/run_engine_probe.sh probes/engine_decode_graph_check.py
#   fleet.sh cancel s                                   stop the waiter and withdraw
#   fleet.sh run --gpu --fleet s [est] [note] -- <ST check>   keep a one-GPU check on the four Sparks
#   fleet.sh kick [--force] [single]                    a dead holder: the fleet's, or the single GPU's
#   fleet.sh st-pair s <sha> [--base <sha>] [est] [note]   the ST engine: one commit against the deployed one, two runs per boot
#   fleet.sh st-chain s [est] [note] -- A=<sha> B=<sha> A B  ST arms in order; a repeated name alternates (A B A B)
#   fleet.sh st-hold s <sha> [est] [note]               boot a commit and keep it for a session's window (end: cancel s)
#   fleet.sh st-probe [--detach] s [sha] [est] [note]   two onepass runs on the LIVE door when idle: D17's sample of the deployed commit
#   fleet.sh window s [MINUTES|off]                      a campaign window: production stays down between this session's tickets
#   fleet.sh run ... --replaces OLD                       a fresh ticket that inherits OLD's place in line (cancel + re-request loses it)
#
# TWO LANES. A boot, a pair, a chain, a live onepass take the fleet: four Sparks, one holder.
# An ST check that needs ONE GPU (probes/run_engine_check.sh, or run_engine_probe.sh without
# --distributed: one container, the four ranks as threads on one card) does not wait for the
# Sparks. It takes the single-GPU lane: ONE Spark beside production -- srv4 by default
# (FLEET_SINGLE_GPU_HOST; set it empty to turn the lane off) -- with its own holder
# (holder-single) and its own evidence. Beside production the GPU is never "free", so the
# evidence is ROOM: that box's MemAvailable less the check's budget (ST_PROBE_GIB, 8 GiB by
# default) must stay above the 16 GiB floor a --test boot keeps (bench/fleet_single.py), and
# only one probe container runs there at a time; a box that cannot answer has no room. On a
# fleet box the two lanes do exclude each other -- a fleet BOOT and a single check never share
# it -- while beside serving they run at once. Elsewhere (a box of its own, such as ost-97x,
# the operator's Windows PC on the tailnet, once it has sshd and an x86_64 image) the lanes
# never wait for each other; the controller's ~/.ssh/config names that alias's address, user
# and port. The supervisor hands the check to probes/run_engine_probe.sh with ST_PROBE_HOST,
# which rsyncs engine/ and probes/ to that host, waits for room, runs the container there on
# the image production runs there, and takes no fleet lease. Say --fleet to keep a one-GPU
# check on the four Sparks (one that needs every rank file, say).
#
# GPU admission accepts current canonical pair, chain, ab-lever, onepass and
# recorded pair/baseline execution. Unknown wrappers, standalone GPU checks,
# sanitizers, custom probe manifests, chain --after and LEGS=none are refused
# before preparation/queueing. Pending edits and execution recheck this policy.
# The ST engine's canonical checks (probes/run_engine_check.sh and run_engine_probe.sh
# over a named, byte-pinned probe) are admitted too: they take the same four nodes, so
# they queue here rather than behind the launcher's own lock. The probe is named in
# fleet_onepass.ST_PROBES -- admitting the runner never admits an arbitrary probe path.
# The fleet LEASE (engine/base/fleet_lease.py) is the one record of who holds the four
# nodes, and this queue is its authority for tickets: GO takes it as queue/<session>, the
# payload's launcher only verifies it, release hands it to the next waiting boot ticket or
# lets it go. Production holds a `production` lease and is asked to hand over only through
# the quiet gate; a session's own boot is never asked (45차 §91). A grant is refused while
# any st-* container is up or the lease is another's.
# --probe is the internal idle-serving scheduling lane; only onepass.py may
# enter it. A rehearsal skips GPUs only for canonical pair/chain/ab-lever.
# No bypass: a FAILed preflight is not queued. Fix the printed cause.
#
# Sessions release immediately after measurements. Recovery belongs exclusively
# to the central controller after 300 seconds without fleet activity; its
# authenticated boot-only maintenance action is separate from experiments.
# A matching baseline is reused; pair and chain target one sample by default.
# The queue retains aging, downstream priority, pending edits and pause/resume.
#
# Operator/control commands retained for owner lifecycle and compatibility:
#   wait s [timeout_min] (registered supervisor only); release s
#   front s; kick [--force]; busy; nodes
#   preflight [--probe] s -- <canonical argv>; deploy s rev; yield s [max_est]
#   restore-needed s (always no); notify s "<cmd>" (event hook)
# Bare request and unvalidated adopt are disabled.
# None of these makes an arbitrary GPU payload an approved experiment.
# Live owners are never killed by ordinary scheduling; dead owners can be kicked.
#
# Source copies ab-lever2.sh and fleet.sh are refreshed during preflight.
# Control scripts and accepted source are pinned before waiting; the holder
# executes the latest accepted command revision. CPU admission uses the fast
# source gate; GPU quality, prefill, decode and acceptance use the same onepass.
# See bench/EXPERIMENTS.md for manifests, repeat samples and evidence semantics.
# Files live under $FLEET_DIR on srv2; mutations hold its .lock.
# srv2's public entry is ~/glm53-logs/fleet.sh.
set -uo pipefail
# Transport metadata must not become a preparation or payload dependency.
unset SSH_CLIENT SSH_CONNECTION SSH_TTY TERM_PROGRAM TERM_PROGRAM_VERSION LC_TERMINAL LC_TERMINAL_VERSION
# Whether the caller named a queue, captured before the default fills it in: naming one
# IS the statement that you mean that queue, wherever you are (tests, a private run).
FLEET_DIR_EXPLICIT=${FLEET_DIR:+1}
FLEET_DIR=${FLEET_DIR:-/home/choiceoh/glm53-logs/fleet}
# The queue is ONE queue and it lives on the controller. Homes are not shared between
# the Sparks, so running this anywhere else silently creates a second, empty queue in a
# directory nobody watches -- on 2026-09-12 three reservations sat in srv4's copy while
# srv2 (the real one, with the ledger and every campaign's heartbeat) said nothing was
# queued, and `status` answered FREE because that copy had no holder. Refuse instead.
FLEET_CONTROLLER=${FLEET_CONTROLLER:-srv2}
wrong_host() {
  [ "$(me)" != "$FLEET_CONTROLLER" ] && [ -z "$FLEET_DIR_EXPLICIT" ] && [ "${FLEET_ALLOW_LOCAL:-0}" != 1 ]
}
require_controller() {
  wrong_host || return 0
  cat >&2 <<EOF
REFUSED: the fleet queue lives on $FLEET_CONTROLLER and homes are not shared, so running it
on $(me) would use a different, empty $FLEET_DIR and answer about nothing.
  ssh $FLEET_CONTROLLER "cd \$REPO && bash bench/fleet.sh $*"
Set FLEET_ALLOW_LOCAL=1 only for a queue you mean to keep on this host.
EOF
  return 1
}
LOGD=${LOGD:-/home/choiceoh/glm53-logs}
export FLEET_DIR LOGD
# The checkout this script belongs to, not a fixed path: a worktree ran its helpers
# against /home/choiceoh/stkernel, which is whatever branch another session left there
# (on 2026-09-12 a branch with no bench/fleet_*.py at all, so every helper errored).
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export REPO
# Submissions return immediately. The detached runner comes back through run,
# preserving preflight, CPU classification and the existing GPU reservation.
case "${1:-}" in
  submit|batch|result|inbox|jobs|stats|plan|ack|collect|retire|estimate|retry) exec python3 "$REPO/bench/experiments.py" "$@";;
  await) shift; exec python3 "$REPO/bench/experiments.py" wait "$@";;
  pause|resume) exec python3 "$REPO/bench/fleet_pause.py" "$@";;
  show|logs|history) exec python3 "$REPO/bench/fleet_inspect.py" "$@";;
  prepare) shift; exec python3 "$REPO/bench/fleet_prepare.py" create --fleet "$REPO/bench/fleet.sh" "$@";;
  edit) shift; exec python3 "$REPO/bench/fleet_pending.py" "$@";;
  priority) exec python3 "$REPO/bench/fleet_priority.py" "$FLEET_DIR";;
esac
# "이 빌드의 기준점이 될 측정이 이미 있으면 알려주는 장치" (operator, 39차): before a
# session spends a boot on a defaults arm, say whether the deployed build already
# has one. bench/baseline.py reads the onepass records (overlay stamp + knobs).
baseline_line() {
  [ -f "$REPO/bench/baseline.py" ] || return 0
  (cd "$REPO" 2>/dev/null && timeout 20 python3 bench/baseline.py --brief 2>/dev/null) || true
}
HEAD_URL=${HEAD_URL:-http://10.10.10.2:8000}
# The single-GPU lane's host and card: one Spark beside production. `${VAR-default}`, not
# `:-`: an explicitly EMPTY host is the switch that turns the lane off, and then a one-GPU
# check takes the four Sparks as it did before. bench/fleet_single.py carries the same
# defaults (a test pins that).
FLEET_SINGLE_GPU_HOST=${FLEET_SINGLE_GPU_HOST-srv4}
FLEET_SINGLE_GPU_NAME=${FLEET_SINGLE_GPU_NAME:-GB10}
export FLEET_SINGLE_GPU_HOST FLEET_SINGLE_GPU_NAME
mkdir -p "$FLEET_DIR"
Q=$FLEET_DIR/queue; H=$FLEET_DIR/holder; HS=$FLEET_DIR/holder-single; L=$FLEET_DIR/log; LK=$FLEET_DIR/.lock
touch "$Q" "$L"
now() { date +%s; }
ts() { date +%F_%T; }
logit() { echo "$(ts) $*" >> "$L"; }
me() { hostname -s; }
LEDGER=$FLEET_DIR/ledger.tsv; JSONL=$LOGD/bracket-onepass.jsonl
hb_file() { echo "$FLEET_DIR/hb.$1"; }
# kind: boot and probe take the fleet (one holder, $H); single takes the one GPU ($HS).
kind_of() { case "${1:-}" in probe|single) echo "$1";; *) echo boot;; esac; }
lane_of() { [ "${1:-}" = single ] && echo single || echo fleet; }
holder_file() { [ "$(lane_of "${1:-}")" = single ] && echo "$HS" || echo "$H"; }   # kind -> its lane's holder
holder_file_of() {  # session -> the holder file naming it; 1 when it holds nothing
  local f; for f in "$H" "$HS"; do [ -s "$f" ] && [ "$(cut -d'|' -f1 "$f")" = "$1" ] && { echo "$f"; return 0; }; done; return 1
}
lane_front() { awk -F'|' -v lane="$(lane_of "${1:-}")" '{ k = ($6 == "single") ? "single" : "fleet" } k == lane { print $2; exit }' "$Q"; }   # kind -> the first queued session of its lane, in the ranked order
single_on_fleet() { case "${FLEET_SINGLE_GPU_ON_FLEET:-}" in 1) return 0;; 0) return 1;; esac; case "${FLEET_SINGLE_GPU_HOST#*@}" in srv[1-4]|srv[1-4].*|spark*|10.10.0.[1-4]|10.10.1.[1-4]|10.10.10.[1-4]|10.10.11.[1-4]) return 0;; *) return 1;; esac; }   # is the single host one of the fleet's own boxes? (= fleet_single.on_fleet)
serving_up() { docker ps --format '{{.Names}}' 2>/dev/null | grep -qE '^(glm53|st-glm53)$'; }   # production: vLLM's or the ST engine's
st_serving_up() { docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^st-glm53$'; }
# ---- the fleet lease: ONE record of who holds the four Sparks (engine/base/fleet_lease.py).
# This queue is its authority for tickets. _try_hold takes it as queue/<session> at GO (or
# finds it already handed to the ticket by a holder that drained), _release hands it to the
# next waiting boot ticket or lets it go, and the payload's launcher only VERIFIES it
# (ST_LEASE_OWNER). Production -- the supervisor, deploy-watch -- holds a `production` lease
# of its own; a session's own boot holds a `session` one. The record's kind decides what the
# queue may ask of its holder: see st_engine_ask. The single-GPU lane is not the fleet and
# takes no lease.
#
# Read from THIS checkout's pinned module, never from launchers/: the helper under launchers/
# was absent from every runner snapshot before PR #768, and "cannot read" rightly counts as
# occupied, so a runner-driven ticket could never be granted at all (2026-09-12, found
# reviewing §91).
LEASE=${FLEET_LEASE_PATH:-/home/choiceoh/glm53-logs/st-fleet.lock}
export FLEET_LEASE_PATH=$LEASE
lease() { python3 "${FLEET_RUNNER_REPO:-$REPO}/engine/base/fleet_lease.py" "$@" --path "$LEASE"; }
lease_state() { lease read 2>/dev/null || echo unreadable; }   # free | free (stale: ..) | <kind> <owner> on <host> since .. | unreadable
lease_kind() { lease kind 2>/dev/null || echo unreadable; }     # free | production | session | queue | probe | unreadable
lease_mine() { lease verify --owner "queue/$1" >/dev/null 2>&1; }   # held by this ticket, and alive
# The ST engine takes the whole fleet (one container per node, named st-*). The queue must
# SEE it whoever started it: holder-empty is not the same as free (2026-09-12, four nodes
# running st-glm53 while status said FREE).
# Containers AND the lease: the two disagreed once and the queue granted while the lease
# was still held, so three reservations died on it in two seconds each (2026-09-12).
#
# Two more ways FREE was wrong, both closed here (2026-09-12 evening):
#   - the look was LOCAL. This queue's node is one of four, so a fleet whose rank 0 had
#     gone while the other three still held their GPUs read exactly like an empty one.
#   - "I could not tell" answered FREE. A missing lease helper and an unreadable lease
#     both fell through to `return 1` -- and free is the single answer an occupancy check
#     may never give from ignorance. Every unknown below is EVIDENCE: it refuses, and the
#     status line says which node or which file could not be read.
# The two cheap sources (this node's docker, the head's lease file) are read every time.
# Only the three remote `docker ps` are cached, because _try_hold asks once a second.
ST_PROBE=$FLEET_DIR/.st-engine-elsewhere
ST_PROBE_TTL=${ST_PROBE_TTL:-20}
st_is_here() { case " $(hostname -I 2>/dev/null) " in *" $1 "*) return 0 ;; esac; return 1; }
st_engine_elsewhere() {   # the nodes this one cannot see, in parallel; cached for TTL seconds
  local age tmp ip
  if [ -f "$ST_PROBE" ]; then
    age=$(( $(now) - $(stat -c %Y "$ST_PROBE" 2>/dev/null || echo 0) ))
    [ "$age" -ge 0 ] && [ "$age" -le "$ST_PROBE_TTL" ] && { cat "$ST_PROBE"; return 0; }
  fi
  tmp=$(mktemp -d) || return 0
  for ip in ${FLEET_NODES_IPS:-10.10.10.1 10.10.10.2 10.10.10.3 10.10.10.4}; do
    st_is_here "$ip" && continue
    ( if out=$(timeout "${ST_PROBE_TIMEOUT:-6}" ssh -o BatchMode=yes -o ConnectTimeout=4 \
                 "choiceoh@$ip" "docker ps --format '{{.Names}} {{.Status}}'" 2>/dev/null); then
        printf '%s\n' "$out" | grep -E '^st-' | head -1 | sed "s|^|$ip: |" > "$tmp/$ip"
      else
        echo "$ip: unreachable -- this node cannot say the fleet is free" > "$tmp/$ip"
      fi ) &
  done
  wait
  cat "$tmp"/* 2>/dev/null | grep -v '^[[:space:]]*$' > "$ST_PROBE.$$"
  rm -rf "$tmp"; mv -f "$ST_PROBE.$$" "$ST_PROBE" 2>/dev/null || rm -f "$ST_PROBE.$$"
  cat "$ST_PROBE" 2>/dev/null
}
st_engine_lease() {   # the lease on the head: silence only when it says free, or it is this ticket's own
  local held; held=$(lease_state)
  case "$held" in
    free|free\ *) return 0 ;;
    unreadable) echo "lease: unreadable at $LEASE -- this node cannot say the fleet is free"; return 0 ;;
  esac
  # a lease handed to the ticket asking (its holder drained and transferred it) is not occupation
  [ -n "${ST_MINE:-}" ] && lease_mine "$ST_MINE" && return 0
  echo "lease: $held"
}
st_engine_evidence() {   # every reason to believe these four nodes are not ours; empty = free
  local here
  # containers of a holder whose lease was just handed to the asking ticket are on their way
  # out: the lease says whose turn it is, the containers say when the last holder has gone
  here=$(docker ps --format '{{.Names}} {{.Status}}' 2>/dev/null | grep -E '^st-' | head -1)
  [ -z "$here" ] || echo "$(me): $here"
  st_engine_lease
  st_engine_elsewhere
}
st_engine_up() { [ -n "$(st_engine_evidence)" ]; }
st_engine_line() { st_engine_evidence | paste -sd';' - | sed 's/;/; /g' | cut -c1-240; }
# ---- the single-GPU lane's occupancy: ROOM beside production on that box -- its MemAvailable
# over ssh less the check's budget must clear the --test floor, and no other probe may be
# there -- remembered for a TTL by bench/fleet_single.py. The fleet's rule holds here too:
# every unknown is EVIDENCE and refuses, and a helper that fails is not an empty (= room) answer.
single_gpu_label() { echo "$FLEET_SINGLE_GPU_NAME on $FLEET_SINGLE_GPU_HOST$(single_on_fleet && echo ' beside production')"; }
single_gpu_evidence() {   # every reason to believe the single GPU is not ours; empty = free
  [ -n "$FLEET_SINGLE_GPU_HOST" ] || { echo "single-GPU lane is off (FLEET_SINGLE_GPU_HOST is empty)"; return 0; }
  local out
  out=$(python3 "${FLEET_RUNNER_REPO:-$REPO}/bench/fleet_single.py" evidence --cache "$FLEET_DIR" 2>&1) && return 0
  echo "${out:-fleet_single.py gave no answer -- this queue cannot say the $FLEET_SINGLE_GPU_NAME is free}"
}
single_gpu_line() { single_gpu_evidence | paste -sd';' - | sed 's/;/; /g' | cut -c1-240; }
single_refused() {  # session reason -- logged once per distinct reason, not once per poll
  local marker=$FLEET_DIR/.single-refused.$1
  [ "$(cat "$marker" 2>/dev/null)" = "$2" ] && return 0
  printf '%s' "$2" > "$marker"; logit "hold refused (single): $2; $1 waits"
}
pace_line() {  # for status: how long production waits after the last ticket, and why
  local pace; pace=$(python3 "${FLEET_RUNNER_REPO:-$REPO}/bench/fleet_pace.py" show "$FLEET_DIR" 2>/dev/null) || return 0
  python3 - "$pace" <<'PY'
import json, sys
p = json.loads(sys.argv[1])
w = f", window {p['window_s'] // 60}m left by {p['window_session']}" if p.get('window_s') else ""
print(f"pace: production returns after {p['grace_s'] // 60}m{p['grace_s'] % 60:02d}s of quiet queue (adaptive {p['adaptive_s']}s{w})")
PY
}
single_line() {   # for status: the lane's holder, else its evidence, else FREE
  [ -n "$FLEET_SINGLE_GPU_HOST" ] || { echo "single: off (FLEET_SINGLE_GPU_HOST is empty; one-GPU checks take the four Sparks)"; return 0; }
  local what
  if [ -s "$HS" ]; then holder_alive "$HS" && what="HELD by $(holder_line "$HS")" || what="held by DEAD $(holder_line "$HS")"
  else
    what=$(single_gpu_line); [ -n "$what" ] || what=FREE
    if single_on_fleet && [ -s "$H" ] && [ "$(cut -d'|' -f7 "$H" | tr -d '\n')" = boot ]; then what="$what -- but the fleet boot $(cut -d'|' -f1 "$H") holds this box too"; fi
  fi
  echo "single ($(single_gpu_label)): $what"
}
# A queue answer is only as new as the copy that computed it, and `status` reads live files
# with whatever rules the caller's checkout happens to carry. On 2026-09-12 a session ran
# `cd ~/stkernel && bash bench/fleet.sh status` on the controller itself and was told FREE
# while four nodes served: that checkout was 5e0216cf, from before the check above existed,
# and the same command from a current tree said "TAKEN by the ST engine".
#
# Hashes and mtimes cannot judge that -- a fresh checkout of an old branch is new by both.
# So the ANSWERS carry a number. Bump FLEET_RULES whenever what this queue reports about
# occupancy or admission changes; a copy below the shared entry's number says so before it
# answers, and preflight's sync (which copies $REPO over $LOGD/fleet.sh) refuses to move the
# shared entry backwards. Copies older than this number cannot warn -- nothing inside them
# knows there is anything to warn about -- but from here on the class reports itself.
#   1  the ST-engine occupancy check (2026-09-12)
#   2  the single-GPU lane: one-GPU checks are admitted to the 5050, not the fleet (2026-09-12)
#   3  the lease is the queue's: taken at GO, asked by kind through the quiet gate, handed on (2026-09-13)
#   4  the single-GPU lane is one Spark beside production: room, not a free GPU, is the evidence,
#      and on a fleet box a fleet boot and a single check never share it (2026-09-13)
FLEET_RULES=4
entry_rules() { sed -n 's/^FLEET_RULES=\([0-9][0-9]*\).*/\1/p' "${1:?file}" 2>/dev/null | head -1; }
entry_line() {
  local entry=$LOGD/fleet.sh theirs
  [ -f "$entry" ] || return 0
  theirs=$(entry_rules "$entry")
  [ -n "$theirs" ] && [ "$theirs" -gt "$FLEET_RULES" ] || return 0
  echo "  OLDER RULES: $0 answers by rules $FLEET_RULES; $entry is at $theirs."
  echo "               Refresh this checkout before trusting FREE -- that is how a tree from"
  echo "               before the ST-engine check reported an empty fleet with four nodes serving."
}
# Refusing is not enough: a queued session would then wait for a human to go and ask.
# Whom the queue may ask is decided by the holder's KIND (engine/base/fleet_lease.py):
#   production  the supervisor's boot, the fleet's default state. Asked only through the
#               QUIET GATE -- nothing outstanding (st:quiet, no request) for FLEET_QUIET_S,
#               the rule deploy-watch applies to its own restarts. An idle engine parks its
#               conversations and hands the lease to the ticket; nobody's answer is cut.
#   session     another session's boot. Never asked (operator, 45차 §91): the ticket waits.
#   queue/probe the queue's own; it hands over by itself at release.
# The ask is made ONCE per ticket (the engine drains on its own from there), through the
# pinned module, and it names the ticket's supervisor -- kind queue, its pid on this host --
# so that the handover transfers the lease to exactly that record and no free moment exists
# in between. An ask that fails is said to have failed: a logged ask nobody made left a
# waiter and a holder each believing the other had been told (45차 §91).
FLEET_QUIET_S=${FLEET_QUIET_S:-120}
QUIET_WHY=""
production_quiet() {  # 0 once the door has answered "nothing outstanding" for FLEET_QUIET_S -- or has nobody to protect
  local load since now body served idle; now=$(now)
  body=$(curl -s -m 5 "$HEAD_URL/metrics" 2>/dev/null)
  load=$(printf '%s\n' "$body" | lease load 2>/dev/null); load=${load:-unknown}
  if [ "$load" != 0 ]; then rm -f "$FLEET_DIR/.quiet-since"; return 1; fi
  # The gate protects a user mid-conversation. An engine that has answered nobody since it booted,
  # or whose last request is already older than the gate, has nobody to protect: it is quiet now.
  # Tickets waited a median 13 minutes at this gate on 2026-09-13, mostly for a production that had
  # booted for no one.
  served=$(printf '%s\n' "$body" | awk '/^vllm:request_success_total(\{[^}]*\})? / {print int($2); exit}')
  idle=$(printf '%s\n' "$body" | awk '/^st:idle_seconds(\{[^}]*\})? / {print int($2); exit}')
  if [ "${served:-x}" = 0 ]; then QUIET_WHY="served nobody since it booted"; return 0; fi
  if [ -n "${idle:-}" ] && [ "$idle" -ge "$FLEET_QUIET_S" ]; then QUIET_WHY="idle for ${idle}s already"; return 0; fi
  [ -f "$FLEET_DIR/.quiet-since" ] || echo "$now" > "$FLEET_DIR/.quiet-since"
  since=$(cat "$FLEET_DIR/.quiet-since" 2>/dev/null || echo "$now")
  QUIET_WHY="quiet for ${FLEET_QUIET_S}s"
  [ $(( now - since )) -ge "$FLEET_QUIET_S" ]
}
st_engine_ask() {  # session pid est note -- under .lock, after a refused hold; logs each state once
  local s=$1 pid=$2 est=$3 note=$4 kind marker="$FLEET_DIR/.asked.$1"
  if lease_mine "$s"; then
    [ -f "$marker.handed" ] || { logit "lease handed to $s; waiting for the last holder's containers to exit"; touch "$marker.handed"; }
    return 0
  fi
  kind=$(lease_kind)
  case "$kind" in
    production) ;;
    free|unreadable) return 0 ;;                          # containers without a lease, or no answer: nobody to ask
    *) [ -f "$marker.waits" ] || { logit "hold refused: $(st_engine_line); $s waits (a $kind holder is not asked)"; touch "$marker.waits"; }; return 0 ;;
  esac
  [ -f "$marker" ] && return 0                            # asked once; the engine drains from here
  if ! production_quiet; then
    [ -f "$marker.busy" ] || { logit "hold refused: production holds the fleet and is not quiet; $s waits for ${FLEET_QUIET_S}s of quiet before asking"; touch "$marker.busy"; }
    return 0
  fi
  if lease yield --requester "queue/$s" --kind queue --pid "$pid" --host "$(me)" --est-minutes "$est" --note "$note" >/dev/null 2>&1; then
    touch "$marker"; logit "hold refused: production holds the fleet; ${QUIET_WHY:-quiet}, asked it to hand over to $s"; _event ask "$s" "$note"
  else
    logit "hold refused: production holds the fleet and could not be asked ($(lease_state))"
  fi
}
serving_idle() {  # a probe may run beside this: healthy, nothing in flight, not booting -- and something must be serving
  # A probe measures a door. With nothing serving, "idle" granted two D17 probes on 2026-09-13
  # (02:57, 03:54) that aborted a second later with "no engine answers": one while the fleet
  # was between tickets, one while production's containers were still being started.
  serving_up || return 1
  booting && return 1
  if st_serving_up; then
    # The ST engine says itself whether anything is outstanding (st:quiet, the same reading as
    # the quiet gate); an engine too old to say it is judged by its request gauges and a door
    # that answers; a door that does not answer is not idle.
    local load; load=$(curl -s -m 5 "$HEAD_URL/metrics" 2>/dev/null | lease load 2>/dev/null); load=${load:-unknown}
    case "$load" in
      0) return 0 ;;
      unknown) [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' "$HEAD_URL/v1/models")" = 200 ] && [ "$(busy_reqs)" = 0 ]; return ;;
      *) return 1 ;;
    esac
  fi
  [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' "$HEAD_URL/health")" = 200 ] && [ "$(busy_reqs)" = 0 ]
}
# expected minutes for a session: median of its last 5 actual holds, else the estimate
expected_min() {  # session est
  local m; m=$(awk -F'\t' -v s="$1" '$2==s {v[++n]=$5} END {if (n) {asort(v); print v[int((n+1)/2)]}}' "$LEDGER" 2>/dev/null)
  echo "${m:-$2}"
}
# ---- preflight: the traps that cost a boot on 09-06, checked before the boot
preflight() {  # [--probe|--single] session [-- cmd...] -> 0 PASS, 1 FAIL
  local ok=1 knobs="" chain="" kind=boot
  case "${1:-}" in --probe|--single) kind=${1#--}; shift;; esac
  echo "preflight $1 [$kind]:"
  shift
  [ "${1:-}" = "--" ] && shift
  if [ $# -gt 0 ]; then
    python3 "${FLEET_RUNNER_REPO:-$REPO}/bench/fleet_onepass.py" \
      --repo "${FLEET_RUNNER_REPO:-$REPO}" --cwd "$PWD" --kind "$kind" -- "$@" || return 1
  fi
  for pair in "ab-lever2.sh:bench/ab-lever.sh" "fleet.sh:bench/fleet.sh"; do
    local copy=$LOGD/${pair%%:*} src=$REPO/${pair#*:} have want
    [ -f "$copy" ] || continue
    # The sync is one-directional in code but not in effect: it copies whatever $REPO the
    # caller ran from over the shared entry, so a stale checkout used to be able to move
    # everyone's copy backwards. The rules number decides -- equal or newer syncs, older
    # is refused and named (2026-09-12).
    have=$(entry_rules "$copy"); want=$(entry_rules "$src")
    if [ -n "$have" ] && [ "$have" -gt "${want:-0}" ]; then
      echo "  FAIL $copy is at rules $have and $src is at ${want:-0}: refusing to move the shared entry back"; ok=0; continue
    fi
    if [ "$(md5sum < "$copy")" = "$(md5sum < "$src")" ]; then echo "  PASS $copy == repo"
    elif cp "$src" "$copy.new" 2>/dev/null && chmod +x "$copy.new" && bash -n "$copy.new" 2>/dev/null && mv "$copy.new" "$copy"; then
      # a stale copy is only ever a stale copy: sync it from the repo (the
      # source of truth) instead of costing the caller a turn (09-06: two
      # queue attempts lost) -- both copies, the tool's own and the runner's
      echo "  SYNC $copy <- repo (was stale)"; logit "preflight synced $(basename "$copy") from the repo"
    else echo "  FAIL $copy differs from $src and could not be synced"; rm -f "$copy.new"; ok=0; fi
  done
  if [ "${1:-}" = bash ] && [ -f "${2:-}" ]; then chain=$2; fi
  if [ -n "$chain" ]; then
    if bash -n "$chain" 2>/dev/null; then echo "  PASS syntax $chain"; else echo "  FAIL syntax $chain"; ok=0; fi
    if grep -qF 'if [[ $touched == 1 ]]; then' "$chain" && grep -q 'RESTORE' "$chain"; then
      echo "  FAIL legacy unconditional restore: guard cleanup with FLEET_RESTORE_MANAGED and use bench/fleet_entry.py idle (see probes/run_gemm_input_cta.sh)"; ok=0
    fi
    # Header examples describe the runner; only executable lines can set knobs.
    knobs=$(grep -vE '^[[:space:]]*#' "$chain" | grep -oE "VLLM_[A-Z0-9_]+=[^ \"'\\]*" | sort -u)
  fi
  [ $# -gt 0 ] && knobs="$knobs $(printf '%s ' "$@" | grep -oE "VLLM_[A-Z0-9_]+=[^ \"']*" | sort -u)"
  # The declared-knob rule is about the LAUNCHER: it forwards only the keys
  # profiles/glm53.env declares, so a boot chain that sets an undeclared knob
  # silently measures the default and costs a boot. A probe has no launcher --
  # run_mk_probe.sh builds its own container and passes its own env -- so the
  # VLLM_* names inside a probe runner are not knobs at all. Checking them
  # FAILed a probe turn on 09-06 over run_mk_probe.sh's own PROBE_CACHE lines
  # (VLLM_CACHE_ROOT, VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR).
  if [ "$kind" != boot ]; then
    echo "  SKIP declared-knob check ($kind: no launcher in the path)"
  else
    # The profile that will serve is the one the chain deploys, and chains pull
    # origin/main at their start (or the holder runs `fleet.sh deploy`): check
    # against origin/main, falling back to the checkout when the fetch is
    # impossible. A key declared only in the checkout (a branch not merged yet)
    # passes with a note; a key declared only by a tree the chain `cd`s into (a
    # PR checkout under ~/mkab) passes with a note; undeclared everywhere FAILs.
    # 09-06: a key merged to main minutes earlier FAILed against the stale checkout.
    local k undeclared="" prof_main="" prof_src=checkout behind=0
    if timeout 20 git -C "$REPO" fetch -q origin 2>/dev/null; then
      prof_main=$(git -C "$REPO" show origin/main:profiles/glm53.env 2>/dev/null) && prof_src=origin/main
      behind=$(git -C "$REPO" rev-list --count HEAD..origin/main 2>/dev/null || echo 0)
    fi
    local prof_here; prof_here=$(cat "$REPO/profiles/glm53.env")
    local profiles="" d
    for d in $( { [ -n "$chain" ] && grep -vE '^\s*#' "$chain"; printf '%s ' "$@"; } 2>/dev/null | grep -oE "cd +[^ ;&|)]+" | awk '{print $2}' | sed "s|^~|$HOME|" | sort -u); do
      [ -f "$d/profiles/glm53.env" ] && profiles="$profiles $d/profiles/glm53.env"
    done
    local only_here="" only_tree=""
    for k in $knobs; do
      if [ -n "$prof_main" ] && grep -qE "^${k%%=*}=" <<< "$prof_main"; then continue; fi
      if grep -qE "^${k%%=*}=" <<< "$prof_here"; then only_here="$only_here ${k%%=*}"
      elif [ -n "$profiles" ] && grep -qE "^${k%%=*}=" $profiles 2>/dev/null; then only_tree="$only_tree ${k%%=*}"
      else undeclared="$undeclared ${k%%=*}"; fi
    done
    if [ -n "$undeclared" ]; then echo "  FAIL undeclared in profiles/glm53.env (the launcher forwards only declared keys; checked $prof_src and checkout):$undeclared"; ok=0
    elif [ -n "$knobs" ]; then echo "  PASS knobs declared in $prof_src: $(echo $knobs | tr ' ' ',')"; fi
    [ -z "$only_here" ] || echo "  NOTE declared only in the checkout (not in origin/main yet):$only_here"
    [ -z "$only_tree" ] || echo "  NOTE declared only by a tree the chain cd's into (a PR checkout):$only_tree"
    [ "${behind:-0}" = 0 ] || echo "  NOTE checkout is $behind commit(s) behind origin/main -- the chain must pull (or fleet.sh deploy) before it boots"
    if [ -f "$REPO/bench/baseline.py" ]; then
      local kv; kv=$(echo $knobs | tr ' ' ',')
      (cd "$REPO" && timeout 20 python3 bench/baseline.py --brief ${kv:+--knobs "$kv"} 2>/dev/null | sed 's/^/  /') || true
    fi
  fi
  [ $ok = 1 ] && { echo "  -> PASS"; return 0; }
  echo "  -> FAIL: fix the cause above and run again (there is no override)"; return 1
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
  if [ "${FLEET_REHEARSE:-0}" = 1 ] && python3 "${FLEET_RUNNER_REPO:-$REPO}/bench/fleet_onepass.py" \
      --repo "${FLEET_RUNNER_REPO:-$REPO}" --cwd "$PWD" --rehearsal-only -- "$@" >/dev/null 2>&1; then
    echo nogpu; return
  fi
  # This reviewed entrypoint invokes nvcc --compile only. A .cu input is not
  # device execution; its argument parser rejects runtime/launcher commands.
  case "${1##*/}" in python|python3|python3.*)
    if [ "${2:-}" = bench/cpu_compile.py ] || [ "${2:-}" = "$REPO/bench/cpu_compile.py" ]; then
      echo nogpu; return
    fi;;
  esac
  local text="$*" f
  for f in "$@"; do
    [ -f "$f" ] || continue
    case "$f" in
      *.py) text="$text $(grep -vE '^\s*#' "$f" 2>/dev/null | grep -oE 'torch\.cuda|\.cuda\(|device=.cuda|--gpus|docker run' | head -3)";;   # code, not docstrings
      *)    text="$text $(grep -vE '^\s*#' "$f" 2>/dev/null)";;
    esac
  done
  local gpu='ab-lever|start-glm53|deploy-overlays|run_mk_probe|run_megakernel_bench|run_engine_probe|run_engine_check|docker run|--gpus|onepass\.py|bracket\.py|bench-dec|torch\.cuda|nvidia-smi|\.cu\b|cuda_'
  local cpu='MK_PROBE_NO_GPU=1|head_pack_accuracy_cpu|baseline\.py|judge\.py|test_logic\.py|b12x_static_compile_check|compile\.sh|nvcc |bash -n|^git |md5sum|proof\.py'
  if echo "$text" | grep -qE "$gpu"; then echo gpu
  elif echo "$text" | grep -qE "$cpu"; then echo nogpu
  else echo unknown; fi
}
audit_line() {  # a stale pin silently turns off CPU reuse and contract narrowing
  local out
  out=$( (cd "$REPO" 2>/dev/null && timeout 20 python3 - <<'PY'
import hashlib, sys
sys.path.insert(0, "bench")
from pathlib import Path
try:
    import cpu_contracts as cc, cpu_evidence as ce
except Exception as exc:                      # noqa: BLE001 -- never take status down
    print(f"audit: unreadable ({type(exc).__name__})"); raise SystemExit
root = Path(".").resolve()
def sha(rel):
    path = root / rel
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""
stale = []
if sha("tests/test_logic.py") != cc.LOGIC_AUDIT:
    stale.append("tests/test_logic.py")
for name in ("LOGIC_SOURCE_AUDIT", "FLEET_AUDIT", "STARTUP_AUDIT"):
    for rel, want in (getattr(ce, name, {}) or {}).items():
        if sha(rel) != want:
            stale.append(rel)
if not stale:
    raise SystemExit
print("audit: STALE -- CPU evidence falls back to full-tree and the planner cannot narrow")
print("       a contract change, so every unrelated commit re-runs the work (it sat like")
print("       this for five days once, seen only as four 'pre-existing' test failures).")
for rel in sorted(set(stale))[:6]:
    print("       drifted: " + rel)
print("       fix: review the change, then update bench/cpu_contracts.LOGIC_AUDIT and")
print("            bench/cpu_evidence.*_AUDIT. tests/test_fleet_source.py says it too.")
PY
  ) 2>/dev/null )
  [ -n "$out" ] && echo "$out" | sed 's/^/  /'
  return 0
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
    out=$(timeout 12 ssh -o BatchMode=yes -o ConnectTimeout=5 "choiceoh@$ip" "nvidia-smi -L >/dev/null 2>&1 && echo gpu=ok || echo gpu=FAIL; echo stray=\$(docker ps --format '{{.Names}}' 2>/dev/null | grep -vc '^glm53\$'); echo ram=\$(free -g | awk 'NR==2{print \$7}')G; for p in ${FLEET_NODE_PATHS:-/home/choiceoh/models/st-glm53-nvidia-tp4-9391}; do [ -e \"\$p\" ] && echo path=ok || echo path=MISSING:\$p; done" 2>/dev/null | tr '\n' ' ')
    [ -n "$out" ] || { out="ssh=FAIL"; }
    echo "  node $ip: $out"
    echo "$out" | grep -qE "FAIL|MISSING|stray=[1-9]" && ok=1
  done
  return $ok
}
restore_needed() {  # session -> always no
  echo "no (only the central controller restores after 300 seconds of idle fleet)"
  return 1
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
  local young='Up ([0-9]+ seconds|Less than a|[0-9] minutes|1[01] minutes)'
  if docker ps --format '{{.Names}} {{.Status}}' 2>/dev/null | grep -E '^st-glm53 ' | grep -qE "$young"; then
    [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' "$HEAD_URL/v1/models")" != 200 ]; return
  fi
  docker ps --format '{{.Names}} {{.Status}}' 2>/dev/null | grep -E '^glm53 ' | grep -qE "$young" \
    && [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' "$HEAD_URL/health")" != 200 ]
}
legacy_busy() { [ "$(busy_procs)" != 0 ] || [ "$(busy_reqs)" != 0 ] || booting; }

# ---- holder liveness. A boot holder of the FLEET holds the lease (queue/<session>), and the
# lease's own evidence rule answers -- one rule, not this file's pid/ssh/3x-estimate guesses
# beside it. A holder from before the queue took leases (no lease at all), a probe holder
# (beside production, no lease) and the single-GPU lane's holder are judged as before: pid on
# this host directly; elsewhere, asked over ssh (evidence first, the window only when the
# node cannot answer).
holder_alive() {  # [holder file], the fleet's by default
  # `local`, or this read lands in the CALLER's variables (bash scopes dynamically): _try_hold's
  # session became the dead holder's and its GO check failed once for nothing, on every kick.
  local hf=${1:-$H} s pid host t0 est note kind
  [ -s "$hf" ] || return 1
  IFS='|' read -r s pid host t0 est note kind < "$hf"
  if [ "$hf" = "$H" ] && [ "${kind:-boot}" = boot ]; then
    lease_mine "$s" && return 0
    case "$(lease_state)" in free|free\ *) ;; *) return 1 ;; esac   # the lease is somebody else's: this holder is not it
  fi
  if [ "$host" = "$(me)" ] && [ -n "$pid" ]; then kill -0 "$pid" 2>/dev/null && return 0; return 1; fi
  # A holder on another node used to be trusted blind for 3x its estimate -- a crashed
  # one blocked the fleet for two hours at est 40, and the recovery the header promises
  # is not installed. Ask instead: evidence first, and only fall back to the window when
  # the node cannot answer (unreachable is not free).
  if [ -n "$pid" ] && [ -n "$host" ]; then
    local answer
    answer=$(holder_probe "$host" "$pid")
    case "$answer" in alive) return 0 ;; gone) return 1 ;; esac
  fi
  [ $(( $(now) - t0 )) -lt $(( ${est:-30} * 60 * 3 )) ]
}

# holder_probe <host> <pid> -> alive|gone|unknown, cached briefly: holder_alive runs on
# every poll of every waiter, and an ssh each time would put the network in the hot path.
holder_probe() {
  local host=$1 pid=$2 cache="$FLEET_DIR/.holder-probe.$1.$2" now age out
  now=$(now)
  if [ -f "$cache" ]; then
    age=$(( now - $(stat -c %Y "$cache" 2>/dev/null || echo 0) ))
    [ "$age" -lt "${HOLDER_PROBE_TTL_S:-20}" ] && { cat "$cache"; return 0; }
  fi
  # /proc, not `kill -0`: that one answers "gone" for a process you do not own, which is
  # the opposite of the truth and would let a waiter take a held fleet.
  out=$(timeout 8 ssh -o BatchMode=yes -o ConnectTimeout=4 "choiceoh@$host" \
          "[ -d /proc/$pid ] && echo alive || echo gone" 2>/dev/null | tail -1)
  case "$out" in alive|gone) ;; *) out=unknown ;; esac
  printf '%s' "$out" > "$cache" 2>/dev/null || true
  printf '%s' "$out"
}
holder_line() { local hf=${1:-$H}; [ -s "$hf" ] && IFS='|' read -r s pid host t0 est note kind < "$hf" && echo "$s${kind:+ [$kind]} (pid $pid@$host, since $(date -d @$t0 +%H:%M), est ${est}m, $note)"; }

with_lock() { ( flock -x 9; "$@" ) 9>"$LK"; }

_enqueue() {  # session est note [kind] [pid] [enqueued_at] -- idempotent per session; a repeat refreshes est/note/kind in place
  local kind pid reconcile_rc; kind=$(kind_of "${4:-}"); pid=${5:-}
  # Parked and resumed records own their original ticket even when the queue
  # projection is absent. Older request/wait fixtures have no parking helper.
  if [ -f "${FLEET_RUNNER_REPO:-$REPO}/bench/fleet_pause.py" ]; then
    local reconcile_args=(); [ -z "$pid" ] || reconcile_args=(--pid "$pid")
    python3 "${FLEET_RUNNER_REPO:-$REPO}/bench/fleet_pause.py" reconcile "$1" ${reconcile_args[@]+"${reconcile_args[@]}"}; reconcile_rc=$?
    case "$reconcile_rc" in 0) return 0;; 1) :;; *) return 2;; esac
  fi
  if grep -q "^[0-9]*|$1|" "$Q"; then
    # two live processes under one session name would merge into one ticket
    # and take one turn between them (09-06: `run fusion` twice); refuse
    local qpid; qpid=$(grep "^[0-9]*|$1|" "$Q" | head -1 | cut -d'|' -f7)
    if [ -n "$qpid" ] && [ -n "$pid" ] && [ "$qpid" != "$pid" ] && kill -0 "$qpid" 2>/dev/null && [ "${FLEET_SAME_SESSION:-0}" != 1 ]; then
      echo "session '$1' is already queued by a live process (pid $qpid): inspect it with fleet.sh show $1; use fleet.sh edit $1 before GO, or another name for different work" >&2
      logit "refused duplicate session $1 (pid $pid vs queued $qpid)"; return 2
    fi
    awk -F'|' -v OFS='|' -v s="$1" -v est="${2:-30}" -v note="${3:-}" -v kind="$kind" -v pid="$pid" '$2==s {$4=est; $5=note; $6=kind; if (pid!="") $7=pid} {print}' "$Q" > "$Q.tmp" && mv "$Q.tmp" "$Q"
    [ "$kind" = single ] || python3 "${FLEET_RUNNER_REPO:-$REPO}/bench/fleet_idle.py" activity "$FLEET_DIR" enqueue || return 1
    return 0
  fi
  echo "$(now)$$|$1|${6:-$(now)}|${2:-30}|${3:-}|$kind|$pid" >> "$Q"; logit "request $1 est=${2:-30}m $3${4:+ [$4]}${6:+ (keeps the place of the ticket it replaces)}"
  # A single-GPU check is not fleet activity: the idle controller's clock is the fleet's.
  [ "$kind" = single ] || python3 "${FLEET_RUNNER_REPO:-$REPO}/bench/fleet_idle.py" activity "$FLEET_DIR" enqueue
}
_dequeue() {
  local existed=0 rowkind
  rowkind=$(grep "^[0-9]*|$1|" "$Q" | head -1 | cut -d'|' -f6)
  grep -q "^[0-9]*|$1|" "$Q" && existed=1
  grep -v "^[0-9]*|$1|" "$Q" > "$Q.tmp"; mv "$Q.tmp" "$Q"
  local marker
  for marker in priority-front priority-yield; do
    [ "$(cat "$FLEET_DIR/$marker" 2>/dev/null)" != "$1" ] || rm -f "$FLEET_DIR/$marker"
  done
  rm -f "$FLEET_DIR/.single-refused.$1"
  [ "$existed" = 0 ] || [ "$rowkind" = single ] || python3 "${FLEET_RUNNER_REPO:-$REPO}/bench/fleet_idle.py" activity "$FLEET_DIR" dequeue
  # A ticket that leaves the line takes its ask back, and a lease already handed to it (but
  # not yet GO: the fleet's holder file is not its) moves on. At GO the holder file IS its,
  # and the lease stays exactly where it is.
  lease withdraw-yield --requester "queue/$1" >/dev/null 2>&1 || true
  rm -f "$FLEET_DIR/.asked.$1" "$FLEET_DIR/.asked.$1".* 2>/dev/null
  [ "$(cut -d'|' -f1 "$H" 2>/dev/null)" = "$1" ] || ! lease_mine "$1" || _lease_pass_on "$1"
}
_withdraw_owned() {  # session pid; an older supervisor cannot erase a reused name
  local rowpid
  rowpid=$(grep "^[0-9]*|$1|" "$Q" | head -1 | cut -d'|' -f7)
  [ "$rowpid" = "$2" ] || return 0
  _dequeue "$1"
}
_position() { grep -n "^[0-9]*|$1|" "$Q" | head -1 | cut -d: -f1; }
_front() { { grep "^[0-9]*|$1|" "$Q"; grep -v "^[0-9]*|$1|" "$Q"; } > "$Q.tmp"; mv "$Q.tmp" "$Q"; echo "$1" > "$FLEET_DIR/priority-front"; logit "front $1"; }

_try_hold() {  # session pid est note [kind] -> 0 when held
  local s=$1 pid=$2 est=$3 note=$4 kind hf; kind=$(kind_of "${5:-}"); hf=$(holder_file "$kind")
  if [ "$kind" != single ] && ! { [ "$kind" = probe ] && [ "$(lease_kind)" = production ]; }; then
    # The fleet lane: the ST engine, a serving container, a legacy chain all occupy the
    # four Sparks. None of that is evidence about the single GPU, so the single lane
    # skips this and asks its own host below. Occupied by someone else -- a lease that is
    # not this ticket's, or st-* containers still up -- the lease's kind decides whether
    # the holder is asked (st_engine_ask) or waited for. A PROBE ticket runs beside
    # production (the live onepass, D17's free base sample): production's own lease is not
    # occupation for it, an idle door is its condition (serving_idle, below), and it takes
    # no lease; behind a session's or a ticket's boot it waits like everything else.
    if ST_MINE=$s st_engine_up; then
      st_engine_ask "$s" "$pid" "$est" "$note"
      return 1
    fi
    # A fleet BOOT never shares a box with a single check: the boot's admission needs that
    # box's free memory, and the check would be what earlyoom finds first. Beside serving
    # (a probe) they run at once, and on a box of its own the lanes never meet.
    if [ "$kind" = boot ] && single_on_fleet && [ -s "$HS" ] && holder_alive "$HS"; then return 1; fi
  elif [ "$kind" = single ] && single_on_fleet && [ -s "$H" ] && [ "$(cut -d'|' -f7 "$H" | tr -d '\n')" = boot ] && holder_alive "$H"; then
    single_refused "$s" "the fleet boot $(cut -d'|' -f1 "$H") holds this box too"; return 1
  fi
  if [ -s "$hf" ]; then
    if holder_alive "$hf"; then return 1; fi
    logit "auto-kick dead holder: $(holder_line "$hf")"; [ "$kind" = single ] || _kick_lease; rm -f "$hf"
    [ "$kind" = single ] || python3 "${FLEET_RUNNER_REPO:-$REPO}/bench/fleet_idle.py" activity "$FLEET_DIR" dead-holder || return 1
  fi
  # We hold .lock and have no live holder in this lane. Every waiter sees the same order;
  # priority cannot interrupt a pair/chain or steal a yielded holder's place. Each lane
  # takes its own head of that order: a one-GPU check behind a queued boot does not wait
  # for the Sparks, and a boot behind a queued check does not wait for the 5050.
  local eligibility=""
  serving_idle || eligibility=--boot-only
  python3 "${FLEET_RUNNER_REPO:-$REPO}/bench/fleet_priority.py" "$FLEET_DIR" --apply ${eligibility:+"$eligibility"} || logit "priority unavailable: retain FIFO"
  [ "$(lane_front "$kind")" = "$s" ] || return 1
  # wait may have captured these before an edit. Read the queue under the
  # admission lock so holder/ledger metadata match the accepted reservation.
  IFS='|' read -r _ _ _ est note kind _ <<< "$(grep "^[0-9]*|$s|" "$Q" | head -1)"
  kind=$(kind_of "$kind"); hf=$(holder_file "$kind")
  [ "$kind" = probe ] && ! serving_idle && return 1
  # never hand the fleet to a dead job (an orphaned waiter whose run process
  # was killed took a turn for pid 3710362 on 09-06 and was auto-kicked 2 s
  # later, dropping the live request with the same session name)
  [ -z "$pid" ] || kill -0 "$pid" 2>/dev/null || return 1
  if [ "$kind" = single ]; then
    # The lane's evidence, right before the grant: that host's own GPU process list.
    # Unreachable is not free. Logged once per distinct reason, not once per poll.
    local why; why=$(single_gpu_line)
    if [ -n "$why" ]; then single_refused "$s" "$why"; return 1; fi
  else
    legacy_busy && return 1
  fi
  python3 "${FLEET_RUNNER_REPO:-$REPO}/bench/fleet_pause.py" admission "$s"; local prepared_rc=$?
  [ "$prepared_rc" != 4 ] || return 4
  [ "$prepared_rc" = 0 ] || { _dequeue "$s"; return 3; }
  # The lease, as this ticket: already handed to it by the holder that drained, or taken now
  # from a free fleet (a stale one is reclaimed by acquire itself). The ticket's supervisor on
  # this host is the record's pid, so its death frees the fleet at once. A probe ticket runs
  # beside production and the single-GPU lane is not the fleet: neither takes one.
  if [ "$kind" = boot ] && ! lease_mine "$s"; then
    lease acquire --owner "queue/$s" --kind queue --pid "$pid" --est-minutes "$est" --note "$note" >/dev/null 2>&1 \
      || { logit "hold refused: the lease could not be taken for $s ($(lease_state))"; return 1; }
  fi
  python3 "${FLEET_RUNNER_REPO:-$REPO}/bench/fleet_handoff.py" admit "$FLEET_DIR" "$s" "$pid" "$kind" "$est" "$note" \
    || { [ "$kind" != boot ] || _lease_pass_on "$s"; return 1; }
  _dequeue "$s"
  if [ "$kind" = single ]; then
    logit "GO $s (pid $pid) [single: $(single_gpu_label)]"
  else
    rm -f "$LOGD"/FLEET-free-for-*.done 2>/dev/null; touch "$LOGD/FLEET-held-by-$s.done"
    logit "GO $s (pid $pid)$( [ "$kind" != boot ] || echo " holding the lease as queue/$s")"
  fi
  _event GO "$s" "$note"; return 0
}
_ledger_row() {  # session [holder file] -- from the holder file, before it is removed
  local s pid host t0 est note kind
  IFS='|' read -r s pid host t0 est note kind < "${2:-$H}"
  local held boots recs; held=$(( ($(now) - t0 + 30) / 60 ))
  boots=$(find "$LOGD" -maxdepth 1 -name 'boot-*.log' -newermt "@$t0" 2>/dev/null | wc -l)
  [ "$boots" = 0 ] && [ "${kind:-boot}" = boot ] && [ -f "$LOGD/glm53.log" ] && [ "$(stat -c %Y "$LOGD/glm53.log")" -ge "$t0" ] && boots=1
  case "${kind:-boot}" in probe|single) boots=0;; esac   # neither boots the fleet
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
_yield_requeue() {
  _enqueue "$1" "$2" "$3" boot "${FLEET_PID:-$PPID}"; _front "$1"
  echo "$4" > "$FLEET_DIR/priority-yield"
}   # the yielding holder resumes immediately after its chosen probe
_wait_work_stop() {  # the wait ended WITHOUT GO (timeout/failure): its CPU work was for this wait
  [ "${WW_ARMED:-0}" = 1 ] && [ "${WW_GO:-0}" != 1 ] && \
    python3 "${FLEET_RUNNER_REPO:-$REPO}/bench/fleet_waitwork.py" stop "$FLEET_DIR" "${WW_SESSION:-}" >/dev/null 2>&1 || true
  return 0
}
# The fleet goes ticket to ticket and returns to production only when nobody waits (operator,
# 2026-09-12: "대기 예약이 없을 때만 되돌리면 되지"). Let go, the production supervisor relaunches
# within 30 s and the next ticket would wait for a whole quiet window again. A waiting PROBE
# ticket runs beside production, so it is a reason to let go, not to hold.
_lease_pass_on() {  # session -- caller holds .lock; the holder file may already be gone
  local next npid
  next=$(python3 "${FLEET_RUNNER_REPO:-$REPO}/bench/fleet_handoff.py" next "$FLEET_DIR" "$1" 2>/dev/null) || next=""
  if [ -n "$next" ]; then
    npid=${next#* }; next=${next%% *}
    if lease transfer --owner "queue/$1" --to "queue/$next" --kind queue --pid "$npid" --host "$(me)" --note "handed from $1" >/dev/null 2>&1; then
      logit "lease handed from $1 to $next (a boot ticket waits; production returns when none does)"; _event handed "$1" "$next"; return 0
    fi
  fi
  if lease release --owner "queue/$1" >/dev/null 2>&1; then logit "lease released by $1 (no boot ticket waits: production restores itself)"; fi
  return 0
}
_kick_lease() {  # the fleet holder file's session loses its lease too (dead, or the operator's word)
  local hs; hs=$(cut -d'|' -f1 "$H" 2>/dev/null); [ -n "$hs" ] || return 0
  lease release --owner "queue/$hs" >/dev/null 2>&1 && logit "lease of $hs released with the kick"
  return 0
}
_release() {  # session -- whichever lane's holder names it
  local hf hkind
  if hf=$(holder_file_of "$1"); then
    hkind=$(cut -d'|' -f7 "$hf")
    _ledger_row "$1" "$hf"
    rm -f "$hf" "$(hb_file "$1")"; _event release "$1" ""
    if [ "$hf" = "$HS" ]; then logit "release $1 [single]"; return 0; fi
    rm -f "$LOGD/FLEET-held-by-$1.done"; logit "release $1"
    [ "${hkind:-boot}" != boot ] || [ "${FLEET_KEEP_LEASE:-0}" = 1 ] || _lease_pass_on "$1"
    python3 "${FLEET_RUNNER_REPO:-$REPO}/bench/fleet_idle.py" activity "$FLEET_DIR" release || return 1
    # The queue's pace, for whoever restores production: how soon the next ticket tends to come
    # after one ends (bench/fleet_pace.py -> restore-grace.json). A constant grace restored
    # production into the next ticket's face 17 times in one night (2026-09-13).
    [ ! -f "${FLEET_RUNNER_REPO:-$REPO}/bench/fleet_pace.py" ] || python3 "${FLEET_RUNNER_REPO:-$REPO}/bench/fleet_pace.py" grace "$FLEET_DIR" >/dev/null 2>&1 || true
    # legacy markers for chains that still poll them
    for p in fusion mkg3 b12x glmfix; do touch "$LOGD/FLEET-free-for-$p.done"; done
    return 0
  fi
  echo "not the holder: $(holder_line 2>/dev/null || echo none)" >&2; return 1
}

_adopt() {  # caller holds .lock throughout the ownership transition
  local s=$1 pid=$2 est=${3:-30} note=${4:-}
  if [ -s "$H" ] && holder_alive; then echo "fleet already held: $(holder_line)" >&2; return 1; fi
  if ST_MINE=$s st_engine_up; then echo "ST engine occupies the fleet: $(st_engine_line)" >&2; return 1; fi
  kill -0 "$pid" 2>/dev/null || { echo "pid $pid is not alive on $(me)" >&2; return 1; }
  lease_mine "$s" || lease acquire --owner "queue/$s" --kind queue --pid "$pid" --est-minutes "$est" --note "$note" >/dev/null 2>&1 \
    || { echo "the lease could not be taken for $s: $(lease_state)" >&2; return 1; }
  printf '%s|%s|%s|%s|%s|%s|boot\n' "$s" "$pid" "$(me)" "$(now)" "$est" "$note" > "$H"
  rm -f "$LOGD"/FLEET-free-for-*.done; touch "$LOGD/FLEET-held-by-$s.done"
  python3 "${FLEET_RUNNER_REPO:-$REPO}/bench/fleet_idle.py" activity "$FLEET_DIR" adopt || return 1
  logit "adopt $s (pid $pid) est=${est}m $note"; echo "held by $s (pid $pid)"
}
_kick() {  # [--force] [single] -- preserve the same lock used by idle recovery and admission
  local force="" lane=fleet a hf
  for a in "$@"; do case "$a" in --force) force=--force;; single|fleet) lane=$a;; "") ;; *) echo "usage: fleet.sh kick [--force] [single]" >&2; return 2;; esac; done
  hf=$(holder_file "$lane")
  if [ ! -s "$hf" ]; then echo "nothing held$([ "$lane" = single ] && echo ' (single)')"; return 0; fi
  if holder_alive "$hf" && [ -z "$force" ]; then echo "holder is ALIVE: $(holder_line "$hf") -- use --force only on the operator's word" >&2; return 1; fi
  logit "kick${force:+ $force}$([ "$lane" = single ] && echo ' [single]') of $(holder_line "$hf")"; [ "$lane" = single ] || _kick_lease; rm -f "$hf"
  if [ "$lane" = fleet ]; then
    python3 "${FLEET_RUNNER_REPO:-$REPO}/bench/fleet_idle.py" activity "$FLEET_DIR" kick || return 1
    touch "$LOGD"/FLEET-free-for-{fusion,mkg3,b12x,glmfix}.done
  fi
  echo "kicked"
}

cmd=${1:-status}; shift || true
# Everything that reads or moves queue STATE must be on the controller; policy-only
# subcommands (classify, preflight, version) are host-independent and stay usable here.
case "$cmd" in
  classify|preflight|version|nodes|busy) ;;
  *) require_controller "$cmd" "$@" || exit 2;;
esac
case "$cmd" in
  request)
    echo 'bare GPU reservations are disabled; use fleet.sh onepass, pair or chain' >&2; exit 2;;
  wait)
    s=${1:?session}; tmo=${2:-720}; pid=${FLEET_PID:-$PPID}
    python3 "${FLEET_RUNNER_REPO:-$REPO}/bench/fleet_onepass.py" --repo "${FLEET_RUNNER_REPO:-$REPO}" \
      --directory "$FLEET_DIR" --wait-owner "$s" "$pid" >/dev/null || exit 2
    est=$(grep "^[0-9]*|$s|" "$Q" | head -1 | cut -d'|' -f4); note=$(grep "^[0-9]*|$s|" "$Q" | head -1 | cut -d'|' -f5)
    kind=$(kind_of "$(grep "^[0-9]*|$s|" "$Q" | head -1 | cut -d'|' -f6)")
    [ -n "$est" ] || { with_lock _enqueue "$s" 30 "" "$kind" "$pid" || exit 6; est=30; note=""; }
    # 큐에서 기다리는 동안 CPU 작업이 자동으로 돈다(운영자: "gpu 없이 할수 있는 작업
    # 같으면 병렬로"). 기본은 step_sim 재검증 — onepass 원장의 최근 기록으로 비용 상수를
    # 다시 폴딩한다(bench/fleet_waitwork.py). GO 에도 끝나지 않았으면 두고 가고(컨트롤러
    # CPU, 플릿와 무관), 포기하는 끝남(TIMEOUT·실패)에서만 끊는다. FLEET_WAIT_WORK 로
    # 바꾸거나 off 로 끈다.
    WW_ARMED=0; WW_GO=0; WW_SESSION=$s
    if python3 "${FLEET_RUNNER_REPO:-$REPO}/bench/fleet_waitwork.py" start \
        --session "$s" --repo "${FLEET_RUNNER_REPO:-$REPO}" --jsonl "$JSONL" \
        --logd "$LOGD" --fleet-dir "$FLEET_DIR"; then
      WW_ARMED=1
    else
      echo "wait work: 시작 실패 — 대기만 한다" >&2
    fi
    trap _wait_work_stop EXIT
    t_end=$(( $(now) + tmo * 60 )); last=""; prep_at=$(( $(now) + 30 ))
    while [ "$(now)" -lt "$t_end" ]; do
      if [ -n "${FLEET_EXPERIMENT_ID:-}" ] && [ "$s" = "exp-$FLEET_EXPERIMENT_ID" ]; then
        python3 "$REPO/bench/experiments.py" pending "$FLEET_EXPERIMENT_ID" || { with_lock _dequeue "$s"; exit 1; }
      fi
      # an orphaned waiter (its run process gone) must not keep polling for a
      # dead pid; a request that vanished (a stale sibling took it, or a
      # cancel) is re-queued at the back instead of waiting forever at "pos /0"
      kill -0 "$pid" 2>/dev/null || { echo "parent $pid is gone; giving up $(ts)" >&2; with_lock _dequeue "$s"; exit 1; }
      # Fetch/declared environment checks stay outside the reservation lock.
      # Fast local source checks run again under admission's lock below.
      if python3 "${FLEET_RUNNER_REPO:-$REPO}/bench/fleet_pause.py" is-paused "$s"; then
        [ "$last" = paused ] || echo "paused: $s; edit and resume to retain this ticket"
        last=paused; before_pause=$(now); sleep 1; t_end=$(( t_end + $(now) - before_pause )); continue
      fi
      [ -n "$(_position "$s")" ] || { with_lock _enqueue "$s" "$est" "$note" "$kind" "$pid" || exit 6; echo "re-queued: $s (entry was gone) $(ts)"; }
      if [ "$(now)" -ge "$prep_at" ]; then
        python3 "${FLEET_RUNNER_REPO:-$REPO}/bench/fleet_prepare.py" check "$s" --refresh --withdraw-failed; prepare_rc=$?
        [ "$prepare_rc" != 4 ] || continue
        [ "$prepare_rc" = 0 ] || exit 3
        prep_at=$(( $(now) + 30 ))
      fi
      with_lock _try_hold "$s" "$pid" "$est" "$note" "$kind"; admission_rc=$?
      if [ "$admission_rc" = 0 ]; then
        WW_GO=1
        python3 "${FLEET_RUNNER_REPO:-$REPO}/bench/fleet_waitwork.py" note "$FLEET_DIR" "$s"
        echo "GO $s $(ts)"; exit 0
      fi
      if [ "$admission_rc" = 3 ]; then exit 3; fi
      [ "$admission_rc" != 4 ] || continue
      why="pos $(_position "$s")/$(grep -c . "$Q")"; hf=$(holder_file "$kind"); [ -s "$hf" ] && why="$why, held by $(holder_line "$hf")"
      if [ "$kind" = single ]; then
        sgl=$(single_gpu_line); [ -z "$sgl" ] || why="$why, $sgl"
        single_on_fleet && [ -s "$H" ] && why="$why, the fleet's $(cut -d'|' -f1 "$H") holds this box too"
      else
        legacy_busy && why="$why, legacy busy ($(busy_procs) procs, $(busy_reqs) reqs$(booting && echo ', booting'))"; why="$why, lease: $(lease_state | cut -c1-120)"
        single_on_fleet && [ -s "$HS" ] && why="$why, a single check holds $FLEET_SINGLE_GPU_HOST"
      fi
      [ "$why" = "$last" ] || { echo "waiting: $why $(ts)"; last=$why; }
      sleep 1
    done
    echo "TIMEOUT $s after ${tmo}m" >&2; exit 1;;
  version) vr=${FLEET_RUNNER_REPO:-$REPO}; sha256sum "$vr/bench/fleet.sh" "$vr/bench/fleet_boot.py" "$vr/bench/fleet_handoff.py"; echo "handoff_protocol=2";;
  release) with_lock _release "${1:?session}";;
  run)
    kind=boot; force=""; detach=0; prepare_spec=""; prepared_manifest=""; lane_force=""; replaces=""
    while :; do case "${1:-}" in --probe) kind=probe; shift;; --cpu|--nogpu) force=nogpu; shift;; --gpu) force=gpu; shift;; --fleet) lane_force=fleet; shift;; --detach) detach=1; shift;; --prepare) prepare_spec=${2:?preparation spec}; shift 2;; --prepared) prepared_manifest=${2:?prepared manifest}; shift 2;; --replaces) replaces=${2:?the session this ticket replaces}; shift 2;; *) break;; esac; done
    s=${1:?session}; shift; est=30; note=""
    [ "${1:-}" != "--" ] && { est=$1; shift; }
    [ "${1:-}" != "--" ] && { note=$1; shift; }
    [ "${1:-}" = "--" ] && shift
    [ $# -gt 0 ] || { echo "usage: fleet.sh run --gpu|--cpu [--probe] [--fleet] <session> [est_min] [note] -- cmd..." >&2; exit 2; }
    if [ "$detach" = 1 ]; then
      detached_args=(run)
      [ "$kind" = probe ] && detached_args+=(--probe)
      [ -z "$lane_force" ] || detached_args+=(--fleet)
      case "$force" in gpu) detached_args+=(--gpu);; nogpu) detached_args+=(--cpu);; esac
      [ -n "$prepare_spec" ] && detached_args+=(--prepare "$prepare_spec")
      [ -n "$prepared_manifest" ] && detached_args+=(--prepared "$prepared_manifest")
      [ -n "$replaces" ] && detached_args+=(--replaces "$replaces")
      detached_args+=("$s" "$est" "$note" -- "$@")
      exec python3 "$REPO/bench/fleet_launch.py" start "$REPO/bench/fleet.sh" "$s" -- "${detached_args[@]}"
    fi
    export FLEET_SESSION=$s
    auto=$(classify_cmd "$@"); cls=${force:-$auto}
    if [ "$force" = nogpu ] && [ "$auto" = gpu ]; then
      echo "REFUSED: you said --cpu but the job shows GPU use (a boot, ab-lever, a probe container, torch.cuda); run it --gpu, or fix the classifier if it is wrong" >&2
      python3 "$REPO/bench/fleet_classify.py" --classification "$auto" -- "$@" >&2
      logit "refused --cpu $s: classifier saw GPU use"; exit 5
    fi
    [ -z "$force" ] && echo "no --gpu/--cpu given: classified as $auto"
    if [ "$cls" != nogpu ]; then
      contract=$(python3 "$REPO/bench/fleet_onepass.py" --repo "$REPO" --cwd "$PWD" --kind "$kind" -- "$@") || exit 2
      # A check that needs ONE GPU does not wait for the four Sparks: it takes the single-GPU
      # lane (the 5050 on ost-97x) unless the caller said --fleet or the lane is off.
      if [ "$kind" = boot ] && [ "$lane_force" != fleet ] && [ -n "$FLEET_SINGLE_GPU_HOST" ] \
          && [ "$(printf '%s' "$contract" | sed -n 's/.*"gpus": *\([0-9][0-9]*\).*/\1/p')" = 1 ]; then
        kind=single
        echo "needs one GPU, not four: single-GPU lane ($(single_gpu_label)); say --fleet to take the four Sparks instead"
      elif [ "$kind" = boot ] && [ "$lane_force" = fleet ] && [ -n "$FLEET_SINGLE_GPU_HOST" ] \
          && [ "$(printf '%s' "$contract" | sed -n 's/.*"gpus": *\([0-9][0-9]*\).*/\1/p')" = 1 ]; then
        # 33 one-GPU checks said --fleet on 2026-09-13: each cost production a drain and a boot, and
        # waited a median 13 minutes for it, while the single lane answered in 0. Say so, once.
        echo "note: this check needs one GPU; --fleet takes the four Sparks (production drains, then reboots) -- the single-GPU lane beside production ($(single_gpu_label)) would take it now"
      fi
    fi
    prep_args=(); [ -n "$prepare_spec" ] && prep_args=(--spec "$prepare_spec")
    [ -n "$prepared_manifest" ] && prep_args+=(--prepared "$prepared_manifest")
    [ "$cls" = nogpu ] || [ "$kind" != boot ] || prep_args+=(--approve-deploy)
    FLEET_PREPARE_MANIFEST=$(python3 "$REPO/bench/fleet_prepare.py" create "$s" --fleet "$REPO/bench/fleet.sh" ${prep_args[@]+"${prep_args[@]}"} -- "$@") || exit 3
    export FLEET_PREPARE_MANIFEST
    if [ "$cls" = nogpu ]; then
      # no GPU needed: run now, in parallel, under nice; no hold, no queue
      logit "nogpu-start $s $note"; _event nogpu-start "$s" "$note"; t0=$(now)
      python3 "$REPO/bench/fleet_launch.py" cpu-started "$s" || exit 3
      nice -n 19 "$@"; rc=$?
      python3 "$REPO/bench/fleet_launch.py" complete "$s" "$rc" || rc=3
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(ts)" "$s" nogpu "$note" $(( ($(now) - t0 + 30) / 60 )) 0 0 0 >> "$LEDGER"
      logit "nogpu-done $s rc=$rc"; _event nogpu-done "$s" "$note"; exit $rc
    fi
    [ "$cls" = unknown ] && echo "no evidence either way -> queued as GPU (say --cpu to run in parallel)"
    pf=(); [ "$kind" = boot ] || pf=(--$kind)
    if ! preflight ${pf[@]+"${pf[@]}"} "$s" -- "$@"; then
      logit "preflight FAIL $s (not queued)"; _event preflight-fail "$s" "$note"; exit 3
    fi
    if [ "$kind" = boot ]; then
      # Sessions validate only their candidate. The central idle controller
      # selects a release-validated recovery when the fleet has been idle.
      export FLEET_VALIDATION_STORE=${FLEET_VALIDATION_STORE:-$FLEET_DIR/validation}
      python3 "$REPO/bench/fleet_prepare.py" validate-targets "$s" --prepared "$FLEET_PREPARE_MANIFEST" >&2 || exit 3
      unset FLEET_RECOVERY_RECEIPT
      export FLEET_VALIDATION_REQUIRED=1 FLEET_VALIDATION_LEVEL=admission
    fi
    runner=$(with_lock python3 "$REPO/bench/fleet_pin.py" "$REPO" "$FLEET_DIR") || exit 3
    inherited=""
    if [ -n "$replaces" ]; then
      # A cancel followed by a fresh name loses the place the old ticket had earned (19 of 77 tickets
      # went that way on 2026-09-13). The new ticket takes the old one's enqueue time; the old one goes.
      inherited=$(grep "^[0-9]*|$replaces|" "$Q" | head -1 | cut -d'|' -f3)
      if [ -n "$inherited" ]; then
        bash "$0" cancel "$replaces" >/dev/null 2>&1 || true
        echo "replaces $replaces: keeps its place in line (queued $(date -d @"$inherited" +%H:%M))"
      else
        echo "replaces $replaces: it is not queued; this ticket starts a place of its own"
      fi
    fi
    with_lock _enqueue "$s" "$est" "$note" "$kind" "$$" ${inherited:+"$inherited"} || exit 6
    export FLEET_RUN_KIND=$kind
    exec python3 "$runner/bench/fleet_boot.py" "$runner/bench/fleet.sh" "$s" "$est" "$note" "$@";;
  status)
    echo "fleet: $( [ -s "$H" ] && { holder_alive && echo "HELD by $(holder_line)" || echo "held by DEAD $(holder_line)"; } \
                  || { st_engine_up && echo "TAKEN by the ST engine, outside this queue -- $(st_engine_line)" || echo FREE; } )"
    echo "lease: $(lease_state)"
    entry_line
    single_line
    pace_line
    remaining=0
    if [ -s "$H" ]; then
      IFS='|' read -r hs hpid hhost ht0 hest hnote hkind < "$H"; held=$(( ($(now) - ht0) / 60 ))
      remaining=$(( hest - held )); [ $remaining -lt 0 ] && remaining=0
      [ $held -gt $(( hest * 2 )) ] && echo "  OVERDUE: held ${held}m against est ${hest}m (a flag, not a kill: operator's kick --force)"
      hbf=$(hb_file "$hs"); [ -f "$hbf" ] && [ $(( $(now) - $(stat -c %Y "$hbf") )) -gt 600 ] && echo "  SILENT: no heartbeat for $(( ($(now) - $(stat -c %Y "$hbf")) / 60 ))m"
    fi
    echo "legacy: $(busy_procs) bench/boot procs, $(busy_reqs) requests in flight$(booting && echo ', head booting')"
    audit_line
    echo "queue ($(grep -c . "$Q")):"; n=0; eta=$remaining; while IFS='|' read -r t s at est note kind qpid; do n=$((n+1)); exp=$(expected_min "$s" "$est"); echo "  $n. $s${kind:+ [$kind]} (since $(date -d @$at +%H:%M), est ${est}m, expect ~${exp}m, ETA ~$(date -d "@$(( $(now) + eta * 60 ))" +%H:%M)) $note"; eta=$(( eta + exp )); done < "$Q"
    ls -t "$LOGD"/FLEET-*.done 2>/dev/null | head -4 | while read -r f; do echo "  marker $(stat -c %y "$f" | cut -c12-16) $(basename "$f")"; done
    python3 "${FLEET_RUNNER_REPO:-$REPO}/bench/fleet_pause.py" list --format text
    echo "log:"; tail -4 "$L" | sed 's/^/  /'
    python3 "${FLEET_RUNNER_REPO:-$REPO}/bench/fleet_idle.py" status "$FLEET_DIR"
    production_line; deployed_line; baseline_line;;
  adopt) echo 'unvalidated GPU adoption is disabled; submit canonical onepass work' >&2; exit 2;;
  front) with_lock _front "${1:?session}"; echo "$1 -> position $(_position "$1")";;
  withdraw)
    if [ "${2:-}" = --pid ]; then with_lock _withdraw_owned "${1:?session}" "${3:?pid}"
    else with_lock _dequeue "${1:?session}"; fi;;
  cancel)
    s=${1:?session}; qpid=$(grep "^[0-9]*|$s|" "$Q" | head -1 | cut -d'|' -f7)
    if [ -z "$qpid" ]; then
      qpid=$(python3 "${FLEET_RUNNER_REPO:-$REPO}/bench/fleet_pause.py" pid "$s") || qpid=""
    fi
    # a live waiter re-queues a vanished entry within 15 s (its wait loop), so
    # the waiter goes first -- it is this tool's own process, recorded at request
    if [ -n "$qpid" ] && kill -0 "$qpid" 2>/dev/null && grep -qE "fleet.sh|fleet_boot.py" "/proc/$qpid/cmdline" 2>/dev/null; then kill "$qpid" 2>/dev/null; sleep 1; echo "stopped waiter pid $qpid"; fi
    with_lock _dequeue "$s"; logit "cancel $s"; echo "cancelled $s";;
  kick) with_lock _kick "$@";;
  busy) echo "$(busy_procs) $(busy_reqs)";;
  preflight)
    [ $# -ge 1 ] || { echo "usage: fleet.sh preflight [--probe|--single] <session> [-- cmd...]" >&2; exit 2; }
    preflight "$@";;
  startup)
    echo 'GPU work requires onepass: use fleet.sh pair/chain for startup knobs; separate startup request campaigns are disabled' >&2; exit 2;;
  onepass)
    s=${1:?session}; name=${2:?NAME}; est=${3:-5}; note=${4:-live onepass $name}
    exec bash "$0" run --gpu --probe "$s" "$est" "$note" -- python3 "$REPO/bench/onepass.py" --name "$name";;
  chain)
    s=${1:?session}; shift; est=30; note=""
    [ "${1:-}" != "--" ] && { est=$1; shift; }
    [ "${1:-}" != "--" ] && { note=$1; shift; }
    [ "${1:-}" = "--" ] && shift
    [ $# -gt 0 ] || { echo "usage: fleet.sh chain <session> [est] [note] -- NAME=KNOBS [...]" >&2; exit 2; }
    [ "${FLEET_REHEARSE:-0}" = 1 ] && lane=--cpu || lane=--gpu
    exec bash "$0" run $lane "$s" "$est" "${note:-chain $*}" -- bash "$REPO/bench/chain.sh" "$@";;
  pair)
    s=${1:?session}; name=${2:?NAME}; knobs=${3:-}; est=${4:-25}; note=${5:-pair $name}
    [ "${FLEET_REHEARSE:-0}" = 1 ] && lane=--cpu || lane=--gpu
    exec bash "$0" run $lane "$s" "$est" "$note" -- bash "$REPO/bench/pair.sh" "$name" "$knobs";;
  # ---- the ST engine's bracket (bench/st_bracket.sh): one committed sha per arm, production
  # shape, two onepass runs per boot (D17). A boot ticket like pair/chain: it takes the fleet
  # lease at GO and the release's own launcher verifies it.
  st-pair)   # fleet.sh st-pair s <sha> [--base <sha>] [est] [note]
    s=${1:?session}; sha=${2:?candidate sha}; shift 2; base=()
    [ "${1:-}" = --base ] && { base=(--base "${2:?base sha}"); shift 2; }
    est=${1:-25}; note=${2:-st-pair $sha}
    [ "${FLEET_REHEARSE:-0}" = 1 ] && lane=--cpu || lane=--gpu
    exec bash "$0" run $lane "$s" "$est" "$note" -- bash "$REPO/bench/st_bracket.sh" pair "$sha" ${base[@]+"${base[@]}"};;
  st-chain)  # fleet.sh st-chain s [est] [note] -- A=<sha> B=<sha> A B     (a repeated name alternates)
    s=${1:?session}; shift; est=45; note=""
    [ "${1:-}" != "--" ] && { est=$1; shift; }
    [ "${1:-}" != "--" ] && { note=$1; shift; }
    [ "${1:-}" = "--" ] && shift
    [ $# -gt 0 ] || { echo "usage: fleet.sh st-chain <session> [est] [note] -- NAME=<sha> [NAME=<sha> ...] [NAME ...]" >&2; exit 2; }
    [ "${FLEET_REHEARSE:-0}" = 1 ] && lane=--cpu || lane=--gpu
    exec bash "$0" run $lane "$s" "$est" "${note:-st-chain $*}" -- bash "$REPO/bench/st_bracket.sh" chain "$@";;
  st-hold)   # fleet.sh st-hold s <sha> [est] [note]: boot a commit and keep it for a session's window; end with cancel
    s=${1:?session}; sha=${2:?sha}; est=${3:-45}; note=${4:-st-hold $sha}
    exec bash "$0" run --gpu "$s" "$est" "$note" -- bash "$REPO/bench/st_bracket.sh" hold "$sha" "$est";;
  window)  # fleet.sh window s [MINUTES|off]: a campaign window -- production stays down between this session's tickets
    s=${1:?session}; m=${2:-30}
    python3 "${FLEET_RUNNER_REPO:-$REPO}/bench/fleet_pace.py" window "$FLEET_DIR" "$s" "$m" >/dev/null || exit 2
    if [ "$m" = off ]; then logit "window $s closed"; echo "window closed by $s"; else logit "window $s ${m}m"; echo "window open: production returns only after ${m}m of quiet queue (or the window closes)"; fi
    pace_line;;
  st-probe)  # fleet.sh st-probe [--detach] s [sha] [est] [note]: two onepass runs on the LIVE door when it is idle
    # -- D17's sample for the deployed commit, no boot, no lease. deploy-watch queues one after
    # every deploy, so st-pair never has to boot the base.
    detach=(); [ "${1:-}" = --detach ] && { detach=(--detach); shift; }
    s=${1:?session}; shift; sha=""
    if [ -n "${1:-}" ] && [[ "$1" =~ ^[0-9a-f]{7,40}$ ]]; then sha=$1; shift; fi
    est=${1:-10}; note=${2:-st-probe ${sha:-deployed}}
    exec bash "$0" run --gpu --probe ${detach[@]+"${detach[@]}"} "$s" "$est" "$note" -- bash "$REPO/bench/st_bracket.sh" probe ${sha:+"$sha"};;
  deploy)
    s=${1:?session}; rev=${2:?rev}
    [ -s "$H" ] && [ "$(cut -d'|' -f1 "$H")" = "$s" ] || { echo "deploy needs the fleet: $s is not the holder ($(holder_line 2>/dev/null || echo none))" >&2; exit 1; }
    ( cd "$REPO" && git fetch -q origin "$rev" && git checkout -q -B ab FETCH_HEAD && git log --oneline -1 && bash launchers/deploy-overlays.sh glm53 2>&1 | tail -3 ) || { logit "deploy FAILED $s $rev"; exit 1; }
    stamp=$(cut -c1-12 "${MK_OVERLAY_STAMP:-$HOME/glm53-cache/.overlay-sha}" 2>/dev/null); sha=$(git -C "$REPO" rev-parse --short HEAD)
    printf '%s\t%s\t%s\t%s\t%s\n' "$(ts)" "$stamp" "$sha" "$s" "$rev" >> "$FLEET_DIR/builds.tsv"
    logit "deploy $s $rev -> build $stamp = $sha"; echo "deployed build $stamp = $sha (registry: $FLEET_DIR/builds.tsv)";;
  yield)
    # The supervisor owns its payload and teardown. Do not drop that ownership
    # for a nested legacy yield; queued work runs immediately at finish.
    [ "${FLEET_RESTORE_MANAGED:-0}" != 1 ] || { echo "yield deferred to supervised finish"; exit 0; }
    s=${1:?session}; max=${2:-15}
    [ -s "$H" ] && [ "$(cut -d'|' -f1 "$H")" = "$s" ] || { echo "not the holder"; exit 0; }
    cand=$(awk -F'|' -v m="$max" '$6=="probe" && $4+0<=m {print $2; exit}' "$Q")
    [ -n "$cand" ] || { echo "nothing to yield to"; exit 0; }
    serving_idle || { echo "serving not idle; not yielding"; exit 0; }
    IFS='|' read -r hs hpid hhost ht0 hest hnote hkind < "$H"
    logit "yield $s -> $cand"; _event yield "$s" "$cand"
    with_lock _yield_requeue "$s" "$hest" "$hnote" "$cand"
    # the holder keeps its LEASE across a yield to a probe: the probe runs beside its serving,
    # and letting the lease go would invite the production supervisor onto the same nodes
    FLEET_KEEP_LEASE=1 FLEET_NO_RESTORE_CHECK=1 with_lock _release "$s"
    echo "yielded to $cand; waiting to resume"
    # give the probe its head start: its waiter polls every 15 s, ours would win the race otherwise
    for i in $(seq 1 15); do [ -s "$H" ] && [ "$(cut -d'|' -f1 "$H")" = "$cand" ] && break; sleep 3; done
    FLEET_PID=${FLEET_PID:-$PPID} bash "$0" wait "$s" "${FLEET_TIMEOUT_MIN:-720}";;
  nodes) nodes_check;;
  notify) s=${1:?session}; shift; [ $# -gt 0 ] && { echo "$*" > "$FLEET_DIR/notify.$s"; echo "hook for $s: $*"; } || { rm -f "$FLEET_DIR/notify.$s"; echo "hook for $s removed"; };;
  events) tail -"${1:-20}" "$FLEET_DIR/events.log" 2>/dev/null;;
  board) (cd "$REPO" && FLEET_DIR="$FLEET_DIR" LOGD="$LOGD" python3 bench/board.py --n "${1:-12}");;
  prune)
    # The queue's own debris: a heartbeat per session ever seen, a launch per detached
    # run, a preparation per prepared input, a pinned runner per distinct source. Dry by
    # default; it never touches the holder, the queue, the paused, or a runner they name.
    exec python3 "$REPO/bench/fleet_prune.py" "$FLEET_DIR" "$@";;
  classify)
    if [ "${1:-}" = --explain ]; then
      shift; cls=$(classify_cmd "$@")
      python3 "$REPO/bench/fleet_classify.py" --classification "$cls" -- "$@"
    else classify_cmd "$@"; fi;;
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
