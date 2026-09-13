#!/usr/bin/env bash
# st_bracket.sh -- the ST engine's bracket on the four Sparks: one COMMITTED sha per arm, in
# production shape, measured the way D17 asks (two onepass runs on one boot).
#
#   bash bench/st_bracket.sh pair  <sha> [--base <sha>]          fleet.sh st-pair  s <sha> [--base <sha>] [est] [note]
#   bash bench/st_bracket.sh chain A=<sha> B=<sha> [A B ...]     fleet.sh st-chain s [est] [note] -- A=<sha> B=<sha> A B
#   bash bench/st_bracket.sh hold  <sha> [minutes]               fleet.sh st-hold  s <sha> [est] [note]
#   bash bench/st_bracket.sh probe [sha]                         fleet.sh st-probe [--detach] s [sha] [est] [note]
#
# An arm is a sha that origin has. The runner cuts it into $ST_RELEASES/<sha12> with
# launchers/st_release.py (the same cut deploy-watch makes, so a winner is promoted by pointing
# production at that very directory), pushes it to the four nodes as ST_ENGINE_DIR, and boots it
# with the shape production serves with ($ST_PRODUCTION_ENV: KV, rows, ST_PRODUCTION=1) on port
# $ST_BRACKET_PORT (8001: nothing outside reaches a candidate by accident), with a tier and a dump
# directory of its own. STK_* knobs are not arms: --production refuses them, and a sha says what it
# is. An arm's tree must carry the ticket-mode launcher (PR #770 or later): the release's own
# launcher is what boots it, and an older one would try to take a lease the queue already holds.
#
# The leg per arm is FIXED: boot -> onepass run 1 (the cold column: TTFT, the compile tail) ->
# POST /v1/prefix/reset -> onepass run 2 (the warm column: decode step/s, warm prefill) -> stop.
# Both runs --require-exclusive. bench/st_judge.py compares warm against warm and prints the
# cold column beside it (45차 §93's table); a chain's first arm is its base.
#
# The lease is the queue's: this runs under `fleet.sh run --gpu`, which took the fleet lease at GO
# and hands ST_LEASE_OWNER here; the release's launcher VERIFIES it, and its `stop` says the boot
# is down without letting the lease go (the queue passes it on or lets it go at the ticket's end).
# pair reuses the base arm when its sha already has a warm sample (a D17 probe ticket after a
# deploy, or an earlier bracket); ST_PAIR_FLOOR_N=1 by default.
# FLEET_REHEARSE=1 boots nothing: records are fabricated (rehearsal=true) so the flow, the judge
# and the queue's handling can be checked without GPUs.
set -uo pipefail
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
LOGD=${LOGD:-/home/choiceoh/glm53-logs}
S=${FLEET_SESSION:-st-bracket}
SOURCE=${ST_SOURCE:-/home/choiceoh/stkernel}
RELEASES=${ST_RELEASES:-/home/choiceoh/st-releases}
STATE=${ST_DEPLOY_STATE:-$RELEASES/deploy-state.json}
PROD_ENV=${ST_PRODUCTION_ENV:-/home/choiceoh/.config/st-glm53.env}
PORT=${ST_BRACKET_PORT:-8001}
RUNS=${ST_BRACKET_RUNS:-2}
BOOT_WAIT=${ST_BRACKET_BOOT_WAIT:-1800}
JSONL=${ONEPASS_JSONL:-$LOGD/bracket-onepass.jsonl}
FLOOR_N=${ST_PAIR_FLOOR_N:-1}
MODEL=${BENCH_MODEL:-glm-5.3-flash}
REHEARSE=${FLEET_REHEARSE:-0}
OUT=$LOGD/st-bracket/$S; mkdir -p "$OUT"
export ONEPASS_JSONL=$JSONL LEGS=onepass
cd "$REPO" || exit 1
say() { echo "== $(date +%T) $*"; }

RELEASE=""; ARM=""; ARM_SHA=""; ARM_TREE=""; DUMPS=""
sha_of() {  # the full commit id when the source tree can say; the name as given otherwise (rehearsal)
  python3 "$REPO/launchers/st_release.py" resolve "$1" --source "$SOURCE" 2>/dev/null || echo "$1"
}
tree_of() {  # the engine/ tree at that commit -- the sample's identity across squashes and fleet-side merges; empty when unknown
  python3 "$REPO/launchers/st_release.py" tree "$1" --source "$SOURCE" 2>/dev/null || true
}
tree_args() {  # --tree for st_judge, when the tree is known
  local tree; tree=$(tree_of "$1"); [ -z "$tree" ] || printf -- '--tree %s' "$tree"
}
release_of() {  # sha -> its release directory, cut if it is not yet; a rehearsal cuts nothing
  [ "$REHEARSE" != 1 ] || { echo "$RELEASES/rehearsal-${1:0:12}"; return 0; }
  python3 "$REPO/launchers/st_release.py" cut "$1" --source "$SOURCE" --releases "$RELEASES"
}
shape() {  # production's shape, minus what an arm decides for itself (tree, image, port, tier, dumps)
  local bracket_port=$PORT
  if [ -f "$PROD_ENV" ]; then set -a; . "$PROD_ENV"; set +a; fi
  local v; for v in $(compgen -v STK_ || true); do unset "$v"; done   # a sha is the arm; --production refuses knobs anyway
  export ST_PRODUCTION=1 PORT=$bracket_port
}
door() { echo "http://127.0.0.1:$PORT"; }
door_up() { curl -fsS --max-time 5 "$(door)/v1/models" 2>/dev/null | grep -q "\"$MODEL\""; }
# The four nodes, named the way the launcher and the supervisor name them (rank order); the head
# cannot ssh to itself. A boot's death is judged on ALL FOUR containers, not on rank 0's alone: a
# worker that dies first leaves rank 0 alive until the control group times out (120 s; the boot
# rendezvous 1800 s), and what rank 0 then dies of is "Connection closed by peer" -- the symptom,
# recorded as the cause for four boots on 2026-09-13 while the dying rank's own words were erased
# by the stop that followed. So the logs are pulled BEFORE the stop, as the supervisor does.
NODES=(${ST_NODES:-10.10.10.2 10.10.10.1 10.10.10.3 10.10.10.4})
SELF_IPS=" $(hostname -I 2>/dev/null) "
node_sh() { local ip=$1; shift
  case "$SELF_IPS" in *" $ip "*) bash -c "$*" </dev/null; return ;; esac
  ssh -n -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new "choiceoh@$ip" "$@"; }
dead_ranks() {  # one line per rank whose st-glm53 container is not running: "r ip exit=N oom=B" (or gone / unreachable)
  local r ip state
  for r in "${!NODES[@]}"; do ip=${NODES[$r]}
    state=$(node_sh "$ip" "docker inspect --format '{{.State.Running}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}}' st-glm53 2>/dev/null" 2>/dev/null) || state=unreachable
    case "$state" in true*) ;; "") echo "$r $ip gone" ;; *) echo "$r $ip ${state#false }" ;; esac
  done
}
first_error() {  # a rank's own last word: its last exception line, else its last stall/refusal/kill line
  grep -a -E "(Error|Exception)(: |$)|STALL rank=|refuse|Killed|different conversations|died" "$1" 2>/dev/null \
    | grep -v -E "error_recovery|errors=0|Warning" | tail -1 | cut -c1-300
}
forensics() {  # <dir>: the four ranks' last 400 lines, pulled before a stop erases them (the supervisor keeps the same)
  local d=$1 r ip; mkdir -p "$d" 2>/dev/null || return 0
  for r in "${!NODES[@]}"; do ip=${NODES[$r]}
    node_sh "$ip" "docker logs --tail=400 st-glm53" > "$d/rank$r-$ip.log" 2>&1 || true
  done
  say "forensics: $d"
}
wait_door() {  # the launcher returns when the containers start; the door answers minutes later (load, capture)
  local waited=0 dead r ip state line
  while [ "$waited" -lt "$BOOT_WAIT" ]; do
    door_up && return 0
    dead=$(dead_ranks)
    if [ -n "$dead" ]; then
      while read -r r ip state; do say "rank $r ($ip) died during boot: $state"; done <<< "$dead"
      forensics "$DUMPS"
      while read -r r ip state; do
        line=$(first_error "$DUMPS/rank$r-$ip.log"); [ -n "$line" ] && say "rank $r said: $line"
      done <<< "$dead"
      say "(rank logs: $DUMPS; launcher: $OUT/boot-$ARM.log)"
      return 1
    fi
    sleep 10; waited=$((waited + 10))
  done
  say "the door did not answer within ${BOOT_WAIT}s"; forensics "$DUMPS"; return 1
}
boot_arm() {  # name sha
  ARM=$1; local sha=$2
  RELEASE=$(release_of "$sha") || { say "ABORT: $sha could not be cut into a release"; return 1; }
  say "arm $ARM = $ARM_SHA -> $RELEASE (port $PORT)"
  [ "$REHEARSE" != 1 ] || return 0
  DUMPS=$LOGD/st-bracket-dumps/$S-$ARM
  ( shape
    export ST_ENGINE_DIR=$RELEASE ST_IMAGE="st-engine:bracket-${ARM_SHA:0:12}" REPO=$RELEASE
    export ST_TIER_DIR=$LOGD/st-bracket-tier/$S-$ARM ST_DUMP_DIR=$DUMPS
    mkdir -p "$ST_TIER_DIR" "$ST_DUMP_DIR"
    bash "$RELEASE/launchers/start-st-glm53.sh" start ) > "$OUT/boot-$ARM.log" 2>&1 \
    || { tail -5 "$OUT/boot-$ARM.log"; say "ABORT: the launcher refused or failed (see $OUT/boot-$ARM.log)"; return 1; }
  wait_door || { stop_arm; return 1; }
  say "door up: $(door)"
}
stop_arm() {  # the release's own stop: ST_LEASE_OWNER is the ticket's, so it stops this boot and no other
  [ "$REHEARSE" != 1 ] || return 0
  [ -n "$RELEASE" ] || return 0
  ( shape; export REPO=$RELEASE; bash "$RELEASE/launchers/start-st-glm53.sh" stop ) >> "$OUT/boot-$ARM.log" 2>&1 \
    || say "stop returned nonzero (see $OUT/boot-$ARM.log)"
}
rehearse_record() {  # run-index: a record shaped like the last real ST one, or a stub, marked rehearsal
  python3 - "$ARM" "$ARM_SHA" "$1" "$JSONL" <<'PY'
import json, os, sys, time
name, sha, run, path = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()] if os.path.exists(path) else []
real = [r for r in rows if r.get("engine") == "st" and not r.get("rehearsal")]
rec = dict(real[-1]) if real else {
    "decode": {"windows_med": 12.0, "tokens_per_step": 3.4, "acc_raw": 0.6}, "quality": {"ok": 9, "total": 9},
    "korean": {"dirty": 0, "n": 5}, "traffic": {"issues": []}, "harness": 42,
    "prefill": [{"ctx": 2000, "cold_s": 6.7, "warm_tok_s": 3000.0}, {"ctx": 32000, "cold_s": 40.0, "warm_tok_s": 3200.0}]}
rec.update({"name": name, "t": time.strftime("%F %T"), "rehearsal": True, "engine": "st", "arm_sha": sha,
            **({"arm_tree": os.environ["ST_BRACKET_TREE"]} if os.environ.get("ST_BRACKET_TREE") else {}),
            "run_index": run, "cold": os.environ.get("ST_BRACKET_COLD", "boot"),
            "session": os.environ.get("FLEET_SESSION", ""), "knobs": {}})
rec.pop("boot_id", None); rec.pop("run_id", None)
os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
with open(path, "a", encoding="utf-8") as fh:
    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
print(f"   rehearsal record {name} run {run} appended (shaped like {real[-1]['name'] if real else 'a stub'})")
PY
}
MEASURE_RECORDED=0
measure() {  # run-index -> one onepass on the candidate's door, exclusive
  local run=$1 offset=0 rc
  MEASURE_RECORDED=0
  [ "$REHEARSE" != 1 ] || { ST_BRACKET_TREE=$ARM_TREE rehearse_record "$run"; return; }
  [ ! -f "$JSONL" ] || offset=$(wc -c < "$JSONL")
  GLM53_API_PORT=$PORT BENCH_MODEL=$MODEL ONEPASS_RUN_INDEX=$run ST_BRACKET_SHA=$ARM_SHA ST_BRACKET_TREE=$ARM_TREE ST_BRACKET_COLD=${ST_BRACKET_COLD:-boot} \
    python3 "$REPO/bench/onepass.py" --name "$ARM" --require-exclusive 2>&1 | tail -40
  rc=${PIPESTATUS[0]}
  # onepass returns 2 after recording quality/evidence issues, but argparse
  # also returns 2 before any request. Only a NEW complete record from this
  # invocation allows the remaining run on the same boot. Failed checks
  # stay failed: leg/probe retain rc=2 after collecting both passes.
  if [ "$rc" = 2 ] && python3 - "$JSONL" "$offset" "$ARM_SHA" "$ARM" "$run" "${FLEET_SESSION:-}" <<'PY'
import json, sys
path, offset, sha, name, run, session = sys.argv[1:]
try:
    with open(path, 'rb') as stream:
        stream.seek(int(offset))
        rows = [json.loads(line) for line in stream if line.strip()]
    if len(rows) != 1:
        sys.exit(1)
    record = rows[0]
    complete = (record.get('engine') == 'st' and record.get('arm_sha') == sha
                and record.get('name') == name and record.get('run_index') == int(run)
                and record.get('session', '') == session
                and record.get('run_id') and record.get('boot_id') and not record.get('rehearsal')
                and record.get('recording', {}).get('status') == 'complete')
    sys.exit(0 if complete else 1)
except (OSError, ValueError, TypeError, AttributeError):
    sys.exit(1)
PY
  then
    MEASURE_RECORDED=1
  fi
  return "$rc"
}
reset_prefix() {  # between the runs: run 2 must not hit the cache run 1 filled (§93)
  curl -fsS -X POST --max-time 30 "$(door)/v1/prefix/reset" >/dev/null 2>&1 \
    || say "prefix reset refused (an engine older than §55?) -- run 2 may hit the cache; the judge cannot tell"
}
leg() {  # name sha -> the fixed leg; 0 when every run recorded
  local name=$1 sha=$2 run rc=0
  ARM_SHA=$(sha_of "$sha"); ARM_TREE=$(tree_of "$sha")
  boot_arm "$name" "$sha" || return 1
  for run in $(seq 1 "$RUNS"); do
    if [ "$run" != 1 ] && [ "$REHEARSE" != 1 ]; then reset_prefix; fi
    say "onepass run $run/$RUNS on $name ($( [ "$run" = 1 ] && echo cold || echo warm ) column)"
    measure "$run" || {
      rc=$?
      if [ "$rc" = 2 ] && [ "$MEASURE_RECORDED" = 1 ]; then
        say "onepass run $run on $name recorded issues (rc=2); retaining this boot for the remaining runs"
      else
        say "onepass run $run on $name failed (rc=$rc)"; break
      fi
    }
  done
  stop_arm
  return $rc
}
judge() {  # cand base -- by commit, and by engine tree: an adopted candidate's records are the base's
  local ct bt; ct=$(tree_of "$1"); bt=$(tree_of "$2")
  python3 "$REPO/bench/st_judge.py" judge --cand "$1" --base "$2" ${ct:+--cand-tree "$ct"} ${bt:+--base-tree "$bt"} --write \
    $( [ "$REHEARSE" = 1 ] && echo --allow-rehearsal )
}
pair() {
  local cand=${1:?pair needs a candidate sha} base=""; shift
  while [ $# -gt 0 ]; do
    case "$1" in --base) base=${2:?--base needs a sha}; shift 2;; *) echo "usage: st_bracket.sh pair <sha> [--base <sha>]" >&2; return 2;; esac
  done
  if [ -z "$base" ]; then
    base=$(python3 "$REPO/launchers/st_release.py" deployed --state "$STATE") \
      || { say "ABORT: no --base and nothing recorded as deployed in $STATE"; return 2; }
  fi
  local cs bs; cs=$(sha_of "$cand"); bs=$(sha_of "$base")
  [ "$cs" != "$bs" ] || { say "ABORT: candidate and base are the same commit ($cs)"; return 2; }
  say "pair: candidate ${cs:0:12} against base ${bs:0:12} (session $S, rehearse=$REHEARSE)"
  leg "ST-${cs:0:12}" "$cand" || return $?
  local have
  have=$(python3 "$REPO/bench/st_judge.py" samples --sha "$bs" $(tree_args "$base") $( [ "$REHEARSE" = 1 ] && echo --allow-rehearsal )) || have=0
  if [ "${have:-0}" -lt "$FLOOR_N" ]; then
    say "base ${bs:0:12} has ${have:-0} warm sample(s), $FLOOR_N wanted: booting it"
    leg "ST-${bs:0:12}" "$base" || return $?
  else
    say "base ${bs:0:12} reused: ${have:-0} warm sample(s) already"
  fi
  say "judge"; judge "$cs" "$bs"
  say "pair done"
}
chain() {  # [--reuse] NAME=<sha> ... [NAME ...]: a repeated name is another boot of the same commit (A B A B alternates)
  local reuse=0; [ "${1:-}" != --reuse ] || { reuse=1; shift; }
  [ $# -gt 0 ] || { echo "usage: st_bracket.sh chain [--reuse] NAME=<sha> [NAME=<sha> ...] [NAME ...]" >&2; return 2; }
  local arm name sha first="" base="" have
  declare -A shas=()
  local -a order=()
  for arm in "$@"; do
    case "$arm" in
      *=*) name=${arm%%=*}; sha=${arm#*=}; shas[$name]=$(sha_of "$sha");;
      *) name=$arm; [ -n "${shas[$name]:-}" ] || { echo "chain: $name names no sha (say NAME=<sha> first)" >&2; return 2; };;
    esac
    order+=("$name"); [ -n "$first" ] || first=$name
  done
  base=${shas[$first]}
  say "chain: ${order[*]} (base = $first = ${base:0:12}, session $S, rehearse=$REHEARSE${reuse:+, reuse=$reuse})"
  for name in "${order[@]}"; do
    if [ "$reuse" = 1 ]; then
      # --reuse: an arm whose commit already has a warm sample is not booted again; the judge takes
      # the samples that exist (and a pooled floor when the base has one boot). A B A B without
      # --reuse still alternates the boots, the way §93 asked, when the spread itself is the question.
      have=$(python3 "$REPO/bench/st_judge.py" samples --sha "${shas[$name]}" $(tree_args "${shas[$name]}") $( [ "$REHEARSE" = 1 ] && echo --allow-rehearsal )) || have=0
      if [ "${have:-0}" -ge 1 ]; then say "$name reused: ${have} warm sample(s) of ${shas[$name]:0:12} already (no --reuse to boot it again)"; continue; fi
    fi
    leg "$name" "${shas[$name]}" || return $?
  done
  local judged=" "
  for name in "${order[@]}"; do
    sha=${shas[$name]}; [ "$sha" != "$base" ] || continue
    case "$judged" in *" $sha "*) continue;; esac; judged="$judged$sha "
    say "judge $name against $first"; judge "$sha" "$base"
  done
  say "chain done"
}
hold() {  # sha [minutes]: boot and keep, for a session's window; ended by the minutes, a stop file, or fleet.sh cancel
  local sha=${1:?hold needs a sha} minutes=${2:-45} t0
  ARM_SHA=$(sha_of "$sha"); ARM_TREE=$(tree_of "$sha")
  boot_arm "hold-${ARM_SHA:0:12}" "$sha" || return 1
  say "holding ${ARM_SHA:0:12} on $(door) for up to ${minutes}m -- end early with: bash bench/fleet.sh cancel $S   (or: touch $OUT/stop)"
  trap 'say "hold interrupted"; stop_arm; exit 143' TERM INT
  t0=$(date +%s)
  while [ $(( $(date +%s) - t0 )) -lt $(( minutes * 60 )) ] && [ ! -f "$OUT/stop" ]; do
    [ "$REHEARSE" != 1 ] || break
    sleep 30
  done
  rm -f "$OUT/stop"
  say "hold over after $(( ($(date +%s) - t0) / 60 ))m"; stop_arm
}
probe() {  # [sha]: two onepass runs on the LIVE production door -- no boot, no lease. The queue's
  # probe lane runs it beside production when the door is idle (fleet.sh st-probe); deploy-watch
  # queues one after every deploy, so the deployed commit always has a warm sample and st-pair
  # never has to boot the base. Run 1 follows a prefix reset, not a boot: it is marked cold=reset
  # and st_judge keeps it out of the cold column, which is a boot's.
  local sha=${1:-} run rc=0
  if [ -z "$sha" ]; then
    sha=$(python3 "$REPO/launchers/st_release.py" deployed --state "$STATE") \
      || { say "ABORT: no sha and nothing recorded as deployed in $STATE"; return 2; }
  fi
  ARM_SHA=$(sha_of "$sha"); ARM_TREE=$(tree_of "$sha"); ARM="d17-${ARM_SHA:0:12}"; PORT=${ST_PROBE_PORT:-8000}; RELEASE=""
  # One run: to the judge a boot is one sample however many runs it carries, and on a live door
  # every run after a reset is warm. The second run bought nothing (ST_PROBE_RUNS=2 to have it).
  local runs=${ST_PROBE_RUNS:-1}
  say "probe: $runs run(s) on the live door $(door) for ${ARM_SHA:0:12} (no boot, no lease; session $S, rehearse=$REHEARSE)"
  [ "$REHEARSE" = 1 ] || door_up || { say "ABORT: no engine answers on $(door)"; return 1; }
  if [ "$REHEARSE" != 1 ]; then
    # A probe's record says arm_sha=<what it was queued for>; the door must be serving exactly
    # that, or the sample is mislabelled (a ticket queued before a deploy and run after it).
    local served; served=$(docker exec st-glm53 printenv ST_RELEASE 2>/dev/null | tr -d '\r' || true)
    if [[ "$served" =~ ^[0-9a-f]{7,40}$ ]] && [ "${served:0:12}" != "${ARM_SHA:0:12}" ]; then
      say "ABORT: the door serves release $served, not ${ARM_SHA:0:12} -- queue the probe for what runs (fleet.sh st-probe s $served)"; return 2
    fi
    [ -n "$served" ] || say "the door does not name its release (a boot older than PR #775?): trusting ${ARM_SHA:0:12}"
  fi
  for run in $(seq 1 "$runs"); do
    [ "$REHEARSE" = 1 ] || reset_prefix
    say "onepass run $run/$runs (after a reset: warm)"
    ST_BRACKET_COLD=reset measure "$run" || {
      rc=$?
      if [ "$rc" = 2 ] && [ "$MEASURE_RECORDED" = 1 ]; then
        say "onepass run $run recorded issues (rc=2); retaining this boot for the remaining runs"
      else
        say "onepass run $run failed (rc=$rc)"; break
      fi
    }
  done
  return $rc
}
case "${1:-}" in
  pair)  shift; pair "$@";;
  chain) shift; chain "$@";;
  hold)  shift; hold "$@";;
  probe) shift; probe "$@";;
  *) sed -n 2,7p "$0" >&2; exit 2;;
esac
