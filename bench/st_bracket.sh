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

RELEASE=""; ARM=""; ARM_SHA=""
sha_of() {  # the full commit id when the source tree can say; the name as given otherwise (rehearsal)
  python3 "$REPO/launchers/st_release.py" resolve "$1" --source "$SOURCE" 2>/dev/null || echo "$1"
}
release_of() {  # sha -> its release directory, cut if it is not yet; a rehearsal cuts nothing
  [ "$REHEARSE" != 1 ] || { echo "$RELEASES/rehearsal-${1:0:12}"; return 0; }
  python3 "$REPO/launchers/st_release.py" cut "$1" --source "$SOURCE" --releases "$RELEASES"
}
shape() {  # production's shape, minus what an arm decides for itself (tree, image, port, tier, dumps)
  if [ -f "$PROD_ENV" ]; then set -a; . "$PROD_ENV"; set +a; fi
  local v; for v in $(compgen -v STK_ || true); do unset "$v"; done   # a sha is the arm; --production refuses knobs anyway
  export ST_PRODUCTION=1 PORT
}
door() { echo "http://127.0.0.1:$PORT"; }
door_up() { curl -fsS --max-time 5 "$(door)/v1/models" 2>/dev/null | grep -q "\"$MODEL\""; }
wait_door() {  # the launcher returns when the containers start; the door answers minutes later (load, capture)
  local waited=0
  while [ "$waited" -lt "$BOOT_WAIT" ]; do
    door_up && return 0
    [ "$(docker inspect --format '{{.State.Running}}' st-glm53 2>/dev/null)" = true ] || { say "rank 0 died during boot (see $OUT/boot-$ARM.log)"; return 1; }
    sleep 10; waited=$((waited + 10))
  done
  say "the door did not answer within ${BOOT_WAIT}s"; return 1
}
boot_arm() {  # name sha
  ARM=$1; local sha=$2
  RELEASE=$(release_of "$sha") || { say "ABORT: $sha could not be cut into a release"; return 1; }
  say "arm $ARM = $ARM_SHA -> $RELEASE (port $PORT)"
  [ "$REHEARSE" != 1 ] || return 0
  ( shape
    export ST_ENGINE_DIR=$RELEASE ST_IMAGE="st-engine:bracket-${ARM_SHA:0:12}" REPO=$RELEASE
    export ST_TIER_DIR=$LOGD/st-bracket-tier/$S-$ARM ST_DUMP_DIR=$LOGD/st-bracket-dumps/$S-$ARM
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
            "run_index": run, "cold": os.environ.get("ST_BRACKET_COLD", "boot"),
            "session": os.environ.get("FLEET_SESSION", ""), "knobs": {}})
rec.pop("boot_id", None); rec.pop("run_id", None)
os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
with open(path, "a", encoding="utf-8") as fh:
    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
print(f"   rehearsal record {name} run {run} appended (shaped like {real[-1]['name'] if real else 'a stub'})")
PY
}
measure() {  # run-index -> one onepass on the candidate's door, exclusive
  local run=$1
  [ "$REHEARSE" != 1 ] || { rehearse_record "$run"; return; }
  GLM53_API_PORT=$PORT BENCH_MODEL=$MODEL ONEPASS_RUN_INDEX=$run ST_BRACKET_SHA=$ARM_SHA ST_BRACKET_COLD=${ST_BRACKET_COLD:-boot} \
    python3 "$REPO/bench/onepass.py" --name "$ARM" --require-exclusive 2>&1 | tail -40
  return "${PIPESTATUS[0]}"
}
reset_prefix() {  # between the runs: run 2 must not hit the cache run 1 filled (§93)
  curl -fsS -X POST --max-time 30 "$(door)/v1/prefix/reset" >/dev/null 2>&1 \
    || say "prefix reset refused (an engine older than §55?) -- run 2 may hit the cache; the judge cannot tell"
}
leg() {  # name sha -> the fixed leg; 0 when every run recorded
  local name=$1 sha=$2 run rc=0
  ARM_SHA=$(sha_of "$sha")
  boot_arm "$name" "$sha" || return 1
  for run in $(seq 1 "$RUNS"); do
    if [ "$run" != 1 ] && [ "$REHEARSE" != 1 ]; then reset_prefix; fi
    say "onepass run $run/$RUNS on $name ($( [ "$run" = 1 ] && echo cold || echo warm ) column)"
    measure "$run" || { rc=$?; say "onepass run $run on $name failed (rc=$rc)"; break; }
  done
  stop_arm
  return $rc
}
judge() {  # cand base
  python3 "$REPO/bench/st_judge.py" judge --cand "$1" --base "$2" --write $( [ "$REHEARSE" = 1 ] && echo --allow-rehearsal )
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
  have=$(python3 "$REPO/bench/st_judge.py" samples --sha "$bs" $( [ "$REHEARSE" = 1 ] && echo --allow-rehearsal )) || have=0
  if [ "${have:-0}" -lt "$FLOOR_N" ]; then
    say "base ${bs:0:12} has ${have:-0} warm sample(s), $FLOOR_N wanted: booting it"
    leg "ST-${bs:0:12}" "$base" || return $?
  else
    say "base ${bs:0:12} reused: ${have:-0} warm sample(s) already"
  fi
  say "judge"; judge "$cs" "$bs"
  say "pair done"
}
chain() {  # NAME=<sha> ... [NAME ...]: a repeated name is another boot of the same commit (A B A B alternates)
  [ $# -gt 0 ] || { echo "usage: st_bracket.sh chain NAME=<sha> [NAME=<sha> ...] [NAME ...]" >&2; return 2; }
  local arm name sha first="" base=""
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
  say "chain: ${order[*]} (base = $first = ${base:0:12}, session $S, rehearse=$REHEARSE)"
  for name in "${order[@]}"; do leg "$name" "${shas[$name]}" || return $?; done
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
  ARM_SHA=$(sha_of "$sha")
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
  ARM_SHA=$(sha_of "$sha"); ARM="d17-${ARM_SHA:0:12}"; PORT=${ST_PROBE_PORT:-8000}; RELEASE=""
  say "probe: $RUNS runs on the live door $(door) for ${ARM_SHA:0:12} (no boot, no lease; session $S, rehearse=$REHEARSE)"
  [ "$REHEARSE" = 1 ] || door_up || { say "ABORT: no engine answers on $(door)"; return 1; }
  for run in $(seq 1 "$RUNS"); do
    [ "$REHEARSE" = 1 ] || reset_prefix
    say "onepass run $run/$RUNS ($( [ "$run" = 1 ] && echo 'after a reset' || echo warm ))"
    ST_BRACKET_COLD=reset measure "$run" || { rc=$?; say "onepass run $run failed (rc=$rc)"; break; }
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
