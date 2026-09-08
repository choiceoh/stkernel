#!/usr/bin/env bash
# 40차: decompose the prefill step cost in ONE hold (39차 DF4's leftover,
# "프리필 스텝 고정 비용(~0.25 s@32K) 자체 절감").
#
# Runs ON srv2. Copy it to ~/glm53-logs/ under a session-tagged name and start
# it through the fleet, never from ~/stkernel: `deploy` rewrites that checkout
# while this script is running, and bash reads a script incrementally.
#
#   scp probes/run_prefill_step_sweep.sh srv2:glm53-logs/pstep-<S>.sh
#   ssh srv2 'FLEET_SESSION=<S> REV=<sha> bash ~/glm53-logs/fleet.sh run --gpu \
#       <S> 90 "40차 프리필 스텝 비용 분해" -- bash ~/glm53-logs/pstep-<S>.sh chain'
#
# chain    holder side: deploy the rev, then one arm with the chunk instrument
#          armed and no leg, with `measure` after it (chain.sh restores).
# measure  after-arm side: the wall-clock surface T(C, ctx) over four chunk
#          sizes and two contexts, then one torch trace at each end of the
#          chunk range for tools/trace_prefill_attribution.py.
set -uo pipefail
MODE=${1:-chain}
LOGD=${LOGD:-/home/choiceoh/glm53-logs}
REPO=${REPO:-/home/choiceoh/stkernel}
CHUNKS=${CHUNKS:-1152,2304,4608,8192}
SWEEP_CTX=${SWEEP_CTX:-32000,128000}
# The trace is the attribution half and only has to contrast the two ends at the
# SAME context; 32K keeps the capture to a few chunks instead of the 111 a
# 128K/1,152 request would write.
# Contexts to capture, one profiler window each, at the boot's own chunk. The
# partial last chunk of each gives the size spread the regression needs; forcing
# a chunk while profiling kills the engine on this build (six attempts, one
# success, and the success was the one that did not force it).
TRACE_CTXS=${TRACE_CTXS:-32000,20000,12000,8000}
# One home for the override file: the probe writes it and `measure` clears it,
# and the scheduler reads it under the container's /prof mount.
CHUNK_FILE=${CHUNK_FILE:-/home/choiceoh/vllm-prof/sched_chunk}
# An attribution from an EARLIER boot, folded into the same split. Only one
# window per boot survives on this build, so the chunk sizes the regression
# needs have to be accumulated across holds. Cross-boot variation (~2%) lands in
# the residual; the reported intervals say whether it mattered.
PRIOR_ATTR=${PRIOR_ATTR:-}

case "$MODE" in
chain)
  S=${FLEET_SESSION:?FLEET_SESSION}
  REV=${REV:?REV (the branch sha to deploy)}
  OUT=$LOGD/pstep-$S
  mkdir -p "$OUT"
  echo "== [pstep] deploy $REV $(date +%T)"
  out=$(bash "$LOGD/fleet.sh" deploy "$S" "$REV" 2>&1); rc=$?
  printf '%s\n' "$out" > "$OUT/deploy.log"
  tail -5 "$OUT/deploy.log"
  # Read the verdict from the captured text, never a pipeline's exit code: on
  # 2026-09-05 `deploy | tail` returned 0 through an ABORT and the chain then
  # rebooted production on the old overlay.
  if [ $rc != 0 ] || grep -q ABORT "$OUT/deploy.log" || ! grep -q "deployed build" "$OUT/deploy.log"; then
    echo "== [pstep] DEPLOY FAILED -- nothing booted"; exit 1
  fi
  cd "$REPO" || exit 1
  # Everything after `chain` is handed to chain.sh verbatim, so a follow-up hold
  # needs no new script -- only its arms and its own --after clauses. The
  # default is the 2026-09-08 pair: PSTEP is the diagnosis (no leg, the sweep IS
  # the measurement) and IDXC6 the first candidate, with onepass because reuse
  # is an approximation and a prefill number alone cannot promote it.
  shift
  [ $# -gt 0 ] || set -- \
    PSTEP="VLLM_GLM53_SCHED_CHUNK_FILE=/prof/sched_chunk" \
    IDXC6="VLLM_GLM53_SCHED_CHUNK_FILE=/prof/sched_chunk INDEX_CACHE_FREQ=6" \
    --legs PSTEP none \
    --after PSTEP "bash $0 measure $OUT" \
    --after IDXC6 "bash $0 measure $OUT/idxc6"
  FLEET_SESSION=$S bash bench/chain.sh "$@"
  ;;
measure)
  OUT=${2:?usage: $0 measure <outdir>}
  mkdir -p "$OUT"
  # the IndexCache arm writes under the diagnosis directory and takes no trace
  case "$OUT" in */idxc6) TRACE_CTXS="" ;; esac
  cd "$REPO" || exit 1
  # A stale override file would silently pin the chunk for whatever boots next
  # (the scheduler reads it every step). Start from a clean slate.
  : > "$CHUNK_FILE"
  echo "== [pstep] instrument armed? $(date +%T)"
  # Two independent receipts: the container's env and the scheduler's own line.
  # grep -F because the anchor has brackets (BRE reads them as a class).
  { docker exec glm53 printenv VLLM_GLM53_SCHED_CHUNK_FILE 2>&1 || echo "printenv FAILED"
    docker logs glm53 2>&1 | grep -aF "[decode-first] scheduler armed" | tail -1
  } | tee "$OUT/knob.txt"
  grep -q "^/prof/sched_chunk$" "$OUT/knob.txt" || echo "!! the knob is NOT in the container: the sweep would measure the boot's own chunking"

  echo "== [pstep] sweep $(date +%T)"
  # TRACE_CTXS="" on the second arm: the attribution only has to be taken once,
  # and a repeat costs the hold minutes for a picture we already have.
  python3 probes/prefill_chunk_sweep.py --ctx "$SWEEP_CTX" --chunks "$CHUNKS" --reps "${REPS:-2}" \
    --chunk-file "$CHUNK_FILE" --json "$OUT/sweep.json" ${TRACE_CTXS:+--trace-ctxs "$TRACE_CTXS"} 2>&1 | tee "$OUT/sweep.log"

  # One trace per window, one window per context, all at the boot's own chunk.
  # Attribute each, then split the fixed cost across all of them together (plus
  # PRIOR_ATTR): the sizes come from each request's partial last chunk, and one
  # size cannot separate fixed from per-token.
  n=0
  for tr in $(python3 -c "
import json
d=json.load(open('$OUT/sweep.json')).get('traces',{})
print(' '.join(r['trace'] for r in d.get('runs',[]) if r.get('trace')))" 2>/dev/null); do
    [ -f "$tr" ] || continue
    n=$((n+1))
    # Out of the shared profiler directory: the next capture on this fleet must
    # not be able to confuse or clobber the evidence for this one.
    mv "$tr" "$OUT/trace-$n.${tr##*.}" && tr="$OUT/trace-$n.${tr##*.}"
    python3 tools/trace_prefill_attribution.py "$tr" --out "$OUT/attr-$n.json" 2>&1 | tail -20 | tee "$OUT/attr-$n.log"
  done
  set -- "$OUT"/attr-*.json "$PRIOR_ATTR"
  ok=""
  for f in "$@"; do [ -s "$f" ] && ok="$ok $f"; done
  if [ -n "$ok" ]; then
    python3 probes/prefill_fixed_cost_attribution.py $ok --json "$OUT/fixed-cost.json" 2>&1 \
      | tee "$OUT/fixed-cost.log"
  else
    echo "no attribution produced: skipping the fixed-cost split"
  fi
  # The arm's own head-log copy is taken BEFORE this step runs, so a serving
  # error during the sweep lands in a log the next boot overwrites -- that is how
  # the 2026-09-08 HTTP 500 became unrecoverable. Keep our own copy here.
  docker logs glm53 > "$OUT/head.log" 2>&1 || echo "docker logs glm53 failed"
  echo "== [pstep] measure done $(date +%T); evidence in $OUT"
  ;;
*) echo "usage: $0 chain|measure <outdir>" >&2; exit 2;;
esac
