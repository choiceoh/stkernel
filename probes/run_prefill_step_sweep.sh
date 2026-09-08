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
TRACE_CTX=${TRACE_CTX:-32000}
TRACE_CHUNKS=${TRACE_CHUNKS:-8192,1152}

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
  case "$OUT" in */idxc6) TRACE_CHUNKS="" ;; esac
  cd "$REPO" || exit 1
  # A stale override file would silently pin the chunk for whatever boots next
  # (the scheduler reads it every step). Start from a clean slate.
  : > /home/choiceoh/vllm-prof/sched_chunk
  echo "== [pstep] instrument armed? $(date +%T)"
  # Two independent receipts: the container's env and the scheduler's own line.
  # grep -F because the anchor has brackets (BRE reads them as a class).
  { docker exec glm53 printenv VLLM_GLM53_SCHED_CHUNK_FILE 2>&1 || echo "printenv FAILED"
    docker logs glm53 2>&1 | grep -aF "[decode-first] scheduler armed" | tail -1
  } | tee "$OUT/knob.txt"
  grep -q "^/prof/sched_chunk$" "$OUT/knob.txt" || echo "!! the knob is NOT in the container: the sweep would measure the boot's own chunking"

  echo "== [pstep] sweep $(date +%T)"
  # TRACE_CHUNKS="" on the second arm: the attribution only has to be taken once,
  # and a repeat costs the hold ~4 minutes for a picture we already have.
  python3 probes/prefill_chunk_sweep.py --ctx "$SWEEP_CTX" --chunks "$CHUNKS" --reps "${REPS:-2}" \
    --json "$OUT/sweep.json" ${TRACE_CHUNKS:+--trace-chunks "$TRACE_CHUNKS"} --trace-ctx "$TRACE_CTX" 2>&1 | tee "$OUT/sweep.log"

  for C in ${TRACE_CHUNKS//,/ }; do
    t=$(python3 -c "import json,sys;print(json.load(open('$OUT/sweep.json')).get('traces',{}).get('$C',{}).get('trace',''))" 2>/dev/null)
    if [ -n "$t" ] && [ -f "$t" ]; then
      python3 tools/trace_prefill_attribution.py "$t" --out "$OUT/attr-$C.json" 2>&1 | tee "$OUT/attr-$C.log"
    else
      echo "no trace captured for chunk $C"
    fi
  done
  # The arm's own head-log copy is taken BEFORE this step runs, so a serving
  # error during the sweep lands in a log the next boot overwrites -- that is how
  # the 2026-09-08 HTTP 500 became unrecoverable. Keep our own copy here.
  docker logs glm53 > "$OUT/head.log" 2>&1 || echo "docker logs glm53 failed"
  echo "== [pstep] measure done $(date +%T); evidence in $OUT"
  ;;
*) echo "usage: $0 chain|measure <outdir>" >&2; exit 2;;
esac
