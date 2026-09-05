#!/usr/bin/env bash
# One lever arm on the PRODUCTION defaults (profiles/glm53.env): boot with the
# given caller env, then the legs. Usage: ab-lever.sh <NAME> "<caller env>" [reps]
#   SKIP_BOOT=1 reuses the live boot. Records --name NAME --tag cand.
#   LEGS=decode,prefill,accept,quality,korean (default: all) picks the legs;
#   LEGS=onepass runs bench/onepass.py instead (all five gates from one
#   workload, ~12 min; 35차) --
#   the 32차 leg matrix: an EXPLORATION arm (a decode-kernel change) runs
#   LEGS=decode,prefill8k QUALITY_CTX=2000,32000 (boot + ~5 min), a PROMOTION
#   arm runs everything. SHORT=1 is the old alias for decode,prefill.
#   REPS default 3: bench/bracket.py samples the engine's step counter in
#   2 s windows, so three reps carry more samples than six used to.
# Lives in the repo as bench/ab-lever.sh; srv2 runs ~/glm53-logs/ab-lever2.sh.
set -uo pipefail
NAME=${1:?usage: ab-lever.sh NAME "ENV" [reps]}
LEVER_ENV=${2:-}
REPS=${3:-3}
LEGS=${LEGS:-decode,prefill,accept,quality,korean}
[ "${SHORT:-0}" = 1 ] && LEGS=decode,prefill
has() { case ",$LEGS," in *",$1,"*) return 0;; esac; return 1; }
REPO=/home/choiceoh/stkernel
LOGD=/home/choiceoh/glm53-logs
OUT=$LOGD/bracket-lever.jsonl
HEAD=10.10.10.2
ARM=$NAME
cd "$REPO" || exit 1
snap() {  # snapshot head + worker logs on a failure, then abort
  D=$LOGD/fail-$NAME-$(date +%H%M%S); mkdir -p "$D"; cp "$LOGD/glm53.log" "$D/glm53.log.srv2" 2>/dev/null
  for ip in 10.10.10.1 10.10.10.3 10.10.10.4; do scp -q -o BatchMode=yes choiceoh@$ip:glm53-logs/glm53.log "$D/glm53.log.$ip" 2>/dev/null; done
  grep -h "Traceback\|Error\|error\|Assertion\|illegal\|died\|CUDA" "$D"/glm53.log.* 2>/dev/null | grep -v "GET /\|POST /\|Unknown vLLM" | cut -c1-200 | tail -12
  echo "ABORT: [$ARM] $1 $(date +%T) -- logs in $D"; exit 1
}
chk() {  # $1 = leg name, $2 = leg output file
  grep -q "HTTP Error 5\|Traceback\|Connection refused\|ConnectionReset" "$2" && snap "$1 leg errored"
  [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' "http://$HEAD:8000/health")" = 200 ] || snap "server died during $1"
}
if [ "${SKIP_BOOT:-0}" = 1 ]; then
  echo "== [$ARM] reusing the live boot (SKIP_BOOT=1) $(date +%T) =="
else
  echo "== [$ARM] boot $(date +%T) caller env: $LEVER_ENV (profile defaults otherwise) =="
  # Production boots pay the prefill JIT tax at boot (PREFILL_WARMUP=1 since
  # 33차); a bracket boot must not. The warmup's six requests land right after
  # health, inside the first decode leg's 2 s windows, and the prefill ladder's
  # cold row IS this harness's cold-tax channel. Export 0 unless the caller
  # explicitly wants to test the warmup itself.
  export PREFILL_WARMUP="${PREFILL_WARMUP:-0}"
  env $LEVER_ENV bash launchers/start-glm53-nvfp4-tp4.sh 2>&1 | tail -40 || true
fi
echo "== [$ARM] wait for health $(date +%T) =="
up=0
# HEALTH_BUDGET_S: a boot on NEW shapes (SPEC_K != 7, a new drafter, a wiped
# cache) JIT-compiles deep_gemm/Triton/cutlass variants serially inside the
# worker at ~1 min each (29차 K5: 45+ min); the caches under /cache persist,
# so a second boot resumes where the first stopped. Default 50 min.
HEALTH_BUDGET_S=${HEALTH_BUDGET_S:-3000}
for i in $(seq 1 $((HEALTH_BUDGET_S / 15))); do
  if [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' "http://$HEAD:8000/health")" = 200 ]; then
    echo "up after $((i*15))s"; up=1; break
  fi
  # a boot that died shows up as the head container gone (the launcher's
  # container exits with the engine); snapshot every node's log at once
  # instead of waiting out the 50-minute health budget
  if [ $i -gt 6 ] && ! docker ps --format '{{.Names}}' | grep -q '^glm53$'; then snap "boot died before health"; fi
  sleep 15
done
[ "$up" = 1 ] || snap "never became healthy"
sleep 20
echo "== [$ARM] fingerprint (head log) =="
grep -hE "armed=|MK W4 packs|fp8-dense\] (Drafter|DFlash|Glm)|drafter lane|mla shadow|mla SHADOW|selftest mla|kda layout gate|kda shadow|KDA shadow|prep-fused|split-K|splitk|union|PDL|pdl|prefetch|early.fc|KV pinned|Available KV|DISARM|drift" "$LOGD/glm53.log" 2>/dev/null | grep -v "GET /\|POST /" | cut -c1-230 | tail -24
if has decode; then
  echo "== [$ARM] decode leg $(date +%T) reps=$REPS =="
  env BENCH_MODEL=glm-5.3-flash python3 bench/bracket.py leg --name "$NAME" --tag cand --reps "$REPS" --out "$OUT" > /tmp/leg.$$ 2>&1; grep -vE "^\s*$" /tmp/leg.$$ | tail -40; chk decode /tmp/leg.$$
fi
if has prefill; then
  echo "== [$ARM] prefill ladder $(date +%T) =="
  BENCH_MODEL=glm-5.3-flash python3 probes/prefill_ladder.py 2048 8192 > /tmp/leg.$$ 2>&1; tail -40 /tmp/leg.$$; chk prefill /tmp/leg.$$
elif has prefill8k; then
  echo "== [$ARM] prefill 8k $(date +%T) =="
  BENCH_MODEL=glm-5.3-flash python3 probes/prefill_ladder.py 8192 > /tmp/leg.$$ 2>&1; tail -20 /tmp/leg.$$; chk prefill /tmp/leg.$$
fi
if has accept; then
  echo "== [$ARM] acceptance profile $(date +%T) =="
  python3 probes/accept_profile.py --label "$ARM" > /tmp/leg.$$ 2>&1; tail -16 /tmp/leg.$$; chk acceptance /tmp/leg.$$
fi
if has quality; then
  echo "== [$ARM] quality $(date +%T) ctx=${QUALITY_CTX:-2000,32000,128000} =="
  python3 bench/check-quality.py > /tmp/leg.$$ 2>&1; tail -6 /tmp/leg.$$; chk quality /tmp/leg.$$
fi
if has korean; then
  echo "== [$ARM] korean $(date +%T) =="
  python3 bench/korean-corruption.py 2 400 > /tmp/leg.$$ 2>&1; tail -12 /tmp/leg.$$; chk korean /tmp/leg.$$
fi
if has onepass; then
  # 35차 (operator): every gate from one workload -- the quality documents are
  # the prefill ladder, the Korean prompts are the decode stream (~12 min)
  echo "== [$ARM] onepass $(date +%T) ctx=${QUALITY_CTX:-2000,32000,128000} =="
  python3 bench/onepass.py --name "$NAME" --korean-extra > /tmp/leg.$$ 2>&1; grep -vE "^\s*$" /tmp/leg.$$ | tail -40; chk onepass /tmp/leg.$$
fi
rm -f /tmp/leg.$$
echo "== [$ARM] gate lines after traffic =="
grep -hE "prep-fused|drift|DISARM|kda shadow|KDA shadow|SHADOW FAIL|union" "$LOGD/glm53.log" 2>/dev/null | grep -v "GET /\|POST /" | cut -c1-200 | tail -12
echo "== [$ARM] done $(date +%T) =="
