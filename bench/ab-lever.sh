#!/usr/bin/env bash
# One lever arm on the PRODUCTION defaults (profiles/glm53.env): boot with the
# given caller env, then bench/onepass.py -- the ONLY leg (operator,
# 2026-09-06: "개별 테스트 없애고 무조건 원패스로 통일"). One workload gives
# every gate at once on ONE Korean workload (Korean documents, 39차): the
# prefill ladder, retrieval quality, Korean corruption, the decode windows
# and the acceptance counters (~2.5 min). The 32차 leg matrix (decode /
# prefill / accept / quality / korean, SHORT=1, REPS) and the separate
# Korean prompt set are gone.
# Usage: ab-lever.sh <NAME> "<caller env>"
#   SKIP_BOOT=1 reuses the live boot. Records --name NAME --tag cand.
#   LEGS=none boots, waits for health and fingerprints only (for a chain
#   that runs onepass itself, e.g. on several arms of one boot).
#   QUALITY_CTX=2000,32000 shortens the ladder for an exploration arm.
# Lives in the repo as bench/ab-lever.sh; srv2 runs ~/glm53-logs/ab-lever2.sh.
set -uo pipefail
NAME=${1:?usage: ab-lever.sh NAME "ENV"}
LEVER_ENV=${2:-}
LEGS=${LEGS:-onepass}
has() { case ",$LEGS," in *",$1,"*) return 0;; esac; return 1; }
REPO=${REPO:-/home/choiceoh/stkernel}
LOGD=${LOGD:-/home/choiceoh/glm53-logs}
HEAD=${HEAD:-10.10.10.2}
ARM=$NAME
cd "$REPO" || exit 1
# 39차 idea 2 -- rehearsal: no boot, no leg; the LAST real record is copied under
# this arm's name with rehearsal=true (judge/baseline ignore such rows unless
# asked), so a chain's flow, judge parsing and log handling can be checked
# without a GPU. fleet.sh runs a FLEET_REHEARSE=1 job in parallel, never holding.
if [ "${FLEET_REHEARSE:-0}" = 1 ]; then
  echo "== [$ARM] REHEARSAL: no boot, no leg; fabricating a record from the last real one $(date +%T) =="
  python3 - "$NAME" "$LEVER_ENV" "${ONEPASS_JSONL:-$LOGD/bracket-onepass.jsonl}" <<'PY'
import json, sys, time, os
name, env, path = sys.argv[1], sys.argv[2], sys.argv[3]
rows = [json.loads(l) for l in open(path) if l.strip()] if os.path.exists(path) else []
real = [r for r in rows if not r.get("rehearsal")]
rec = dict(real[-1]) if real else {"decode": {"windows_med": 20.0, "tokens_per_step": 3.4, "acc_raw": 0.48},
                                   "quality": {"ok": 9, "total": 9}, "korean": {"dirty": 0, "n": 5}, "prefill": [], "harness": 39}
knobs = dict(kv.split("=", 1) for kv in env.split() if "=" in kv)
rec.update({"name": name, "t": time.strftime("%F %T"), "rehearsal": True, "knobs": knobs,
            "session": os.environ.get("FLEET_SESSION", ""), "proof_ok": f"{len(knobs)}/{len(knobs)}" if knobs else None})
rec["proof"] = {k: True for k in knobs}
with open(path, "a", encoding="utf-8") as fh:
    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
print(f"   rehearsal record {name} appended (copied from {real[-1]['name'] if real else 'nothing'})")
PY
  echo "== [$ARM] done (rehearsal) $(date +%T) =="; exit 0
fi
# 39차 idea 4 -- the first boot after a deploy compiles cold (~12 min) and inflates
# the cold prefill column; tell onepass so the record carries cold_compile=true.
STAMP_FILE=${MK_OVERLAY_STAMP:-$HOME/glm53-cache/.overlay-sha}
SEEN=$LOGD/.boot-stamps
_stamp=$(cut -c1-12 "$STAMP_FILE" 2>/dev/null)
if [ -n "$_stamp" ] && ! grep -qx "$_stamp" "$SEEN" 2>/dev/null; then
  export MK_COLD_COMPILE=1; echo "== [$ARM] first boot on build $_stamp: cold compile (cold prefill column not comparable) =="
  echo "$_stamp" >> "$SEEN"
fi
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
# 35차: the lever's PROOF is the container's environment, not the caller's
# line (a lever once read as not applied and the cause was never found)
if [ -n "${LEVER_ENV:-}" ]; then
  _keys=$(for kv in $LEVER_ENV; do echo "${kv%%=*}"; done | paste -sd'|')
  echo "== [$ARM] lever env inside the head container =="
  docker exec glm53 env 2>/dev/null | grep -E "^($_keys)=" || echo "  (none of $_keys set in the container!)"
fi
echo "== [$ARM] fingerprint (head log) =="
grep -hE "armed=|MK W4 packs|fp8-dense\] (Drafter|DFlash|Glm)|drafter lane|mla shadow|mla SHADOW|selftest mla|kda layout gate|kda shadow|KDA shadow|prep-fused|split-K|splitk|union|PDL|pdl|prefetch|early.fc|KV pinned|Available KV|DISARM|drift" "$LOGD/glm53.log" 2>/dev/null | grep -v "GET /\|POST /" | cut -c1-230 | tail -24
# 33차: the W4 packs' provenance -- gptq=N is the serving proof of the GPTQ
# packs; gptq=0 with calibration dumps on disk means the packer fell back
# (cache purged, names mismatched) and production is silently RTN
grep -hE "megakernel packs:" "$LOGD/glm53.log" 2>/dev/null | sed -E "s/^.*\[fp8-dense\] //" | cut -c1-200 | tail -2
# 37차: packs loaded from the cache report gptq=0 cached=N -- those ARE the
# GPTQ packs (the cache key carries the lever set); only gptq=0 AND cached=0
# is a fallback to RTN.
if grep -hE "megakernel packs:" "$LOGD/glm53.log" 2>/dev/null | grep "Glm5Next.*gptq=0" | grep -q "cached=0 " \
   && [ -n "$(ls "$HOME/glm53-cache/mkcalib/rank0" 2>/dev/null)" ]; then
  echo "WARNING: [$ARM] the target's W4 packs are RTN (gptq=0) although calibration dumps exist -- check VLLM_GLM53_MK_PACK_GPTQ and the pack cache"
fi
if has onepass; then
  # 35차 (operator): every gate from one workload -- the quality documents are
  # the prefill ladder, the Korean prompts are the decode stream. onepass
  # derives tokens/step from the engine's counters (spec_k_eff); SPEC_K is
  # only its fallback, so hand it the profile's value when the caller has none.
  _k=$(sed -nE 's/^SPEC_K=([0-9]+).*/\1/p' profiles/glm53.env | tail -1)
  echo "== [$ARM] onepass $(date +%T) ctx=${QUALITY_CTX:-2000,32000,128000} k=${SPEC_K:-${_k:-7}} =="
  env SPEC_K="${SPEC_K:-${_k:-7}}" BENCH_MODEL=glm-5.3-flash python3 bench/onepass.py --name "$NAME" > /tmp/leg.$$ 2>&1; grep -vE "^\s*$" /tmp/leg.$$ | tail -40; chk onepass /tmp/leg.$$
  echo "== [$ARM] acceptance counters =="
  curl -s -m 5 "http://$HEAD:8000/metrics" | grep -E "^vllm:spec_decode_num_accepted_tokens_per_pos_total|^vllm:spec_decode_num_(drafts|draft_tokens|accepted_tokens)_total" | sed "s/{[^}]*}//"
fi
rm -f /tmp/leg.$$
# 39차: the head log is the server's stdout, rewritten by every boot -- keep this
# arm's copy (boot-<NAME>.log) BEFORE the next boot erases it, and prove the arm's
# lanes from it with fixed-string markers (bench/proof-markers.tsv): armed != serving.
cp "$LOGD/glm53.log" "$LOGD/boot-$NAME.log" 2>/dev/null && echo "== [$ARM] head log kept: $LOGD/boot-$NAME.log =="
echo "== [$ARM] serving proof (bench/proof.py) =="
python3 bench/proof.py --log "$LOGD/boot-$NAME.log" 2>&1 | tail -20 || true
echo "== [$ARM] gate lines after traffic =="
grep -hE "prep-fused|drift|DISARM|kda shadow|KDA shadow|SHADOW FAIL|union" "$LOGD/glm53.log" 2>/dev/null | grep -v "GET /\|POST /" | cut -c1-200 | tail -12
echo "== [$ARM] done $(date +%T) =="
