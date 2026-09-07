#!/usr/bin/env bash
set -euo pipefail
: "${REPO:?}" "${FLEET_SESSION:?}" "${RETRY_JOB:?}" "${RETRY_REV:?}" "${IMAGE:?}"
export FLEET="$REPO/bench/fleet.sh" LEVER="$REPO/bench/ab-lever.sh"
export ONEPASS_JSONL="$RETRY_JOB/onepass.jsonl" ONEPASS_MEMORY_DIR="$RETRY_JOB/memory"
export ONEPASS_REQUIRE_EXCLUSIVE=1 QUALITY_CTX=2000,4000,8000,32000,128000
export KV_TOKENS=524288 MAX_LEN=262144
export PREFILL_WARMUP=0
export GLM53_API_PORT=18000 GLM53_API_HOST=127.0.0.1 HEAD=127.0.0.1
changed=0
finish() {
  rc=$?
  trap - EXIT
  if [ "$changed" = 1 ] && bash "$FLEET" restore-needed "$FLEET_SESSION" >/dev/null 2>&1; then
    echo "== restoring full production cache capacity $(date +%T)"
    export KV_TOKENS=2000000 MAX_LEN=1048576
    export GLM53_API_PORT=8000 GLM53_API_HOST=0.0.0.0 HEAD=10.10.10.2
    LEGS=none bash "$LEVER" SPFR20907PROD "" > "$RETRY_JOB/production-restore.log" 2>&1 || { restore_rc=$?; [ "$rc" != 0 ] || rc=$restore_rc; }
    tail -15 "$RETRY_JOB/production-restore.log"
  fi
  echo "$rc" > "$RETRY_JOB/exit_code"
  exit "$rc"
}
trap finish EXIT
for host in local choiceoh@10.10.10.1 choiceoh@10.10.10.3 choiceoh@10.10.10.4; do
  if [ "$host" = local ]; then available=$(df -Pk /home/choiceoh | awk 'NR==2{print $4}')
  else available=$(ssh -o BatchMode=yes -o ConnectTimeout=3 "$host" "df -Pk /home/choiceoh" | awk 'NR==2{print $4}'); fi
  [[ "$available" =~ ^[0-9]+$ ]] && [ "$available" -ge 33554432 ] || { echo "REFUSED: $host has less than 32 GiB spare disk before deploy"; exit 3; }
done
bash "$FLEET" deploy "$FLEET_SESSION" "$RETRY_REV"
[ "$(git -C "$REPO" rev-parse HEAD)" = "$RETRY_REV" ]
changed=1
bash "$REPO/bench/chain.sh" SPFR20907B1="" \
 SPFR20907A="VLLM_GLM53_PREFILL_SP_FUSE_MHC=1 VLLM_GLM53_PREFILL_SP_FP8_AG_MIN_TOKENS=2048 VLLM_GLM53_PREFILL_SP_FP8_RS_MIN_TOKENS=4096" \
 SPFR20907B2="" \
 --after SPFR20907B1 "python3 '$RETRY_JOB/attest.py' SPFR20907B1" \
 --after SPFR20907A "python3 '$RETRY_JOB/attest.py' SPFR20907A" \
 --after SPFR20907B2 "python3 '$RETRY_JOB/attest.py' SPFR20907B2"
