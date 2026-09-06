#!/usr/bin/env bash
# Run only under fleet.sh run --gpu. Same code/profile, cache-off -> cold -> hit.
# Every arm uses the canonical Korean onepass workload; no separate request set.
set -euo pipefail
REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
LOGD=${LOGD:-/home/choiceoh/glm53-logs}
FLEET_DIR=${FLEET_DIR:-$LOGD/fleet}
SESSION=${FLEET_SESSION:?run through fleet.sh run --gpu}
EVIDENCE=${STARTUP_CACHE_EVIDENCE:-$LOGD/startup-cache-$(date +%Y%m%d-%H%M%S)}
PREFIX=${STARTUP_CACHE_PREFIX:-STARTCACHE}
monitor_pid=
cd "$REPO"
[ "$(cut -d'|' -f1 "$FLEET_DIR/holder")" = "$SESSION" ] || { echo 'not the fleet holder'; exit 2; }
mkdir -p "$EVIDENCE"
export PREFILL_WARMUP=0 QUALITY_CTX=${QUALITY_CTX:-2000,32000}
export REPO LOGD
printf '%s\n' "$(git rev-parse HEAD)" > "$EVIDENCE/source-commit.txt"
cp profiles/glm53.env "$EVIDENCE/profile.env"

snapshot() {
  local arm=$1 ip container
  cp "$LOGD/glm53.log" "$EVIDENCE/$arm-srv2.log" || true
  docker inspect --format '{{.State.Status}} {{.State.ExitCode}} {{.State.OOMKilled}}' glm53 > "$EVIDENCE/$arm-srv2.state" 2>&1 || true
  docker exec glm53 sha256sum /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/glm53_startup_cache.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/glm53_rank_cache.py > "$EVIDENCE/$arm-srv2.sha256" 2>&1 || true
  for ip in 1 3 4; do
    scp -q -o BatchMode=yes -o ConnectTimeout=8 "choiceoh@10.10.10.$ip:glm53-logs/glm53.log" "$EVIDENCE/$arm-srv$ip.log" || true
    ssh -o BatchMode=yes -o ConnectTimeout=8 "choiceoh@10.10.10.$ip" 'docker inspect --format "{{.State.Status}} {{.State.ExitCode}} {{.State.OOMKilled}}" glm53-worker; df -B1 /home/choiceoh/glm53-cache | tail -1; grep -E "MemFree:|MemAvailable:" /proc/meminfo; docker exec glm53-worker sha256sum /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/glm53_startup_cache.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/glm53_rank_cache.py' > "$EVIDENCE/$arm-srv$ip.state" 2>&1 || true
  done
  docker inspect --format '{{json .Config.Env}}' glm53 | python3 -c 'import json,sys; print(json.dumps([v for v in json.load(sys.stdin) if v.startswith(("VLLM_GLM53_FP8_CACHE=", "VLLM_GLM53_RANK_CACHE="))]))' > "$EVIDENCE/$arm-cache-env.json" || true
}

failed() {
  local rc=$?
  trap - EXIT
  if [ -n "$monitor_pid" ]; then kill "$monitor_pid" 2>/dev/null || true; wait "$monitor_pid" 2>/dev/null || true; fi
  if [ "$rc" != 0 ]; then
    echo "startup-cache trial failed rc=$rc; restoring cache-off serving"
    snapshot "${current_arm:-failure}"
    LEGS=none HEALTH_BUDGET_S=1800 bash bench/ab-lever.sh "${PREFIX}RESTORE" 'VLLM_GLM53_FP8_CACHE=0 VLLM_GLM53_RANK_CACHE=0' > "$EVIDENCE/restore.log" 2>&1 || true
    snapshot RESTORE
  fi
  printf '%s\n' "$rc" > "$EVIDENCE/exit-code"
  exit "$rc"
}
trap failed EXIT

for stage in BASE COLD WARM; do
  current_arm=${PREFIX}${stage}
  knobs=''
  [ "$stage" != BASE ] || knobs='VLLM_GLM53_FP8_CACHE=0 VLLM_GLM53_RANK_CACHE=0'
  start=$(date +%s)
  previous=$(docker inspect --format '{{.Id}}' glm53 2>/dev/null || true)
  (
    while (( $(date +%s) - start < 1800 )); do
      current=$(docker inspect --format '{{.Id}}' glm53 2>/dev/null || true)
      if [ -n "$current" ] && [ "$current" != "$previous" ] && [ "$(curl -s -m 2 -o /dev/null -w '%{http_code}' http://10.10.10.2:8000/health)" = 200 ]; then
        printf '%s\t%s\n' "$current_arm" "$(( $(date +%s) - start ))" >> "$EVIDENCE/health-wall-seconds.tsv"
        exit 0
      fi
      sleep 1
    done
    exit 1
  ) &
  monitor_pid=$!
  echo "=== $current_arm $(date -Is) ==="
  LEGS=none HEALTH_BUDGET_S=1800 bash bench/ab-lever.sh "$current_arm" "$knobs" > "$EVIDENCE/$current_arm-boot.out" 2>&1
  wait "$monitor_pid"
  monitor_pid=
  snapshot "$current_arm"
  if [ "$stage" != BASE ]; then
    python3 - "$EVIDENCE" "$current_arm" "$stage" <<'PY'
from pathlib import Path
import re, sys
root, arm, stage = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
for node in (1, 2, 3, 4):
    text = (root / f"{arm}-srv{node}.log").read_text()
    assert re.search(r"\[rank-cache\] (?:hit|saved) rank=", text), f"srv{node}: no rank artifact used/saved"
    rows = re.findall(r"\[fp8-cache\].*?enabled=True hit=(\d+) miss=(\d+) errors=(\d+)", text)
    assert len(rows) >= 2, f"srv{node}: target/drafter FP8 cache receipts missing"
    assert all(int(e) == 0 for h, m, e in rows), f"srv{node}: FP8 cache errors: {rows}"
    if stage == "WARM":
        assert re.search(r"\[rank-cache\] hit rank=", text), f"srv{node}: rank cache missed"
        assert all(int(h) > 0 and int(m) == 0 for h, m, e in rows), f"srv{node}: FP8 warm misses: {rows}"
print("all four nodes have the required cache receipts")
PY
  fi
  STARTUP_CACHE_RESPONSES="$EVIDENCE/$current_arm-responses.jsonl" \
    python3 bench/startup_cache_onepass.py --name "$current_arm" --ctx "$QUALITY_CTX" --out "$EVIDENCE/onepass.jsonl" > "$EVIDENCE/$current_arm-onepass.out" 2>&1
  tail -1 "$EVIDENCE/onepass.jsonl" >> "$LOGD/bracket-onepass.jsonl"
  cat "$EVIDENCE/$current_arm-onepass.out"
  snapshot "$current_arm"
  python3 - "$EVIDENCE/onepass.jsonl" <<'PY'
import json, sys
row = json.loads(open(sys.argv[1]).readlines()[-1])
assert row['quality']['ok'] == row['quality']['total'], row['quality']
assert row['korean']['dirty'] == 0, row['korean']
PY
  echo "=== $current_arm complete $(date -Is) ==="
done
