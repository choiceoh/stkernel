#!/usr/bin/env bash
# This task's collection window. It asks production to hand over at its quiet
# boundary and never takes another session's fleet. Invoke from the frozen tree
# on srv2; no private prompts or responses are written into this checkout.
set -euo pipefail
MODE=${1:-collect}
case "$MODE" in collect|compare) ;; *) echo 'usage: collect_window.sh collect|compare' >&2; exit 2;; esac
TREE=$(cd "$(dirname "$0")/../.." && pwd)
OWNER=session/q38gptq-0919
LOCK=/home/choiceoh/glm53-logs/st-fleet.lock
OUT=/home/choiceoh/glm53-logs/qwen38-gptq-20260919
PRIVATE=/home/choiceoh/st-calibration-private/qwen38-gptq-20260919
PACK=/cache/qwen38-gptq-20260919
export PORT=8001 ST_ENGINE_DIR=/home/choiceoh/st-engine-qwen38-gptq-4436
export ST_IMAGE=st-engine:qwen38-gptq-4436 ST_TAP_MTP_INPUTS=0
export ST_SPEC_K=3 ST_HC_FP8=0 ST_MTP_PRECISION=bf16 ST_MTP_EXPERTS=bf16
export ST_SHARED_OVERLAP=one ST_DRAFT_CANDIDATES=0 ST_DRAFT_THRESHOLD=0.1
unset ST_MTP_TUNED ST_DRAFT_INDEX
URL=http://127.0.0.1:$PORT
NODES=(10.10.10.2 10.10.10.1 10.10.10.3 10.10.10.4)
cd "$TREE"
mkdir -p "$OUT"
lease() { python3 engine/base/fleet_lease.py "$@" --path "$LOCK"; }
node() { local ip=$1; shift; if [ "$ip" = 10.10.10.2 ]; then bash -c "$*"; else ssh -n -o BatchMode=yes -o ConnectTimeout=8 "choiceoh@$ip" "$@"; fi; }
owned=0
finish() {
  local rc=$?
  trap - EXIT
  if [ "$owned" = 1 ] && lease verify --owner "$OWNER" >/dev/null 2>&1; then
    for r in 0 1 2 3; do node "${NODES[$r]}" "docker logs st-qwen38 2>&1" > "$OUT/final-$MODE-rank$r.log" 2>&1 || true; done
    ST_LEASE_OWNER="$OWNER" bash launchers/start-st-qwen38.sh stop >> "$OUT/stop.log" 2>&1 || true
    lease release --owner "$OWNER" || true
  else
    lease withdraw-yield --requester "$OWNER" >/dev/null 2>&1 || true
  fi
  exit "$rc"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# A session owned by somebody else cannot be asked to stop for this run.
python3 - "$LOCK" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
record = json.loads(path.read_text()) if path.exists() else {}
if record.get('yield_to'):
    raise SystemExit('another requester already waits for this fleet')
queue = pathlib.Path('/home/choiceoh/glm53-logs/fleet/queue')
if queue.is_file() and queue.read_text().strip():
    raise SystemExit('canonical fleet tickets already wait; do not pass them')
PY
kind=$(lease kind)
if [ "$MODE" = collect ]; then
  estimate=50; note='Qwen real-input GPTQ collection: separate fit and held-out statistics'
else
  estimate=90; note='Qwen GPTQ repack and held-out error, then RTN/GPTQ/RTN canonical onepass'
fi
case "$kind" in
  free) lease acquire --owner "$OWNER" --kind session --pid $$ --host "$(hostname -s)" --est-minutes "$estimate" \
          --note "$note" ;;
  production)
    lease yield --requester "$OWNER" --kind session --pid $$ --host "$(hostname -s)" --est-minutes "$estimate" \
          --note "$note"
    for ((i=0; i<120; i++)); do lease verify --owner "$OWNER" >/dev/null 2>&1 && break; sleep 5; done
    lease verify --owner "$OWNER" ;;
  *) echo "fleet belongs to another session: $(lease read)" >&2; exit 3 ;;
esac
owned=1
export ST_LEASE_OWNER="$OWNER"
for ((i=0; i<120; i++)); do
  busy=0
  for ip in "${NODES[@]}"; do
    names=$(node "$ip" "docker ps --format '{{.Names}}'")
    if [[ "$names" =~ (^|$'\n')(st-|glm53|q38|vllm) ]]; then busy=1; fi
  done
  [ "$busy" = 0 ] && break
  sleep 5
done
[ "$busy" = 0 ] || { echo 'old containers have not left' >&2; exit 4; }

git rev-parse HEAD > "$OUT/$MODE-source.sha"
git status --porcelain > "$OUT/$MODE-source-status.txt"
[ ! -s "$OUT/$MODE-source-status.txt" ] || { echo 'the source snapshot is dirty' >&2; exit 5; }
for ip in "${NODES[@]}"; do
  node "$ip" "install -d -m 700 /home/choiceoh/glm53-cache/qwen38-gptq-20260919; mkdir -p '$OUT'"
  if [ "$ip" = 10.10.10.2 ]; then cp probes/qwen38_gptq_audit.py "$OUT/audit-$MODE.py";
  else scp -q probes/qwen38_gptq_audit.py "choiceoh@$ip:$OUT/audit-$MODE.py"; fi
  if [ "$MODE" = compare ]; then
    node "$ip" "mkdir -p '$OUT/tools/probes'; touch '$OUT/tools/probes/__init__.py'"
    if [ "$ip" = 10.10.10.2 ]; then cp probes/qwen38_gptq_{score,feed}.py "$OUT/tools/probes/";
    else scp -q probes/qwen38_gptq_{score,feed}.py "choiceoh@$ip:$OUT/tools/probes/"; fi
  fi
done

boot() {
  local label=$1
  echo "== $label boot $(date -Is)"
  bash launchers/start-st-qwen38.sh > "$OUT/$label-launch.log" 2>&1
  ready=0
  for ((i=0; i<360; i++)); do
    if curl -fsS --max-time 3 "$URL/v1/models" > "$OUT/$label-models.json" 2>/dev/null; then ready=1; break; fi
    if (( i % 6 == 0 )); then
      for ip in "${NODES[@]}"; do
        state=$(node "$ip" "docker inspect --format '{{.State.Running}}' st-qwen38")
        [ "$state" = true ] || { echo "$label: a rank exited during boot" >&2; return 1; }
      done
    fi
    sleep 5
  done
  [ "$ready" = 1 ] || { echo "$label did not become ready" >&2; return 1; }
}

receipts() {
  local label=$1
  for r in 0 1 2 3; do
    ip=${NODES[$r]}
    node "$ip" "mkdir -p '$OUT'; cp /home/choiceoh/glm53-logs/st-qwen38-dumps/boot-rank$r.json '$OUT/$label-boot-rank$r.json'"
    node "$ip" "docker inspect --format '{{.Image}}' st-qwen38" > "$OUT/$label-image-rank$r.txt"
  done
}

collect() {
  local label=$1 split=$2 minimum=$3
  export ST_PACK_ROOT=$PACK/$label ST_SELF_CALIBRATE=1
  boot "$label"
  python3 -u probes/qwen38_gptq_feed.py --dataset "$PRIVATE/$split.jsonl" --out "$PRIVATE/$label-collection" \
    --url "$URL" --min-rows "$minimum" | tee "$OUT/$label-collection.log"
  # The save control is asynchronous. Each rank's complete audit is its receipt.
  receipts "$label"
  for r in 0 1 2 3; do
    ip=${NODES[$r]}
    passed=0
    for ((attempt=0; attempt<12; attempt++)); do
      if node "$ip" "docker exec -e PYTHONPATH=/repo st-qwen38 python3 '$OUT/audit-$MODE.py' --root '$ST_PACK_ROOT' \
        --rank $r --ckpt /home/choiceoh/models/st-qwen38-tep4 --boot '$OUT/$label-boot-rank$r.json' \
        --out '$OUT/$label-audit-rank$r.json' --min-rows $minimum" > "$OUT/$label-audit-rank$r.log" 2>&1; then passed=1; break; fi
      sleep 5
    done
    [ "$passed" = 1 ] || { echo "$label rank $r filing audit failed" >&2; return 1; }
  done
  for r in 0 1 2 3; do node "${NODES[$r]}" "docker logs st-qwen38 2>&1" > "$OUT/$label-rank$r.log"; done
  bash launchers/start-st-qwen38.sh stop >> "$OUT/stop.log" 2>&1
  echo "== $label collected and verified $(date -Is)"
}

compare() {
  export ST_SELF_CALIBRATE=0 GLM53_API_PORT=$PORT BENCH_MODEL=qwen3.8-flash-next SPEC_K=3
  export ONEPASS_PROFILE=extended ONEPASS_JSONL=$OUT/onepass.jsonl
  export ST_BRACKET_SHA=$(git rev-parse HEAD) ST_BRACKET_TREE=$(git rev-parse HEAD:engine)
  export FLEET_SESSION=q38gptq-0919
  for label in Bpack A1 B A2; do
    if [[ "$label" == B* ]]; then export ST_PACK_ROOT=$PACK/fit; else export ST_PACK_ROOT=$PACK/rtn; fi
    boot "$label"
    receipts "$label"
    for r in 0 1 2 3; do
      ip=${NODES[$r]}
      if [[ "$label" == B* ]]; then expected=--expect-gptq; else expected=--expect-rtn; fi
      node "$ip" "docker exec -e PYTHONPATH=/repo st-qwen38 python3 '$OUT/audit-$MODE.py' --root '$ST_PACK_ROOT' \
        --rank $r --ckpt /home/choiceoh/models/st-qwen38-tep4 --boot '$OUT/$label-boot-rank$r.json' \
        --out '$OUT/$label-audit-rank$r.json' $expected" > "$OUT/$label-audit-rank$r.log" 2>&1
      if [ "$label" != Bpack ]; then cmp "$OUT/Bpack-image-rank$r.txt" "$OUT/$label-image-rank$r.txt"; fi
    done
    if [ "$label" = Bpack ]; then
      echo "== actual GPTQ packs verified; held-out projection scoring $(date -Is)"
      jobs=()
      for r in 0 1 2 3; do
        ip=${NODES[$r]}
        node "$ip" "docker exec -e PYTHONPATH=/repo:$OUT/tools st-qwen38 python3 -m probes.qwen38_gptq_score \
          --fit '$PACK/fit' --heldout '$PACK/heldout' --weights /home/choiceoh/models/st-qwen38-tep4/rank${r}of4.safetensors \
          --audit '$OUT/Bpack-audit-rank$r.json' --out '$OUT/projection-rank$r.json' --device cuda --owner '$OWNER'" \
          > "$OUT/projection-rank$r.log" 2>&1 &
        jobs+=("$!")
      done
      failed=0; for pid in "${jobs[@]}"; do wait "$pid" || failed=1; done
      [ "$failed" = 0 ] || { echo 'held-out projection scoring failed' >&2; return 1; }
      bash launchers/start-st-qwen38.sh stop >> "$OUT/stop.log" 2>&1
      continue
    fi
    for run in 1 2; do
      # One full C=1/C=4 run and a second C=1 run in the same boot (D17).
      echo "== $label onepass $run $(date -Is)"
      curl -fsS -X POST --max-time 30 "$URL/v1/prefix/reset" > /dev/null
      offset=0; [ ! -f "$ONEPASS_JSONL" ] || offset=$(wc -c < "$ONEPASS_JSONL")
      set +e
      ONEPASS_RUN_INDEX=$run python3 -u bench/onepass.py --name "q38gptq-$label" --num-spec 3 --require-exclusive \
        > "$OUT/$label-onepass-$run.log" 2>&1
      rc=$?
      set -e
      echo "$rc" > "$OUT/$label-onepass-$run.rc"
      [ "$rc" = 0 ] || [ "$rc" = 2 ] || { echo 'onepass runtime failure' >&2; return "$rc"; }
      # rc=2 is retained as a failed quality/evidence result, never a passing gate.
      python3 - "$ONEPASS_JSONL" "$offset" "$ST_BRACKET_SHA" "q38gptq-$label" "$run" <<'PY'
import json, sys
path, offset, sha, name, run = sys.argv[1:]
with open(path, 'rb') as stream:
    stream.seek(int(offset))
    rows = [json.loads(line) for line in stream if line.strip()]
assert len(rows) == 1, 'onepass did not append exactly one result'
r = rows[0]
assert (r.get('engine') == 'st' and r.get('arm_sha') == sha and r.get('name') == name
        and r.get('run_index') == int(run) and r.get('run_id') and r.get('boot_id')
        and r.get('recording', {}).get('status') == 'complete' and not r.get('rehearsal')), 'incomplete onepass'
PY
    done
    for r in 0 1 2 3; do node "${NODES[$r]}" "docker logs st-qwen38 2>&1" > "$OUT/$label-rank$r.log"; done
    bash launchers/start-st-qwen38.sh stop >> "$OUT/stop.log" 2>&1
  done
}

if [ "$MODE" = collect ]; then
  collect fit train 131072
  collect heldout test 4096
  echo 'Both real-input Hessian sets are filed. Repacking and the quality/speed bracket remain.'
else
  compare
fi
