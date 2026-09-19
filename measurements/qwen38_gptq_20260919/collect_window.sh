#!/usr/bin/env bash
# This task's collection window. It asks production to hand over at its quiet
# boundary and never takes another session's fleet. Invoke from the frozen tree
# on srv2; no private prompts or responses are written into this checkout.
set -euo pipefail
MODE=${1:-collect}
case "$MODE" in collect|compare|serve|expanded|collect330|serve330|fleet330) ;; *) echo 'usage: collect_window.sh collect|compare|serve|expanded|collect330|serve330|fleet330' >&2; exit 2;; esac
case "${2:-}" in ''|--preflight) ;; *) echo 'only optional second argument is --preflight' >&2; exit 2;; esac
TREE=$(cd "$(dirname "$0")/../.." && pwd)
OWNER=${ST_LEASE_OWNER:-session/q38gptq-0919}
PARENT=${ST_WINDOW_PARENT:-0}
LOCK=/home/choiceoh/glm53-logs/st-fleet.lock
OUT=/home/choiceoh/glm53-logs/qwen38-gptq-20260919
# Keep the failed first scoring attempt beside (not underneath) the final run.
if [ "$MODE" = compare ]; then OUT=$OUT/compare; fi
if [ "$MODE" = serve ]; then OUT=$OUT/serving; fi
PRIVATE=/home/choiceoh/st-calibration-private/qwen38-gptq-20260919
PACK=/cache/qwen38-gptq-20260919
export PORT=8001 ST_ENGINE_DIR=/home/choiceoh/st-engine-qwen38-gptq-4436
export ST_IMAGE=st-engine:qwen38-gptq-4436 ST_TAP_MTP_INPUTS=0
EXPANDED=0
FIRST_COLLECTION=fit131
if [[ "$MODE" = expanded || "$MODE" = collect330 || "$MODE" = serve330 || "$MODE" = fleet330 ]]; then
  EXPANDED=1
  [ "$MODE" = expanded ] || FIRST_COLLECTION=fit330
  OUT=/home/choiceoh/glm53-logs/qwen38-gptq-330k-20260919
  PRIVATE=/home/choiceoh/st-calibration-private/qwen38-gptq-330k-20260919
  PACK=/cache/qwen38-gptq-330k-20260919
  export ST_ENGINE_DIR=/home/choiceoh/st-engine-qwen38-gptq-330k-4436
  export ST_IMAGE=st-engine:qwen38-gptq-330k-4436
fi
if [ "$MODE" = fleet330 ]; then OUT=$OUT/fleet-20260920b; fi
export ST_SPEC_K=3 ST_HC_FP8=0 ST_MTP_PRECISION=bf16 ST_MTP_EXPERTS=bf16
export ST_SHARED_OVERLAP=one ST_DRAFT_CANDIDATES=0 ST_DRAFT_THRESHOLD=0.1
unset ST_MTP_TUNED ST_DRAFT_INDEX
URL=http://127.0.0.1:$PORT
NODES=(10.10.10.2 10.10.10.1 10.10.10.3 10.10.10.4)
cd "$TREE"
if [ "$EXPANDED" = 1 ]; then
  python3 - "$PRIVATE" "$MODE" <<'PY'
import json, pathlib, sys
from probes.qwen38_gptq_feed import load_split
root = pathlib.Path(sys.argv[1])
raw = (root / 'manifest.json').read_bytes()
assert raw == pathlib.Path('measurements/qwen38_gptq_20260919/expanded-dataset-manifest.json').read_bytes(), 'private manifest changed'
m = json.loads(raw)
assert m['version'] == 2 and m['expansion']['evaluation_bytes_preserved']
assert m['expansion']['original_snapshot_prefixes_verified']
for split, expected in [('train', 330234), ('validation', 55441), ('test', 50512)]:
    rows = load_split(root / (split + '.jsonl'), m)
    assert sum(r['prompt_tokens'] for r in rows) == expected == m['prompt_tokens'][split]
mode = sys.argv[2]
labels = ('fit131', 'fit240', 'fit330', 'validation', 'heldout') if mode == 'expanded' else ('fit330',)
if mode != 'serve330':
    for label in labels:
        assert not (root / (label + '-collection')).exists(), 'collection evidence must not be overwritten'
else:
    for rank in range(4):
        assert pathlib.Path(f'/home/choiceoh/glm53-logs/qwen38-gptq-330k-20260919/offline-installed-rank{rank}.json').is_file(), '5050 packs not installed and verified'
print('expanded private input hashes and token counts verified', flush=True)
PY
fi
# Load real CPU dependencies before requesting any fleet downtime. --help exits
# before these imports and the mocked request tests cannot detect missing files.
if [ "$MODE" != collect ]; then
  PYTHONPATH=bench BENCH_MODEL=qwen3.8-flash-next GLM53_API_PORT=$PORT python3 - <<'PY'
import onepass
for filename in ('korean-corruption.py', 'check-quality.py', 'onepass_metrics.py'):
    onepass._load(filename, 'gptq_preflight_' + filename.replace('-', '_').replace('.', '_'))
print('canonical onepass CPU dependencies loaded', flush=True)
PY
fi
if [ "${2:-}" = --preflight ]; then
  echo 'CPU preflight complete; no fleet lease or serving state changed'
  exit 0
fi
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
    [ "$PARENT" = 1 ] || lease release --owner "$OWNER" || true
  else
    lease withdraw-yield --requester "$OWNER" >/dev/null 2>&1 || true
  fi
  exit "$rc"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [ "$PARENT" = 1 ]; then
  [[ "$OWNER" == queue/* ]] || { echo 'parent window must be a canonical queue reservation'; exit 3; }
  lease verify --owner "$OWNER"
  python3 - "$OWNER" <<'PY'
import pathlib, sys
holder = pathlib.Path('/home/choiceoh/glm53-logs/fleet/holder').read_text().strip().split('|')
assert holder[0] == sys.argv[1].removeprefix('queue/') and holder[6] == 'boot', 'canonical boot holder differs'
PY
else
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
if [ "$MODE" = collect330 ]; then
  estimate=20; note='Qwen 330K real-input statistics only; release fleet before RTX 5050 packing/scoring'
elif [ "$MODE" = expanded ]; then
  estimate=180; note='Qwen GPTQ size comparison: 131K/240K/330K, fixed validation and canonical consumer controls'
elif [ "$MODE" = collect ]; then
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
fi
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
  host_pack=/home/choiceoh/glm53-cache/${PACK#/cache/}
  node "$ip" "install -d -m 700 '$host_pack'; mkdir -p '$OUT'"
  if [ "$MODE" = expanded ]; then
    node "$ip" "test ! -e '$host_pack/fit131/mkcalib' && test ! -e '$host_pack/fit240/mkcalib' && test ! -e '$host_pack/fit330/mkcalib' && test ! -e '$host_pack/validation/mkcalib' && test ! -e '$host_pack/heldout/mkcalib'"
  fi
  if [[ "$MODE" = collect330 || "$MODE" = fleet330 ]]; then
    node "$ip" "test ! -e '$host_pack/fit330/mkcalib'"
  fi
  if [ "$ip" = 10.10.10.2 ]; then cp probes/qwen38_gptq_audit.py "$OUT/audit-$MODE.py";
  else scp -q probes/qwen38_gptq_audit.py "choiceoh@$ip:$OUT/audit-$MODE.py"; fi
  if [ "$MODE" != collect ]; then
    node "$ip" "mkdir -p '$OUT/tools/probes'; touch '$OUT/tools/probes/__init__.py'"
    if [ "$ip" = 10.10.10.2 ]; then cp probes/qwen38_gptq_{score,feed,offline,subset}.py "$OUT/tools/probes/";
    else scp -q probes/qwen38_gptq_{score,feed,offline,subset}.py "choiceoh@$ip:$OUT/tools/probes/"; fi
    if [ "$EXPANDED" = 1 ]; then
      if [ "$ip" = 10.10.10.2 ]; then cp tests/test_engine_qwen38_precision_port.py "$OUT/tools/gptq_precision_test.py";
      else scp -q tests/test_engine_qwen38_precision_port.py "choiceoh@$ip:$OUT/tools/gptq_precision_test.py"; fi
    fi
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
    node "$ip" "docker exec st-qwen38 cat /opt/st/runtime-manifest.json" > "$OUT/$label-runtime-rank$r.json"
    if [ "$EXPANDED" = 1 ] && [ "$label" = "$FIRST_COLLECTION" ]; then
      node "$ip" "python3 -c 'import json,sys,runpy; counters=runpy.run_path(\"$OUT/audit-$MODE.py\")[\"counters\"]; old=json.load(open(sys.argv[1]))[\"weights_id\"]; new=counters(json.load(open(sys.argv[2]))[\"root\"])[\"calibration_weights_id\"]; assert old == new, \"checkpoint changed since the first calibration campaign\"' '/home/choiceoh/glm53-logs/qwen38-gptq-20260919/fit-audit-rank$r.json' '$OUT/$label-boot-rank$r.json'"
      # Each node builds locally. Image IDs include build metadata; compare
      # source/package/seed manifests across ranks, and retain each rank's exact
      # image ID across subsequent boots below.
      cmp "$OUT/$FIRST_COLLECTION-runtime-rank0.json" "$OUT/$FIRST_COLLECTION-runtime-rank$r.json"
    fi
    if [ "$EXPANDED" = 1 ] && [ "$label" != "$FIRST_COLLECTION" ]; then
      cmp "$OUT/$FIRST_COLLECTION-image-rank$r.txt" "$OUT/$label-image-rank$r.txt"
      cmp "$OUT/$FIRST_COLLECTION-runtime-rank$r.json" "$OUT/$label-runtime-rank$r.json"
      node "$ip" "python3 -c 'import json,sys,runpy; counters=runpy.run_path(\"$OUT/audit-$MODE.py\")[\"counters\"]; a,b=[counters(json.load(open(p))[\"root\"])[\"calibration_weights_id\"] for p in sys.argv[1:]]; assert a == b, \"weight identity changed within size comparison\"' '$OUT/$FIRST_COLLECTION-boot-rank$r.json' '$OUT/$label-boot-rank$r.json'"
    fi
  done
}

gather_expanded() {
  for r in 1 2 3; do
    scp -q "choiceoh@${NODES[$r]}:$OUT/*-rank$r.json" "$OUT/"
  done
}

collect() {
  local label=$1 split=$2 minimum=$3
  export ST_PACK_ROOT=$PACK/$label ST_SELF_CALIBRATE=1 ST_CALIBRATION_ROWS=${4:-131072}
  boot "$label"
  receipts "$label"
  if [ "$EXPANDED" = 1 ] && [ "$label" = "$FIRST_COLLECTION" ]; then
    for r in 0 1 2 3; do
      lease verify --owner "$OWNER" >/dev/null
      node "${NODES[$r]}" "docker exec -e PYTHONPATH=/repo:$OUT/tools st-qwen38 python3 -m unittest gptq_precision_test.GpuPrecisionPortTests.test_native_prefill_collection_at_hidden_and_padded_width" \
        > "$OUT/row-target-native-rank$r.log" 2>&1
    done
  fi
  python3 -u probes/qwen38_gptq_feed.py --dataset "$PRIVATE/$split.jsonl" --out "$PRIVATE/$label-collection" \
    --url "$URL" --min-rows "$minimum" --owner "$OWNER" | tee "$OUT/$label-collection.log"
  # The save control is asynchronous. Each rank's complete audit is its receipt.
  for r in 0 1 2 3; do
    ip=${NODES[$r]}
    passed=0
    for ((attempt=0; attempt<12; attempt++)); do
      if node "$ip" "docker exec -e PYTHONPATH=/repo st-qwen38 python3 '$OUT/audit-$MODE.py' --root '$ST_PACK_ROOT' \
        --rank $r --ckpt /home/choiceoh/models/st-qwen38-tep4 --boot '$OUT/$label-boot-rank$r.json' \
        --out '$OUT/$label-audit-rank$r.json' --min-rows $minimum --expect-row-target $ST_CALIBRATION_ROWS" > "$OUT/$label-audit-rank$r.log" 2>&1; then passed=1; break; fi
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
  export ONEPASS_ST_CONTAINER=st-qwen38
  export ONEPASS_PROFILE=extended ONEPASS_JSONL=$OUT/onepass.jsonl
  export ST_BRACKET_SHA=$(git rev-parse HEAD) ST_BRACKET_TREE=$(git rev-parse HEAD:engine)
  export FLEET_SESSION=${OWNER#*/}
  local labels=(Bpack A1 B A2) pack_receipts=$OUT first_pack=Bpack
  if [ "$MODE" = expanded ]; then
    labels=(B131pack B240pack B330pack A1 B131 B330 A2)
    first_pack=B131pack
  fi
  if [ "$MODE" = serve330 ]; then
    labels=(A1 B330 A2)
    first_pack=fit330
  fi
  if [ "$MODE" = fleet330 ]; then
    labels=(B330pack A1 B330 A2)
    first_pack=B330pack
  fi
  if [ "$MODE" = serve ]; then
    # Resume consumer tests only after all four held-out scores succeeded.
    labels=(A1 B A2)
    pack_receipts=/home/choiceoh/glm53-logs/qwen38-gptq-20260919/compare
    for r in 0 1 2 3; do
      node "${NODES[$r]}" "test -s '$pack_receipts/projection-rank$r.json'"
    done
  fi
  for label in "${labels[@]}"; do
    case "$label" in
      B131*) export ST_PACK_ROOT=$PACK/fit131 ;;
      B240*) export ST_PACK_ROOT=$PACK/fit240 ;;
      B330*) export ST_PACK_ROOT=$PACK/fit330 ;;
      B*) export ST_PACK_ROOT=$PACK/fit ;;
      *) export ST_PACK_ROOT=$PACK/rtn ;;
    esac
    boot "$label"
    receipts "$label"
    for r in 0 1 2 3; do
      ip=${NODES[$r]}
      if [[ "$label" == B* ]]; then expected=--expect-gptq; else expected=--expect-rtn; fi
      offline_flag=''
      if [ "$MODE" = fleet330 ] && [[ "$label" == B* ]]; then offline_flag='--min-rows 330000'; fi
      if [ "$MODE" = serve330 ] && [ "$label" = B330 ]; then
        offline_flag="--offline-manifest '$OUT/offline-result-rank$r.json'"
      fi
      node "$ip" "docker exec -e PYTHONPATH=/repo:$OUT/tools st-qwen38 python3 '$OUT/audit-$MODE.py' --root '$ST_PACK_ROOT' \
        --rank $r --ckpt /home/choiceoh/models/st-qwen38-tep4 --boot '$OUT/$label-boot-rank$r.json' \
        --out '$OUT/$label-audit-rank$r.json' $expected $offline_flag" > "$OUT/$label-audit-rank$r.log" 2>&1
      if [ "$label" != "$first_pack" ]; then
        cmp "$pack_receipts/$first_pack-image-rank$r.txt" "$OUT/$label-image-rank$r.txt"
        node "$ip" "python3 -c 'import json,sys; a,b=map(lambda p: json.load(open(p)), sys.argv[1:]); assert a[\"weights_id\"] == b[\"weights_id\"], \"checkpoint identity changed since scoring\"' '$pack_receipts/$first_pack-audit-rank$r.json' '$OUT/$label-audit-rank$r.json'"
      fi
    done
    if [[ "$label" == *pack ]]; then
      echo "== actual GPTQ packs verified; held-out projection scoring $(date -Is)"
      scores=(heldout)
      if [ "$MODE" = expanded ]; then
        scores=(validation)
        [ "$label" != B330pack ] || scores+=(heldout)
      fi
      for split in "${scores[@]}"; do
      held_root=$PACK/$split
      [ "$MODE" != fleet330 ] || held_root=/cache/qwen38-gptq-20260919/heldout
      score_prefix=projection
      [ "$MODE" != expanded ] || score_prefix=$label-projection-$split
      jobs=()
      for r in 0 1 2 3; do
        lease verify --owner "$OWNER" >/dev/null
        ip=${NODES[$r]}
        node "$ip" "docker exec -e PYTHONPATH=/repo:$OUT/tools st-qwen38 python3 -m probes.qwen38_gptq_score \
          --fit '$ST_PACK_ROOT' --heldout '$held_root' --weights /home/choiceoh/models/st-qwen38-tep4/rank${r}of4.safetensors \
          --audit '$OUT/$label-audit-rank$r.json' --out '$OUT/$score_prefix-rank$r.json' --device cuda --owner '$OWNER' --parent-verified" \
          > "$OUT/$score_prefix-rank$r.log" 2>&1 &
        jobs+=("$!")
      done
      failed=0; for pid in "${jobs[@]}"; do wait "$pid" || failed=1; done
      lease verify --owner "$OWNER" >/dev/null
      [ "$failed" = 0 ] || { echo 'held-out projection scoring failed' >&2; return 1; }
      done
      bash launchers/start-st-qwen38.sh stop >> "$OUT/stop.log" 2>&1
      if [ "$MODE" = expanded ] && [ "$label" = B330pack ]; then
        gather_expanded
        python3 measurements/qwen38_gptq_20260919/summarize_expanded.py --root "$OUT" \
          --out "$OUT/size-comparison.json" > "$OUT/size-comparison.log"
      fi
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

if [[ "$MODE" = collect330 || "$MODE" = fleet330 ]]; then
  collect fit330 train 330000 330000
  gather_expanded
  python3 - "$OUT" "$MODE" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
audits = [json.loads((root / f'fit330-audit-rank{r}.json').read_bytes()) for r in range(4)]
assert all(a['statistics_valid'] and a['minimum_rows'] >= 330000 and a['sites'] == 193 for a in audits)
(root / 'collection-complete.json').write_text(json.dumps(dict(stage='statistics_ready', fit='fit330',
    minimum_rows_by_rank=[a['minimum_rows'] for a in audits],
    source_sha=(root / (sys.argv[2] + '-source.sha')).read_text().strip()), indent=2) + '\n')
PY
  if [ "$MODE" = fleet330 ]; then
    compare
    gather_expanded
  else
    echo '330K statistics audited; releasing fleet before offline packing on RTX 5050'
  fi
elif [ "$MODE" = expanded ]; then
  collect fit131 train 131072 131072
  collect fit240 train 240490 240490
  collect fit330 train 330000 330000
  collect validation validation 55441 55441
  collect heldout test 50512 50512
  compare
  gather_expanded
elif [ "$MODE" = collect ]; then
  collect fit train 131072
  collect heldout test 4096
  echo 'Both real-input Hessian sets are filed. Repacking and the quality/speed bracket remain.'
else
  compare
fi
