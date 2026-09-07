#!/usr/bin/env bash
# Run only under fleet.sh run --gpu. Artifacts: off -> cold -> hit.
# pack-io: prime, then candidate/control/control/candidate on warm artifacts.
# pack-key: prime SHA aliases, then control/candidate/candidate/control.
# Every arm uses the canonical Korean onepass workload; no separate request set.
set -euo pipefail
REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
LOGD=${LOGD:-/home/choiceoh/glm53-logs}
FLEET_DIR=${FLEET_DIR:-$LOGD/fleet}
SESSION=${FLEET_SESSION:?run through fleet.sh run --gpu}
EVIDENCE=${STARTUP_CACHE_EVIDENCE:-$LOGD/startup-cache-$(date +%Y%m%d-%H%M%S)}
PREFIX=${STARTUP_CACHE_PREFIX:-STARTCACHE}
MODE=${STARTUP_CACHE_MODE:-artifacts}
case "$MODE" in
  artifacts) stages=(BASE COLD WARM); restore_knobs='VLLM_GLM53_FP8_CACHE=0 VLLM_GLM53_RANK_CACHE=0' ;;
  pack-io) stages=(PRIME FAST1 BASE1 BASE2 FAST2); restore_knobs='VLLM_GLM53_MK_PACK_FAST_IO=0' ;;
  pack-key) stages=(PRIME BASE1 FAST1 FAST2 BASE2); restore_knobs='VLLM_GLM53_MK_PACK_SHA256=0 VLLM_GLM53_MK_PACK_FAST_IO=1' ;;
  renderer-warmup) stages=(PRIME BASE1 FAST1 FAST2 BASE2); restore_knobs='VLLM_GLM53_EARLY_MM_WARMUP=0' ;;
  graph-profile) stages=(PRIME BASE1 FAST1 FAST2 BASE2); restore_knobs='VLLM_GLM53_SKIP_UNUSED_GRAPH_PROFILE=0' ;;
  overlay-deploy) stages=(PRIME BASE1 FAST1 FAST2 BASE2); restore_knobs='VLLM_GLM53_SKIP_UNUSED_GRAPH_PROFILE=0' ;;
  campaign) stages=(); restore_knobs='' ;;
  *) echo "unknown startup mode: $MODE"; exit 2 ;;
esac
monitor_pid=
cd "$REPO"
[ "$(cut -d'|' -f1 "$FLEET_DIR/holder")" = "$SESSION" ] || { echo 'not the fleet holder'; exit 2; }
mkdir -p "$EVIDENCE"
export PREFILL_WARMUP=0 QUALITY_CTX=${QUALITY_CTX:-2000,32000}
export REPO LOGD
printf '%s\n' "$(git rev-parse HEAD)" > "$EVIDENCE/source-commit.txt"
cp profiles/glm53.env "$EVIDENCE/profile.env"
declare -A campaign_knobs=()
if [ "$MODE" = campaign ]; then
  python3 bench/startup_campaign.py plan --spec "${STARTUP_CAMPAIGN_SPEC:?provide a campaign JSON}" --evidence "$EVIDENCE" > "$EVIDENCE/plan.tsv"
  while IFS=$'\t' read -r stage knobs; do stages+=("$stage"); campaign_knobs[$stage]=$knobs; done < "$EVIDENCE/plan.tsv"
fi

snapshot() {
  local arm=$1 ip container
  cp "$LOGD/glm53.log" "$EVIDENCE/$arm-srv2.log" || true
  docker inspect --format '{{.State.Status}} {{.State.ExitCode}} {{.State.OOMKilled}} {{.Image}}' glm53 > "$EVIDENCE/$arm-srv2.state" 2>&1 || true
  docker exec glm53 sha256sum /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/glm53_startup_cache.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/glm53_rank_cache.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/glm53_megakernel.py > "$EVIDENCE/$arm-srv2.sha256" 2>&1 || true
  if [ "$MODE" = renderer-warmup ]; then
    docker exec glm53 sha256sum /usr/local/lib/python3.12/dist-packages/vllm/v1/engine/async_llm.py /usr/local/lib/python3.12/dist-packages/vllm/renderers/glm53_renderer_warmup.py >> "$EVIDENCE/$arm-srv2.sha256" 2>&1 || true
  fi
  if [[ "$MODE" = graph-profile || "$MODE" = overlay-deploy ]]; then
    docker exec glm53 sha256sum /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py /usr/local/lib/python3.12/dist-packages/deneb_boot_stamps.py >> "$EVIDENCE/$arm-srv2.sha256"
  fi
  for ip in 1 3 4; do
    scp -q -o BatchMode=yes -o ConnectTimeout=8 "choiceoh@10.10.10.$ip:glm53-logs/glm53.log" "$EVIDENCE/$arm-srv$ip.log" || true
    ssh -o BatchMode=yes -o ConnectTimeout=8 "choiceoh@10.10.10.$ip" 'docker inspect --format "{{.State.Status}} {{.State.ExitCode}} {{.State.OOMKilled}} {{.Image}}" glm53-worker; df -B1 /home/choiceoh/glm53-cache | tail -1; grep -E "MemFree:|MemAvailable:" /proc/meminfo; docker exec glm53-worker sha256sum /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/glm53_startup_cache.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/glm53_rank_cache.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/glm53_megakernel.py' > "$EVIDENCE/$arm-srv$ip.state" 2>&1 || true
  done
  if [[ "$MODE" = graph-profile || "$MODE" = overlay-deploy ]]; then
    for ip in 1 3 4; do
      ssh -o BatchMode=yes -o ConnectTimeout=8 "choiceoh@10.10.10.$ip" 'docker exec glm53-worker sha256sum /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py /usr/local/lib/python3.12/dist-packages/deneb_boot_stamps.py' >> "$EVIDENCE/$arm-srv$ip.state"
    done
    docker inspect --format '{{json .Config.Env}}' glm53 | python3 -c 'import json,sys; print(json.dumps([v for v in json.load(sys.stdin) if v.startswith(("VLLM_GLM53_SKIP_UNUSED_GRAPH_PROFILE=", "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS="))]))' > "$EVIDENCE/$arm-graph-env.json"
  fi
  docker inspect --format '{{json .Config.Env}}' glm53 | python3 -c 'import json,sys; print(json.dumps([v for v in json.load(sys.stdin) if v.startswith("VLLM_")]))' > "$EVIDENCE/$arm-cache-env.json" || true
}

failed() {
  local rc=$?
  trap - EXIT
  if [ -n "$monitor_pid" ]; then kill "$monitor_pid" 2>/dev/null || true; wait "$monitor_pid" 2>/dev/null || true; fi
  if [ "$rc" != 0 ]; then
    echo "startup-cache trial failed rc=$rc; restoring control serving"
    snapshot "${current_arm:-failure}" || true
    if [ "${FLEET_RESTORE_MANAGED:-0}" != 1 ]; then
      LEGS=none HEALTH_BUDGET_S=1800 bash bench/ab-lever.sh "${PREFIX}RESTORE" "$restore_knobs" > "$EVIDENCE/restore.log" 2>&1 || true
      snapshot RESTORE || true
    fi
  fi
  printf '%s\n' "$rc" > "$EVIDENCE/exit-code"
  exit "$rc"
}
trap failed EXIT
# A handled signal exits after the foreground boot returns, so failed() can
# restore control without leaving an orphan boot or recording exit code zero.
trap 'exit 130' INT
trap 'exit 143' TERM

for stage in "${stages[@]}"; do
  current_arm=${PREFIX}${stage}
  knobs=''
  [ "$stage" != BASE ] || knobs='VLLM_GLM53_FP8_CACHE=0 VLLM_GLM53_RANK_CACHE=0'
  if [ "$MODE" = pack-io ]; then
    knobs='VLLM_GLM53_MK_PACK_FAST_IO=0'
    [[ "$stage" != FAST* ]] || knobs='VLLM_GLM53_MK_PACK_FAST_IO=1'
  fi
  if [ "$MODE" = pack-key ]; then
    knobs='VLLM_GLM53_MK_PACK_FAST_IO=1 VLLM_GLM53_MK_PACK_SHA256=1'
    [[ "$stage" != BASE* ]] || knobs='VLLM_GLM53_MK_PACK_FAST_IO=1 VLLM_GLM53_MK_PACK_SHA256=0'
  fi
  if [ "$MODE" = renderer-warmup ]; then
    knobs='VLLM_GLM53_EARLY_MM_WARMUP=1'
    [[ "$stage" != BASE* ]] || knobs='VLLM_GLM53_EARLY_MM_WARMUP=0'
  fi
  if [ "$MODE" = graph-profile ]; then
    knobs='VLLM_GLM53_SKIP_UNUSED_GRAPH_PROFILE=1'
    [[ "$stage" != BASE* ]] || knobs='VLLM_GLM53_SKIP_UNUSED_GRAPH_PROFILE=0'
  fi
  if [ "$MODE" = overlay-deploy ]; then
    knobs='VLLM_GLM53_SKIP_UNUSED_GRAPH_PROFILE=0'
    python3 bench/startup_deploy_receipts.py "$EVIDENCE/$current_arm-before-deploy.json"
    deploy_start=$(date +%s)
    preserve=1
    [[ "$stage" != BASE* ]] || preserve=0
    DEPLOY_PRESERVE_IDENTICAL=$preserve bash launchers/deploy-overlays.sh glm53 > "$EVIDENCE/$current_arm-deploy.log" 2>&1
    printf '%s\t%s\n' "$current_arm" "$(( $(date +%s) - deploy_start ))" >> "$EVIDENCE/deployment-seconds.tsv"
    python3 bench/startup_deploy_receipts.py "$EVIDENCE/$current_arm-after-deploy.json"
    python3 - "$EVIDENCE" "$current_arm" "$stage" <<'DEPLOY_GATE'
import json, sys
from pathlib import Path
root, arm, stage = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
before = json.loads((root / f"{arm}-before-deploy.json").read_text())
after = json.loads((root / f"{arm}-after-deploy.json").read_text())
assert set(before) == set(after) == {"srv1", "srv2", "srv3", "srv4"}
if stage != "PRIME":
    for node in before:
        a, b = before[node]["files"], after[node]["files"]
        assert a.keys() == b.keys()
        assert all(a[k]["sha256"] == b[k]["sha256"] for k in a), (node, "runtime content changed")
        if stage.startswith("FAST"):
            assert all(a[k]["mtime_ns"] == b[k]["mtime_ns"] and a[k]["inode"] == b[k]["inode"] for k in a), (node, "identical source rewritten")
        else:
            assert all(a[k]["mtime_ns"] != b[k]["mtime_ns"] for k in a if k.endswith(".cu")), (node, "control did not rewrite CUDA sources")
print("all-rank same-content deployment path verified")
DEPLOY_GATE
  fi
  [ "$MODE" != campaign ] || knobs=${campaign_knobs[$stage]}
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
    python3 - "$EVIDENCE" "$current_arm" "$stage" "$MODE" <<'PY'
from pathlib import Path
import re, sys
root, arm, stage = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
mode = sys.argv[4]
for node in (1, 2, 3, 4):
    text = (root / f"{arm}-srv{node}.log").read_text()
    assert re.search(r"\[rank-cache\] (?:hit|saved) rank=", text), f"srv{node}: no rank artifact used/saved"
    rows = re.findall(r"\[fp8-cache\].*?enabled=True hit=(\d+) miss=(\d+) errors=(\d+)", text)
    assert len(rows) >= 2, f"srv{node}: target/drafter FP8 cache receipts missing"
    assert all(int(e) == 0 for h, m, e in rows), f"srv{node}: FP8 cache errors: {rows}"
    if stage == "WARM" or (mode in ("pack-io", "pack-key", "renderer-warmup", "graph-profile", "overlay-deploy", "campaign") and stage != "PRIME"):
        assert re.search(r"\[rank-cache\] hit rank=", text), f"srv{node}: rank cache missed"
        assert all(int(h) > 0 and int(m) == 0 for h, m, e in rows), f"srv{node}: FP8 warm misses: {rows}"
    if mode == "campaign":
        import json
        env = dict(v.split("=", 1) for v in json.loads((root / f"{arm}-cache-env.json").read_text()))
        if "VLLM_GLM53_MK_PACK_FAST_IO" in env:
            fast = int(env["VLLM_GLM53_MK_PACK_FAST_IO"])
            io = re.findall(r"\[mk-pack-io\].*?fast=(\d+).*?fast_hits=(\d+) legacy_hits=(\d+)", text)
            assert len(io) >= 2, f"srv{node}: pack IO receipts missing"
            assert all(int(f) == fast and (stage == "PRIME" or int(h if fast else l) > 0)
                       and int(l if fast else h) == 0 for f, h, l in io), f"srv{node}: campaign pack IO mismatch: {io}"
        if env.get("VLLM_GLM53_MK_PACK_SHA256") == "1" and stage != "PRIME":
            keys = re.findall(r"sha_hits=(\d+) md5_fallback=(\d+) aliases=(\d+) alias_errors=(\d+)", text)
            assert len(keys) >= 2 and all(int(h) > 0 and int(m) == int(a) == int(e) == 0 for h,m,a,e in keys), f"srv{node}: campaign SHA alias miss"
        assert not re.search(r"pack cache .*?unreadable|pack cache key failed|MK W4 pack build FAILED", text), f"srv{node}: campaign pack failure"
    if mode == "pack-io":
        fast = int(stage.startswith("FAST"))
        io = re.findall(r"\[mk-pack-io\].*?fast=(\d+).*?fast_hits=(\d+) legacy_hits=(\d+)", text)
        assert len(io) >= 2, f"srv{node}: pack IO receipts missing"
        assert all(int(f) == fast and (stage == "PRIME" or int(h if fast else l) > 0)
                   and int(l if fast else h) == 0 for f, h, l in io), f"srv{node}: wrong pack IO path: {io}"
        assert not re.search(r"pack cache .*?unreadable|MK W4 pack build FAILED", text), f"srv{node}: pack restore failure"
    if mode == "pack-key":
        sha = int(stage == "PRIME" or stage.startswith("FAST"))
        io = re.findall(r"\[mk-pack-io\].*?fast=(\d+) sha256=(\d+).*?fast_hits=(\d+) legacy_hits=(\d+) sha_hits=(\d+) md5_fallback=(\d+) aliases=(\d+) alias_errors=(\d+)", text)
        assert len(io) >= 2, f"srv{node}: pack key receipts missing"
        assert all(int(f) == 1 and int(s) == sha and int(h) > 0 and int(l) == 0 and int(e) == 0
                   for f, s, h, l, sh, md, a, e in io), f"srv{node}: wrong pack key path: {io}"
        if stage != "PRIME":
            assert all(int(md) == 0 and int(a) == 0 and (int(sh) == int(h) if sha else int(sh) == 0)
                       for f, s, h, l, sh, md, a, e in io), f"srv{node}: pack key warm miss: {io}"
        packs = re.findall(r"packs: rtn=(\d+) gptq=(\d+) gptq_failed=(\d+) cached=(\d+)", text)
        assert len(packs) >= 2 and all(int(r) == int(g) == int(e) == 0 and int(h) > 0
                                    for r, g, e, h in packs), f"srv{node}: unexpected repack: {packs}"
        assert not re.search(r"pack cache .*?unreadable|pack cache key failed|MK W4 pack build FAILED", text), f"srv{node}: pack restore failure"
    if mode in ("renderer-warmup", "graph-profile", "overlay-deploy"):
        packs = re.findall(r"packs: rtn=(\d+) gptq=(\d+) gptq_failed=(\d+) cached=(\d+)", text)
        assert len(packs) >= 2 and all(int(r) == int(g) == int(e) == 0 and int(h) > 0
                                    for r, g, e, h in packs), f"srv{node}: unexpected repack"
        assert not re.search(r"pack cache .*?unreadable|pack cache key failed|MK W4 pack build FAILED", text)
        if node == 2 and mode == "renderer-warmup":
            early = stage == "PRIME" or stage.startswith("FAST")
            if early:
                assert "[early-mm-warmup] submitted processors=2 before engine startup" in text
                assert "[early-mm-warmup] completed processors=2/2" in text
                assert len(re.findall(r"\[early-mm-warmup\] reused .*? join_s=", text)) == 2
                assert text.index("[early-mm-warmup] completed") < text.index("[boot-stamp] load-model took"), "warmup did not overlap model startup"
                assert not re.search(r"\[early-mm-warmup\].*(?:failed|skipped|unavailable)", text)
            else:
                assert "[early-mm-warmup]" not in text
            assert "multi-modal warmup failed" not in text.lower()
    if mode in ("graph-profile", "overlay-deploy"):
        fast = mode == "graph-profile" and (stage == "PRIME" or stage.startswith("FAST"))
        assert ("[glm53-graph-profile] skipped unused estimate" in text) == fast, f"srv{node}: wrong graph profile path"
        assert ("[boot-stamp] cudagraph-memory-profile took" in text) != fast, f"srv{node}: wrong dry capture path"
        for phase in ("encoder-profile", "profile-run", "cudagraph-capture", "compile+warmup"):
            assert f"[boot-stamp] {phase} took" in text, f"srv{node}: missing {phase}"
        assert "Traceback (most recent call last)" not in text, f"srv{node}: startup traceback"
print("all four nodes have the required cache receipts")
PY
  fi
  if [[ "$MODE" = graph-profile || "$MODE" = overlay-deploy ]]; then
    python3 bench/startup_first_requests.py --out "$EVIDENCE/$current_arm-first-requests.json" > "$EVIDENCE/$current_arm-first-requests.out" 2>&1
  fi
  STARTUP_CACHE_RESPONSES="$EVIDENCE/$current_arm-responses.jsonl" \
    python3 bench/startup_cache_onepass.py --name "$current_arm" --ctx "$QUALITY_CTX" --out "$EVIDENCE/onepass.jsonl" > "$EVIDENCE/$current_arm-onepass.out" 2>&1
  tail -1 "$EVIDENCE/onepass.jsonl" >> "$LOGD/bracket-onepass.jsonl"
  cat "$EVIDENCE/$current_arm-onepass.out"
  snapshot "$current_arm"
  if [ "$MODE" = overlay-deploy ]; then
    python3 bench/startup_deploy_receipts.py "$EVIDENCE/$current_arm-after-boot.json"
    python3 - "$EVIDENCE" "$current_arm" "$stage" <<'NINJA_GATE'
import json, sys
from pathlib import Path
root, arm, stage = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
before = json.loads((root / f"{arm}-after-deploy.json").read_text())
after = json.loads((root / f"{arm}-after-boot.json").read_text())
if stage != "PRIME":
    for node in before:
        a, b = before[node]["ninja"], after[node]["ninja"]
        assert a and b, (node, "missing compile receipts")
        changed = [k for k in b if k not in a or a[k]["sha256"] != b[k]["sha256"]]
        assert bool(changed) != stage.startswith("FAST"), (node, stage, changed)
        print(node, "Ninja logs changed:", changed)
NINJA_GATE
  fi
  python3 - "$EVIDENCE/onepass.jsonl" <<'PY'
import json, sys
row = json.loads(open(sys.argv[1]).readlines()[-1])
assert row['quality']['ok'] == row['quality']['total'], row['quality']
assert row['korean']['dirty'] == 0, row['korean']
PY
  if [ "$stage" = PRIME ] && { [ "$MODE" = pack-io ] || { [ "$MODE" = campaign ] && [[ $knobs == *VLLM_GLM53_MK_PACK_FAST_IO=* ]]; }; }; then
    docker cp "$REPO/probes/glm53_pack_io_check.py" glm53:/tmp/glm53_pack_io_check.py
    docker exec glm53 python3 /tmp/glm53_pack_io_check.py > "$EVIDENCE/pack-io-gpu.json" 2> "$EVIDENCE/pack-io-gpu.log"
    cat "$EVIDENCE/pack-io-gpu.json"
  fi
  if [ "$stage" = PRIME ] && { [ "$MODE" = pack-key ] || { [ "$MODE" = campaign ] && [[ $knobs == *VLLM_GLM53_MK_PACK_SHA256=1* ]]; }; }; then
    docker cp "$REPO/probes/glm53_pack_key_check.py" glm53:/tmp/glm53_pack_key_check.py
    docker exec glm53 python3 /tmp/glm53_pack_key_check.py --out /tmp/pack-key-gpu.json > "$EVIDENCE/pack-key-gpu.log" 2>&1
    docker cp glm53:/tmp/pack-key-gpu.json "$EVIDENCE/pack-key-gpu.json"
    cat "$EVIDENCE/pack-key-gpu.json"
  fi
  if [ "$MODE" = campaign ]; then
    python3 bench/startup_campaign.py check --evidence "$EVIDENCE" --stage "$stage"
  fi
  echo "=== $current_arm complete $(date -Is) ==="
done

if [ "$MODE" = campaign ]; then
  python3 bench/startup_campaign.py summarize --evidence "$EVIDENCE"
fi
