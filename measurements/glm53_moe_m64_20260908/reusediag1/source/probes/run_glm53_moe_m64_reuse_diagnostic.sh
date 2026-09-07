#!/usr/bin/env bash
# Local numerical collection, with working error detectors, under an owned hold.
set -euo pipefail
[[ $# == 0 ]] || exit 2
REPO=$(cd "$(dirname "$0")/.." && pwd)
python3 -c 'import sys;sys.path.insert(0,sys.argv[1]+"/probes");from glm53_offline_checks import check_holder;check_holder()' "$REPO"
revision=$(git -C "$REPO" rev-parse HEAD)
IMAGE=sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211
log_dir=$(mktemp -d /tmp/glm53-m64-reuse.XXXXXX)
name="moe-m64-${FLEET_SESSION}-reuse-$$"
trap 'docker rm -f "$name" >/dev/null 2>&1 || true' EXIT
echo "M64 reuse diagnostic logs: $log_dir source=$revision image=$IMAGE"
for node in local 10.10.10.1 10.10.10.3 10.10.10.4; do
  code='import pathlib,subprocess,sys;p=sys.argv[1];r=sys.argv[2];assert subprocess.check_output(["git","-C",p,"rev-parse","HEAD"],text=True).strip()==r;assert not subprocess.check_output(["git","-C",p,"status","--porcelain"],text=True).strip();m=dict(l.split(":",1) for l in pathlib.Path("/proc/meminfo").read_text().splitlines());assert int(m["MemAvailable"].split()[0])>=16*1024*1024;print("source and memory PASS")'
  args=(python3 -c "$code" "$REPO" "$revision")
  if [[ $node == local ]]; then "${args[@]}"; else
    printf -v command '%q ' "${args[@]}"
    ssh -o BatchMode=yes -o ConnectTimeout=5 "choiceoh@$node" "$command" </dev/null
  fi
done
bash "$REPO/probes/run_glm53_cuda_driver_lookup_check.sh" --canaries-only >"$log_dir/detectors.log" 2>&1
python3 - "$log_dir/detectors.log" "$REPO/probes" <<'DETECTORS'
import json,pathlib,sys
sys.path.insert(0,sys.argv[2])
from glm53_sanitizer_report import validate_canaries
print(json.dumps(validate_canaries(pathlib.Path(sys.argv[1]).read_text(),sys.argv[2])),flush=True)
DETECTORS
tool_dir=/usr/local/cuda/compute-sanitizer
"$tool_dir/compute-sanitizer" --version
sha256sum "$tool_dir/compute-sanitizer"
for mode in plain memcheck racecheck; do
  args=(docker run --rm --name "$name" --gpus all --network none
    --cpus 4 --memory 16g --shm-size 1g -v "$REPO:/repo:ro"
    -e CUTE_DSL_ARCH=sm_121a -e OMP_NUM_THREADS=1 -e PYTHONPATH=/repo/probes)
  while IFS=$'\t' read -r source target _; do
    [[ -z $source || $source == \#* ]] && continue
    args+=(-v "$REPO/build/glm53/$source:$target:ro")
  done < "$REPO/build/glm53/manifest.tsv"
  if [[ $mode == plain ]]; then
    args+=(--entrypoint python3 "$IMAGE")
  else
    args+=(-v "$tool_dir:/opt/glm-probe-sanitizer:ro"
      --entrypoint /opt/glm-probe-sanitizer/compute-sanitizer "$IMAGE"
      --error-exitcode 99 --tool "$mode" python3)
  fi
  args+=(/repo/probes/glm53_moe_m64_sanitize.py --reuse-diagnostic)
  limit=15m
  if [[ $mode == plain ]]; then limit=3m; fi
  if ! timeout --signal=TERM --kill-after=30s "$limit" "${args[@]}" >"$log_dir/$mode.log" 2>&1; then
    tail -c 2200 "$log_dir/$mode.log"; exit 1
  fi
  docker run --rm -i --runtime runc --network none --memory 4g --cpus 2 \
    -v "$REPO:/repo:ro" -v "$log_dir:/evidence:ro" --entrypoint python3 "$IMAGE" \
    - "/evidence/$mode.log" "$mode" /repo <<'VERIFY'
import json,pathlib,sys
sys.path.insert(0,sys.argv[3]+'/probes')
from glm53_moe_m64_reuse_diagnostic import verify_log
log=pathlib.Path(sys.argv[1]).read_text();mode=sys.argv[2]
report=verify_log(log,sys.argv[3])
if mode=='memcheck':assert 'ERROR SUMMARY: 0 errors' in log
if mode=='racecheck':assert 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' in log
print(json.dumps(dict(mode=mode,**report)),flush=True)
VERIFY
done
echo MOE_M64_REUSE_COLLECTION_COMPLETE
