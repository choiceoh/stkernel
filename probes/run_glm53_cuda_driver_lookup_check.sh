#!/usr/bin/env bash
# Isolate sanitizer API initialization reports under a normal owned boot hold.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
python3 -c 'import sys;sys.path.insert(0,sys.argv[1]+"/probes");from glm53_offline_checks import check_holder;check_holder()' "$REPO"
IMAGE=sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211
log_dir=$(mktemp -d /tmp/glm53-driver-lookup.XXXXXX)
name="moe-m64-${FLEET_SESSION}-driver-$$"
trap 'docker rm -f "$name" >/dev/null 2>&1 || true' EXIT
tool_dir=/usr/local/cuda/compute-sanitizer
echo "Driver lookup logs: $log_dir"
"$tool_dir/compute-sanitizer" --version
sha256sum "$tool_dir/compute-sanitizer"
for mode in torch-only driver-only torch-driver driver-torch torch-driver-invalid driver-torch-invalid; do
  rc=0
  timeout --signal=TERM --kill-after=10s 90s docker run --rm --name "$name" --gpus all --network none \
    --cpus 2 --memory 4g --shm-size 1g -v "$REPO:/repo:ro" \
    -v "$tool_dir:/opt/glm-probe-sanitizer:ro" \
    --entrypoint /opt/glm-probe-sanitizer/compute-sanitizer "$IMAGE" \
    --error-exitcode 99 --tool memcheck python3 /repo/probes/glm53_cuda_driver_lookup_check.py --mode "$mode" \
    >"$log_dir/$mode.log" 2>&1 || rc=$?
  python3 - "$log_dir/$mode.log" "$mode" "$rc" <<'PARSE'
import collections,hashlib,json,pathlib,re,sys
p=pathlib.Path(sys.argv[1]);lines=p.read_text().splitlines();mode=sys.argv[2];rc=int(sys.argv[3])
records=[json.loads(l) for l in lines if l.startswith('{')]
final=[r for r in records if r.get('kind')=='DRIVER_LOOKUP_PROGRAM_COMPLETE']
assert len(final)==1 and final[0]['mode']==mode and rc in (0,99)
summaries=[int(m.group(1)) for l in lines if (m:=re.fullmatch(r'========= ERROR SUMMARY: (\d+) errors',l))]
assert len(summaries)==1
errors=[l for l in lines if l.startswith('========= Program hit ')]
report=dict(mode=mode,exit_code=rc,error_count=summaries[0],api_reports=dict(collections.Counter(errors)),
    program=final[0],log_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
    serving_gate=False,numerical_acceptance=False)
p.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
PARSE
done
for tool in memcheck racecheck; do
  rc=0
  timeout --signal=TERM --kill-after=10s 120s docker run --rm --name "$name" --gpus all --network none \
    --cpus 2 --memory 4g --shm-size 1g -v "$REPO:/repo:ro" \
    -e GLM53_SANITIZER_CANARY=1 -e CUTE_DSL_ARCH=sm_121a \
    -v "$tool_dir:/opt/glm-probe-sanitizer:ro" \
    --entrypoint /opt/glm-probe-sanitizer/compute-sanitizer "$IMAGE" \
    --error-exitcode 99 --tool "$tool" python3 /repo/probes/glm53_sanitizer_order_canary.py --tool "$tool" \
    >"$log_dir/$tool-canary.log" 2>&1 || rc=$?
  python3 - "$log_dir/$tool-canary.log" "$tool" "$rc" <<'CANARY'
import hashlib,json,pathlib,re,sys
p=pathlib.Path(sys.argv[1]);text=p.read_text();tool=sys.argv[2];rc=int(sys.argv[3])
records=[json.loads(l) for l in text.splitlines() if l.startswith('{')]
assert any(r.get('kind')=='INTENTIONAL_BAD_KERNEL_END' for r in records)
detected='Invalid __global__ write' in text if tool=='memcheck' else bool(re.search(r'RACECHECK SUMMARY: [1-9][0-9]* hazards',text))
report=dict(tool=tool,exit_code=rc,deliberate_error_detected=detected,log_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
    serving_gate=False,numerical_acceptance=False)
p.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
CANARY
done
echo DRIVER_LOOKUP_DIAGNOSTIC_COMPLETE
