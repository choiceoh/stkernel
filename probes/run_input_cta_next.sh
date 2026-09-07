#!/usr/bin/env bash
# Run a bounded isolated kernel probe under the fleet's idle-serving lane.
set -euo pipefail
cd /home/choiceoh/stkernel-input-cta-next-20260908
export REPO=$PWD
out=${INPUT_NEXT_OUT:-/home/choiceoh/glm53-logs/INPUTNEXT0908}
session=${FLEET_SESSION:?}
image=sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211
IFS='|' read -r held _pid _host _start _est _note kind < /home/choiceoh/glm53-logs/fleet/holder
[[ $held == "$session" && $kind == probe ]] || exit 2
[[ -z $(git status --porcelain) ]] || exit 2
[[ ! -e $out/source.commit ]] || { echo 'ABORT: fresh evidence required'; exit 2; }
mkdir -p "$out/build"
git rev-parse HEAD > "$out/source.commit"
snapshot() {
  local target=$1
  curl -fsS --max-time 5 http://127.0.0.1:8000/health >/dev/null
  curl -fsS --max-time 5 http://127.0.0.1:8000/metrics > "$out/$target.metrics"
  docker inspect glm53 --format '{{.Id}} {{.Image}} {{.State.StartedAt}}' > "$out/$target.boot"
  python3 - "$out/$target.metrics" <<'PY'
import sys
from pathlib import Path
lines=Path(sys.argv[1]).read_text().splitlines()
for key in ('num_requests_running','num_requests_waiting'):
    values=[float(l.rsplit(' ',1)[1]) for l in lines if l.startswith('vllm:'+key+'{')]
    assert values and sum(values)==0,(key,values)
PY
}
snapshot before
[[ $(docker inspect glm53 --format '{{.Image}}') == "$image" ]] || exit 2
available_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
(( available_kib >= 8*1024*1024 )) || { echo 'ABORT: less than 8 GiB available'; exit 2; }
cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  docker stop -t 2 "inputnext-$session" >/dev/null 2>&1 || true
  echo "$rc" > "$out/runner.exit"
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
args=(run --rm --name "inputnext-$session" --gpus device=0 --network=none
      --cpuset-cpus=14-17 --memory=5g --shm-size=1g
      --mount "type=bind,src=$REPO,dst=/repo,readonly"
      --mount "type=bind,src=$out,dst=/evidence"
      --mount "type=bind,src=$out/build,dst=/build"
      --mount 'type=bind,src=/usr/local/cuda/compute-sanitizer,dst=/san,readonly'
      --workdir /repo)
timeout 420 docker "${args[@]}" --entrypoint python3 "$image" \
  /repo/probes/gemm_input_cta_next.py > "$out/probe.log" 2>&1
for tool in racecheck memcheck; do
  timeout 240 docker "${args[@]}" --entrypoint /san/compute-sanitizer "$image" \
    --tool "$tool" --target-processes application-only --error-exitcode 77 \
    --kernel-name kns=mk_gemm_input_next_kernel \
    python3 /repo/probes/gemm_input_cta_next.py --check-only \
    --out "/evidence/$tool.json" > "$out/$tool.log" 2>&1
done
snapshot after
cmp "$out/before.boot" "$out/after.boot"
python3 - "$out" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
def finished(name):
    return {l.rsplit(' ',1)[0]:float(l.rsplit(' ',1)[1]) for l in (p/name).read_text().splitlines()
            if l.startswith('vllm:request_success_total{')}
before,after=finished('before.metrics'),finished('after.metrics')
assert before and before==after, 'serving traffic changed during the kernel probe'
assert all(json.loads((p/(name+'.json')).read_text())['status']=='PASS' for name in ('result','racecheck','memcheck'))
(p/'service-proof.json').write_text(json.dumps({'same_boot':True,'idle_before_after':True,
    'finished_requests_unchanged':True,'service_restarted':False},indent=2)+'\n')
print('PASS actual CTA=2 baseline, exact candidates, sanitizers; idle serving untouched')
PY
