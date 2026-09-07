#!/usr/bin/env bash
# Called only by glm53_offline_checks.py within our normal fleet boot hold.
set -euo pipefail
diagnostic=0
probe_args=()
transports=(bf16 fp8-v3)
if [[ $# == 1 && $1 == --fp8-diagnostic ]]; then
  diagnostic=1
  probe_args+=(--fp8-diagnostic)
  transports=(fp8-v3)
elif [[ $# != 0 ]]; then
  echo 'only --fp8-diagnostic is accepted' >&2; exit 2
fi
REPO=$(cd "$(dirname "$0")/.." && pwd)
python3 -c 'import sys;sys.path.insert(0,sys.argv[1]+"/probes");from glm53_offline_checks import check_holder;check_holder()' "$REPO"
revision=$(git -C "$REPO" rev-parse HEAD)
IMAGE=sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211
TARGET_PREFIX=/usr/local/lib/python3.12/dist-packages
port=29675
if [[ -n $(ss -ltnH "sport = :$port") ]]; then echo "port $port is in use" >&2; exit 1; fi
ips=(10.10.10.2 10.10.10.1 10.10.10.3 10.10.10.4)
log_dir=$(mktemp -d /tmp/glm53-moe-m64.XXXXXX)
echo "TP4 M64 logs: $log_dir source=$revision image=$IMAGE"
run_id="moe-m64-${FLEET_SESSION}-$$"
cleanup() {
  docker rm -f "$run_id-0" >/dev/null 2>&1 || true
  for rank in 1 2 3; do
    ssh -o BatchMode=yes -o ConnectTimeout=5 "choiceoh@${ips[$rank]}" \
      "docker rm -f $run_id-$rank >/dev/null 2>&1 || true" </dev/null &
  done
  wait || true
}
trap cleanup EXIT
# Refuse dirty/stale worker source and low memory before any CUDA process.
for rank in 0 1 2 3; do
  code='import json,pathlib,subprocess,sys; p=sys.argv[1]; r=sys.argv[2]; assert subprocess.check_output(["git","-C",p,"rev-parse","HEAD"],text=True).strip()==r; assert not subprocess.check_output(["git","-C",p,"status","--porcelain"],text=True).strip(); m=dict(l.split(":",1) for l in pathlib.Path("/proc/meminfo").read_text().splitlines()); assert int(m["MemAvailable"].split()[0])>=16*1024*1024; print("source and memory PASS")'
  args=(python3 -c "$code" "$REPO" "$revision")
  if [[ $rank == 0 ]]; then "${args[@]}"; else
    printf -v command '%q ' "${args[@]}"
    ssh -o BatchMode=yes -o ConnectTimeout=5 "choiceoh@${ips[$rank]}" "$command" </dev/null
  fi
done
for transport in "${transports[@]}"; do
  pids=()
  for rank in 0 1 2 3; do
    args=(docker run --rm --name "$run_id-$rank" --gpus all --network host
      --cpus 4 --memory 16g --shm-size 1g --entrypoint python3
      --device /dev/infiniband:/dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1:-1
      -v "$REPO:/repo:ro"
      -e NCCL_NET=IB -e NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0
      -e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e GLOO_SOCKET_IFNAME=enP2p1s0f0np0
      -e NCCL_IB_GID_INDEX=3 -e NCCL_IB_ADDR_FAMILY=AF_INET
      -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_DISABLE=0 -e NCCL_CROSS_NIC=1
      -e NCCL_PROTO=LL,LL128,Simple -e NCCL_MIN_NCHANNELS=16 -e NCCL_MAX_NCHANNELS=16
      -e NCCL_NCHANNELS_PER_NET_PEER=4 -e NCCL_CUMEM_ENABLE=0 -e NCCL_NVLS_ENABLE=0
      -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_P2P_LEVEL=SYS
      -e NCCL_DEBUG=WARN -e CUTE_DSL_ARCH=sm_121a -e OMP_NUM_THREADS=1 -e PYTHONPATH=/repo/probes)
    while IFS=$'\t' read -r source target _; do
      [[ -z $source || $source == \#* ]] && continue
      args+=(-v "$REPO/build/glm53/$source:$target:ro")
    done < "$REPO/build/glm53/manifest.tsv"
    args+=("$IMAGE" -m torch.distributed.run --nnodes=4 --nproc-per-node=1
      --node-rank="$rank" --master-addr=10.10.10.2 --master-port="$port"
      /repo/probes/glm53_moe_m64_check.py --transport "$transport" "${probe_args[@]}")
    if [[ $rank == 0 ]]; then
      timeout 900 "${args[@]}" >"$log_dir/$transport-rank-$rank.log" 2>&1 &
    else
      printf -v command '%q ' "${args[@]}"
      timeout 900 ssh -o BatchMode=yes -o ConnectTimeout=5 "choiceoh@${ips[$rank]}" \
        "$command" >"$log_dir/$transport-rank-$rank.log" 2>&1 &
    fi
    pids+=("$!")
  done
  for rank in 0 1 2 3; do
    if ! wait "${pids[$rank]}"; then
      tail -60 "$log_dir/$transport-rank-$rank.log";exit 1
    fi
  done
  cat "$log_dir/$transport-rank-0.log"
done
if [[ $diagnostic == 1 ]]; then
  python3 - "$log_dir/fp8-v3-rank-0.log" "$REPO/probes" <<'DIAGNOSTIC'
import json,pathlib,sys
sys.path.insert(0,sys.argv[2])
from glm53_moe_m64_fp8_diagnostic import MARKER,completion
records=[json.loads(l) for l in pathlib.Path(sys.argv[1]).read_text().splitlines() if l.startswith('{')]
trials=[r for r in records if r.get('kind')=='MOE_M64_FP8_DIAGNOSTIC_TRIAL']
reports=[r for r in records if r.get('verdict')==MARKER]
assert len(reports)==1
report=reports[0]
assert completion(trials,report['provenance'])==report
assert report['serving_gate'] is False and report['numerical_acceptance'] is False
print(MARKER)
DIAGNOSTIC
  exit 0
fi
# The image does not contain compute-sanitizer; use the pinned host tool,
# as the MLA probe does. Version/hash are evidence, not GPU acceptance.
tool_dir=/usr/local/cuda/compute-sanitizer
[[ -x $tool_dir/compute-sanitizer ]] || { echo 'host sanitizer missing' >&2; exit 3; }
sha256sum "$tool_dir/compute-sanitizer"
for sanitizer in memcheck racecheck; do
  args=(docker run --rm --name "$run_id-0" --gpus all --network none
    --cpus 4 --memory 16g --shm-size 1g
    -v "$REPO:/repo:ro" -v "$tool_dir:/opt/glm-probe-sanitizer:ro"
    -e CUTE_DSL_ARCH=sm_121a -e OMP_NUM_THREADS=1 -e PYTHONPATH=/repo/probes
    --entrypoint /opt/glm-probe-sanitizer/compute-sanitizer)
  while IFS=$'\t' read -r source target _; do
    [[ -z $source || $source == \#* ]] && continue
    args+=(-v "$REPO/build/glm53/$source:$target:ro")
  done < "$REPO/build/glm53/manifest.tsv"
  args+=("$IMAGE" --error-exitcode 99 --tool "$sanitizer"
    python3 /repo/probes/glm53_moe_m64_sanitize.py)
  if ! timeout --signal=TERM --kill-after=30s 15m "${args[@]}" >"$log_dir/$sanitizer.log" 2>&1; then
    tail -60 "$log_dir/$sanitizer.log";exit 1
  fi
  python3 - "$log_dir/$sanitizer.log" "$sanitizer" <<'CHECK'
import json,pathlib,sys
log=pathlib.Path(sys.argv[1]).read_text();tool=sys.argv[2]
summary='ERROR SUMMARY: 0 errors' if tool=='memcheck' else 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)'
assert summary in log, 'missing clean sanitizer summary'
reports=[json.loads(l) for l in log.splitlines() if l.startswith('{')]
record=next(r for r in reports if r.get('verdict')=='MOE_M64_SANITIZER_CASES_PASS')
assert [(r['rows'],r['skew'],r['bad_rows']) for r in record['results']]==[(n,s,0) for n in (6144,6912,8192) for s in (False,True)]
print(json.dumps(dict(sanitizer=tool,summary=summary,**record)))
CHECK
  echo "MOE_M64_${sanitizer^^}_PASS"
done
echo MOE_M64_ALL_GATES_PASS
