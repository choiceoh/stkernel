#!/usr/bin/env bash
# Run from srv2 while holding a fleet.sh GPU probe turn. The same isolated
# checkout must exist at REPO on all nodes. No serving restart or deployment.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
transport=${1:-fp8-v3}
case "$transport" in bf16|fp8|fp8-v2|fp8-v3) ;; *) echo 'unknown transport' >&2; exit 2;; esac
port=${PREFILL_PROBE_PORT:-29673}
if [[ -n $(ss -ltnH "sport = :$port") ]]; then echo "port $port is in use" >&2; exit 1; fi
eval "$(
  . "$REPO/profiles/glm53.env"
  printf 'IMAGE=%q\nTARGET_PREFIX=%q\n' "$PROFILE_IMAGE" "$TARGET_PREFIX"
)"
ips=(10.10.10.2 10.10.10.1 10.10.10.3 10.10.10.4)
run_id="prefill-tp4-$(date +%s)-$$"
log_dir=$(mktemp -d "${TMPDIR:-/tmp}/glm53-prefill-tp4.XXXXXX")
echo "TP4 probe logs: $log_dir; run=$run_id; transport=$transport"
# Names are unique to this invocation; cleanup only touches these containers.
cleanup() {
  docker stop -t 3 "$run_id-0" >/dev/null 2>&1 || true
  for rank in 1 2 3; do
    ssh -o BatchMode=yes -o ConnectTimeout=5 "choiceoh@${ips[$rank]}" \
      "docker stop -t 3 $run_id-$rank >/dev/null 2>&1 || true" </dev/null &
  done
  wait || true
}
trap cleanup EXIT
pids=()
for rank in 0 1 2 3; do
  args=(docker run --rm --name "$run_id-$rank" --gpus all --network host
        --cpus 4 --memory 6g --shm-size 1g --entrypoint python3
        --device /dev/infiniband:/dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1:-1
        -v "$REPO:/repo:ro"
        -e NCCL_NET=IB -e NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0
        -e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e GLOO_SOCKET_IFNAME=enP2p1s0f0np0
        -e NCCL_IB_GID_INDEX=3 -e NCCL_IB_ADDR_FAMILY=AF_INET
        -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_DISABLE=0 -e NCCL_CROSS_NIC=1
        -e NCCL_PROTO=LL,LL128,Simple -e NCCL_MIN_NCHANNELS=16 -e NCCL_MAX_NCHANNELS=16
        -e NCCL_NCHANNELS_PER_NET_PEER=4 -e NCCL_CUMEM_ENABLE=0 -e NCCL_NVLS_ENABLE=0
        -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_P2P_LEVEL=SYS
        -e NCCL_DEBUG=WARN -e OMP_NUM_THREADS=1)
  for pair in \
    'glm53_prefill_collectives.py:device_communicators/glm53_prefill_collectives.py' \
    'cuda_communicator.py:device_communicators/cuda_communicator.py' \
    'parallel_state.py:parallel_state.py'; do
    args+=(-v "$REPO/overlay/modules/glm53_runtime/${pair%%:*}:${TARGET_PREFIX%/}/vllm/distributed/${pair#*:}:ro")
  done
  args+=("$IMAGE" -m torch.distributed.run --nnodes=4 --nproc-per-node=1
         --node-rank="$rank" --master-addr=10.10.10.2 --master-port="$port"
         /repo/probes/glm53_prefill_collectives_check.py --transport "$transport"
         --rows 128 129 8185 --timing)
  if [[ $rank == 0 ]]; then
    # srv2 need not have an SSH key authorized for itself.
    timeout 240 "${args[@]}" >"$log_dir/rank-$rank.log" 2>&1 &
  else
    printf -v remote_command '%q ' "${args[@]}"
    timeout 240 ssh -o BatchMode=yes -o ConnectTimeout=5 "choiceoh@${ips[$rank]}" \
      "$remote_command" >"$log_dir/rank-$rank.log" 2>&1 &
  fi
  pids+=("$!")
done
result=0
for rank in 0 1 2 3; do
  if wait "${pids[$rank]}"; then
    echo "rank $rank: exited successfully"
  else
    result=1
    echo "rank $rank: FAILED"
    tail -60 "$log_dir/rank-$rank.log"
    break  # EXIT cleanup stops peers instead of waiting out rendezvous timeouts.
  fi
done
if [[ $result == 0 ]]; then
  cat "$log_dir/rank-0.log"
fi
exit "$result"
