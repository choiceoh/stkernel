#!/usr/bin/env bash
# Window 6's training rank (2026-09-19/20): from the original head (no --init) on the v3 answer-only runs (data6).
# train_rank6.sh RANK STEPS EVAL_EVERY LR EVAL_WINDOWS -- this node's rank of the fifth four-node MTP tuning, detached
# as container q38tune: the checkpoint's own head as the start (the trainer clamps the warmup to a tenth of the run).
set -euo pipefail
r=$1; steps=$2; every=$3; lr=$4; windows=${5:-96}
W=/home/choiceoh/q38mtp-train-0919
mkdir -p "$W/run6" "$W/cache"
NCCL_ENV=(-e NCCL_P2P_LEVEL=SYS -e TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600 -e NCCL_NET=IB -e NCCL_IB_DISABLE=0
          -e NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0 -e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e GLOO_SOCKET_IFNAME=enP2p1s0f0np0
          -e NCCL_CROSS_NIC=1 -e NCCL_PROTO=LL,LL128,Simple -e NCCL_CUMEM_ENABLE=0 -e NCCL_IB_ROCE_VERSION_NUM=2
          -e NCCL_IB_ADDR_FAMILY=AF_INET -e NCCL_NVLS_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN
          -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True)
docker rm -f q38tune >/dev/null 2>&1 || true
docker run -d --name q38tune --gpus all --network host --ipc host --shm-size 16g --ulimit memlock=-1:-1 \
  --ulimit nofile=524288:524288 --cap-add IPC_LOCK --device /dev/infiniband:/dev/infiniband \
  --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -e RANK="$r" -e WORLD_SIZE=4 -e MASTER_ADDR=10.10.10.2 -e MASTER_PORT=29611 -e LOCAL_RANK=0 "${NCCL_ENV[@]}" \
  -v /home/choiceoh/st-engine-qwen38:/repo:ro -v "$W/base":/ckpt:ro -v "$W/data6":/data:ro -v "$W/run6":/out -v "$W/cache":/cache \
  --entrypoint /bin/bash st-engine:qwen38 -lc 'source /repo/launchers/lib/common-tp4.sh; eval "$CT_GID_PRELUDE"; cd /repo && PYTHONPATH=/repo exec python3 -u -m engine.profiles.qwen38.mtp_tune train --data /data --ckpt /ckpt --out /out --steps '"$steps"' --accumulate 2 --eval-every '"$every"' --eval-windows '"$windows"' --lr '"$lr" >/dev/null
echo "rank $r started"
