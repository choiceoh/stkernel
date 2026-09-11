#!/usr/bin/env bash
set -euo pipefail
rank=${1:?rank required}
cd /home/choiceoh/st-engine-f4d7-20260911
mkdir -p evidence
trap 'docker rm -f st-engine-f4d7-model >/dev/null 2>&1 || true' EXIT
timeout --kill-after=10s 180 docker run --rm --name st-engine-f4d7-model --gpus all --network host \
  --device /dev/infiniband --ulimit memlock=-1 --shm-size=256m \
  --memory=8g --memory-swap=8g --cpus=2 --entrypoint /bin/bash \
  --mount type=bind,src=$PWD,dst=/repo,readonly \
  --mount type=bind,src=$PWD/ranks,dst=/ranks,readonly \
  --mount type=bind,src=$PWD/config.json,dst=/home/choiceoh/models/glm53-redhat-nvfp4/config.json,readonly \
  -e OMP_NUM_THREADS=2 -e RANK="$rank" -e WORLD_SIZE=4 \
  -e MASTER_ADDR=10.10.10.2 -e MASTER_PORT=29673 \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0 \
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e GLOO_SOCKET_IFNAME=enP2p1s0f0np0 \
  -e NCCL_CUMEM_ENABLE=0 -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET \
  glm53:v13-b12x-it -lc '. /repo/launchers/lib/common-tp4.sh; eval "$CT_GID_PRELUDE"; exec python3 /repo/engine/profiles/glm53/check.py --distributed --layers 0,3 --tokens 64 --chunk 32 --ranks /ranks' \
  > "evidence/fleet-reference-rank$rank.log" 2>&1
