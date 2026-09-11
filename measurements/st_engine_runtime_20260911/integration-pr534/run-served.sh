#!/usr/bin/env bash
set -euo pipefail
rank=${1:?rank required}
cd /home/choiceoh/st-engine-f4d7-20260911
mkdir -p cache evidence/integration-pr534
source launchers/lib/common-tp4.sh
export NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0
eval "$CT_GID_PRELUDE"
export PROFILE=glm53 CACHE_HOST=$PWD/cache MAX_JOBS=2 PROBE_CACHE=1
export VLLM_GLM53_MEGAKERNEL=1 VLLM_GLM53_MK_MLA=1
export MK_PROBE_DOCKER_ARGS="--name st-engine-f4d7-fleet-served --network host --device /dev/infiniband --ulimit memlock=-1 --shm-size=256m --memory=12g --memory-swap=12g --cpus=4 -e OMP_NUM_THREADS=2 --mount type=bind,src=$PWD/ranks-v534,dst=/ranks,readonly --mount type=bind,src=$PWD/config.json,dst=/home/choiceoh/models/glm53-redhat-nvfp4/config.json,readonly -e RANK=$rank -e WORLD_SIZE=4 -e MASTER_ADDR=10.10.10.2 -e MASTER_PORT=29673 -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=$NCCL_IB_HCA -e NCCL_IB_GID_INDEX=$NCCL_IB_GID_INDEX -e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e GLOO_SOCKET_IFNAME=enP2p1s0f0np0 -e NCCL_CUMEM_ENABLE=0 -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET"
trap 'docker rm -f st-engine-f4d7-fleet-served >/dev/null 2>&1 || true' EXIT
timeout --kill-after=10s 360 bash probes/run_mk_probe.sh engine/profiles/glm53/check.py --distributed --lanes served --layers 0,3 --tokens 64 --chunk 32 --ranks /ranks > "evidence/integration-pr534/fleet-served-rank$rank.log" 2>&1
