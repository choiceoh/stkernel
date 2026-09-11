#!/usr/bin/env bash
# Boot the ST engine's GLM-5.3 on the four Sparks: one container per node,
# inside the glm53 judge image (the served kernels live there), the engine
# tree mounted at /repo, this node's rank file, the composed overlay sources
# on their target paths, /cache for the JIT builds, and the production
# launcher's NCCL/RoCE environment (start-glm53-nvfp4-tp4.sh 431-445).
#
#   bash launchers/start-st-glm53.sh            # start all four (rank r on 10.10.10.(r+1))
#   bash launchers/start-st-glm53.sh stop       # docker rm -f st-glm53 on every node
#   bash launchers/start-st-glm53.sh logs [r]   # tail rank r's container log
#
# Rank order is base/comm.NODES (srv1..srv4); the head (MASTER_ADDR) is srv2.
# Never beside a serving vLLM or a q38 stack: check `docker ps` on every node
# first -- this script refuses if a glm53*/q38* container is up.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
NODES=(10.10.10.1 10.10.10.2 10.10.10.3 10.10.10.4)
IMAGE=${IMAGE:-glm53:v13-b12x-it}
PORT=${PORT:-8000}
RANKS_DIR=${RANKS_DIR:-/home/choiceoh/models/glm53-redhat-nvfp4-tp4}
CKPT=${CKPT:-/home/choiceoh/models/glm53-redhat-nvfp4}
DRAFTER=${DRAFTER:-/home/choiceoh/models/GLM-5.3-Flash-DFlash2}
ENGINE_DIR=/home/choiceoh/st-engine                     # the engine tree, rsynced to every node
OVERLAY_DIR=${OVERLAY_DIR:-/home/choiceoh/overlays/glm53}   # deploy-overlays.sh glm53 puts the composed files here
CACHE_DIR=${CACHE_DIR:-/home/choiceoh/glm53-cache}
SSHOPT="-o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new"
NAME=st-glm53

node_sh() { local ip=$1; shift; ssh $SSHOPT "choiceoh@$ip" "$@"; }

case "${1:-start}" in
  stop)
    for ip in "${NODES[@]}"; do node_sh "$ip" "docker rm -f $NAME >/dev/null 2>&1 && echo '$ip: stopped' || echo '$ip: none'"; done; exit 0 ;;
  logs)
    r=${2:-0}; node_sh "${NODES[$r]}" "docker logs --tail 60 $NAME"; exit 0 ;;
  start) ;;
  *) echo "usage: $0 [start|stop|logs r]" >&2; exit 2 ;;
esac

# refuse to share the fleet
for ip in "${NODES[@]}"; do
  busy=$(node_sh "$ip" "docker ps --format '{{.Names}}' | grep -E '^(glm53|q38|vllm)' || true")
  [ -z "$busy" ] || { echo "ABORT: $ip runs $busy -- the fleet is taken (hand off the queue, do not squat)" >&2; exit 1; }
done

# the overlay manifest (composed here) says which files go where
bash "$REPO/launchers/compose-overlays.sh" glm53 >&2
MANIFEST="$REPO/build/glm53/manifest.tsv"
# the served kernels the lanes bind: KDA (fla fork), megakernel MLA, tilelang mHC, the kpool indexer, and the whole
# b12x MoE family on the flashinfer side (the lane calls flashinfer.fused_moe.b12x_fused_moe directly)
sources=(glm53_megakernel.py glm53_megakernel.cu kda.py chunk_delta_h.py tilelang.py tilelang_kernels.py
         sparse_attn_indexer_kpool.py glm53_kpool_indexer.py
         moe_micro_kernel.py moe_dispatch.py b12x_moe.py moe_static_common.py moe_sf_pack.py moe_static_kernel_v4.py moe_static_kernel_v5.py moe_dynamic_gated_tiled.py moe_dynamic_gated_sf6.py moe_dynamic_gated_sf6_q0.py moe_dynamic_prefill.py moe_dynamic_prefill_n128.py moe_dynamic_ep_local.py glm53_ep_route_remap.py moe_reform_sf_pack.py glm53_ep_local_selftest.py glm53_tp_sf6_q0_selftest.py glm53_ep_tiled.py moe_static_ep_tiled.py glm53_ep_tiled_selftest.py)
mounts=""
for src in "${sources[@]}"; do
  target=$(awk -F '\t' -v s="$src" '$1 == s {print $2}' "$MANIFEST")
  [ -n "$target" ] || { echo "ABORT: $src missing from $MANIFEST" >&2; exit 1; }
  mounts="$mounts -v $OVERLAY_DIR/$src:$target:ro"
done

NCCL_ENV="-e NCCL_P2P_LEVEL=SYS -e TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200 \
-e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0 \
-e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e GLOO_SOCKET_IFNAME=${GLOO_IFNAME:-enP2p1s0f0np0} \
-e NCCL_CROSS_NIC=1 -e NCCL_PROTO=LL,LL128,Simple -e NCCL_CUMEM_ENABLE=0 \
-e NCCL_IB_GID_INDEX=3 -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET \
-e NCCL_NVLS_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN \
-e NCCL_MIN_NCHANNELS=16 -e NCCL_MAX_NCHANNELS=16 -e NCCL_NCHANNELS_PER_NET_PEER=4 \
-e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 -e VLLM_GLM53_MK_MLA=1 \
-e TRITON_CACHE_DIR=/cache/triton -e VLLM_CACHE_ROOT=/cache/vllm -e FLASHINFER_WORKSPACE_BASE=/cache"

for r in "${!NODES[@]}"; do
  ip=${NODES[$r]}
  echo "== rank $r on $ip"
  rsync -a --delete -e "ssh $SSHOPT" --exclude __pycache__ "$REPO/engine" "$REPO/overlay" "choiceoh@$ip:$ENGINE_DIR/"
  node_sh "$ip" "test -s $RANKS_DIR/rank${r}of4.safetensors" || { echo "ABORT: $ip lacks rank${r}of4.safetensors (fanout-st-ranks.sh)" >&2; exit 1; }
  node_sh "$ip" "test -s $DRAFTER/model.safetensors && test -f $CKPT/tokenizer.json" || { echo "ABORT: $ip lacks the drafter or the tokenizer" >&2; exit 1; }
  node_sh "$ip" "docker rm -f $NAME >/dev/null 2>&1 || true; docker run -d --name $NAME --gpus all --restart no \
    --network host --ipc host --shm-size 32g --ulimit memlock=-1:-1 --ulimit nofile=524288:524288 --cap-add IPC_LOCK \
    --device /dev/infiniband:/dev/infiniband \
    -e RANK=$r -e WORLD_SIZE=4 -e MASTER_ADDR=10.10.10.2 -e MASTER_PORT=29555 -e LOCAL_RANK=0 $NCCL_ENV \
    -v $ENGINE_DIR:/repo:ro -v $RANKS_DIR:$RANKS_DIR:ro -v $CKPT:$CKPT:ro -v $DRAFTER:$DRAFTER:ro -v $CACHE_DIR:/cache \
    -v /home/choiceoh/glm53-logs:/home/choiceoh/glm53-logs $mounts \
    --entrypoint /bin/bash $IMAGE -lc 'cd /repo && PYTHONPATH=/repo exec python3 engine/profiles/glm53/boot.py --port $PORT' >/dev/null && echo '$ip: started'"
done
echo "head: http://10.10.10.2:$PORT/v1/completions  (GET / for status)"
