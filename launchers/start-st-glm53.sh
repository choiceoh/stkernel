#!/usr/bin/env bash
# Boot the ST engine's GLM-5.3 on the four Sparks: one container per node,
# in the ST image (engine/runtime/Dockerfile: the fleet's pinned seed image
# with vLLM removed and DeepGEMM promoted to a library; the engine tree and
# its kernels are COPIED in, nothing is mounted over a framework namespace),
# this node's rank file, the drafter, /cache for the JIT builds, and the
# production launcher's NCCL/RoCE environment (start-glm53-nvfp4-tp4.sh 431-445).
#
#   bash launchers/start-st-glm53.sh            # rsync + build the image on every node, start all four (rank 0=srv2, rank 1=srv1, then srv3/srv4)
#   bash launchers/start-st-glm53.sh stop       # docker rm -f st-glm53 on every node
#   bash launchers/start-st-glm53.sh logs [r]   # tail rank r's container log
#
# Rank order is base/comm.NODES (srv2, srv1, srv3, srv4): rank 0 hosts the rendezvous store, so it is the head.
# Never beside a serving vLLM or a q38 stack: check `docker ps` on every node
# first -- this script refuses if a glm53*/q38*/vllm* container is up.
# The image is built on each node from the rsynced tree (engine/runtime/build.sh,
# seconds: two thin layers on the seed every node already has); the seed's
# image ID is pinned there, so a node with a different seed refuses.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
NODES=(10.10.10.2 10.10.10.1 10.10.10.3 10.10.10.4)
IMAGE=${IMAGE:-st-engine:glm53}
PORT=${PORT:-8000}
RANKS_DIR=${RANKS_DIR:-/home/choiceoh/models/glm53-redhat-nvfp4-tp4}
CKPT=${CKPT:-/home/choiceoh/models/glm53-redhat-nvfp4}
DRAFTER=${DRAFTER:-/home/choiceoh/models/GLM-5.3-Flash-DFlash2}
ENGINE_DIR=/home/choiceoh/st-engine                     # the engine tree, rsynced to every node (= the image's build context)
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

# the RoCE/NCCL environment of the production launcher; the JIT cache dirs come from the image (engine/runtime/Dockerfile ENV)
NCCL_ENV="-e NCCL_P2P_LEVEL=SYS -e TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200 \
-e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0 \
-e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e GLOO_SOCKET_IFNAME=${GLOO_IFNAME:-enP2p1s0f0np0} \
-e NCCL_CROSS_NIC=1 -e NCCL_PROTO=LL,LL128,Simple -e NCCL_CUMEM_ENABLE=0 \
-e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET \
-e NCCL_NVLS_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN \
-e NCCL_MIN_NCHANNELS=16 -e NCCL_MAX_NCHANNELS=16 -e NCCL_NCHANNELS_PER_NET_PEER=4 \
-e TORCH_NCCL_ASYNC_ERROR_HANDLING=1"

# the checkpoint's metadata travels with the engine tree: a node needs its rank file, the drafter and these few files,
# not the 185 GB HF checkpoint (srv1 has 29 GB free)
META="$REPO/build/st-glm53-meta"; mkdir -p "$META"
cp "$CKPT"/config.json "$CKPT"/tokenizer.json "$CKPT"/tokenizer_config.json "$CKPT"/generation_config.json "$META"/ 2>/dev/null
cp "$CKPT"/chat_template*.jinja "$META"/ 2>/dev/null || true
for r in "${!NODES[@]}"; do
  ip=${NODES[$r]}
  echo "== rank $r on $ip"
  rsync -a --delete -e "ssh $SSHOPT" --exclude __pycache__ "$REPO/engine" "$REPO/launchers" "$META" "choiceoh@$ip:$ENGINE_DIR/"
  node_sh "$ip" "test -s $RANKS_DIR/rank${r}of4.safetensors" || { echo "ABORT: $ip lacks rank${r}of4.safetensors (fanout-st-ranks.sh)" >&2; exit 1; }
  node_sh "$ip" "test -s $DRAFTER/model.safetensors" || { echo "ABORT: $ip lacks the DFlash2 drafter at $DRAFTER" >&2; exit 1; }
  node_sh "$ip" "ST_IMAGE=$IMAGE bash $ENGINE_DIR/engine/runtime/build.sh >/dev/null 2>&1 || ST_IMAGE=$IMAGE bash $ENGINE_DIR/engine/runtime/build.sh 2>&1 | tail -5" \
    || { echo "ABORT: $ip could not build $IMAGE (seed image pinned in engine/runtime/build.sh)" >&2; exit 1; }
  node_sh "$ip" "docker rm -f $NAME >/dev/null 2>&1 || true; docker run -d --name $NAME --gpus all --restart no \
    --network host --ipc host --shm-size 32g --ulimit memlock=-1:-1 --ulimit nofile=524288:524288 --cap-add IPC_LOCK \
    --device /dev/infiniband:/dev/infiniband \
    -e RANK=$r -e WORLD_SIZE=4 -e MASTER_ADDR=10.10.10.2 -e MASTER_PORT=29555 -e LOCAL_RANK=0 $NCCL_ENV \
    -v $ENGINE_DIR:/repo:ro -v $RANKS_DIR:$RANKS_DIR:ro -v $DRAFTER:$DRAFTER:ro -v $CACHE_DIR:/cache \
    -v /home/choiceoh/glm53-logs:/home/choiceoh/glm53-logs \
    --entrypoint /bin/bash $IMAGE -lc 'source /repo/launchers/lib/common-tp4.sh; eval \"\$CT_GID_PRELUDE\"; cd /opt/st && PYTHONPATH=/opt/st exec python3 -m engine.profiles.glm53.boot --port $PORT --ckpt-meta /repo/st-glm53-meta' >/dev/null && echo '$ip: started'"
done
echo "head: http://10.10.10.2:$PORT/v1/completions  (GET / for status)"
