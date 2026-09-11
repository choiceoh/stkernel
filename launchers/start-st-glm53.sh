#!/usr/bin/env bash
# Boot the ST engine's GLM-5.3 on the four Sparks: one container per node,
# inside the standalone ST image, the engine tree mounted at /repo,
# this node's rank file, /cache for the JIT builds, and the production
# launcher's NCCL/RoCE environment (start-glm53-nvfp4-tp4.sh 431-445).
# The image is built on each node from the rsynced tree (engine/runtime/build.sh) before the container starts.
#
#   bash launchers/start-st-glm53.sh            # start all four (rank 0=srv2, rank 1=srv1, then srv3/srv4)
#   bash launchers/start-st-glm53.sh stop       # docker rm -f st-glm53 on every node
#   bash launchers/start-st-glm53.sh logs [r]   # tail rank r's container log
#
# Rank order is base/comm.NODES (srv2, srv1, srv3, srv4): rank 0 hosts the rendezvous store, so it is the head.
# Never beside a serving vLLM or a q38 stack: check `docker ps` on every node
# first -- this script refuses if a glm53*/q38* container is up.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
NODES=(10.10.10.2 10.10.10.1 10.10.10.3 10.10.10.4)
IMAGE=${ST_IMAGE:-${IMAGE:-st-engine:glm53}}
PORT=${PORT:-8000}
RANKS_DIR=${RANKS_DIR:-/home/choiceoh/models/glm53-redhat-nvfp4-tp4-up-gate-v1}
CKPT=${CKPT:-/home/choiceoh/models/glm53-redhat-nvfp4}
DRAFTER=${DRAFTER:-/home/choiceoh/models/GLM-5.3-Flash-DFlash2}
ENGINE_DIR=/home/choiceoh/st-engine                     # the engine tree, rsynced to every node
CACHE_DIR=${CACHE_DIR:-/home/choiceoh/glm53-cache}
SSHOPT="-o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new"
NAME=st-glm53

node_sh() { local ip=$1; shift; ssh $SSHOPT "choiceoh@$ip" "$@"; }

case "${1:-start}" in
  stop)
    for ip in "${NODES[@]}"; do node_sh "$ip" "docker rm -f $NAME >/dev/null 2>&1 && echo '$ip: stopped' || echo '$ip: none'"; done
    node_sh "${NODES[0]}" "rm -f /home/choiceoh/st-fleet.lock"; exit 0 ;;
  logs)
    r=${2:-0}; node_sh "${NODES[$r]}" "docker logs --tail 60 $NAME"; exit 0 ;;
  start) ;;
  *) echo "usage: $0 [start|stop|logs r]" >&2; exit 2 ;;
esac

# refuse to share the fleet: a serving/other container on any node, or another runner's lock on the head
# (the lock is a file on rank 0's node; `stop` removes it; every fleet runner -- every session -- honours it)
LOCK=/home/choiceoh/st-fleet.lock
for ip in "${NODES[@]}"; do
  busy=$(node_sh "$ip" "docker ps --format '{{.Names}}' | grep -E '^(glm53|q38|vllm|st-)' || true")
  [ -z "$busy" ] || { echo "ABORT: $ip runs $busy -- the fleet is taken (hand off the queue, do not squat)" >&2; exit 1; }
done
held=$(node_sh "${NODES[0]}" "cat $LOCK 2>/dev/null || true")
[ -z "$held" ] || { echo "ABORT: the fleet is locked by '$held' ($LOCK on ${NODES[0]}); wait or 'stop' from that side" >&2; exit 1; }
node_sh "${NODES[0]}" "echo '$(whoami)@$(hostname) st-glm53 $(date '+%F %T')' > $LOCK"

NCCL_ENV="-e NCCL_P2P_LEVEL=SYS -e TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200 \
-e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0 \
-e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e GLOO_SOCKET_IFNAME=${GLOO_IFNAME:-enP2p1s0f0np0} \
-e NCCL_CROSS_NIC=1 -e NCCL_PROTO=LL,LL128,Simple -e NCCL_CUMEM_ENABLE=0 \
-e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET \
-e NCCL_NVLS_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN \
-e NCCL_MIN_NCHANNELS=16 -e NCCL_MAX_NCHANNELS=16 -e NCCL_NCHANNELS_PER_NET_PEER=4 \
-e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
-e TRITON_CACHE_DIR=/cache/triton -e TILELANG_CACHE_DIR=/cache/tilelang \
-e DG_JIT_CACHE_DIR=/cache/deep_gemm -e ST_MLA_BUILD_ROOT=/cache/mla -e FLASHINFER_WORKSPACE_BASE=/cache"
# the profile's declared D11 knobs (STK_*, boot.declared) travel from this shell into every rank; an undeclared one kills the boot
for v in $(compgen -v STK_ || true); do NCCL_ENV="$NCCL_ENV -e $v=${!v}"; done

# the checkpoint's metadata travels with the engine tree: a node needs its rank file, the drafter and these few files,
# not the full HF checkpoint
META="$REPO/build/st-glm53-meta"; mkdir -p "$META"
cp "$CKPT"/config.json "$CKPT"/tokenizer.json "$CKPT"/tokenizer_config.json "$CKPT"/generation_config.json "$META"/ 2>/dev/null
cp "$CKPT"/chat_template*.jinja "$META"/ 2>/dev/null || true
# Every node prepares and starts in PARALLEL. A node's rsync, image build and container start depend
# on no other node's, but rank 0 waits at the rendezvous for the last node to arrive -- so a sequential
# loop put its own stagger straight into rank 0's boot: measured 9.0 s of a 90.2 s boot (2026-09-11,
# srv2 :39.3 / srv1 :42.9 / srv3 :46.1 / srv4 :48.3). Each node's output is buffered and printed in rank
# order so the log stays readable, and one node's failure stops the whole fleet rather than leaving a
# partial one behind.
stage=$(mktemp -d); trap 'rm -rf "$stage"' EXIT

start_rank() {
  local r=$1 ip=${NODES[$1]}
  echo "== rank $r on $ip"
  rsync -a --delete -e "ssh $SSHOPT" --exclude __pycache__ "$REPO/engine" "$REPO/launchers" "$META" "choiceoh@$ip:$ENGINE_DIR/" \
    || { echo "ABORT: $ip could not receive the engine tree (rsync)" >&2; return 1; }
  # the ST image is built on the node from the tree just rsynced: seconds (two thin layers on the seed every node has); the seed ID is pinned in build.sh
  node_sh "$ip" "ST_IMAGE=$IMAGE bash $ENGINE_DIR/engine/runtime/build.sh >/dev/null 2>&1 || ST_IMAGE=$IMAGE bash $ENGINE_DIR/engine/runtime/build.sh 2>&1 | tail -5" \
    || { echo "ABORT: $ip could not build $IMAGE (engine/runtime/build.sh)" >&2; return 1; }
  node_sh "$ip" "test -s $RANKS_DIR/rank${r}of4.safetensors" || { echo "ABORT: $ip lacks rank${r}of4.safetensors (fanout-st-ranks.sh)" >&2; return 1; }
  node_sh "$ip" "test -s $DRAFTER/model.safetensors" || { echo "ABORT: $ip lacks the DFlash2 drafter at $DRAFTER" >&2; return 1; }
  node_sh "$ip" "docker rm -f $NAME >/dev/null 2>&1 || true; docker run -d --name $NAME --gpus all --restart no \
    --network host --ipc host --shm-size 32g --ulimit memlock=-1:-1 --ulimit nofile=524288:524288 --cap-add IPC_LOCK \
    --device /dev/infiniband:/dev/infiniband \
    -e RANK=$r -e WORLD_SIZE=4 -e MASTER_ADDR=10.10.10.2 -e MASTER_PORT=29555 -e LOCAL_RANK=0 $NCCL_ENV \
    -v $ENGINE_DIR:/repo:ro -v $RANKS_DIR:$RANKS_DIR:ro -v $DRAFTER:$DRAFTER:ro -v $CACHE_DIR:/cache \
    -v /home/choiceoh/glm53-logs:/home/choiceoh/glm53-logs \
    --entrypoint /bin/bash $IMAGE -lc 'source /repo/launchers/lib/common-tp4.sh; eval \"\$CT_GID_PRELUDE\"; cd /repo && PYTHONPATH=/repo exec python3 -u engine/profiles/glm53/boot.py --port $PORT --ranks $RANKS_DIR --ckpt-meta /repo/st-glm53-meta --drafter-dir $DRAFTER' >/dev/null && echo '$ip: started'"
}

pids=()
for r in "${!NODES[@]}"; do
  start_rank "$r" >"$stage/rank$r.log" 2>&1 &
  pids[$r]=$!
done
failed=""
for r in "${!NODES[@]}"; do
  wait "${pids[$r]}" || failed="$failed $r"
  cat "$stage/rank$r.log"
done
if [ -n "$failed" ]; then
  echo "ABORT: rank(s)$failed did not start; stopping the rest so no partial fleet is left" >&2
  bash "$0" stop >/dev/null 2>&1 || true
  exit 1
fi
echo "head: http://10.10.10.2:$PORT/v1/completions  (GET / for status)"
