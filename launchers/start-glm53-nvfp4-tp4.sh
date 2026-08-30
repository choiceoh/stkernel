#!/bin/bash
# GLM-5.3-Flash NVFP4 TP=4 launcher for the Deneb fleet.
# Ported from tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark (Lane A:
# fp8 KV + native MTP, ~55 tok/s structured on identical hardware).
# Their serve flags are kept verbatim (block-size 2304, marlin, glm47 parser
# are all load-bearing per their deploy report); the fabric env is OURS —
# the dual-HCA config proven at 86 tok/s on dsv4. Self-distributing like
# start-qwen38-nvfp4-tp4.sh: run on the head, launches workers via ssh
# WORKER-FIRST, head last.
set -euo pipefail

IMAGE="${IMAGE:-glm53:v10-dflash2}"
NAME_HEAD=glm53
NAME_WORKER=glm53-worker
MODEL_HOST_PATH=/home/choiceoh/models/glm-5.3-flash-nvfp4
INDEXER_PATCH=/home/choiceoh/patches/sparse_attn_indexer_kpool.py
MODEL_PATH=/models/glm-5.3-flash-nvfp4
CACHE_HOST_PATH=/home/choiceoh/glm53-cache
LOG_HOST_DIR=/home/choiceoh/glm53-logs
HEAD_IP=10.10.10.2
WORKER_IPS=(10.10.10.1 10.10.10.3 10.10.10.4)
MPORT=29521
PORT=8000
GMU="${GMU:-0.85}"
MOE_BACKEND="${MOE_BACKEND:-marlin}"
EAGER="${EAGER:-1}"
GRAPH_CAP="${GRAPH_CAP:-256}"
MAX_BATCHED="${MAX_BATCHED:-2048}"
MAX_SEQS="${MAX_SEQS:-6}"
MM_LIMIT="${MM_LIMIT:-{\"image\":0,\"video\":0}}"
COMPILE_CFG="${COMPILE_CFG:-{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"custom_ops\":[\"all\"],\"pass_config\":{\"fuse_gemm_comms\":true,\"fuse_allreduce_rms\":true,\"fuse_attn_quant\":true}}}"
# DFLASH2=1: block-diffusion drafter (2.15x over MTP-4 at TP2, acceptance 74%).
# num_speculative_tokens MUST be 7 (drafter block 8 minus the verified token).
DFLASH2="${DFLASH2:-1}"
DRAFT_HOST_PATH=/home/choiceoh/models/GLM-5.3-Flash-DFlash2
KV_DTYPE="${KV_DTYPE:-fp8_e4m3}"   # auto = bf16, for isolating KV quantization
KV_BYTES="${KV_BYTES:-auto}"          # auto = let vLLM profile per node
MAX_LEN="${MAX_LEN:-1048576}"
SPEC_K="${SPEC_K:-4}"
SSHOPT="-o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=8"

test -f "$MODEL_HOST_PATH/config.json"
test -f "$MODEL_HOST_PATH/chat_template_mm.jinja" || {
  echo "ABORT: chat_template_mm.jinja missing in model dir (copy from ~/glm53-4x/)"; exit 1; }

# Refuse to start over a live dsv4/q38 stack.
for ip in $HEAD_IP "${WORKER_IPS[@]}"; do
  run() { if [ "$ip" = "$HEAD_IP" ]; then bash -c "$1"; else ssh $SSHOPT choiceoh@"$ip" "$1"; fi; }
  if run 'docker ps --format "{{.Names}}"|grep -qE "^(hy4|q38)"'; then
    echo "ABORT: $ip runs hy4/q38 — stop production/experiment first"; exit 1; fi
  run "test -f $MODEL_HOST_PATH/config.json" || { echo "ABORT: $ip missing weights"; exit 1; }
  run "test -f $INDEXER_PATCH" || { echo "ABORT: $ip missing SM121 indexer patch"; exit 1; }
  run "docker image inspect $IMAGE >/dev/null 2>&1" || { echo "ABORT: $ip missing image"; exit 1; }
done

# Our proven fabric env (dual HCA), their runtime env.
ENVV="-e HF_HOME=/cache/huggingface -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
-e VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=/cache/flashinfer_autotune -e VLLM_CACHE_ROOT=/cache/vllm \
-e FLASHINFER_WORKSPACE_BASE=/cache -e TRITON_CACHE_DIR=/cache/triton \
-e CUTE_DSL_ARCH=sm_121a -e VLLM_B12X_CUDAGRAPH_PIECEWISE_PREWARM=0 \
-e VLLM_USE_AOT_COMPILE=1 -e VLLM_USE_MEGA_AOT_ARTIFACT=0 -e VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
-e VLLM_USE_FLASHINFER_SAMPLER=1 -e NCCL_P2P_LEVEL=SYS \
-e TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200 \
-e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
-e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
-e FLASHINFER_DISABLE_VERSION_CHECK=1 \
-e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0 \
-e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e GLOO_SOCKET_IFNAME=enp1s0f0np0 \
-e TP_SOCKET_IFNAME=enp1s0f0np0 -e MN_IF_NAME=enp1s0f0np0 \
-e NCCL_CROSS_NIC=1 -e NCCL_PROTO=LL,LL128,Simple -e NCCL_CUMEM_ENABLE=0 \
-e NCCL_IB_GID_INDEX=3 -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET \
-e NCCL_NVLS_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN \
-e TORCH_NCCL_ASYNC_ERROR_HANDLING=1"

COMMON="--gpus all -d --restart no --network host --ipc host --shm-size 32g \
--memory 112g --memory-swap 112g --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
--device /dev/infiniband:/dev/infiniband \
-v $MODEL_HOST_PATH:$MODEL_PATH:ro -v CACHEDIR:/cache -v $LOG_HOST_DIR:/glmlogs \
-v $INDEXER_PATCH:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer_kpool.py:ro"

# SPEC=0 serves straight from the target model -- no drafter, no accept path.
# Only for isolating a defect: it costs the whole speculative speedup.
if [ "${SPEC:-1}" = 0 ]; then
  SPECCFG_VAL=""
elif [ "$DFLASH2" = 1 ]; then
  test -f "$DRAFT_HOST_PATH/config.json" || { echo "ABORT: DFlash2 drafter missing at $DRAFT_HOST_PATH"; exit 1; }
  COMMON="$COMMON -v $DRAFT_HOST_PATH:/models/dflash2-draft:ro"
  SPECCFG_VAL="--speculative-config '{\"method\":\"dflash\",\"model\":\"/models/dflash2-draft\",\"num_speculative_tokens\":7}'"
else
  SPECCFG_VAL="--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":$SPEC_K}'"
fi

if [ "$EAGER" = 1 ]; then
  EAGER_FLAG="--enforce-eager"
else
  EAGER_FLAG="--max-cudagraph-capture-size $GRAPH_CAP --compilation-config '$COMPILE_CFG'"
fi
KV_FLAG=""; [ "$KV_BYTES" != auto ] && KV_FLAG="--kv-cache-memory $KV_BYTES"
# ASYNC_SCHED=0 removes the async scheduler while leaving the drafter in
# place. vLLM registers bools via BooleanOptionalAction, hence --no-.
ASYNC_FLAG=""; [ "${ASYNC_SCHED:-1}" = 0 ] && ASYNC_FLAG="--no-async-scheduling"
SERVE_ARGS="$MODEL_PATH \
--served-model-name glm-5.3-flash \
--host 0.0.0.0 --port $PORT \
--trust-remote-code \
--tensor-parallel-size 4 \
--gpu-memory-utilization $GMU \
--max-model-len $MAX_LEN \
--max-num-seqs $MAX_SEQS --max-num-batched-tokens $MAX_BATCHED --block-size 2304 --moe-backend $MOE_BACKEND \
$SPECCFG_VAL \
--kv-cache-dtype $KV_DTYPE $KV_FLAG \
$ASYNC_FLAG \
$EAGER_FLAG --enable-flashinfer-autotune \
--limit-mm-per-prompt '$MM_LIMIT' \
--tool-call-parser glm47 --enable-auto-tool-choice \
--reasoning-parser glm45 --chat-template $MODEL_PATH/chat_template_mm.jinja \
--distributed-executor-backend mp \
--nnodes 4 --master-addr $HEAD_IP --master-port $MPORT"

mkdir -p "$CACHE_HOST_PATH" "$LOG_HOST_DIR"

# Reclaim page cache on every node before the engine measures free memory.
# NVRM allocates against MemFree, so cached file pages are memory the engine
# cannot use, and pulling this image alone accounts for ~11 GiB of it.
# SKIP_PREFLIGHT=1 opts out.
PREFLIGHT=/home/choiceoh/stkernel/launchers/memfree-preflight.sh
if [ "${SKIP_PREFLIGHT:-0}" != 1 ] && [ -x "$PREFLIGHT" ]; then
  echo "== memfree preflight =="
  # Report goes to stderr (straight to the terminal); the computed GMU is the
  # only thing on stdout, so $(...) captures it alone.
  if GMU_SAFE=$("$PREFLIGHT" 3); then
    if awk "BEGIN{exit !($GMU > $GMU_SAFE)}" 2>/dev/null; then
      echo "  ! GMU=$GMU exceeds what free memory supports ($GMU_SAFE) -- boot may OOM"
    fi
  else
    echo "  preflight refused (a node was unreachable); continuing with GMU=$GMU"
  fi
fi


# Workers first (rank 1..3), head last — their documented order.
rank=1
for ip in "${WORKER_IPS[@]}"; do
  W_B64=$(printf '%s' "vllm serve $SERVE_ARGS --node-rank $rank --headless > /glmlogs/glm53.log 2>&1" | base64 -w0)
  ssh $SSHOPT choiceoh@"$ip" "mkdir -p $CACHE_HOST_PATH $LOG_HOST_DIR; docker rm -f $NAME_WORKER 2>/dev/null; \
    docker run --name $NAME_WORKER ${COMMON/CACHEDIR/$CACHE_HOST_PATH} $ENVV -e VLLM_HOST_IP=$ip \
    --entrypoint /bin/bash $IMAGE -c 'echo $W_B64 | base64 -d > /tmp/serve.sh; bash /tmp/serve.sh'"
  echo "worker rank=$rank @$ip launched"
  rank=$((rank+1))
done
sleep 8
docker rm -f $NAME_HEAD 2>/dev/null || true
SERVE_B64=$(printf '%s' "vllm serve $SERVE_ARGS --node-rank 0 > /glmlogs/glm53.log 2>&1" | base64 -w0)
docker run --name $NAME_HEAD ${COMMON/CACHEDIR/$CACHE_HOST_PATH} $ENVV -e VLLM_HOST_IP=$HEAD_IP \
  --entrypoint /bin/bash $IMAGE -c "echo $SERVE_B64 | base64 -d > /tmp/serve.sh; bash /tmp/serve.sh"
echo "head rank=0 launched — poll :$PORT/v1/models (cold boot ~15min)"
