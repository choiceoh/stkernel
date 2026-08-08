#!/bin/bash
# DeepSeek-V4-Flash-0731 TP=4 on the PROD-PROVEN hybrid-1.6 stack (aidendle94 sparkrun fork).
# Faithful port of ~/hybrid-stack compose.{head,worker}.yaml from TP2(srv2+srv3) to
# TP4(srv2 head + srv3/srv1/srv4 workers): same image, overlays, kernel caches, env,
# dspark speculative decode (SPEC_TOKENS=5). First boot recompiles kernels for TP4
# shapes — expect a LONG warmup; watchdog-disable envs carried over. RUN ON srv2.
set -euo pipefail

IMAGE="aidendle94/sparkrun-vllm-ds4-gb10:production-hybrid-1.6"
HEAD_IP=10.10.10.2
WORKERS="10.10.10.3:1 10.10.10.1:2 10.10.10.4:3"
MODEL_PATH=/home/choiceoh/models/DeepSeek-V4-Flash-0731
SERVED_NAME=deepseek-v4-flash
TP_SIZE=4
GPU_MEM="${GPU_MEM:-0.60}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-430000}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_NUM_BATCHED="${MAX_NUM_BATCHED:-4096}"
GRAPH_CAP=256
SPEC_TOKENS="${SPEC_TOKENS:-5}"
SSHOPT="-o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=no"

overlay_dir() { case "$1" in 10.10.10.3) echo /home/choiceoh/hybrid-stack/overlay;; *) echo /home/choiceoh/hybrid-stack-port/overlay;; esac; }

ENVV="-e CUDA_VISIBLE_DEVICES=0 -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUTE_DSL_ARCH=sm_121a \
-e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0 \
-e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e GLOO_SOCKET_IFNAME=enp1s0f0np0 -e TP_SOCKET_IFNAME=enp1s0f0np0 \
-e MN_IF_NAME=enp1s0f0np0 -e NCCL_CROSS_NIC=1 -e NCCL_PROTO=LL,LL128,Simple -e NCCL_CUMEM_ENABLE=0 \
-e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN -e NCCL_NVLS_ENABLE=0 -e NCCL_P2P_LEVEL=SYS \
-e HF_HUB_OFFLINE=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
-e VLLM_USE_AOT_COMPILE=1 \
-e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 -e VLLM_DISABLE_PERSISTENT_TOPK=1 \
-e VLLM_DSV4_INDEX_TOPK_FREQ=0 -e VLLM_USE_MEGA_AOT_ARTIFACT=0 -e VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
-e VLLM_USE_V2_MODEL_RUNNER=${V2RUNNER:-1} -e VLLM_USE_FLASHINFER_SAMPLER=1 -e VLLM_USE_B12X_MOE=0 \
-e VLLM_DSPARK_REPLICATE_MARKOV_W1=1 \
-e TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200 -e TORCH_NCCL_DUMP_ON_TIMEOUT=0 -e TORCH_NCCL_ASYNC_ERROR_HANDLING=0 \
-e MODEL_PATH=$MODEL_PATH -e SERVED_MODEL_NAME=$SERVED_NAME -e PORT=8000 -e TP_SIZE=$TP_SIZE \
-e GPU_MEM=$GPU_MEM -e SPEC_TOKENS=$SPEC_TOKENS -e TEMPERATURE=0.95 -e TOP_P=0.44 \
-e MAX_MODEL_LEN=$MAX_MODEL_LEN -e MAX_NUM_SEQS=$MAX_NUM_SEQS -e MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED \
-e GRAPH_CAP=$GRAPH_CAP -e ASYNC_SCHED=1 -e MASTER_ADDR=$HEAD_IP -e SP=${SP:-} -e MOE=${MOE:-b12x} -e IDXFREQ=${IDXFREQ:-} -e EPFLAG=${EPFLAG:-} -e VLLM_DSV4_INDEXER_SP=${IDXSP:-1} -e VLLM_B12X_INDEXER_STREAM=${IDXSTREAM:-} -e VLLM_B12X_KV_STREAM=${KVSTREAM:-} -e VLLM_B12X_MLA_CKV_GATHER=${CKVG:-} -e VLLM_B12X_CUDAGRAPH_PIECEWISE_PREWARM=${PREWARM:-0}"
RDMA_FLAGS="--device=/dev/infiniband:/dev/infiniband --cap-add=IPC_LOCK --ulimit memlock=-1:-1"
COMMON="--runtime nvidia --gpus all --network host --ipc host --restart unless-stopped"

mounts_for() { local ov="$1"; echo "-v /home/choiceoh/models:/home/choiceoh/models:ro \
-v /home/choiceoh/.cache/huggingface:/root/.cache/huggingface \
-v /home/choiceoh/.cache/vllm-hybrid:/cache \
-v /home/choiceoh/.cache/tilelang-hybrid:/root/.tilelang \
-v ${ov}-b12x/attention.py:/opt/venv/lib/python3.12/site-packages/vllm/models/deepseek_v4/attention.py:ro \
-v ${ov}-b12x/flashinfer_sparse.py:/opt/venv/lib/python3.12/site-packages/vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py:ro \
-v ${ov}-b12x/indexer.py:/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/mla/indexer.py:ro \
-v ${ov}-b12x/sparse_swa_dsv4.py:/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/mla/sparse_swa.py:ro"; }

echo "=== [0/5] preflight: image + model + overlays on all nodes ==="
HID=$(docker image inspect "$IMAGE" --format '{{.Id}}')
for w in $WORKERS; do
  ip=${w%%:*}
  WID=$(ssh $SSHOPT choiceoh@$ip "docker image inspect $IMAGE --format '{{.Id}}'" 2>/dev/null || true)
  [ "$WID" = "$HID" ] || { echo "ABORT: image missing/skewed on $ip"; exit 1; }
  ssh $SSHOPT choiceoh@$ip "test -f $(overlay_dir $ip)/attention.py && test -f $MODEL_PATH/config.json && mkdir -p ~/.cache/huggingface ~/.cache/vllm-hybrid ~/.cache/tilelang-hybrid" \
    || { echo "ABORT: overlay/model missing on $ip"; exit 1; }
done
echo "preflight OK (${HID:0:19})"

echo "=== [1/5] retire old vllm-dsv4 containers (free memory) ==="
docker rm -f vllm-dsv4 2>/dev/null || true
for w in $WORKERS; do ip=${w%%:*}; ssh $SSHOPT choiceoh@$ip "docker rm -f vllm-dsv4-worker 2>/dev/null; true"; done
sleep 3

echo "=== [2/5] write serve script (compose-faithful) ==="
cat > /tmp/serve-hy4.sh <<'SERVEEOF'
#!/bin/bash
set -euo pipefail
unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS 2>/dev/null || true
export VLLM_ENABLE_PCIE_ALLREDUCE=0 VLLM_PCIE_ALLREDUCE_BACKEND=cpp
# Auto-detect the RoCE-v2 IPv4 GID index (re-numbers across reboots).
for HCA in $(echo "${NCCL_IB_HCA}" | tr ',' ' '); do
  for i in $(seq 0 15); do
    t=$(cat /sys/class/infiniband/$HCA/ports/1/gid_attrs/types/$i 2>/dev/null || true)
    g=$(cat /sys/class/infiniband/$HCA/ports/1/gids/$i 2>/dev/null || true)
    case "$t" in *"RoCE v2"*) case "$g" in *0000:0000:0000:0000:0000:ffff:*) export NCCL_IB_GID_INDEX=$i; break 2;; esac;; esac
  done
done
echo "[hy4] NODE_RANK=${NODE_RANK} SPEC=dspark/${SPEC_TOKENS} GID=${NCCL_IB_GID_INDEX:-unset}"
if [ "${ASYNC_SCHED:-1}" = "1" ]; then ASYNC_ARG="--async-scheduling"; else ASYNC_ARG="--no-async-scheduling"; fi
exec vllm serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME:-deepseek-v4-flash}" \
  --host 0.0.0.0 --port "${PORT}" --trust-remote-code --hf-overrides "{\"use_index_cache\": true, \"index_topk_freq\": ${IDXFREQ:-4}}" \
  --kv-cache-dtype fp8 --block-size 256 --load-format auto \
  --tensor-parallel-size "${TP_SIZE}" \
  --gpu-memory-utilization "${GPU_MEM}" \
  --max-model-len "${MAX_MODEL_LEN}" --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --max-cudagraph-capture-size "${GRAPH_CAP}" \
  --compilation-config "{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"custom_ops\":[\"all\"],\"pass_config\":{\"fuse_gemm_comms\":true,\"fuse_allreduce_rms\":true,\"fuse_rope_kvcache_cat_mla\":true,\"fuse_attn_quant\":true${SP:+,\"enable_sp\":true,\"sp_min_token_num\":512}}}" \
  ${ASYNC_ARG} --no-scheduler-reserve-full-isl \
  --enable-chunked-prefill --enable-prefix-caching --enable-flashinfer-autotune \
  --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --default-chat-template-kwargs.temperature=${TEMPERATURE} \
  --default-chat-template-kwargs.top_p=${TOP_P} \
  --default-chat-template-kwargs.thinking=true --default-chat-template-kwargs.reasoning_effort=high \
  --attention-backend FLASHINFER_MLA_SPARSE_DSV4 --moe-backend "${MOE:-b12x}" ${EPFLAG:-} \
  --disable-custom-all-reduce \
  --nnodes 4 --node-rank "${NODE_RANK}" --master-addr "${MASTER_ADDR}" --master-port 25000 \
  ${HEADLESS:+--headless} \
  --speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":${SPEC_TOKENS},\"draft_sample_method\":\"probabilistic\"}"
SERVEEOF
chmod +x /tmp/serve-hy4.sh

echo "=== [3/5] create containers ==="
mkdir -p ~/.cache/huggingface ~/.cache/vllm-hybrid ~/.cache/tilelang-hybrid
docker rm -f hy4 2>/dev/null || true
docker run -d --name hy4 $COMMON $RDMA_FLAGS $ENVV -e VLLM_HOST_IP=$HEAD_IP \
  $(mounts_for /home/choiceoh/hybrid-stack/overlay) \
  --entrypoint /bin/bash $IMAGE -c "sleep infinity" >/dev/null
docker cp /tmp/serve-hy4.sh hy4:/tmp/serve.sh
for w in $WORKERS; do
  ip=${w%%:*}
  ov=$(overlay_dir $ip)
  ssh $SSHOPT choiceoh@$ip "docker rm -f hy4-worker 2>/dev/null; docker run -d --name hy4-worker $COMMON $RDMA_FLAGS $ENVV -e VLLM_HOST_IP=$ip \
    $(mounts_for $ov) --entrypoint /bin/bash $IMAGE -c 'sleep infinity' >/dev/null"
  scp $SSHOPT -q /tmp/serve-hy4.sh choiceoh@$ip:/tmp/serve-hy4.sh
  ssh $SSHOPT choiceoh@$ip "docker cp /tmp/serve-hy4.sh hy4-worker:/tmp/serve.sh"
  echo "  $ip container ready"
done

echo "=== [4/5] launch WORKERS first, then HEAD ==="
for w in $WORKERS; do
  ip=${w%%:*}; rank=${w##*:}
  ssh $SSHOPT choiceoh@$ip "docker exec -d -e NODE_RANK=$rank -e HEADLESS=1 hy4-worker bash -c 'bash /tmp/serve.sh > /tmp/hy4.log 2>&1'"
  echo "  worker $ip rank $rank launched"
done
sleep 3
docker exec -d -e NODE_RANK=0 hy4 bash -c 'bash /tmp/serve.sh > /tmp/hy4.log 2>&1'
echo "  head rank 0 launched"
echo "=== [5/5] done — watch: docker exec hy4 tail -f /tmp/hy4.log ; poll :8000/v1/models ==="
