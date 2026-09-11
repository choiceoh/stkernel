#!/bin/bash
# Qwen3.8-Flash-Next NVFP4 on 4x DGX Spark GB10 -- TP=4 with EXPERT PARALLEL.
#
# "TEP=4" is TP=4 with EP on, and the EP is not an option here.
#
#   moe_intermediate_size 640, TP-split 4 ways -> 160 per rank
#     -> 320 gate+up rows, 320 % 128 = 64. The FLASHINFER_CUTLASS / b12x NvFp4
#        backends cannot tile that without padding w1/w3, and padding this
#        branch is RECORDED AS DESTROYING THE MODEL: 'Padding intermediate size
#        from 160 to 192' boots and HEALTH-OKs, and then '안녕하세요' answers
#        '1' and '360과 168의 최대공약수' loops forever (MEASUREMENTS.md).
#   EP hands each rank 128 whole experts of 512 instead, so the intermediate
#     stays 640 -> 1280 gate+up rows, 1280 % 128 = 0. No padding, nothing to
#     be wrong about.
#
# The shared expert used to be the hole in that: it is TP-split the same way,
# to the same misaligned 160. qwen38_moe fuses it into the routed grouped GEMM
# as an eleventh, rank-local slot, so it inherits the 640 as well. See that
# module's README; the algebra is bit-exact against the two-computation oracle.
#
# Derived from the bring-up launcher that lived outside the repo on srv2. Its
# findings are kept as comments here rather than rediscovered.
set -euo pipefail

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_repo="$(cd "$_here/.." && pwd)"
# shellcheck source=lib/common-tp4.sh
. "$_here/lib/common-tp4.sh"

ct_load_profile "$_repo/profiles/qwen38.env" \
  IMAGE MODEL_HOST_PATH SERVED_NAME MOE_BACKEND EXPERT_PARALLEL MAX_MODEL_LEN \
  MAX_NUM_SEQS MAX_NUM_BATCHED KV_DTYPE GPU_MEM PLE_CPU_OFFLOAD \
  PLE_SSD PLE_SSD_DIR \
  FORCE_FP8_EMBED CUDAGRAPH_MODE SPEC_TOKENS ADAPTIVE_SPEC NGRAM_FIX \
  QSA_MAX_SPLITS ALL2ALL ASYNC AUTOTUNE LOAD_FORMAT FUSE SHARED_FUSE

IMAGE="${IMAGE:-${PROFILE_IMAGE:-vllm/vllm-openai:qwen38-flash-next}}"
MODEL_PATH="${MODEL_HOST_PATH:-${PROFILE_MODEL_PATH:-/home/choiceoh/models/qwen38-flash-next-nvfp4}}"
SERVED_NAME="${SERVED_NAME:-${PROFILE_SERVED_NAME:-Qwen3.8-Flash-Next}}"

HEAD_IP=10.10.10.2
TP_SIZE="${TP_SIZE:-4}"
case "$TP_SIZE" in
  4) WORKERS="10.10.10.3:1 10.10.10.1:2 10.10.10.4:3" ;;
  2) WORKERS="10.10.10.3:1" ;;
  1) WORKERS="" ;;
  *) echo "ABORT: TP_SIZE must be 1, 2 or 4 (got $TP_SIZE)" >&2; exit 2 ;;
esac
# This launcher only does the right thing on the head node. The model and image
# checks and every docker command run LOCALLY, so from any other node it quietly
# builds a different cluster: the head lands on the wrong machine carrying
# VLLM_HOST_IP=$HEAD_IP, the workers rendezvous at an address nobody serves, and
# the first visible failure is an unrelated-looking ssh error on whichever
# worker this node happens to lack a key for.
if [ "${DRY_RUN:-0}" != 1 ] && ! ip -4 -o addr show 2>/dev/null | grep -qw "$HEAD_IP"; then
  _mine=$(ip -4 -o addr show scope global 2>/dev/null | sed 's|.* inet \([0-9.]*\)/.*|\1|' | paste -sd" ")
  echo "ABORT: 이 런처는 head 노드($HEAD_IP)에서 실행해야 합니다 — 여기는 $(hostname) [${_mine}]" >&2
  echo "       ssh <head> 'bash /home/choiceoh/stkernel/launchers/start-qwen38-nvfp4-tep4.sh'" >&2
  exit 1
fi

PORT="${PORT:-8000}"
MASTER_PORT="${MASTER_PORT:-29501}"
SSHOPT="-o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=no"

# --- the EP decision, enforced rather than defaulted --------------------
EXPERT_PARALLEL="${EXPERT_PARALLEL:-1}"
if [ "$EXPERT_PARALLEL" != 1 ] && [ "$TP_SIZE" != 1 ]; then
  cat >&2 <<'WARN'
ABORT: EXPERT_PARALLEL=0 at TP>1.
  moe_intermediate_size 640 split over the ranks is not 128-aligned, and the
  only way the FP4 MoE backends accept it is padding -- which boots and then
  destroys the model (MEASUREMENTS.md). Set EXPERT_PARALLEL=1, or TP_SIZE=1.
WARN
  exit 2
fi
EP_FLAG=""; [ "$EXPERT_PARALLEL" = 1 ] && EP_FLAG="--enable-expert-parallel"

# --- MoE backend --------------------------------------------------------
# b12x is the SM12x CuteDSL FP4 lane. vLLM's NvFp4 oracle deliberately
# EXCLUDES it from auto-selection, so it only runs when named. Marlin -- the
# default when this is empty -- is weight-only FP4: it dequantizes to bf16 and
# leaves the chip's FP4 tensor cores idle.
MOE_BACKEND="${MOE_BACKEND:-flashinfer_b12x}"
MOE_FLAG=""; [ -n "$MOE_BACKEND" ] && MOE_FLAG="--moe-backend $MOE_BACKEND"

# --- memory window ------------------------------------------------------
# auto: probe every node and take the floor. Six boots died in one day to a
# hand-picked value; failing in 2 seconds beats failing in 12 minutes.
GPU_MEM="${GPU_MEM:-auto}"
if [ "$GPU_MEM" = auto ]; then
  if [ -x /home/choiceoh/tp4-mem.sh ] || [ -f /home/choiceoh/tp4-mem.sh ]; then
    GPU_MEM=$(bash /home/choiceoh/tp4-mem.sh gmu) || {
      echo "ABORT: no memory window -- run: bash ~/tp4-mem.sh plan" >&2; exit 4; }
    echo "tp4-mem: GPU_MEM=$GPU_MEM (auto)"
  else
    GPU_MEM=0.80
    echo "tp4-mem.sh absent; GPU_MEM=$GPU_MEM (fixed default)"
  fi
fi

MAX_MODEL_LEN="${MAX_MODEL_LEN:-196608}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
MAX_NUM_BATCHED="${MAX_NUM_BATCHED:-16384}"
KV_DTYPE="${KV_DTYPE:-bfloat16}"
# 1 = keep the 51 GiB PLE n-gram table in host RAM instead of device memory.
PLE_CPU_OFFLOAD="${PLE_CPU_OFFLOAD:-1}"
# Build the FP8 PLE embedding method on the ON-DEVICE path for an NVFP4
# checkpoint, which upstream only does for Fp8Config. REQUIRED for TP>1: the
# offload path -- the only other reader of the PLE scale -- is gated to
# nnodes=1. See overlay/modules/qwen38_ple/.
FORCE_FP8_EMBED="${FORCE_FP8_EMBED:-1}"
# The image defaults to FULL_AND_PIECEWISE, which on this hybrid (36 linear +
# 12 full attention) model breaks during capture on a dynamic shape:
# "Constraints violated (L[query_start_loc].size()[0])". PIECEWISE is pinned
# for the same reason the dsv4 stack pins it.
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-PIECEWISE}"
EAGER="${EAGER:-0}"
FUSE="${FUSE:-0}"
if [ "$EAGER" = 1 ]; then
  COMPILE_FLAG="--enforce-eager"
elif [ "$FUSE" = 1 ]; then
  COMPILE_FLAG="--compilation-config '{\"cudagraph_mode\":\"$CUDAGRAPH_MODE\",\"pass_config\":{\"fuse_gemm_comms\":true,\"fuse_allreduce_rms\":true,\"enable_sp\":true}}'"
else
  COMPILE_FLAG="--compilation-config '{\"cudagraph_mode\":\"$CUDAGRAPH_MODE\"}'"
fi

# Speculative decoding via the checkpoint's own MTP head (mtp_num_hidden_layers
# 1, hybrid full_attention). 0 disables.
SPEC_TOKENS="${SPEC_TOKENS:-0}"
SPEC_FLAG=""
if [ "$SPEC_TOKENS" != 0 ]; then
  SPEC_JSON="{\"method\":\"mtp\",\"num_speculative_tokens\":$SPEC_TOKENS"
  [ -n "${DRAFT_TP:-}" ] && SPEC_JSON="$SPEC_JSON,\"draft_tensor_parallel_size\":$DRAFT_TP"
  SPEC_FLAG="--speculative-config '$SPEC_JSON}'"
fi
ASYNC="${ASYNC:-0}";  ASYNC_FLAG=""; [ "$ASYNC" = 1 ] && ASYNC_FLAG="--async-scheduling"
AUTOTUNE="${AUTOTUNE:-1}"; AUTOTUNE_FLAG=""; [ "$AUTOTUNE" = 0 ] && AUTOTUNE_FLAG="--no-enable-flashinfer-autotune"
ALL2ALL="${ALL2ALL:-}"; A2A_FLAG=""; [ -n "$ALL2ALL" ] && A2A_FLAG="--all2all-backend $ALL2ALL"
LOAD_FORMAT="${LOAD_FORMAT:-}"; LOAD_FLAG=""; [ -n "$LOAD_FORMAT" ] && LOAD_FLAG="--load-format $LOAD_FORMAT"
# KV_CACHE_MEMORY (bytes): size the KV cache to a fixed number instead of
# "whatever the utilization fraction leaves after the profile peak". On GB10
# unified memory the profile peak at MAX_NUM_BATCHED=16384 measured ~26 GiB on
# top of ~32 GiB of weights+PLE per rank (GPU_MEM=0.50 left < 2.44 GiB for KV),
# and every allocation beyond the budget is a driver NV_ERR_NO_MEMORY the
# kernel journal records. tp4-mem prints the matching number.
KV_FLAG=""; [ -n "${KV_CACHE_MEMORY:-}" ] && KV_FLAG="--kv-cache-memory-bytes $KV_CACHE_MEMORY"
# fastsafetensors on this fleet means the overlay's LOCAL mode (see the profile):
# without DENEB_FST_LOCAL=1 the loader partitions reads across ranks and
# redistributes over its own NCCL, which dies in ncclSystemError on 4-node RoCE.
_FST=""; [ "$LOAD_FORMAT" = fastsafetensors ] && _FST="-e DENEB_FST_LOCAL=${FST_LOCAL:-1}"
# Knobs the overlays read. With these at 0 the overlays' added branches are
# dead code identical to upstream, so mounting them unconditionally is safe.
ADAPTIVE_SPEC="${ADAPTIVE_SPEC:-0}"
NGRAM_FIX="${NGRAM_FIX:-0}"
QSA_MAX_SPLITS="${QSA_MAX_SPLITS:-0}"
SHARED_FUSE="${SHARED_FUSE:-0}"

# --- overlays, from the composed build ----------------------------------
OVDIR="$_repo/build/qwen38"
OVMOUNTS=""
if [ -f "$OVDIR/manifest.tsv" ]; then
  while IFS=$'\t' read -r src tgt _pre; do
    case "$src" in ''|\#*) continue ;; esac
    [ -f "$OVDIR/$src" ] || { echo "ABORT: $OVDIR/$src missing -- run compose-overlays.sh qwen38" >&2; exit 3; }
    OVMOUNTS="$OVMOUNTS -v $OVDIR/$src:$tgt:ro"
  done < "$OVDIR/manifest.tsv"
else
  echo "NOTE: $OVDIR/manifest.tsv absent -- running on stock image code"
fi

LOGDIR=/home/choiceoh/q38-logs
CACHE_DIR="${CACHE_DIR:-/home/choiceoh/q38-cache}"
mkdir -p "$LOGDIR" "$CACHE_DIR"

# Same fabric wiring as the proven dsv4 stack: RoCE via NCCL_NET=IB on the
# CX-7 HCAs, cuMem off, NVLS off (no NVLink between Sparks).
# LAUNCH_BLOCKING=1: every kernel launch synchronous, so an asynchronous fault
# is reported at the kernel that raised it rather than at the next launch that
# happened to notice (the first TEP=4 request's IMA surfaced in the QSA Triton
# loader). Diagnostic only -- it serializes the whole step.
_LB=""; [ "${LAUNCH_BLOCKING:-0}" = 1 ] && _LB="-e CUDA_LAUNCH_BLOCKING=1"
ENVV="-e CUDA_VISIBLE_DEVICES=0 -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUTE_DSL_ARCH=sm_121a $_LB $_FST \
-e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0 \
-e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e GLOO_SOCKET_IFNAME=enp1s0f0np0 -e TP_SOCKET_IFNAME=enp1s0f0np0 \
-e MN_IF_NAME=enp1s0f0np0 -e NCCL_CROSS_NIC=1 -e NCCL_PROTO=LL,LL128,Simple -e NCCL_CUMEM_ENABLE=0 \
-e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN -e NCCL_NVLS_ENABLE=0 -e NCCL_P2P_LEVEL=SYS \
-e HF_HUB_OFFLINE=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
-e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e VLLM_PLE_CPU_OFFLOAD=$PLE_CPU_OFFLOAD \
-e DENEB_PLE_FORCE_FP8_EMBED=$FORCE_FP8_EMBED -e DENEB_ADAPTIVE_SPEC=$ADAPTIVE_SPEC \
-e DENEB_PLE_SSD=${PLE_SSD:-0} -e DENEB_PLE_SSD_DIR=${PLE_SSD_DIR:-} \
-e DENEB_QSA_MAX_SPLITS=$QSA_MAX_SPLITS -e DENEB_NGRAM_FIX=$NGRAM_FIX \
-e DENEB_Q38_SHARED_FUSE=$SHARED_FUSE"
COMMON="--runtime nvidia --gpus all --ipc host --network host --cap-add IPC_LOCK --ulimit memlock=-1:-1 --shm-size 32g"
# EXTRA_DOCKER_ARGS: appended verbatim to every container's docker run (head and
# workers) -- diagnostic mounts and envs such as a PYTHONPATH hook that dumps
# kernel arguments. Empty in production; whatever it names must exist on every node.
COMMON="$COMMON ${EXTRA_DOCKER_ARGS:-}"
RDMA_FLAGS="--device /dev/infiniband"
MOUNTS="-v $MODEL_PATH:$MODEL_PATH:ro -v $CACHE_DIR:/root/.cache/vllm -v $LOGDIR:/q38logs $OVMOUNTS"
# PLE on SSD: each rank reads its own block (tools/qwen38_ple_shard.py) from
# PLE_SSD_DIR; the dir is mounted read-only and every node must hold its rank's file.
if [ "${PLE_SSD:-0}" = 1 ]; then
  [ -n "${PLE_SSD_DIR:-}" ] || { echo "ABORT: PLE_SSD=1 needs PLE_SSD_DIR" >&2; exit 2; }
  MOUNTS="$MOUNTS -v $PLE_SSD_DIR:$PLE_SSD_DIR:ro"
fi

echo "=== [0/5] preflight ==="
echo "  image      $IMAGE"
echo "  model      $MODEL_PATH"
echo "  TP=$TP_SIZE  EP=$EXPERT_PARALLEL  moe=${MOE_BACKEND:-marlin(default)}  gmu=$GPU_MEM"
echo "  PLE offload=$PLE_CPU_OFFLOAD  force_fp8_embed=$FORCE_FP8_EMBED  ple_ssd=${PLE_SSD:-0}  shared_fuse=$SHARED_FUSE"
echo "  overlays   $(printf '%s' "$OVMOUNTS" | grep -o ' -v ' | wc -l) file(s)"

_wips=""; for _w in $WORKERS; do _wips="$_wips ${_w%%:*}"; done
ct_refuse_foreign_stacks '^(glm53|hy4)(-|$)' Q38 "$SSHOPT" "$HEAD_IP" $_wips

for ip in $HEAD_IP $_wips; do
  if [ "$ip" = "$HEAD_IP" ]; then run() { bash -c "$1"; }; else run() { ssh $SSHOPT choiceoh@"$ip" "$1"; }; fi
  run "[ -f $MODEL_PATH/config.json ]" || { echo "ABORT: $ip missing $MODEL_PATH/config.json" >&2; exit 1; }
  run "docker image inspect $IMAGE >/dev/null 2>&1" || { echo "ABORT: $ip missing image $IMAGE" >&2; exit 1; }
done
echo "  all nodes have the model and the image"
if [ "${PLE_SSD:-0}" = 1 ]; then
  # (set -e trap: a `[ .. ] && echo` loop ends false on a non-matching last
  # worker, which under -e killed the script silently inside $(...).)
  _rank_of() {
    if [ "$1" = "$HEAD_IP" ]; then echo 0; return 0; fi
    for _w in $WORKERS; do
      if [ "${_w%%:*}" = "$1" ]; then echo "${_w##*:}"; return 0; fi
    done
    echo "ABORT: $1 is neither head nor a listed worker" >&2; return 1
  }
  for ip in $HEAD_IP $_wips; do
    if [ "$ip" = "$HEAD_IP" ]; then run() { bash -c "$1"; }; else run() { ssh $SSHOPT choiceoh@"$ip" "$1"; }; fi
    _f="$PLE_SSD_DIR/ple-r$(_rank_of "$ip")of$TP_SIZE.weight"
    run "[ -f $_f ]" || { echo "ABORT: $ip missing its PLE block $_f -- tools/qwen38_ple_shard.py build" >&2; exit 1; }
  done
  echo "  every node holds its PLE block (PLE on SSD)"
fi

echo "=== [1/5] write serve.sh ==="
cat > /tmp/serve-q38.sh <<EOF
#!/bin/bash
exec vllm serve $MODEL_PATH \\
  --served-model-name $SERVED_NAME \\
  --trust-remote-code $LOAD_FLAG \\
  --tensor-parallel-size $TP_SIZE --pipeline-parallel-size 1 \\
  --distributed-executor-backend mp \\
  $EP_FLAG \\
  --max-model-len $MAX_MODEL_LEN \\
  --max-num-seqs $MAX_NUM_SEQS --max-num-batched-tokens $MAX_NUM_BATCHED \\
  --gpu-memory-utilization $GPU_MEM $KV_FLAG \\
  --kv-cache-dtype $KV_DTYPE \\
  $MOE_FLAG \\
  --enable-prefix-caching --enable-chunked-prefill \\
  $COMPILE_FLAG \\
  $SPEC_FLAG $ASYNC_FLAG $A2A_FLAG $AUTOTUNE_FLAG \\
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \\
  --host 0.0.0.0 --port $PORT \\
  --nnodes $TP_SIZE --node-rank "\${NODE_RANK}" --master-addr $HEAD_IP --master-port $MASTER_PORT \\
  \${HEADLESS:+--headless}
EOF
chmod +x /tmp/serve-q38.sh
echo "  written"

# DRY_RUN=1 stops here: everything above is checks and a file in /tmp;
# everything below creates containers on four machines. (It used to skip only
# the head-node guard, which is how a "dry run" from srv4 once built a head
# container here and root-owned bind-mount stubs on srv3.)
if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "=== DRY_RUN: would start head $HEAD_IP + workers [$WORKERS] with:"
  sed 's/^/    /' /tmp/serve-q38.sh
  exit 0
fi

echo "=== [2/5] head container ==="
docker rm -f q38 2>/dev/null || true
docker run -d --name q38 $COMMON $RDMA_FLAGS $ENVV -e VLLM_HOST_IP=$HEAD_IP $MOUNTS \
  --entrypoint /bin/bash "$IMAGE" -c "sleep infinity" >/dev/null
docker cp /tmp/serve-q38.sh q38:/tmp/serve.sh
echo "  q38 ready"

echo "=== [3/5] worker containers ==="
for w in $WORKERS; do
  ip=${w%%:*}
  ssh $SSHOPT choiceoh@"$ip" "mkdir -p $LOGDIR $CACHE_DIR; docker rm -f q38-worker 2>/dev/null; \
    docker run -d --name q38-worker $COMMON $RDMA_FLAGS $ENVV -e VLLM_HOST_IP=$ip $MOUNTS \
    --entrypoint /bin/bash $IMAGE -c 'sleep infinity' >/dev/null"
  scp $SSHOPT -q /tmp/serve-q38.sh choiceoh@"$ip":/tmp/serve-q38.sh
  ssh $SSHOPT choiceoh@"$ip" "docker cp /tmp/serve-q38.sh q38-worker:/tmp/serve.sh"
  echo "  $ip ready"
done

echo "=== [4/5] launch workers, then head ==="
for w in $WORKERS; do
  ip=${w%%:*}; rank=${w##*:}
  ssh $SSHOPT choiceoh@"$ip" "docker exec -d -e NODE_RANK=$rank -e HEADLESS=1 q38-worker bash -c 'bash /tmp/serve.sh > /q38logs/q38.log 2>&1'"
  echo "  worker $ip rank $rank"
done
sleep 3
docker exec -d -e NODE_RANK=0 q38 bash -c 'bash /tmp/serve.sh > /q38logs/q38.log 2>&1'
echo "  head rank 0"

echo "=== [5/5] done -- watch: docker exec q38 tail -f /q38logs/q38.log ; poll :$PORT/v1/models ==="
