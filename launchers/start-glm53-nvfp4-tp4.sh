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

# Whether GMU was pinned by the caller, recorded before the profile and the
# default below can fill it in. The preflight adopts its measured value only
# when it was not.
_GMU_PINNED=${GMU:+1}

# Defaults come from the profile; compose-overlays.sh reads the same file, so
# the serving knobs have one home. Caller env wins over it -- a bare source
# would clobber an explicit override, which is how the diagnostic runs are made.
PROFILE_ENV="${PROFILE_ENV:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/profiles/glm53.env}"
# Running a copy of this script from somewhere other than the repo resolves
# PROFILE_ENV to a path that does not exist, and every profile value -- the
# module list's env knobs included -- silently falls back to the literals below.
# That is the failure this whole session kept finding, so it says so.
if [ ! -f "$PROFILE_ENV" ]; then
  echo "WARNING: profile not found at $PROFILE_ENV -- using built-in defaults."
  echo "         Run launchers/start-glm53-nvfp4-tp4.sh from the checkout, or set PROFILE_ENV."
fi
if [ -f "$PROFILE_ENV" ]; then
  # The profile's VLLM_* knobs are read from the file rather than a fixed list,
  # so a knob added to a profile reaches the container without editing this.
  # No VLLM_* in the profile is normal (dsv4 has none), and an empty grep
  # exits 1 -- which under `set -euo pipefail` ends the script silently.
  _vllm_keys=$(grep -oE '^VLLM_[A-Z0-9_]+' "$PROFILE_ENV" 2>/dev/null | sort -u || true)
  _caller=""
  for _v in IMAGE MOE_BACKEND ENABLE_EP EAGER GRAPH_CAP MAX_SEQS MAX_BATCHED MAX_LEN \
            GMU SPEC_K KV_DTYPE KV_BYTES DFLASH2 SPEC ASYNC_SCHED ATTN_BACKEND \
            MODEL_HOST_PATH SERVED_NAME DRAFT_TP DRAFT_KV CUSTOM_OPS_AXIS COMPILE_CFG \
            EXTRA_ENV LOAD_FORMAT DRAFT_SAMPLE REJECT_METHOD PREFIX_CACHE \
            PREFILL_WARMUP PREFILL_WARMUP_LENS $_vllm_keys; do
    if [ -n "${!_v:-}" ]; then _caller="$_caller $_v=$(printf %q "${!_v}")"; fi
  done
  # shellcheck disable=SC1090
  . "$PROFILE_ENV"
  [ -n "$_caller" ] && eval "$_caller"
  IMAGE="${IMAGE:-${PROFILE_IMAGE:-}}"
fi

IMAGE="${IMAGE:-glm53:v13-b12x}"
NAME_HEAD=glm53
NAME_WORKER=glm53-worker
# Last-resort host path, used only when no profile is in play. It named the
# LibertAIDAI build, which is being removed from the fleet for corrupting
# Korean (see profiles/glm53.env), so a profile-less run would have aborted on
# missing weights.
MODEL_HOST_PATH="${MODEL_HOST_PATH:-${PROFILE_MODEL_PATH:-/home/choiceoh/models/glm53-redhat-nvfp4}}"
# Mount point inside the container. Deliberately left at the old name: it is
# bound from whatever MODEL_HOST_PATH points at, and it keys the compile cache.
MODEL_PATH=/models/glm-5.3-flash-nvfp4
SERVED_NAME="${SERVED_NAME:-${PROFILE_SERVED_NAME:-glm-5.3-flash}}"
CACHE_HOST_PATH=/home/choiceoh/glm53-cache
LOG_HOST_DIR=/home/choiceoh/glm53-logs
HEAD_IP=10.10.10.2
WORKER_IPS=(10.10.10.1 10.10.10.3 10.10.10.4)
MPORT=29521
PORT=8000

# This launcher only does the right thing on the head node. Every HEAD_IP
# command runs through `bash -c` (see run() below) and the head container is
# started with a plain `docker run` here, so from any other node it quietly
# builds a different cluster: the head lands on the wrong machine carrying
# VLLM_HOST_IP=$HEAD_IP, the workers rendezvous at an address nobody serves,
# and the first thing that actually fails is an unrelated-looking ssh error on
# whichever worker this node happens to lack a key for.
if [ "${DRY_RUN:-0}" != 1 ] && ! ip -4 -o addr show 2>/dev/null | grep -qw "$HEAD_IP"; then
  _here=$(ip -4 -o addr show scope global 2>/dev/null | sed 's|.* inet \([0-9.]*\)/.*|\1|' | paste -sd" ")
  echo "ABORT: 이 런처는 head 노드($HEAD_IP)에서 실행해야 합니다 — 여기는 $(hostname) [${_here}]"
  echo "       ssh <head> 'bash /home/choiceoh/stkernel/launchers/start-glm53-nvfp4-tp4.sh'"
  exit 1
fi
GMU="${GMU:-0.73}"                # 0.85 does not boot; weights+act ~78 GiB/rank
MOE_BACKEND="${MOE_BACKEND:-flashinfer_b12x}"
# b12x EP: remaps global top-k onto a local-only wrapper (+ dummy expert).
# Off by default -- the TP-sharded path is the measured one. ENABLE_EP=1
# sends --enable-expert-parallel (MoE TP becomes 1, experts shard).
# Weight loading is the whole boot: 340 s of an 11 min cold start, of which
# ~161 s is reading 185 GB at 1.15 GB/s and the rest is per-tensor work. vLLM's
# "instanttensor" does distributed loading with pipelined prefetch and direct
# I/O; a deploy report on identical hardware measured 15x, which would take the
# boot under 5 min. That report also says it can cause SILENT RANK DEATH in
# multi-node, so it stays opt-in and every boot that uses it must clear the
# generation check before its numbers are read.
LOAD_FORMAT="${LOAD_FORMAT:-auto}"
case "$LOAD_FORMAT" in
  auto|safetensors|instanttensor) ;;
  *) echo "ABORT: LOAD_FORMAT must be auto, safetensors or instanttensor (got $LOAD_FORMAT)" >&2; exit 2 ;;
esac
[ "$LOAD_FORMAT" = auto ] || echo "load-format: $LOAD_FORMAT (default is auto)"

ENABLE_EP="${ENABLE_EP:-0}"
case "$ENABLE_EP" in
  0|1) ;;
  *) echo "ABORT: ENABLE_EP must be 0 or 1 (got $ENABLE_EP)" >&2; exit 2 ;;
esac
EP_FLAG=""
if [ "$ENABLE_EP" = 1 ]; then
  EP_FLAG="--enable-expert-parallel"
fi
# Empty = let vLLM pick. It picks FLASHINFER_MLA_SPARSE_SM90 on GB10, and that
# is not a fallback -- that backend declares `capability.major in (9, 12)`, so
# it claims Blackwell deliberately and sits ahead of SM120 in the order.
#
# FLASHINFER_MLA_SPARSE_SM120 is nonetheless eligible here: has_flashinfer_
# sparse_mla_sm120() is True in this image, dtype bf16, kv_cache_dtype fp8_e4m3
# and index_topk 2048 all pass its supports_combination(). Which of the two is
# actually faster on sm_121 is unmeasured, so this knob exists to find out
# rather than to assert an answer.
ATTN_BACKEND="${ATTN_BACKEND:-}"   # GLM spells b12x this way; marlin is gone
EAGER="${EAGER:-0}"
GRAPH_CAP="${GRAPH_CAP:-16}"      # 256 is sized for MAX_SEQS=6
MAX_BATCHED="${MAX_BATCHED:-2048}"
MAX_SEQS="${MAX_SEQS:-4}"
MM_LIMIT="${MM_LIMIT:-{\"image\":0,\"video\":0}}"
# The old value asked for FULL_AND_PIECEWISE *and* fuse_attn_quant, which vLLM
# refuses together while use_inductor_graph_partition is off -- it dropped the
# piecewise half and logged it. FULL is refused anyway by the sparse indexer
# backend (UNIFORM_BATCH only), so what actually ran was FULL_DECODE_ONLY. This
# declares that, instead of asking for something and silently getting less.
#
# PIECEWISE=1 takes the other branch: drop the attention-quant fusion and get
# piecewise graphs over prefill. Which is faster here has never been measured.
if [ "${PIECEWISE:-0}" = 1 ]; then
  # Not ${COMPILE_CFG:-{...}}: the JSON's own braces close the expansion early
  # and the leftover "}}" is appended to whatever the caller passed, so
  # overriding this produced "trailing characters" from the JSON parser and the
  # knob could never be used.
  # #108's fusion-barrier A/B: elementwise 483/step (25.6%) survive because
  # custom_ops:["all"] makes every registered op an opaque inductor wall.
  # CUSTOM_OPS_AXIS="" removes the walls entirely (fusion arm);
  # the default "all" is the control arm. One boot decides.
  [ -n "${COMPILE_CFG:-}" ] || COMPILE_CFG='{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"],"pass_config":{"fuse_gemm_comms":true,"fuse_allreduce_rms":true}}'
  if [ -n "${CUSTOM_OPS_AXIS+x}" ]; then
    COMPILE_CFG=$(printf '%s' "$COMPILE_CFG" | sed 's/\"all\"/'"\"${CUSTOM_OPS_AXIS:-}\""'/')
  fi
else
  [ -n "${COMPILE_CFG:-}" ] || COMPILE_CFG='{"cudagraph_mode":"FULL_DECODE_ONLY","custom_ops":["all"],"pass_config":{"fuse_gemm_comms":true,"fuse_allreduce_rms":true,"fuse_attn_quant":true}}'
  if [ -n "${CUSTOM_OPS_AXIS+x}" ]; then
    COMPILE_CFG=$(printf '%s' "$COMPILE_CFG" | sed 's/\"all\"/'"\"${CUSTOM_OPS_AXIS:-}\""'/')
  fi
fi
# Compact drops dummy slots when tokens*top_k > 640. That path is eager and
# data-dependent, so a captured batch must stay at or below the cutover.
# GRAPH_CAP=32 * top_k=8 = 256 is safe. PIECEWISE graphs prefill, so compact
# stays off there and zero-weight local repeats keep a fixed shape.
B12X_EP_TOPK="${B12X_EP_TOPK:-8}"
if [ "$ENABLE_EP" = 1 ]; then
  if [ "${PIECEWISE:-0}" = 1 ]; then
    : "${VLLM_B12X_EP_COMPACT:=0}"
    echo "ENABLE_EP=1 + PIECEWISE=1: VLLM_B12X_EP_COMPACT=$VLLM_B12X_EP_COMPACT (prefill graphs keep fixed-shape repeats)"
  fi
  if [ "${EAGER:-0}" != 1 ] && [ "${VLLM_B12X_EP_COMPACT:-1}" != 0 ]; then
    _ep_routed=$((GRAPH_CAP * B12X_EP_TOPK))
    if [ "$_ep_routed" -gt 640 ]; then
      echo "ABORT: ENABLE_EP=1 GRAPH_CAP=$GRAPH_CAP captures $_ep_routed pairs > 640 compact cutover — GRAPH_CAP<=80, EAGER=1, or VLLM_B12X_EP_COMPACT=0" >&2
      exit 2
    fi
  fi
fi
# DFLASH2=1: block-diffusion drafter (2.15x over MTP-4 at TP2, acceptance 74%).
# num_speculative_tokens MUST be 7 (drafter block 8 minus the verified token).
DFLASH2="${DFLASH2:-1}"
DRAFT_HOST_PATH=/home/choiceoh/models/GLM-5.3-Flash-DFlash2
# DFlash/DSpark synthesize their context KV from target hidden states. Tokens
# restored by automatic prefix caching do not run through the target, so this
# image leaves their draft KV slots unwritten while draft attention still
# reads them. That is upstream vLLM #47926: long shared-prefix workloads can
# collapse to position-0-only acceptance. Until that draft PR's four-file
# state/block-table repair lands, fail closed and make every prompt token flow
# through the target. PREFIX_CACHE=1 is an explicit throughput-first rollback;
# it may restore TTFT reuse at the cost of DFlash2 acceptance on cache hits.
PREFIX_CACHE="${PREFIX_CACHE:-0}"
case "$PREFIX_CACHE" in
  0) PREFIX_CACHE_FLAG="--no-enable-prefix-caching" ;;
  1) PREFIX_CACHE_FLAG="--enable-prefix-caching" ;;
  *) echo "ABORT: PREFIX_CACHE must be 0 or 1 (got $PREFIX_CACHE)" >&2; exit 2 ;;
esac
[ "$PREFIX_CACHE" = 1 ] || echo "prefix-cache: disabled for DFlash2 draft-KV safety"
# The former AUDIT overlay replaced V1 files, but this image runs V2 Model
# Runner.  Refuse the old switch instead of claiming an audit that cannot run.
[ "${AUDIT:-0}" = 0 ] || {
  echo "ABORT: AUDIT targets the inactive V1 runner and is unsupported on glm53:v13-b12x" >&2
  exit 2
}
# b12x picks static vs dynamic MoE by routed_rows = tokens * top_k against a
# cutover of 640. A speculative verify step is 8 tokens -> 64 routed rows ->
# static, while prefill goes dynamic and a non-speculative decode step (8 rows)
# goes direct_micro. Setting this to 0 sends everything to dynamic, which is
# how to find out whether static is where the Korean fragments come from.
MOE_CUTOVER="${MOE_CUTOVER:-}"
# Both cudagraph wrappers already check that a replay sees the same input
# tensor addresses it was captured with, and both gate that check on
# is_debugging_mode = (VLLM_LOGGING_LEVEL == "DEBUG"). Off by default, so a
# graph reading a stale address is silent. GRAPH_DEBUG=1 turns it into an
# assertion. Only positional tensor args are covered -- firing is proof,
# staying quiet is not a clean bill of health. Very verbose; diagnostic only.
GRAPH_DEBUG="${GRAPH_DEBUG:-0}"
KV_DTYPE="${KV_DTYPE:-fp8_e4m3}"   # auto = bf16, for isolating KV quantization
KV_BYTES="${KV_BYTES:-auto}"          # auto = let vLLM profile per node
MAX_LEN="${MAX_LEN:-1048576}"
SPEC_K="${SPEC_K:-7}"             # the comment above is not advisory
SSHOPT="-o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=8"

# Overlays come from the profile's composed manifest, put on every node by
# launchers/deploy-overlays.sh. This used to be a single hardcoded bind while
# the profile named eleven modules.
OVERLAY_DIR="${OVERLAY_DIR:-${PROFILE_OVERLAY_DIR:-/home/choiceoh/overlays/glm53}}"
OVERLAY_MANIFEST="$OVERLAY_DIR/manifest.tsv"
OVFILES=(); OVTARGETS=(); OVBASES=()

load_overlay_manifest() {
  local manifest="$1" source target base extra seen
  [ -f "$manifest" ] || { echo "ABORT: overlay manifest missing ($manifest) -- run deploy-overlays.sh glm53"; exit 1; }
  while IFS=$'\t' read -r source target base extra \
      || [ -n "${source:-}${target:-}${base:-}${extra:-}" ]; do
    [[ -z "$source" || "$source" == \#* ]] && continue
    [ -n "$target" ] && [ -n "$base" ] && [ -z "${extra:-}" ] \
      || { echo "ABORT: malformed overlay manifest row: $source $target $base ${extra:-}"; exit 1; }
    case "$source" in *[!A-Za-z0-9._-]*|.*) echo "ABORT: unsafe overlay source: $source"; exit 1;; esac
    case "$target" in
      "${TARGET_PREFIX:-/usr/local/lib/python3.12/dist-packages/}"*) ;;
      *) echo "ABORT: overlay target outside the package root: $target"; exit 1;;
    esac
    case "$target" in *[!A-Za-z0-9_./-]*) echo "ABORT: unsafe character in target: $target"; exit 1;; esac
    [ "$base" = absent ] || [[ "$base" =~ ^[0-9a-f]{64}$ ]] \
      || { echo "ABORT: invalid base contract for $source: $base"; exit 1; }
    for seen in ${OVFILES[@]+"${OVFILES[@]}"}; do
      [ "$seen" != "$source" ] || { echo "ABORT: duplicate overlay source: $source"; exit 1; }
    done
    for seen in ${OVTARGETS[@]+"${OVTARGETS[@]}"}; do
      [ "$seen" != "$target" ] || { echo "ABORT: duplicate overlay target: $target"; exit 1; }
    done
    OVFILES+=("$source"); OVTARGETS+=("$target"); OVBASES+=("$base")
  done < "$manifest"
  (( ${#OVFILES[@]} > 0 )) || { echo "ABORT: overlay manifest is empty"; exit 1; }
}
if [ "${DRY_RUN:-0}" = 1 ] && [ ! -f "$OVERLAY_MANIFEST" ]; then
  echo "note: no overlay manifest at $OVERLAY_MANIFEST (deploy-overlays.sh glm53)"
else
  load_overlay_manifest "$OVERLAY_MANIFEST"
fi

[ "${DRY_RUN:-0}" = 1 ] || test -f "$MODEL_HOST_PATH/config.json"
[ "${DRY_RUN:-0}" = 1 ] || test -f "$MODEL_HOST_PATH/chat_template_mm.jinja" || {
  echo "ABORT: chat_template_mm.jinja missing in model dir (copy from ~/glm53-4x/)"; exit 1; }

# Refuse to start over a live dsv4/q38 stack. Skipped under DRY_RUN: these ask
# the machines about themselves, which a config print has no use for.
for ip in $([ "${DRY_RUN:-0}" = 1 ] || echo "$HEAD_IP ${WORKER_IPS[@]}"); do
  run() { if [ "$ip" = "$HEAD_IP" ]; then bash -c "$1"; else ssh $SSHOPT choiceoh@"$ip" "$1"; fi; }
  if run 'docker ps --format "{{.Names}}"|grep -qE "^(hy4|q38)"'; then
    echo "ABORT: $ip runs hy4/q38 — stop production/experiment first"; exit 1; fi
  run "test -f $MODEL_HOST_PATH/config.json" || { echo "ABORT: $ip missing weights"; exit 1; }
  if (( ${#OVFILES[@]} )); then
    _sum=$(cd "$OVERLAY_DIR" && sha256sum manifest.tsv "${OVFILES[@]}" 2>/dev/null)
    [ "$(run "cd $OVERLAY_DIR && sha256sum manifest.tsv ${OVFILES[*]} 2>/dev/null")" = "$_sum" ] \
      || { echo "ABORT: $ip overlays differ from head -- run deploy-overlays.sh glm53"; exit 1; }
  fi
  run "docker image inspect $IMAGE >/dev/null 2>&1" || { echo "ABORT: $ip missing image"; exit 1; }
done

# The manifest pins each target's sha256 as the image ships it. A mismatch means
# the overlay was written against a different build and would replace code it has
# never seen; "absent" means the overlay adds a file, so the image must NOT have
# one. One container start covers every row.
if [ "${DRY_RUN:-0}" != 1 ] && (( ${#OVFILES[@]} )); then
  _got=$(docker run --rm --entrypoint sha256sum "$IMAGE" "${OVTARGETS[@]}" 2>/dev/null || true)
  for _i in "${!OVFILES[@]}"; do
    _have=$(printf '%s\n' "$_got" | awk -v t="${OVTARGETS[$_i]}" '$2==t{print $1}')
    if [ "${OVBASES[$_i]}" = absent ]; then
      [ -z "$_have" ] || { echo "ABORT: ${OVFILES[$_i]} is declared new but the image already has ${OVTARGETS[$_i]}"; exit 1; }
    else
      [ "$_have" = "${OVBASES[$_i]}" ] \
        || { echo "ABORT: base preimage mismatch for ${OVFILES[$_i]} (image ${_have:-missing}, manifest ${OVBASES[$_i]})"; exit 1; }
    fi
  done
  echo "overlays: ${#OVFILES[@]} base contracts verified against $IMAGE"
fi

# Our proven fabric env (dual HCA), their runtime env.
# Torch profiler. bench/profile-step.py drives it over /start_profile and
# /stop_profile; hy4 has had this wired since the dsv4 kernel campaign and
# glm53 never did, which is why the 33 ms of the decode step that is not memory
# traffic has stayed unattributed. Zero cost until a capture is started -- the
# env arms the worker profiler, and the HTTP routes come from --profiler-config
# below (env alone does not attach them).
ENVV="-e VLLM_TORCH_PROFILER_DIR=/prof -e HF_HOME=/cache/huggingface -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
-e VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=/cache/flashinfer_autotune -e VLLM_CACHE_ROOT=/cache/vllm \
-e FLASHINFER_WORKSPACE_BASE=/cache -e TRITON_CACHE_DIR=/cache/triton \
-e CUTE_DSL_ARCH=sm_121a \
-e VLLM_USE_AOT_COMPILE=1 -e VLLM_USE_MEGA_AOT_ARTIFACT=0 -e VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
-e VLLM_USE_FLASHINFER_SAMPLER=1 -e NCCL_P2P_LEVEL=SYS \
-e TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200 \
-e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
-e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
-e FLASHINFER_DISABLE_VERSION_CHECK=1 \
# Channel count is the only NCCL axis that moved on this fabric (4-node
# torch all-reduce sweep, 2026-09-03, engine down): the default channel
# count leaves the 200G links at 129-131 Gb/s ring-effective on the 64 MB
# prefill payload; 16 channels with 4 per net peer reaches 142.3 Gb/s
# (-8.2% on that all-reduce) and cuts a 256 KB message 0.272 -> 0.157 ms.
# 8/24/32 channels, LL128-only, extra QPs and a 16 MB buffer were all
# neutral or worse; LL128-only is 3.5x SLOWER. Prefill AR is 16% of a
# 32K prefill, so this is ~1.3% of prefill -- env only, no numerics.
-e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0 \
-e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e GLOO_SOCKET_IFNAME=enp1s0f0np0 \
-e TP_SOCKET_IFNAME=enp1s0f0np0 -e MN_IF_NAME=enp1s0f0np0 \
-e NCCL_CROSS_NIC=1 -e NCCL_PROTO=LL,LL128,Simple -e NCCL_CUMEM_ENABLE=0 \
-e NCCL_IB_GID_INDEX=3 -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET \
-e NCCL_NVLS_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN \
-e NCCL_MIN_NCHANNELS=16 -e NCCL_MAX_NCHANNELS=16 -e NCCL_NCHANNELS_PER_NET_PEER=4 \
-e TORCH_NCCL_ASYNC_ERROR_HANDLING=1"

# Profile-declared VLLM_* knobs. Until now the profile set them and nothing
# carried them, so every module they gate ran its stock path.
for _k in ${_vllm_keys:-}; do
  if [ -n "${!_k:-}" ]; then ENVV="$ENVV -e $_k=${!_k}"; fi
done

# Diagnostic env passthrough. EXTRA_ENV="A=1 B=2" becomes -e A=1 -e B=2.
# For one-shot debugging only -- CUDA_LAUNCH_BLOCKING=1 pins an async kernel
# failure to its real launch site instead of surfacing at the next sync
# (this lane has now lost two boots to that signature). Production knobs
# belong in the profile, not here.
EXTRA_ENV_FLAGS=""
# _vllm_keys is newline-separated (grep | sort -u); flatten it so the
# membership test below can use a space-delimited case pattern.
_vllm_keys_sp=" $(printf '%s ' ${_vllm_keys:-})"
for _kv in ${EXTRA_ENV:-}; do
  case "$_kv" in
    # A comma-joined list is the natural mistake, and it passes the KEY=VALUE
    # shape: the whole string lands in ONE -e whose value is
    # "1,NEXT_KEY=1,..." -- so every knob reads as off and the boot measures
    # the baseline while the log says the knobs were set. Refuse it here.
    [A-Za-z_]*=*,[A-Za-z_]*=*)
      echo "ABORT: EXTRA_ENV is space-separated, not comma-separated: $_kv"
      echo "       use EXTRA_ENV=\"A=1 B=2\""
      exit 1 ;;
    [A-Za-z_]*=*)
      # A profile-declared VLLM_* key is emitted as its own -e further down,
      # and docker takes the LAST -e for a name. So EXTRA_ENV silently loses
      # to the profile for exactly the knobs a sweep wants to move, and the
      # boot measures the profile value while the caller believes otherwise.
      # Pass those as caller environment variables instead: the _caller
      # restore above re-applies them after the profile is sourced.
      case "$_vllm_keys_sp" in
        *" ${_kv%%=*} "*)
          echo "ABORT: ${_kv%%=*} is declared in the profile, so EXTRA_ENV cannot"
          echo "       override it (docker takes the last -e). Pass it directly:"
          echo "         ${_kv%%=*}=${_kv#*=} bash launchers/start-glm53-nvfp4-tp4.sh"
          exit 1 ;;
      esac
      EXTRA_ENV_FLAGS="$EXTRA_ENV_FLAGS -e $_kv" ;;
    *) echo "ABORT: EXTRA_ENV entry is not KEY=VALUE: $_kv"; exit 1 ;;
  esac
done
[ -n "$EXTRA_ENV_FLAGS" ] && echo "extra env:$EXTRA_ENV_FLAGS"

# Docker defaults the soft nofile to 1024 while the host allows 500k and the
# image permits 524288. NCCL opens a socket per peer connection and a loader
# that prefetches shards in parallel opens many descriptors at once; together
# they exhaust 1024 and NCCL then fails far from the cause -- with
# LOAD_FORMAT=instanttensor three ranks died at once as ncclSystemError
# "Call to socket failed: Too many open files". Raise it to the image hard
# limit; nothing here needs the low default.
COMMON="--gpus all -d --restart no --network host --ipc host --shm-size 32g \
--memory 112g --memory-swap 112g --ulimit memlock=-1:-1 --ulimit nofile=524288:524288 --cap-add IPC_LOCK \
--device /dev/infiniband:/dev/infiniband \
-v $MODEL_HOST_PATH:$MODEL_PATH:ro -v CACHEDIR:/cache -v $LOG_HOST_DIR:/glmlogs \
-v /home/choiceoh/vllm-prof:/prof"
COMMON="$COMMON $EXTRA_ENV_FLAGS"
for _i in ${OVFILES[@]+"${!OVFILES[@]}"}; do
  COMMON="$COMMON -v $OVERLAY_DIR/${OVFILES[$_i]}:${OVTARGETS[$_i]}:ro"
done

if [ "$GRAPH_DEBUG" = 1 ]; then ENVV="$ENVV -e VLLM_LOGGING_LEVEL=DEBUG"; fi
if [ -n "$MOE_CUTOVER" ]; then
  ENVV="$ENVV -e FLASHINFER_B12X_STATIC_COMPACT_CUTOVER_PAIRS=$MOE_CUTOVER"
fi
if [ "$ENABLE_EP" = 1 ]; then
  ENVV="$ENVV -e VLLM_B12X_EP_COMPACT=${VLLM_B12X_EP_COMPACT:-1}"
fi

# SPEC=0 serves straight from the target model -- no drafter, no accept path.
# Only for isolating a defect: it costs the whole speculative speedup.
if [ "${SPEC:-1}" = 0 ]; then
  SPECCFG_VAL=""
elif [ "$DFLASH2" = 1 ]; then
  [ "${DRY_RUN:-0}" = 1 ] || test -f "$DRAFT_HOST_PATH/config.json" || { echo "ABORT: DFlash2 drafter missing at $DRAFT_HOST_PATH"; exit 1; }
  COMMON="$COMMON -v $DRAFT_HOST_PATH:/models/dflash2-draft:ro"
  # The drafter emits a block of 8 and one of them is the verified token, so
  # this path only works at 7. It was a literal before, which meant SPEC_K was
  # silently ignored rather than checked.
  # The drafter's dflash_config.block_size is 8, so it PROPOSES 7. Whether the
  # target has to VERIFY all 7 is a different question, and it has never been
  # tested. vLLM does not reject a smaller k for dflash -- dspark raises
  # explicitly on num_speculative_tokens < dspark_block_size, dflash has no
  # such branch, and speculator.py sizes every buffer from
  # num_speculative_tokens with no hardcoded 7.
  #
  # The measured reason to want a smaller k: the step's variable part is
  # 0.706 ms x DISTINCT EXPERTS and the verify batch is (k+1) tokens, so the
  # expert tax scales with k. SPEC=0 (1 token) is a 31.3 ms step, k=7
  # (8 tokens) is 66.1 -- k=3 should land near 46.
  #
  # The risk is not a crash, it is a distribution shift: the drafter was
  # trained on blocks of 8 and would see a shorter one. That shows up as
  # acceptance collapse in the per-position counters, not as an error, so the
  # experiment is only readable with /metrics beside the throughput.
  if [ "$SPEC_K" != 7 ] && [ "${SPEC_K_FORCE:-0}" != 1 ]; then
    echo "ABORT: dflash requires SPEC_K=7 (drafter block 8 minus the verified token), got $SPEC_K -- set SPEC_K_FORCE=1 to open it for an experiment"
    exit 1
  fi
  _spec_extra=""
  [ -n "${DRAFT_TP:-}" ] && _spec_extra="$_spec_extra,\"draft_tensor_parallel_size\":$DRAFT_TP"
  [ -n "${DRAFT_KV:-}" ] && [ "${DRAFT_KV}" != auto ] && _spec_extra="$_spec_extra,\"kv_cache_dtype\":\"$DRAFT_KV\""
  # draft_sample_method: the image default is "greedy", which treats the
  # drafter's distribution as one-hot in rejection sampling -- accept prob
  # degenerates to p_target(draft_token), fine at temp 0 and lossy at temp>0
  # (Capicua25x/glm53-dspark SERVE-SPARKS.md boot-death #9: prose accept
  # length 1.03 -> ~1.9 from this one field; the DFlash2 reference config
  # ships probabilistic). Our bench runs at temp 0.95, so every acceptance
  # number this lane has recorded was taken under the degenerate setting.
  # DRAFT_SAMPLE=greedy restores the old behavior for an A/B.
  DRAFT_SAMPLE="${DRAFT_SAMPLE:-probabilistic}"
  # rejection_sample_method: "standard" verifies the draft one token at a
  # time and stops at the first rejection. "block" is block verification
  # (Sun et al. 2024, arXiv 2403.10444): it verifies the whole block jointly
  # and accepts at least as many tokens for the same draft and target
  # distributions. The kernel only takes that branch when temp > 0 -- our
  # bench runs at 0.95 -- and it reads the cached draft logits, which exist
  # because DRAFT_SAMPLE defaults to probabilistic.
  case "${REJECT_METHOD:-}" in
    "" ) ;;
    standard|block )
      _spec_extra="$_spec_extra,\"rejection_sample_method\":\"$REJECT_METHOD\"" ;;
    * ) echo "ABORT: REJECT_METHOD must be standard or block, got '$REJECT_METHOD'"; exit 1 ;;
  esac
  SPECCFG_VAL="--speculative-config '{\"method\":\"dflash\",\"model\":\"/models/dflash2-draft\",\"num_speculative_tokens\":$SPEC_K,\"draft_sample_method\":\"$DRAFT_SAMPLE\"$_spec_extra}'"
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
# Reclaim stale containers, then page cache, then size GMU -- in that order,
# and before SERVE_ARGS bakes the number in.
#
# The order is the point. This used to run after SERVE_ARGS was built, so the
# measured value could not reach --gpu-memory-utilization at all: the preflight
# printed a recommendation and the boot ignored it. And it ran before the
# launch loop removes the previous run's workers, so it measured with those
# still resident -- which is where "an unexpected tenant" readings and free
# memory that swung by 70 GiB between boots came from.
#
# NVRM allocates against MemFree, so cached file pages are memory the engine
# cannot use. SKIP_PREFLIGHT=1 opts out of the whole step.
PREFLIGHT=/home/choiceoh/stkernel/launchers/memfree-preflight.sh
if [ "${SKIP_PREFLIGHT:-0}" != 1 ] && [ -x "$PREFLIGHT" ] && [ "${DRY_RUN:-0}" != 1 ]; then
  echo "== 이전 부팅 잔여 컨테이너 회수 =="
  for _ip in "$HEAD_IP" "${WORKER_IPS[@]}"; do
    if [ "$_ip" = "$HEAD_IP" ]; then
      docker rm -f $NAME_HEAD $NAME_WORKER >/dev/null 2>&1 || true
    else
      ssh $SSHOPT choiceoh@"$_ip" "docker rm -f $NAME_HEAD $NAME_WORKER >/dev/null 2>&1; true" || true
    fi
  done

  echo "== memfree preflight =="
  # Report goes to stderr (straight to the terminal); the computed GMU is the
  # only thing on stdout, so $(...) captures it alone.
  if GMU_SAFE=$("$PREFLIGHT" 3); then
    if [ "${_GMU_PINNED:-}" = 1 ]; then
      if awk "BEGIN{exit !($GMU > $GMU_SAFE)}" 2>/dev/null; then
        echo "  ! GMU=$GMU 를 호출자가 지정했고 실측 상한($GMU_SAFE)을 넘습니다 — 그대로 진행"
      fi
    elif awk "BEGIN{exit !($GMU != $GMU_SAFE)}" 2>/dev/null; then
      # Both directions. Backing GMU off is the intuitive move and the wrong
      # one: weights and activations come out of the same budget, so KV is
      # (GMU x total - overhead) and shrinking GMU drives it negative. Four
      # boots died that way before this adopted the measured value.
      echo "  GMU $GMU -> $GMU_SAFE (실측 채택)"
      GMU=$GMU_SAFE
    fi
  else
    echo "  preflight refused (a node was unreachable); continuing with GMU=$GMU"
  fi
fi

PROF_CFG="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"/prof\",\"torch_profiler_with_stack\":false}"
SERVE_ARGS="$MODEL_PATH \
--served-model-name $SERVED_NAME \
--host 0.0.0.0 --port $PORT \
--trust-remote-code \
--tensor-parallel-size 4 \
--gpu-memory-utilization $GMU \
--profiler-config '$PROF_CFG' \
${ATTN_BACKEND:+--attention-backend $ATTN_BACKEND }\
--max-model-len $MAX_LEN \
--max-num-seqs $MAX_SEQS --max-num-batched-tokens $MAX_BATCHED --block-size 2304 --moe-backend $MOE_BACKEND \
$PREFIX_CACHE_FLAG \
--load-format $LOAD_FORMAT \
${EP_FLAG:+$EP_FLAG }\
$SPECCFG_VAL \
--kv-cache-dtype $KV_DTYPE $KV_FLAG \
$ASYNC_FLAG \
$EAGER_FLAG --enable-flashinfer-autotune \
--limit-mm-per-prompt '$MM_LIMIT' \
--tool-call-parser glm47 --enable-auto-tool-choice \
--reasoning-parser glm45 --chat-template $MODEL_PATH/chat_template_mm.jinja \
--distributed-executor-backend mp \
--nnodes 4 --master-addr $HEAD_IP --master-port $MPORT"

# DRY_RUN=1 prints what the knobs resolved to and stops -- before the cache
# reclaim, before any container. A default that points at a deleted image or a
# deleted backend is otherwise invisible until the boot fails.
if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "profile   : ${PROFILE_ENV:-<none>}"
  for _k in IMAGE MOE_BACKEND ENABLE_EP VLLM_B12X_EP_COMPACT VLLM_B12X_EP_NO_DUMMY KV_DTYPE EAGER GRAPH_CAP GMU MAX_SEQS \
            MAX_BATCHED MAX_LEN DFLASH2 SPEC SPEC_K ASYNC_SCHED PREFIX_CACHE \
            DRAFT_SAMPLE REJECT_METHOD; do
    printf '  %-12s %s\n' "$_k" "${!_k:-<unset>}"
  done
  echo "overlays  :"
  printf '%s\n' "${COMMON:-}" | tr ' ' '\n' | grep -A0 "dist-packages" | sed 's/^/    /'
  echo "spec flag : ${SPECCFG_VAL:-<none>}"
  echo "graph flag: ${EAGER_FLAG:-<none>}"
  exit 0
fi

mkdir -p "$CACHE_HOST_PATH" "$LOG_HOST_DIR"

# torch.compile caches under the mounted /cache and outlives boots, but the
# overlays that shape the compiled graph are not part of its key -- so after an
# overlay change a stale entry is still found and then fails to load
# ("Compiling model again due to a load failure"). Stamp the manifest sha beside
# the cache and clear it when that moves.
# Maintenance, not a precondition: a stale cache costs time, a launcher that
# exits costs the boot. Anything in here reports and continues.
best_effort() {
  local what="$1"; shift
  if ! "$@" >/dev/null 2>&1; then
    echo "  ! $what 실패 — 계속 진행합니다 (부팅을 막을 이유가 아님)"
    return 0
  fi
}

if [ -f "$OVERLAY_MANIFEST" ]; then
  _ov_sha=$(sha256sum "$OVERLAY_MANIFEST" | cut -d" " -f1)
  _stamp="$CACHE_HOST_PATH/.overlay-sha"
  if [ "$(cat "$_stamp" 2>/dev/null)" != "$_ov_sha" ]; then
    echo "overlays changed -> clearing torch.compile cache"
    # The container writes this cache as root, so the host user cannot remove
    # it. Delete from inside a container instead of reaching for sudo.
    best_effort "컴파일 캐시 삭제" \
      docker run --rm -v "$CACHE_HOST_PATH":/cache --entrypoint rm "$IMAGE" \
        -rf /cache/vllm/torch_compile_cache
    best_effort "오버레이 sha 기록" \
      bash -c "printf '%s' \"$_ov_sha\" > \"$_stamp\""
  fi
fi

# Workers first (rank 1..3), head last — their documented order.
rank=1
for ip in "${WORKER_IPS[@]}"; do
  W_B64=$(printf '%s' "vllm serve $SERVE_ARGS --node-rank $rank --headless > /glmlogs/glm53.log 2>&1" | base64 -w0)
  ssh $SSHOPT choiceoh@"$ip" "mkdir -p $CACHE_HOST_PATH $LOG_HOST_DIR /home/choiceoh/vllm-prof; docker rm -f $NAME_WORKER 2>/dev/null; \
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

# deneb fork (prefill-warmup): pay the 11-41% cold JIT tax (PR #192 ledger's
# one open prefill target) at boot instead of on the first user request:
# one real prefill per production length once the server answers, exact token
# counts, distinct content per request. Backgrounded on purpose -- cold boot
# takes ~15min and the launcher must return. rep1-vs-rep2 in the log is the
# receipt that the tax existed and is now paid.
if [ "${PREFILL_WARMUP:-0}" = "1" ]; then
  LAUNCH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  nohup python3 "$LAUNCH_DIR/prefill-warmup.py" --port "$PORT"     ${PREFILL_WARMUP_LENS:+--lens "$PREFILL_WARMUP_LENS"}     > "$LOG_HOST_DIR/prefill-warmup.log" 2>&1 &
  echo "prefill-warmup armed (log: $LOG_HOST_DIR/prefill-warmup.log)"
fi
