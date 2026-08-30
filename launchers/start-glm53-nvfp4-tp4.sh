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
  for _v in IMAGE MOE_BACKEND EAGER GRAPH_CAP MAX_SEQS MAX_BATCHED MAX_LEN \
            GMU SPEC_K KV_DTYPE KV_BYTES DFLASH2 SPEC ASYNC_SCHED ATTN_BACKEND \
            MODEL_HOST_PATH SERVED_NAME DRAFT_TP DRAFT_KV $_vllm_keys; do
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
  COMPILE_CFG="${COMPILE_CFG:-{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"custom_ops\":[\"all\"],\"pass_config\":{\"fuse_gemm_comms\":true,\"fuse_allreduce_rms\":true}}}"
else
  COMPILE_CFG="${COMPILE_CFG:-{\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"custom_ops\":[\"all\"],\"pass_config\":{\"fuse_gemm_comms\":true,\"fuse_allreduce_rms\":true,\"fuse_attn_quant\":true}}}"
fi
# DFLASH2=1: block-diffusion drafter (2.15x over MTP-4 at TP2, acceptance 74%).
# num_speculative_tokens MUST be 7 (drafter block 8 minus the verified token).
DFLASH2="${DFLASH2:-1}"
DRAFT_HOST_PATH=/home/choiceoh/models/GLM-5.3-Flash-DFlash2
# AUDIT=1 mounts overlay/modules/glm53_drop_audit and turns it on. It reports
# a sampled token discarded past prefill, and a break in the contiguity that
# three accepted-token counts assume. Diagnostic only.
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
AUDIT="${AUDIT:-0}"
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
-e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0 \
-e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e GLOO_SOCKET_IFNAME=enp1s0f0np0 \
-e TP_SOCKET_IFNAME=enp1s0f0np0 -e MN_IF_NAME=enp1s0f0np0 \
-e NCCL_CROSS_NIC=1 -e NCCL_PROTO=LL,LL128,Simple -e NCCL_CUMEM_ENABLE=0 \
-e NCCL_IB_GID_INDEX=3 -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET \
-e NCCL_NVLS_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN \
-e TORCH_NCCL_ASYNC_ERROR_HANDLING=1"

# Profile-declared VLLM_* knobs. Until now the profile set them and nothing
# carried them, so every module they gate ran its stock path.
for _k in ${_vllm_keys:-}; do
  if [ -n "${!_k:-}" ]; then ENVV="$ENVV -e $_k=${!_k}"; fi
done

COMMON="--gpus all -d --restart no --network host --ipc host --shm-size 32g \
--memory 112g --memory-swap 112g --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
--device /dev/infiniband:/dev/infiniband \
-v $MODEL_HOST_PATH:$MODEL_PATH:ro -v CACHEDIR:/cache -v $LOG_HOST_DIR:/glmlogs \
-v /home/choiceoh/vllm-prof:/prof"
for _i in ${OVFILES[@]+"${!OVFILES[@]}"}; do
  COMMON="$COMMON -v $OVERLAY_DIR/${OVFILES[$_i]}:${OVTARGETS[$_i]}:ro"
done

# glm53_drop_audit ships in the manifest; this only arms it.
[ "$AUDIT" = 1 ] && ENVV="$ENVV -e VLLM_DENEB_DROP_AUDIT=1"
if [ "$GRAPH_DEBUG" = 1 ]; then ENVV="$ENVV -e VLLM_LOGGING_LEVEL=DEBUG"; fi
if [ -n "$MOE_CUTOVER" ]; then
  ENVV="$ENVV -e FLASHINFER_B12X_STATIC_COMPACT_CUTOVER_PAIRS=$MOE_CUTOVER"
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
  [ "$SPEC_K" = 7 ] || { echo "ABORT: dflash requires SPEC_K=7 (drafter block 8 minus the verified token), got $SPEC_K"; exit 1; }
  _spec_extra=""
  [ -n "${DRAFT_TP:-}" ] && _spec_extra="$_spec_extra,\"draft_tensor_parallel_size\":$DRAFT_TP"
  [ -n "${DRAFT_KV:-}" ] && [ "${DRAFT_KV}" != auto ] && _spec_extra="$_spec_extra,\"kv_cache_dtype\":\"$DRAFT_KV\""
  SPECCFG_VAL="--speculative-config '{\"method\":\"dflash\",\"model\":\"/models/dflash2-draft\",\"num_speculative_tokens\":$SPEC_K$_spec_extra}'"
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

SERVE_ARGS="$MODEL_PATH \
--served-model-name $SERVED_NAME \
--host 0.0.0.0 --port $PORT \
--trust-remote-code \
--tensor-parallel-size 4 \
--gpu-memory-utilization $GMU \
--profiler-config '{"profiler":"torch","torch_profiler_dir":"/prof","torch_profiler_with_stack":false}' \
${ATTN_BACKEND:+--attention-backend $ATTN_BACKEND }\
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

# DRY_RUN=1 prints what the knobs resolved to and stops -- before the cache
# reclaim, before any container. A default that points at a deleted image or a
# deleted backend is otherwise invisible until the boot fails.
if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "profile   : ${PROFILE_ENV:-<none>}"
  for _k in IMAGE MOE_BACKEND KV_DTYPE EAGER GRAPH_CAP GMU MAX_SEQS \
            MAX_BATCHED MAX_LEN DFLASH2 SPEC SPEC_K ASYNC_SCHED; do
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
