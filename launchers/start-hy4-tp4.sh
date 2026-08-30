#!/bin/bash
# DeepSeek-V4-Flash-0731 TP=4 on the PROD-PROVEN hybrid-1.6 stack (aidendle94 sparkrun fork).
# Faithful port of ~/hybrid-stack compose.{head,worker}.yaml from TP2(srv2+srv3) to
# TP4(srv2 head + srv3/srv1/srv4 workers): same image, overlays, kernel caches, env,
# dspark speculative decode (SPEC_TOKENS=5). First boot recompiles kernels for TP4
# shapes — expect a LONG warmup; watchdog-disable envs carried over. RUN ON srv2.
set -euo pipefail

# Identity comes from the profile; compose-overlays.sh and deploy-overlays.sh
# already read the same file, so this removes the second copy rather than adding
# one. Behaviour is unchanged today -- the profile's image, model path and
# served name are character-for-character what was hardcoded here. Caller env
# still wins, as everywhere else.
PROFILE_ENV="${PROFILE_ENV:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/profiles/dsv4.env}"
if [ -f "$PROFILE_ENV" ]; then
  # No VLLM_* in the profile is normal (dsv4 has none), and an empty grep
  # exits 1 -- which under `set -euo pipefail` ends the script silently.
  _vllm_keys=$(grep -oE '^VLLM_[A-Z0-9_]+' "$PROFILE_ENV" 2>/dev/null | sort -u || true)
  _caller=""
  for _v in IMAGE MODEL_PATH SERVED_NAME $_vllm_keys; do
    if [ -n "${!_v:-}" ]; then _caller="$_caller $_v=$(printf %q "${!_v}")"; fi
  done
  # shellcheck disable=SC1090
  . "$PROFILE_ENV"
  [ -n "$_caller" ] && eval "$_caller"
  IMAGE="${IMAGE:-${PROFILE_IMAGE:-}}"
  MODEL_PATH="${MODEL_PATH:-${PROFILE_MODEL_PATH:-}}"
  SERVED_NAME="${SERVED_NAME:-${PROFILE_SERVED_NAME:-}}"
fi

IMAGE="${IMAGE:-aidendle94/sparkrun-vllm-ds4-gb10:production-hybrid-1.6}"
EXPECTED_IMAGE_ID="sha256:b763d81b57f7611378a514fa0faf859c3b0d0ec1010f8c5115bea11a60d49ec3"
HEAD_IP=10.10.10.2
WORKERS="10.10.10.3:1 10.10.10.1:2 10.10.10.4:3"
MODEL_PATH="${MODEL_PATH:-/home/choiceoh/models/DeepSeek-V4-Flash-0731}"
SERVED_NAME="${SERVED_NAME:-deepseek-v4-flash}"
TP_SIZE=4
GPU_MEM="${GPU_MEM:-0.60}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-430000}"
# MAX_NUM_SEQS 16->32 adopted 2026-08-11: aggregate decode keeps scaling
# past the old cap (C=16 290 -> C=24 340 -> C=32 386 tok/s, +33%, raw acc
# flat ~31-32%, C=16 identical across caps = zero cap overhead). M=32x6=192
# stays inside GRAPH_CAP=256 (ceiling ~C=42); per-stream latency at C=32 is
# ~12 tok/s — an admission cap, so low-concurrency behavior is unchanged.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
MAX_NUM_BATCHED="${MAX_NUM_BATCHED:-4096}"
GRAPH_CAP=256
SPEC_TOKENS="${SPEC_TOKENS:-5}"
MODEL_VOCAB_SIZE=129280
FP8HEAD="${FP8HEAD:-0}"
# MARKOV_TOPK=512 adopted 2026-08-10: two independent boots reproduce
# +1.5% C=1 decode (acc-normalized) with residual-sigma collapse; trace shows
# the markov W2 GEMV leaving the step budget (-0.95 ms/step in the gemm
# bucket). Quality 9/9 + 256K needles 3/3 + prefill unaffected. 0 disarms.
MARKOV_TOPK="${MARKOV_TOPK:-512}"
# GATEFUSE (VLLM_DSV4_GATE_FUSED) adopted 2026-08-10, default 1: fused
# small-M MoE router gate (triton SK=1, fp32-out direct) replaces the Tier-4
# bf16 F.linear + splitKreduce + fp32-cast chain on GB10 (sm_121 fails every
# GateLinear specialized-tier device gate). Bracket: C=1 acc50-normalized
# 62.11 / 64.28 / 62.14 = +3.5%; quality 9/9; greedy divergence 93 chars =
# natural boot-to-boot 96 (indistinguishable); prefill neutral (M>32 path
# untouched). GATEFUSE=0 disarms.
# REFINE (VLLM_DSPARK_REFINE_PASS, default 0, MEASURED-REJECTED 2026-08-12):
# two-pass draft refinement — pass-1 tokens replace the noise slots and the
# backbone + sequential sampling re-run with the SAME Gumbel keys. The comment
# claimed "target invariant by construction, only acceptance moves [up]" — the
# A/B measured the OPPOSITE: tg128 median 58.6->36.5 tok/s (-38%), raw accept
# 21.4%->9.0% (halved), all positions worse. Worker logs confirm the feature
# activates ("DSpark(v2) two-pass ... enabled") but APIServer warns "Unknown
# vLLM env VLLM_DSPARK_REFINE_PASS" => half-wired in production-hybrid-1.6; the
# refinement corrupts drafts (coupling doesn't hold) rather than improving them.
# DO NOT enable on this image. Re-test with scratchpad refine_ab.py if a future
# author image reimplements it. (MEASUREMENTS: REFINE_PASS A/B entry.)
REFINE="${REFINE:-0}"
# MARKOV_SIDELOAD (VLLM_DSPARK_MARKOV_SIDELOAD, default empty, UNMEASURED):
# absolute path (under /home/choiceoh/models, ro-mounted on all nodes) to a
# tools/markov_refit.py payload replacing the draft Markov W1/W2. Proposal-q
# only — verification distribution untouched.
MARKOV_SIDELOAD="${MARKOV_SIDELOAD:-}"
V2RUNNER="${V2RUNNER:-1}"
case "$FP8HEAD" in
  0|1) ;;
  *) echo "ABORT: FP8HEAD must be 0 or 1 (got $FP8HEAD)"; exit 2 ;;
esac
case "$REFINE" in
  0|1) ;;
  *) echo "ABORT: REFINE must be 0 or 1 (got $REFINE)"; exit 2 ;;
esac
if [ -n "$MARKOV_SIDELOAD" ]; then
  case "$MARKOV_SIDELOAD" in
    /home/choiceoh/models/*) ;;
    *)
      echo "ABORT: MARKOV_SIDELOAD must be an absolute path under /home/choiceoh/models (got $MARKOV_SIDELOAD)"
      exit 2 ;;
  esac
  case "$MARKOV_SIDELOAD" in
    *[!A-Za-z0-9_./-]*|*..*)
      echo "ABORT: unsafe character in MARKOV_SIDELOAD: $MARKOV_SIDELOAD"
      exit 2 ;;
  esac
fi
if [[ ! "$MARKOV_TOPK" =~ ^(0|[1-9][0-9]*)$ ]] \
   || ((${#MARKOV_TOPK} > ${#MODEL_VOCAB_SIZE})); then
  echo "ABORT: MARKOV_TOPK must be 0 or an integer in [1,$MODEL_VOCAB_SIZE] (got $MARKOV_TOPK)"
  exit 2
fi
MARKOV_TOPK=$((10#$MARKOV_TOPK))
if ((MARKOV_TOPK > MODEL_VOCAB_SIZE)); then
  echo "ABORT: MARKOV_TOPK exceeds model vocab ($MARKOV_TOPK > $MODEL_VOCAB_SIZE)"
  exit 2
fi
case "$V2RUNNER" in
  0|1) ;;
  *) echo "ABORT: V2RUNNER must be 0 or 1 (got $V2RUNNER)"; exit 2 ;;
esac
if ((FP8HEAD == 1 || MARKOV_TOPK > 0 || REFINE == 1)) \
    || [ -n "$MARKOV_SIDELOAD" ]; then
  if [ "$V2RUNNER" != "1" ]; then
    echo "ABORT: DSpark FP8/top-k/refine/sideload overlays require V2RUNNER=1"
    exit 2
  fi
fi
# VLLM_DSPARK_IMPL: the production-hybrid-1.6 IMAGE bakes upstream as its ENV
# default (docker inspect evidence, 2026-08-10) — production has always run
# the upstream (registry-loaded) DSpark impl; the "fork" reading of an unset
# env only applies when the image ENV is absent. Passing it explicitly here
# documents that reality and pins it against future image drift. The speed
# knobs require it; forcing fork with a knob armed aborts.
DSPARK_IMPL="${DSPARK_IMPL:-upstream}"
if ((FP8HEAD == 1 || MARKOV_TOPK > 0 || REFINE == 1)) \
    || [ -n "$MARKOV_SIDELOAD" ]; then
  if [ "$DSPARK_IMPL" != "upstream" ]; then
    echo "ABORT: FP8HEAD/MARKOV_TOPK/REFINE/MARKOV_SIDELOAD require the upstream DSpark impl (got DSPARK_IMPL=$DSPARK_IMPL)"
    exit 2
  fi
fi
case "$DSPARK_IMPL" in
  fork|upstream) ;;
  *) echo "ABORT: DSPARK_IMPL must be fork or upstream (got $DSPARK_IMPL)"; exit 2 ;;
esac
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
-e VLLM_USE_V2_MODEL_RUNNER=$V2RUNNER -e VLLM_DSPARK_IMPL=$DSPARK_IMPL \
-e VLLM_USE_FLASHINFER_SAMPLER=1 -e VLLM_USE_B12X_MOE=0 \
-e VLLM_DSPARK_REPLICATE_MARKOV_W1=1 -e VLLM_DSV4_COMPRESSOR_FP8=${COMPRESSOR_FP8:-1} -e VLLM_DSV4_TARGET_LM_HEAD_FP8=${TGT_HEAD_FP8:-1} \
-e VLLM_DSV4_GATE_FUSED=${GATEFUSE:-1} \
-e VLLM_DSV4_MHC_SMALLM_TUNED=${MHCTUNE:-1} -e VLLM_DSV4_MHC_TUNED_R2=${MHCTUNE2:-1} -e VLLM_DSV4_MHC_BIGFUSE_TUNED=${MHCTUNE3:-1} \
-e VLLM_DSV4_ONESHOT_AR=${ONESHOT:-1} -e VLLM_DSV4_ONESHOT_SHADOW=${OSAR_SHADOW:-0} \
-e VLLM_DSV4_FREE_BF16_LM_HEAD=${HEADFREE:-0} -e VLLM_DSV4_FREE_BF16_COMPRESSOR=${COMPFREE:-0} \
-e VLLM_DSPARK_FP8_DRAFT_HEAD=$FP8HEAD -e VLLM_DSPARK_DRAFT_TOPK=$MARKOV_TOPK \
-e VLLM_DSPARK_REFINE_PASS=$REFINE -e VLLM_DSPARK_MARKOV_SIDELOAD=$MARKOV_SIDELOAD \
-e TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200 -e TORCH_NCCL_DUMP_ON_TIMEOUT=0 -e TORCH_NCCL_ASYNC_ERROR_HANDLING=0 \
-e MODEL_PATH=$MODEL_PATH -e SERVED_MODEL_NAME=$SERVED_NAME -e PORT=8000 -e TP_SIZE=$TP_SIZE \
$(for _k in ${_vllm_keys:-}; do if [ -n "${!_k:-}" ]; then printf -- "-e %s=%s " "$_k" "${!_k}"; fi; done) \
-e GPU_MEM=$GPU_MEM -e SPEC_TOKENS=$SPEC_TOKENS -e TEMPERATURE=${TEMP:-0.8} -e REASONING_EFFORT=${EFFORT:-} \
-e MAX_MODEL_LEN=$MAX_MODEL_LEN -e MAX_NUM_SEQS=$MAX_NUM_SEQS -e MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED \
-e GRAPH_CAP=$GRAPH_CAP -e ASYNC_SCHED=1 -e MASTER_ADDR=$HEAD_IP -e MOE=${MOE:-b12x} -e IDXFREQ=${IDXFREQ:-} -e VLLM_DSV4_INDEXER_SP=${IDXSP:-1} -e VLLM_B12X_INDEXER_STREAM=${IDXSTREAM:-} -e VLLM_B12X_KV_STREAM=${KVSTREAM:-} -e VLLM_B12X_MLA_CKV_GATHER=${CKVG:-} -e VLLM_B12X_CUDAGRAPH_PIECEWISE_PREWARM=${PREWARM:-0} \
-e VLLM_TORCH_PROFILER_DIR=/prof \
-e VLLM_SERVER_DEV_MODE=${DEVMODE:-1} -e VLLM_ENGINE_READY_TIMEOUT_S=3600"
# The rowwise FP8 experiment and the image's older DeepGEMM FP8 copy must not
# coexist. Top-k global row gathers require W2 to be full on every TP rank.
if ((FP8HEAD == 1)); then
  ENVV+=" -e VLLM_DSPARK_FP8_LM_HEAD=0"
fi
if ((MARKOV_TOPK > 0)); then
  ENVV+=" -e VLLM_DSPARK_REPLICATE_MARKOV_W2=1"
fi
# TEMPERATURE (TEMP knob, default 0.8): the SERVING sampling default is set
# via --override-generation-config below — measured on hard prose: acceptance
# 15.9->18.9%, decode +7% vs 0.95; 0.7 adds nothing (saturation). Without the
# override, the effective default was generation_config's t1.0/top_p1.0
# (battery W0==W1 evidence). Explicit client temperature still wins.
# Effective default top_p stays 1.0 — production has always run 1.0.
# The old default-chat-template-kwargs temperature/top_p were REMOVED
# 2026-08-11: the dsv4 template wrapper reads neither kwarg (encoding audit
# vs the checkpoint's reference encoding/) — dead config, zero prompt-byte
# effect, never sampling params.
# EFFORT knob (REASONING_EFFORT template kwarg, default EMPTY = reference
# "low" = no effort preamble = byte-identical prompts to before): the stock
# wrapper silently no-opped "high" (only max/xhigh injected anything, and
# with the reference *high* text); the tok_deepseek_v4* overlays restore the
# reference low/high/max levels. EFFORT=high|max now injects the reference
# reasoning-effort preamble at conversation start — A/B (quality gate +
# decode bracket) before adopting a non-empty default: longer thinking,
# different prompt prefix bytes.
# The 2026-08-09 incident kill-switches (TRIMIDX/C4AREUSE/SPFAST) are gone:
# all three were measured and not adopted (rejected / neutral / neutral,
# MEASUREMENTS.md 08-09), and their overlay code paths were removed 08-11.
# Memory-reclaim knobs (default OFF, unmeasured — flip ONE per boot):
# HEADFREE=1 drops the bf16 lm_head shard (~265MB/rank) once the load-time
# fp8 copies exist; COMPFREE=1 drops the attention-compressor bf16
# originals (~0.7GB/rank) at their lazy fp8 quant (memory-profile pass).
# Both fail closed at boot while any bf16 consumer remains (they require
# TGT_HEAD_FP8=1 / COMPRESSOR_FP8=1 — the current defaults). This is the
# KV-capacity axis, not speed: verify via boot-log "GPU KV cache size"
# growth (same GPU_MEM!) + 9/9 + bench-dec/prefill no-regress
# (MEASUREMENTS.md A/B queue #5/#6).
# VLLM_ENGINE_READY_TIMEOUT_S=3600: the image default is 600s (envs.py:27),
# below our cold-recompile boot times (supervisor history has
# 'boot grace exceeded'); dead engines still fail fast via the supervisor.
# NOTE: pass_config's fuse_gemm_comms:true is DECORATIVE — the fork resolves
# it to False (boot log evidence); fuse_norm_quant/fuse_act_quant
# auto-enable, and fuse_attn_quant forces splitting_ops=[] so the effective
# cudagraph mode is FULL_DECODE_ONLY (no piecewise prefill graphs). A/B
# candidates in MEASUREMENTS.md.
# IDXFREQ default 4->6 adopted 2026-08-11: 128K decode +7~11% (two boots,
# matched-seed 3x3 vs freq=4 baseline: 66.0 -> 70.9 / 73.4 avg), 2K/32K
# neutral, TTFT unchanged, retrieval 9/9 (2K/32K/128K). 256K needles NOT
# re-verified at 6 (freq=4-era 3/3 only). IDXFREQ=4 restores the old default.
# The /start_profile,/stop_profile routes are attached ONLY when the serve
# CLI passes --profiler-config with a non-null profiler (see
# entrypoints/serve/profile/api_router.py attach_router). Env vars alone
# (VLLM_TORCH_PROFILER_DIR / VLLM_SERVER_DEV_MODE) do NOT attach them.
# VLLM_TORCH_PROFILER_DIR arms the worker torch profiler; the HTTP routes
# themselves live in entrypoints/serve/profile and are gated by
# VLLM_SERVER_DEV_MODE (fork keeps them dev-only). Both are needed:
# without DEV_MODE the endpoints 404. DEVMODE=0 disables the dev routes.
# (original note) VLLM_TORCH_PROFILER_DIR only ENABLES the /start_profile,/stop_profile
# endpoints (bench/profile-step.py); zero overhead until a capture is started.
# NOTE: SP/EPFLAG knobs removed — b12x MoE is TP-only, so the enable_sp
# compilation pass (#46789) and EP/DP are structurally impossible on this stack.
RDMA_FLAGS="--device=/dev/infiniband:/dev/infiniband --cap-add=IPC_LOCK --ulimit memlock=-1:-1"
COMMON="--runtime nvidia --gpus all --network host --ipc host --restart unless-stopped"

MANIFEST_NAME=manifest.tsv
HEAD_OV=/home/choiceoh/hybrid-stack/overlay-b12x
OVERLAY_MANIFEST="$HEAD_OV/$MANIFEST_NAME"

load_overlay_manifest() {
  local manifest="$1" source target base_contract extra seen
  [ -f "$manifest" ] || { echo "ABORT: overlay manifest missing ($manifest)"; exit 1; }
  OVFILES=()
  OVTARGETS=()
  OVBASES=()
  while IFS=$'\t' read -r source target base_contract extra \
      || [ -n "${source:-}${target:-}${base_contract:-}${extra:-}" ]; do
    [[ -z "$source" || "$source" == \#* ]] && continue
    [ -n "$target" ] && [ -n "$base_contract" ] && [ -z "${extra:-}" ] \
      || { echo "ABORT: malformed overlay manifest row: $source $target $base_contract ${extra:-}"; exit 1; }
    case "$source" in
      *[!A-Za-z0-9._-]*|.*)
        echo "ABORT: unsafe overlay source in manifest: $source"; exit 1 ;;
    esac
    case "$target" in
      /opt/venv/lib/python3.12/site-packages/vllm/*) ;;
      *) echo "ABORT: unsafe overlay target in manifest: $target"; exit 1 ;;
    esac
    case "$target" in
      *[!A-Za-z0-9_./-]*)
        echo "ABORT: unsafe character in overlay target: $target"; exit 1 ;;
    esac
    if [ "$base_contract" != "absent" ] \
        && [[ ! "$base_contract" =~ ^[0-9a-f]{64}$ ]]; then
      echo "ABORT: invalid base preimage contract for $source: $base_contract"
      exit 1
    fi
    for seen in "${OVFILES[@]}"; do
      [ "$seen" != "$source" ] \
        || { echo "ABORT: duplicate overlay source in manifest: $source"; exit 1; }
    done
    for seen in "${OVTARGETS[@]}"; do
      [ "$seen" != "$target" ] \
        || { echo "ABORT: duplicate overlay target in manifest: $target"; exit 1; }
    done
    OVFILES+=("$source")
    OVTARGETS+=("$target")
    OVBASES+=("$base_contract")
  done < "$manifest"
  ((${#OVFILES[@]} > 0)) || { echo "ABORT: overlay manifest is empty"; exit 1; }
}
load_overlay_manifest "$OVERLAY_MANIFEST"

mounts_for() {
  local ov="$1" i
  printf '%s' "-v /home/choiceoh/models:/home/choiceoh/models:ro \
-v /home/choiceoh/.cache/huggingface:/root/.cache/huggingface \
-v /home/choiceoh/.cache/vllm-hybrid:/cache \
-v /home/choiceoh/.cache/tilelang-hybrid:/root/.tilelang \
-v /home/choiceoh/vllm-prof:/prof"
  for i in "${!OVFILES[@]}"; do
    printf ' -v %s-b12x/%s:%s:ro' "$ov" "${OVFILES[$i]}" "${OVTARGETS[$i]}"
  done
  printf '\n'
}

echo "=== [0/5] preflight: image + model + overlays on all nodes ==="
HID=$(docker image inspect "$IMAGE" --format '{{.Id}}')
[ "$HID" = "$EXPECTED_IMAGE_ID" ] || {
  echo "ABORT: production-hybrid-1.6 image ID drifted"
  echo "  expected: $EXPECTED_IMAGE_ID"
  echo "  actual:   $HID"
  exit 1
}
# The manifest pins the unmounted base preimage for every replaced file and
# explicitly marks newly-created targets absent. Check before any bind mount
# can hide the image bytes.
BASE_SPECS=()
for i in "${!OVFILES[@]}"; do
  BASE_SPECS+=("${OVBASES[$i]}:${OVTARGETS[$i]}")
done
docker run --rm --entrypoint /bin/sh "$IMAGE" -c '
  set -eu
  for spec do
    contract=${spec%%:*}
    target=${spec#*:}
    if [ "$contract" = absent ]; then
      if [ -e "$target" ] || [ -L "$target" ]; then
        echo "ABORT: expected absent base target exists: $target" >&2
        exit 1
      fi
      continue
    fi
    if [ ! -f "$target" ] || [ -L "$target" ]; then
      echo "ABORT: base preimage is not a regular non-symlink file: $target" >&2
      exit 1
    fi
    actual=$(sha256sum "$target")
    actual=${actual%% *}
    if [ "$actual" != "$contract" ]; then
      echo "ABORT: base preimage skew: $target" >&2
      echo "  expected: $contract" >&2
      echo "  actual:   $actual" >&2
      exit 1
    fi
  done
' sh "${BASE_SPECS[@]}" \
  || { echo "ABORT: unmounted base source attestation failed"; exit 1; }
# The manifest is the only overlay inventory. Hash it together with every
# listed source so list skew, missing files, and byte skew all fail closed.
HEAD_OVSUM=$(cd "$HEAD_OV" && sha256sum "$MANIFEST_NAME" "${OVFILES[@]}") \
  || { echo "ABORT: manifest/overlays missing on head ($HEAD_OV)"; exit 1; }
if [ -n "$MARKOV_SIDELOAD" ] && [ ! -f "$MARKOV_SIDELOAD" ]; then
  echo "ABORT: MARKOV_SIDELOAD missing on head ($MARKOV_SIDELOAD)"
  exit 1
fi
for w in $WORKERS; do
  ip=${w%%:*}
  WID=$(ssh $SSHOPT choiceoh@$ip "docker image inspect $IMAGE --format '{{.Id}}'" 2>/dev/null || true)
  [ "$WID" = "$HID" ] || { echo "ABORT: image missing/skewed on $ip"; exit 1; }
  WOVSUM=$(ssh $SSHOPT choiceoh@$ip "cd $(overlay_dir $ip)-b12x && sha256sum $MANIFEST_NAME ${OVFILES[*]}" 2>/dev/null || true)
  [ "$WOVSUM" = "$HEAD_OVSUM" ] || { echo "ABORT: overlay missing/skewed on $ip ($(overlay_dir $ip)-b12x)"; exit 1; }
  ssh $SSHOPT choiceoh@$ip "test -f $MODEL_PATH/config.json && mkdir -p ~/.cache/huggingface ~/.cache/vllm-hybrid ~/.cache/tilelang-hybrid ~/vllm-prof" \
    || { echo "ABORT: model/caches missing on $ip"; exit 1; }
  if [ -n "$MARKOV_SIDELOAD" ]; then
    ssh $SSHOPT choiceoh@$ip "test -f $MARKOV_SIDELOAD" \
      || { echo "ABORT: MARKOV_SIDELOAD missing on $ip ($MARKOV_SIDELOAD)"; exit 1; }
  fi
done
echo "preflight OK (${HID:0:19}, ${#OVFILES[@]} base-attested overlays + manifest in sync x4)"

echo "=== [1/5] retire old vllm-dsv4 containers (free memory) ==="
docker rm -f vllm-dsv4 2>/dev/null || true
for w in $WORKERS; do ip=${w%%:*}; ssh $SSHOPT choiceoh@$ip "docker rm -f vllm-dsv4-worker 2>/dev/null; true"; done
sleep 3

echo "=== [1.5/5] drop reclaimable page cache on all nodes (UMA memory check) ==="
# GB10 UMA: vLLM's startup free-memory probe treats reclaimable page cache as
# unavailable, so the KV pool size varies boot-to-boot (4.22->5.34 GiB spreads
# reported on identical config — NVIDIA forum #378890). Best-effort drop +
# free log makes KV capacity deterministic and measurable across restarts.
DROPCMD='sync; if sudo -n true 2>/dev/null; then echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null; echo "caches dropped"; else echo "no passwordless sudo - skipped"; fi; free -g | head -2'
bash -c "$DROPCMD" 2>&1 | sed 's/^/  head: /'
for w in $WORKERS; do ip=${w%%:*}; ssh $SSHOPT choiceoh@$ip "$DROPCMD" 2>&1 | sed "s/^/  $ip: /"; done

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
echo "[hy4] DSpark speed FP8_HEAD=${VLLM_DSPARK_FP8_DRAFT_HEAD:-0} TOPK=${VLLM_DSPARK_DRAFT_TOPK:-0} REFINE=${VLLM_DSPARK_REFINE_PASS:-0} SIDELOAD=${VLLM_DSPARK_MARKOV_SIDELOAD:-none}"
if [ "${ASYNC_SCHED:-1}" = "1" ]; then ASYNC_ARG="--async-scheduling"; else ASYNC_ARG="--no-async-scheduling"; fi
exec vllm serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME:-deepseek-v4-flash}" \
  --profiler-config "{\"profiler\": \"torch\", \"torch_profiler_dir\": \"/prof\", \"torch_profiler_with_stack\": false}" --host 0.0.0.0 --port "${PORT}" --trust-remote-code --hf-overrides "{\"use_index_cache\": true, \"index_topk_freq\": ${IDXFREQ:-6}}" \
  --kv-cache-dtype fp8 --block-size 256 --load-format auto \
  --tensor-parallel-size "${TP_SIZE}" \
  --gpu-memory-utilization "${GPU_MEM}" \
  --max-model-len "${MAX_MODEL_LEN}" --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --max-cudagraph-capture-size "${GRAPH_CAP}" \
  --compilation-config "{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"custom_ops\":[\"all\"],\"pass_config\":{\"fuse_gemm_comms\":true,\"fuse_allreduce_rms\":true,\"fuse_rope_kvcache_cat_mla\":true,\"fuse_attn_quant\":true}}" \
  ${ASYNC_ARG} --no-scheduler-reserve-full-isl \
  --enable-chunked-prefill --enable-prefix-caching --enable-flashinfer-autotune \
  --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --override-generation-config "{\"temperature\": ${TEMPERATURE}}" \
  --default-chat-template-kwargs.thinking=true \
  ${REASONING_EFFORT:+--default-chat-template-kwargs.reasoning_effort=${REASONING_EFFORT}} \
  --attention-backend FLASHINFER_MLA_SPARSE_DSV4 --moe-backend "${MOE:-b12x}" \
  --disable-custom-all-reduce \
  --nnodes 4 --node-rank "${NODE_RANK}" --master-addr "${MASTER_ADDR}" --master-port 25000 \
  ${HEADLESS:+--headless} \
  --speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":${SPEC_TOKENS},\"draft_sample_method\":\"probabilistic\"}"
SERVEEOF
chmod +x /tmp/serve-hy4.sh

echo "=== [3/5] create containers ==="
mkdir -p ~/.cache/huggingface ~/.cache/vllm-hybrid ~/.cache/tilelang-hybrid ~/vllm-prof
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
