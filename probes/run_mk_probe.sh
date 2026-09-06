#!/usr/bin/env bash
# Run ANY offline probe against the composed glm53 overlay sources, in a fresh
# container -- the general form of run_megakernel_bench.sh (which is pinned to
# the ladder-step-2 probe and its six files). Mounts the megakernel driver AND
# the b12x MoE files, so a probe can drive the served MoE path and the MK lane
# side by side.
#
#   bash probes/run_mk_probe.sh probes/<probe>.py [probe args...]
#
# Never the serving container (docker ps must be empty of glm53*), never while
# a TP=4 boot is up on any host: MEASUREMENTS 11차 lost two measurements to a
# serving boot on another node. VLLM_GLM53_*/VLLM_DSV4_* set in this shell are
# forwarded (a probe arms itself with setdefault, so a forwarded knob WINS).
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
PROBE=${1:?usage: run_mk_probe.sh probes/<probe>.py [args...]}
shift
[ -f "$REPO/$PROBE" ] || { echo "ABORT: no such probe: $REPO/$PROBE" >&2; exit 1; }

PROFILE=${PROFILE:-glm53}
ENVFILE="$REPO/profiles/$PROFILE.env"
[ -f "$ENVFILE" ] || { echo "ABORT: no such profile: $ENVFILE" >&2; exit 1; }
PROFILE_IMAGE=""
TARGET_PREFIX="/opt/venv/lib/python3.12/site-packages/"
eval "$(
  # shellcheck disable=SC1090
  . "$ENVFILE"
  printf 'PROFILE_IMAGE=%q\nTARGET_PREFIX=%q\n' \
    "${PROFILE_IMAGE:-}" "${TARGET_PREFIX:-/opt/venv/lib/python3.12/site-packages/}"
)"
IMAGE=${IMAGE:-$PROFILE_IMAGE}
[ -n "$IMAGE" ] || { echo "ABORT: $PROFILE names no image" >&2; exit 1; }

BUILD="$REPO/build/$PROFILE"
MANIFEST="$BUILD/manifest.tsv"
bash "$REPO/launchers/compose-overlays.sh" "$PROFILE" >&2

# The megakernel driver and everything its arm touches, plus the served MoE
# path (vLLM's b12x experts, flashinfer's wrapper and its sm12x dispatch).
sources=(glm53_megakernel.py glm53_megakernel.cu
         glm5next_kda.py kda.py chunk_delta_h.py
         tilelang.py tilelang_kernels.py glm53_fp8_dense.py glm53_nvfp4_scale.py
         glm53_nvfp4_bproj.py
         flashinfer_b12x_moe.py b12x_moe.py moe_dispatch.py moe_micro_kernel.py
         moe_dynamic_prefill.py moe_dynamic_prefill_n128.py
         parallel_state.py glm53_prefill_collectives.py
         moe_static_common.py moe_static_kernel_v4.py moe_sf_pack.py
         moe_static_kernel_v5.py moe_dynamic_gated_tiled.py)
mounts=()
for source in "${sources[@]}"; do
  target=$(awk -F '\t' -v source="$source" '$1 == source {print $2}' "$MANIFEST")
  [ -n "$target" ] || { echo "ABORT: $source is missing from $MANIFEST" >&2; exit 1; }
  [ "${target#*$'\n'}" = "$target" ] \
    || { echo "ABORT: duplicate $source rows in $MANIFEST" >&2; exit 1; }
  [ -f "$BUILD/$source" ] \
    || { echo "ABORT: composed source missing: $BUILD/$source" >&2; exit 1; }
  mounts+=(--mount "type=bind,src=$BUILD/$source,dst=$target,readonly")
done

case "${VLLM_GLM53_MEGAKERNEL-1}" in
  0|""|false|FALSE|no|off)
    echo "ABORT: VLLM_GLM53_MEGAKERNEL='${VLLM_GLM53_MEGAKERNEL}' would disarm the probe; unset it" >&2
    exit 1 ;;
esac

envs=(-e "MK_PKG_PATH=${TARGET_PREFIX%/}")
# A runner-owned report directory is the only extra writable evidence mount.
# Preserve the fresh challenge through the container boundary; ordinary manual
# probes keep their existing behavior.
if [ -n "${FLEET_PROBE_REPORT:-}" ]; then
  report_dir=$(dirname "$FLEET_PROBE_REPORT")
  mounts+=(--mount "type=bind,src=$report_dir,dst=/fleet-report")
  envs+=(-e FLEET_PROBE_REPORT=/fleet-report/probe-report.json)
  for v in FLEET_PROBE_NONCE FLEET_PROBE_BINDING FLEET_EXPERIMENT_ID; do
    envs+=(-e "$v=${!v}")
  done
fi
# PROBE_CACHE=1: the serving boot's persistent caches (the launcher mounts
# $HOME/glm53-cache at /cache and points flashinfer / triton / vLLM at it),
# so a probe that JIT-compiles -- probes/nvfp4_prefill_warm.py builds the
# cutlass mm_fp4 kernels the NVFP4P boot otherwise compiles inside the serve
# process, where it cost srv3 its memory (32차) -- warms the cache the next
# serve boot reads. MAX_JOBS bounds the parallel nvcc jobs on this host.
if [ "${PROBE_CACHE:-0}" = 1 ]; then
  CACHE_HOST=${CACHE_HOST:-$HOME/glm53-cache}
  [ -d "$CACHE_HOST" ] || { echo "ABORT: PROBE_CACHE=1 but no $CACHE_HOST" >&2; exit 1; }
  mounts+=(--mount "type=bind,src=$CACHE_HOST,dst=/cache")
  envs+=(-e FLASHINFER_WORKSPACE_BASE=/cache -e VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=/cache/flashinfer_autotune
         -e TRITON_CACHE_DIR=/cache/triton -e VLLM_CACHE_ROOT=/cache/vllm -e "MAX_JOBS=${MAX_JOBS:-4}")
fi
_fwd=""
for v in $(compgen -v | grep -E '^VLLM_(GLM53|DSV4)_'); do
  envs+=(-e "$v=${!v}")
  _fwd="$_fwd $v=${!v}"
done
echo "profile=$PROFILE image=$IMAGE probe=$PROBE files=${#sources[@]} args=$*" >&2
echo "forwarded:${_fwd:- (none)}" >&2

cmd="python3 /repo/$PROBE"
for a in "$@"; do cmd="$cmd $(printf %q "$a")"; done

# MK_PROBE_NO_GPU=1: no GPU in the container -- for the cute-dsl compile
# checks (probes/b12x_static_compile_check.py) that build a kernel to its .o
# on the CPU; CUTE_DSL_ARCH names the target the device query cannot
gpu_args=(--gpus all)
if [ "${MK_PROBE_NO_GPU:-0}" = 1 ]; then
  gpu_args=()
  envs+=(-e "CUTE_DSL_ARCH=${CUTE_DSL_ARCH:-sm_121a}" -e MK_PROBE_NO_GPU=1)
fi
# MK_PROBE_DOCKER_ARGS: extra docker run flags (e.g. --cpuset-cpus 16-19 to
# keep a CPU-only compile check off the cores a serving worker uses)
# shellcheck disable=SC2206
extra_args=(${MK_PROBE_DOCKER_ARGS:-})
exec docker run --rm "${gpu_args[@]}" "${extra_args[@]}" --entrypoint /bin/bash \
  --mount "type=bind,src=$REPO,dst=/repo,readonly" \
  "${envs[@]}" "${mounts[@]}" \
  "$IMAGE" -lc "$cmd"
