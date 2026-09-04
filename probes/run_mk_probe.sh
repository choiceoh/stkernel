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
         glm5next_kda.py tilelang.py tilelang_kernels.py glm53_fp8_dense.py
         flashinfer_b12x_moe.py b12x_moe.py moe_dispatch.py moe_micro_kernel.py)
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
_fwd=""
for v in $(compgen -v | grep -E '^VLLM_(GLM53|DSV4)_'); do
  envs+=(-e "$v=${!v}")
  _fwd="$_fwd $v=${!v}"
done
echo "profile=$PROFILE image=$IMAGE probe=$PROBE files=${#sources[@]} args=$*" >&2
echo "forwarded:${_fwd:- (none)}" >&2

cmd="python3 /repo/$PROBE"
for a in "$@"; do cmd="$cmd $(printf %q "$a")"; done

exec docker run --rm --gpus all --entrypoint /bin/bash \
  --mount "type=bind,src=$REPO,dst=/repo,readonly" \
  "${envs[@]}" "${mounts[@]}" \
  "$IMAGE" -lc "$cmd"
