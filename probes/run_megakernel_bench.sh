#!/usr/bin/env bash
# Run the megakernel probe against the exact composed overlay sources of a
# PROFILE. Ladder step 2 of overlay/modules/glm53_megakernel/README.md --
# srv4 scratch container only, never the serving one.
#
#   bash probes/run_megakernel_bench.sh [--profile glm53|dsv4] [probe args...]
#
# The profile decides four things, and getting any of them from somewhere else
# is how a run measures the wrong stack: the IMAGE, the composed build tree,
# the sources to bind (a profile has no row for a file its model does not
# have), and the package root the probe imports from (GLM's image installs to
# dist-packages, dsv4's to the venv site-packages).
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)

PROFILE=${PROFILE:-glm53}
args=()
while [ $# -gt 0 ]; do
  case "$1" in
    --profile) PROFILE=${2:?--profile needs a value}; shift 2 ;;
    --profile=*) PROFILE=${1#--profile=}; shift ;;
    *) args+=("$1"); shift ;;
  esac
done

ENVFILE="$REPO/profiles/$PROFILE.env"
[ -f "$ENVFILE" ] || { echo "ABORT: no such profile: $ENVFILE" >&2; exit 1; }

# Sourced in a subshell: the profile sets serving knobs (VLLM_*) and pulling
# them into THIS shell would forward the profile's values as the probe's env
# -- the probe arms its own segments and must not inherit a boot's.
# Pre-set both names: the subshell inherits `set -e`, so a profile that fails
# to source produces an empty eval, and without these the next line would die
# on "PROFILE_IMAGE: unbound variable" instead of naming the real problem.
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

# What the probe needs mounted at its real image path, per profile. The probe
# drives the driver (.py) which compiles the .cu next to it and diffs against
# the paths it replaces, so every file a selected segment touches has to be
# here. Naming a file the profile has no row for ABORTS: a silently skipped
# mount would measure the image's stock file while the log says otherwise.
case "$PROFILE" in
  glm53)
    sources=(glm53_megakernel.py glm53_megakernel.cu
             glm5next_kda.py tilelang.py tilelang_kernels.py
             glm53_fp8_dense.py)
    defaults=()
    ;;
  dsv4)
    # MHC is the only segment this model reaches: no linear-attention layer
    # (kda), no e2m1 dense pack (gemm/exact), different MLA geometry. And its
    # stock pair is SWEPT, so the dispatch arm -- the wrapper's own choice --
    # is the reference that matters here; raw stays for kernel comparability.
    sources=(glm53_megakernel.py glm53_megakernel.cu mhc_tilelang.py)
    defaults=(--segments mhc --stock both)
    ;;
  *)
    echo "ABORT: $PROFILE has no megakernel probe recipe" >&2; exit 1 ;;
esac

bash "$REPO/launchers/compose-overlays.sh" "$PROFILE" >&2

mounts=()
for source in "${sources[@]}"; do
  target=$(awk -F '\t' -v source="$source" '$1 == source {print $2}' "$MANIFEST")
  [ -n "$target" ] \
    || { echo "ABORT: $source is missing from $MANIFEST" >&2; exit 1; }
  [ "${target#*$'\n'}" = "$target" ] \
    || { echo "ABORT: duplicate $source rows in $MANIFEST" >&2; exit 1; }
  [ -f "$BUILD/$source" ] \
    || { echo "ABORT: composed source missing: $BUILD/$source" >&2; exit 1; }
  mounts+=(--mount "type=bind,src=$BUILD/$source,dst=$target,readonly")
done

# --entrypoint: the image's default ENTRYPOINT is the vllm server, not a
# shell (the glm53 image layout bit the osar build the same way)
# docker run does NOT inherit the caller's environment. Without this the
# VLLM_GLM53_MK_* knobs set on the host silently do nothing inside the
# container -- a sweep over them then reports four identical numbers and
# reads as "the knob had no effect" rather than "the knob never arrived".
# VLLM_DSV4_* rides along because this lane's STOCK arm is tuned by three of
# them (MHC_SMALLM_TUNED / TUNED_R2 / BIGFUSE_TUNED, all default 1): a sweep
# that wants the untuned reference has to be able to say so.
# The probe arms itself with os.environ.setdefault, so a forwarded knob WINS.
# That is what a sweep wants -- and it is also how a probe run measures
# nothing: profiles/dsv4.env declares VLLM_GLM53_MEGAKERNEL=0, so an operator
# who sourced the profile in this shell (the normal way to get IMAGE and
# MODEL_PATH) would ship the master switch OFF into the container, the driver
# would never arm, and the failure would read as a kernel problem.
case "${VLLM_GLM53_MEGAKERNEL:-1}" in
  0|""|false|FALSE)
    echo "ABORT: VLLM_GLM53_MEGAKERNEL=${VLLM_GLM53_MEGAKERNEL} is set in this" >&2
    echo "       shell and would be forwarded, leaving the probe disarmed." >&2
    echo "       unset it (a profile sourced here is the usual cause):" >&2
    echo "         unset VLLM_GLM53_MEGAKERNEL" >&2
    exit 1 ;;
esac
for _k in VLLM_GLM53_MK_MHC VLLM_GLM53_MK_GEMM VLLM_GLM53_MK_KDA VLLM_GLM53_MK_MLA; do
  case "${!_k:-1}" in
    0|"") echo "WARNING: $_k=0 is set here and will disarm that segment" >&2 ;;
  esac
done

envs=(-e "MK_PKG_PATH=${TARGET_PREFIX%/}")
for v in $(compgen -v | grep -E '^VLLM_(GLM53|DSV4)_'); do
  envs+=(-e "$v=${!v}")
done

echo "profile=$PROFILE image=$IMAGE pkg=${TARGET_PREFIX%/}" \
     "files=$((${#mounts[@]} / 2)) args=${defaults[*]-}${args[*]+ ${args[*]}}" >&2

# Build the command string explicitly. The old form interpolated "$*" into the
# -lc string AND passed "$@" after it; once this wrapper consumed --profile
# with shift, "$*" was empty and every probe argument would have vanished
# silently -- the run would then measure the default segments and read as if
# the caller's flags had had no effect.
cmd="python3 /repo/probes/megakernel_glm53_bench.py"
for a in ${defaults[@]+"${defaults[@]}"} ${args[@]+"${args[@]}"}; do
  cmd="$cmd $(printf %q "$a")"
done

exec docker run --rm --gpus all --entrypoint /bin/bash \
  --mount "type=bind,src=$REPO,dst=/repo,readonly" \
  "${envs[@]}" \
  "${mounts[@]}" \
  "$IMAGE" -lc "$cmd"
