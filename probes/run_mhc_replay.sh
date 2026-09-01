#!/usr/bin/env bash
# Run the GLM53 megakernel probe against the exact composed overlay sources.
# Ladder step 2 of overlay/modules/glm53_megakernel/README.md -- srv4 scratch
# container only, never the serving one.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
IMAGE=${IMAGE:-glm53:v13-b12x}
BUILD="$REPO/build/glm53"
MANIFEST="$BUILD/manifest.tsv"

bash "$REPO/launchers/compose-overlays.sh" glm53 >&2

# The probe drives the driver (.py) which compiles the .cu next to it, and
# diffs against the TileLang pair + fp8-dense stock path -- so it needs the
# composed driver, the .cu, the mhc tilelang pair and the fp8-dense module
# mounted at their real image paths, exactly as a serving boot would see
# them (the probe also re-checks the mounted SHA-256s against the manifest).
mounts=()
for source in glm53_megakernel.py glm53_megakernel.cu \
              glm5next_kda.py tilelang.py tilelang_kernels.py \
              glm53_fp8_dense.py; do
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
envs=()
for v in $(compgen -v | grep '^VLLM_GLM53_'); do envs+=(-e "$v=${!v}"); done

exec docker run --rm --gpus all --entrypoint /bin/bash \
  --mount "type=bind,src=$REPO,dst=/repo,readonly" \
  "${envs[@]}" \
  "${mounts[@]}" \
  "$IMAGE" -lc "python3 /repo/probes/mhc_replay.py $*" _ "$@"
