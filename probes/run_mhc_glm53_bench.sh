#!/usr/bin/env bash
# Run the GLM53 MHC probe against the exact composed overlay sources.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
IMAGE=${IMAGE:-glm53:v13-b12x}
BUILD="$REPO/build/glm53"
MANIFEST="$BUILD/manifest.tsv"

bash "$REPO/launchers/compose-overlays.sh" glm53 >&2

mounts=()
for source in tilelang.py tilelang_kernels.py; do
  target=$(awk -F '\t' -v source="$source" '$1 == source {print $2}' "$MANIFEST")
  [ -n "$target" ] \
    || { echo "ABORT: $source is missing from $MANIFEST" >&2; exit 1; }
  [ "${target#*$'\n'}" = "$target" ] \
    || { echo "ABORT: duplicate $source rows in $MANIFEST" >&2; exit 1; }
  [ -f "$BUILD/$source" ] \
    || { echo "ABORT: composed source missing: $BUILD/$source" >&2; exit 1; }
  mounts+=(--mount "type=bind,src=$BUILD/$source,dst=$target,readonly")
done

exec docker run --rm --gpus all \
  --mount "type=bind,src=$REPO,dst=/repo,readonly" \
  "${mounts[@]}" \
  -e STKERNEL_MHC_OVERLAY_BUILD=/repo/build/glm53 \
  "$IMAGE" python3 /repo/probes/mhc_glm53_bench.py "$@"
