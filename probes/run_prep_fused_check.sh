#!/usr/bin/env bash
# glm53_prep_fused numerics probe in a fresh container (srv4 scratch, never the
# serving one). Composes the glm53 overlay and bind-mounts EVERY row of the
# composed manifest at its real image path -- the fused prep module imports
# through vllm.models.glm5next.nvidia, which pulls the whole mounted model
# tree, so a partial mount list would silently test the image's own files.
#
#   IMAGE=glm53:sm121-fi618 bash probes/run_prep_fused_check.sh [--trials 60]
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
IMAGE=${IMAGE:-glm53:v13-b12x}
BUILD="$REPO/build/glm53"
MANIFEST="$BUILD/manifest.tsv"

bash "$REPO/launchers/compose-overlays.sh" glm53 >&2

mounts=()
while IFS=$'\t' read -r source target contract || [ -n "${source:-}" ]; do
  [[ -z "$source" || "$source" == \#* ]] && continue
  [ -f "$BUILD/$source" ] \
    || { echo "ABORT: composed source missing: $BUILD/$source" >&2; exit 1; }
  mounts+=(--mount "type=bind,src=$BUILD/$source,dst=$target,readonly")
done < "$MANIFEST"

envs=()
for v in $(compgen -v | grep '^VLLM_GLM53_'); do envs+=(-e "$v=${!v}"); done

exec docker run --rm --gpus all --entrypoint /bin/bash \
  --mount "type=bind,src=$REPO,dst=/repo,readonly" \
  "${envs[@]}" \
  "${mounts[@]}" \
  "$IMAGE" -lc "python3 /repo/probes/prep_fused_check.py $*" _ "$@"
