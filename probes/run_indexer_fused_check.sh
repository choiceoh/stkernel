#!/usr/bin/env bash
# glm53_kpool_tail_select fused tail-select probe in a fresh container (never the serving one).
# Composes the glm53 overlay and bind-mounts EVERY row of the composed manifest
# at its real image path -- the fused prep module imports through
# vllm.models.glm5next.nvidia, which pulls the whole mounted model tree, so a
# partial mount list would silently test the image's own files -- but only
# after verifying each row's base preimage against the image exactly as the
# serving launcher does: a mismatch means the overlay would replace code it has
# never seen, and a probe on such a hybrid proves nothing about the fleet.
#
#   bash probes/run_indexer_fused_check.sh [--trials 60]      # IMAGE = profile's
#
# The image must be the profile's (glm53:v13-b12x, on srv2); srv4's
# glm53:sm121-fi618 differs in five mounted files and is refused.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
PROFILE_IMAGE=$(grep -oE '^PROFILE_IMAGE="[^"]+"' "$REPO/profiles/glm53.env" | cut -d'"' -f2)
IMAGE=${IMAGE:-${PROFILE_IMAGE:-glm53:v13-b12x}}
BUILD="$REPO/build/glm53"
MANIFEST="$BUILD/manifest.tsv"

bash "$REPO/launchers/compose-overlays.sh" glm53 >&2

sources=() targets=() bases=()
while IFS=$'\t' read -r source target contract || [ -n "${source:-}" ]; do
  contract=${contract%$'\r'} target=${target%$'\r'}
  [[ -z "$source" || "$source" == \#* ]] && continue
  [ -f "$BUILD/$source" ] \
    || { echo "ABORT: composed source missing: $BUILD/$source" >&2; exit 1; }
  sources+=("$source"); targets+=("$target"); bases+=("$contract")
done < "$MANIFEST"

# base contracts, one container start for every row (launcher lines 270-286)
_got=$(docker run --rm --entrypoint sha256sum "$IMAGE" "${targets[@]}" 2>/dev/null || true)
for _i in "${!sources[@]}"; do
  _have=$(printf '%s\n' "$_got" | awk -v t="${targets[$_i]}" '$2==t{print $1}')
  if [ "${bases[$_i]}" = absent ]; then
    [ -z "$_have" ] \
      || { echo "ABORT: ${sources[$_i]} is declared new but $IMAGE already has ${targets[$_i]}" >&2; exit 1; }
  else
    [ "$_have" = "${bases[$_i]}" ] \
      || { echo "ABORT: base preimage mismatch for ${sources[$_i]} (image ${_have:-missing}, manifest ${bases[$_i]}) -- run on the profile image" >&2; exit 1; }
  fi
done
echo "overlays: ${#sources[@]} base contracts verified against $IMAGE" >&2

mounts=()
for _i in "${!sources[@]}"; do
  mounts+=(--mount "type=bind,src=$BUILD/${sources[$_i]},dst=${targets[$_i]},readonly")
done

envs=()
for v in $(compgen -v | grep '^VLLM_GLM53_'); do envs+=(-e "$v=${!v}"); done

exec docker run --rm --gpus all --entrypoint /bin/bash \
  --mount "type=bind,src=$REPO,dst=/repo,readonly" \
  "${envs[@]}" \
  "${mounts[@]}" \
  "$IMAGE" -lc 'python3 /repo/probes/indexer_decode_fused_check.py "$@"' _ "$@"
