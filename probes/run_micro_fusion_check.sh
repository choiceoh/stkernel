#!/usr/bin/env bash
# Offline gate for launch-count bundle 2 (RUNBOOK EXP-20): mounts the composed
# glm53 overlay sources the bundle touches into a fresh container and runs
# probes/micro_fusion_check.py (numerics + CUDA-graph timing of the stock vs
# fused chains). Never the serving container; a shared GPU (srv4 with the
# production worker resident) only makes the RELATIVE timing meaningful.
#
#   bash probes/run_micro_fusion_check.sh [--sections dual,kda,kpool] [--iters 20]
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
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

sources=(glm53_kda_onepass.py glm5next_kda.py sparse_attn_indexer_kpool.py)
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

# The probe arms every bundle knob itself (setdefault); a caller's explicit
# value wins so a sweep can hold one at 0.
envs=(-e "MK_PKG_PATH=${TARGET_PREFIX%/}")
# `|| true`: with pipefail a shell that carries no VLLM_* knob must not abort
# the probe (grep exits 1 on no match)
for v in $(compgen -v | grep -E '^VLLM_(GLM53|DSV4)_' || true); do
  envs+=(-e "$v=${!v}")
done
echo "profile=$PROFILE image=$IMAGE files=${#sources[@]} args=$*" >&2

cmd="python3 /repo/probes/micro_fusion_check.py"
for a in "$@"; do cmd="$cmd $(printf %q "$a")"; done

exec docker run --rm --gpus all --entrypoint /bin/bash \
  --mount "type=bind,src=$REPO,dst=/repo,readonly" \
  "${envs[@]}" "${mounts[@]}" \
  "$IMAGE" -lc "$cmd"
