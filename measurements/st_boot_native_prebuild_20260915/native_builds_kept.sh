#!/usr/bin/env bash
# Kept (warm) native builds, serial and parallel, in the production image against a copy of srv4's /cache/cu132 build
# directories (the fleet's paths, so Ninja sees its own up-to-date graph). Usage: native_builds_kept.sh TREE COPY LOG
set -euo pipefail
TREE=$1; COPY=$2; LOG=$3
exec > >(tee "$LOG") 2>&1
HERE=$(cd "$(dirname "$0")" && pwd)
IMAGE=st-engine:glm53
echo "image $(docker image inspect -f '{{.Id}}' $IMAGE); tree $(cat "$TREE/.commit" 2>/dev/null); host $(hostname); $(date -u +%FT%TZ)"
sample() {
  docker run --rm --network none -e CUDA_VISIBLE_DEVICES= -e PYTHONPATH=/repo -e MAX_JOBS=2 -e CUTE_DSL_ARCH=sm_121a \
    -e ST_DENSE_BUILD_ROOT=/cache/cu132/st-dense -e ST_ONESHOT_BUILD_ROOT=/cache/cu132/st-oneshot \
    -e ST_MLA_BUILD_ROOT=/cache/cu132/mla -e ST_NATIVE_BUILD_ROOT=/cache/cu132/st-native \
    -v "$TREE":/repo:ro -v "$HERE":/bench:ro -v "$COPY":/cache/cu132 -w /repo \
    --entrypoint python3 "$IMAGE" /bench/native_builds_cpu.py "$1" 2>&1 | grep -v "No CUDA runtime is found"
}
for mode in serial parallel parallel serial; do
  sample "$mode"
done
