#!/usr/bin/env bash
# Two containers, one kept /cache: the first run compiles the four natives, the second must reuse them.
# usage: flock -w 7200 /tmp/c2opt-heavy.lock bash measurements/st_native_build_root_20260915/native_twice.sh [image]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
IMAGE=${1:-st-engine:bracket-9c45086a0622}
CACHE=$(mktemp -d)
for run in cold warm; do
  echo "== $run $(date -u +%H:%M:%S) MemAvailable $(awk '/MemAvailable/ {print $2}' /proc/meminfo) kB"
  docker run --rm -e CUDA_VISIBLE_DEVICES= -e CUTE_DSL_ARCH=sm_121a -e PYTHONPATH=/repo \
    -e ST_NATIVE_BUILD_ROOT=/cache/cu132/st-native \
    -v "$REPO":/repo:ro -v "$CACHE":/cache -v "$HERE":/scratch:ro -w /repo \
    --entrypoint python3 "$IMAGE" /scratch/native_twice.py "$run"
done
# hand the kept builds back to the invoking user (the container wrote them as root)
docker run --rm -v "$CACHE":/cache --entrypoint chown "$IMAGE" -R "$(id -u):$(id -g)" /cache
echo "== done $(date -u +%H:%M:%S) (builds kept in $CACHE)"
