#!/usr/bin/env bash
# Cold serial, cold parallel and kept parallel native builds (native_builds_cpu.py) in the ST image, CUDA hidden.
# Usage: native_builds_cpu.sh TREE WORK LOG
set -euo pipefail
TREE=$1; WORK=$2; LOG=$3
exec > >(tee "$LOG") 2>&1
HERE=$(cd "$(dirname "$0")" && pwd)
IMAGE=st-engine:glm53
mkdir -p "$WORK"
sample() {  # root mode
  local root=$1 mode=$2
  mkdir -p "$root"
  docker run --rm --network none --memory 16g -e CUDA_VISIBLE_DEVICES= -e PYTHONPATH=/repo -e MAX_JOBS=2 \
    -e CUTE_DSL_ARCH=sm_121a \
    -e ST_DENSE_BUILD_ROOT=/work/root/st-dense -e ST_ONESHOT_BUILD_ROOT=/work/root/st-oneshot \
    -e ST_MLA_BUILD_ROOT=/work/root/mla -e ST_NATIVE_BUILD_ROOT=/work/root/st-native \
    -v "$TREE":/repo:ro -v "$HERE":/bench:ro -v "$root":/work/root -w /repo \
    --entrypoint python3 "$IMAGE" /bench/native_builds_cpu.py "$mode" 2>&1 | grep -v "No CUDA runtime is found"
}
echo "image $(docker image inspect -f '{{.Id}}' $IMAGE); tree $(cat "$TREE/.commit" 2>/dev/null); host $(hostname); $(date -u +%FT%TZ); MemAvailable $(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo) GiB"
sample "$WORK/parallel" parallel     # cold: a fresh root
sample "$WORK/parallel" parallel     # kept: the same root again
sample "$WORK/serial" serial         # cold, one after another, in first-use order
echo "done $(date -u +%FT%TZ)"
