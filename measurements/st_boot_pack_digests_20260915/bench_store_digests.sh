#!/usr/bin/env bash
# CPU A/B of the pack store's identity work (bench_store_digests.py) in the ST image, CUDA hidden.
# Usage: bench_store_digests.sh TREE WORK LOG   (TREE: the branch checkout mounted at /repo; WORK: scratch root)
set -euo pipefail
TREE=$1; WORK=$2; LOG=$3
exec > >(tee "$LOG") 2>&1
HERE=$(cd "$(dirname "$0")" && pwd)
IMAGE=st-engine:bracket-9c45086a0622
mkdir -p "$WORK/old"
# the old store, with the packing.py its algorithm identity reads beside it (unchanged on this branch)
git -C "$HERE/../.." show bb72154d:engine/kernels/dense/store.py > "$WORK/old/store_bb72154d.py"
git -C "$HERE/../.." show bb72154d:engine/kernels/dense/packing.py > "$WORK/old/packing.py"
run() {
  docker run --rm --network none -e CUDA_VISIBLE_DEVICES= -e PYTHONPATH=/repo \
    -v "$TREE":/repo:ro -v "$HERE":/bench:ro -v "$WORK":/work -v "$WORK/old":/old:ro \
    -v /home/choiceoh/models/st-glm53-9391-up-gate-full/rank3of4.safetensors:/models/rank3of4.safetensors:ro \
    -v /home/choiceoh/models/GLM-5.3-Flash-DFlash2/model.safetensors:/drafter/model.safetensors:ro \
    -v /home/choiceoh/glm53-cache/mkcalib:/calibration:ro \
    --entrypoint python3 "$IMAGE" /bench/bench_store_digests.py "$@" 2>&1 | grep -v "No CUDA runtime"
}
echo "image $(docker image inspect -f '{{.Id}}' $IMAGE); tree $(cat "$TREE/.commit" 2>/dev/null); host $(hostname); $(date -u +%FT%TZ)"
run setup /work/root
for variant in old new new old; do
  run run /work/root "$variant"
done
