#!/usr/bin/env bash
# tokenizers_info.py in the ST image, CUDA hidden. Usage: decoded_vocab_diff.sh LOG
set -euo pipefail
LOG=$1
exec > >(tee "$LOG") 2>&1
HERE=$(cd "$(dirname "$0")" && pwd)
IMAGE=st-engine:glm53
echo "image $(docker image inspect -f '{{.Id}}' $IMAGE); host $(hostname); $(date -u +%FT%TZ)"
docker run --rm --network none -e CUDA_VISIBLE_DEVICES= -v "$HERE":/bench:ro \
  -v /home/choiceoh/st-engine/st-glm53-meta:/meta:ro --entrypoint python3 "$IMAGE" /bench/tokenizers_info.py /meta 2>&1 \
  | grep -v "No CUDA runtime is found"
