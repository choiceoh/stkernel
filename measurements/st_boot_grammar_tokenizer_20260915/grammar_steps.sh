#!/usr/bin/env bash
# grammar_steps.py twice (a cold and a warm process) in the ST image, CUDA hidden. Usage: grammar_steps.sh LOG
set -euo pipefail
LOG=$1
exec > >(tee "$LOG") 2>&1
HERE=$(cd "$(dirname "$0")" && pwd)
IMAGE=st-engine:bracket-9c45086a0622
echo "image $(docker image inspect -f '{{.Id}}' $IMAGE); host $(hostname); $(date -u +%FT%TZ)"
for i in 1 2; do
  docker run --rm --network none -e CUDA_VISIBLE_DEVICES= -v "$HERE":/bench:ro \
    -v /home/choiceoh/st-engine/st-glm53-meta:/meta:ro --entrypoint python3 "$IMAGE" /bench/grammar_steps.py /meta 2>&1 \
    | grep -v "No CUDA runtime is found"
done
