#!/usr/bin/env bash
# CPU-only release gate in the exact serving image, before any overlay writes.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODEL_DIR=${1:?usage: check-glm53-chat.sh <tokenizer/model directory> <serving image>}
IMAGE=${2:?usage: check-glm53-chat.sh <tokenizer/model directory> <serving image>}
test -f "$MODEL_DIR/tokenizer_config.json"
docker run --rm --network none --memory 3g --cpus 2 \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$REPO:/checks:ro" -v "$MODEL_DIR:/models/glm:ro" \
  --entrypoint /bin/bash "$IMAGE" -c '
    set -euo pipefail
    python3 -u /checks/tests/test_glm53_chat.py
    python3 -u /checks/probes/glm53_chat_contract.py --tokenizer /models/glm
  '
