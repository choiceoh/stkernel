#!/usr/bin/env bash
# CPU-only release gate in the exact serving image, before any overlay writes.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODEL_DIR=${1:?usage: check-glm53-chat.sh <tokenizer/model directory> <serving image>}
IMAGE=${2:?usage: check-glm53-chat.sh <tokenizer/model directory> <serving image>}
test -f "$MODEL_DIR/tokenizer_config.json"
# Gate the candidate parser in an isolated container, not the unpatched image.
PARSER_BASE=$(awk -F '\t' '$1 == "glm47_moe.py" {print $3}' "$REPO/overlay/modules/glm53_runtime/manifest.tsv")
PARSER_TARGET=/usr/local/lib/python3.12/dist-packages/vllm/parser/glm47_moe.py
ACTUAL_BASE=$(docker run --rm --network none --entrypoint sha256sum "$IMAGE" "$PARSER_TARGET")
test "${ACTUAL_BASE%% *}" = "$PARSER_BASE" || {
  echo "ABORT: GLM parser base changed; review the overlay against this image" >&2
  exit 1
}
docker run --rm --network none --memory 3g --cpus 2 \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$REPO:/checks:ro" -v "$MODEL_DIR:/models/glm:ro" \
  -v "$REPO/overlay/modules/glm53_runtime/glm47_moe.py:$PARSER_TARGET:ro" \
  --entrypoint /bin/bash "$IMAGE" -c '
    set -euo pipefail
    python3 -u /checks/tests/test_glm53_chat.py
    python3 -u /checks/tests/test_glm53_tool_acceptance.py
    python3 -u /checks/probes/glm53_chat_contract.py --tokenizer /models/glm
  '
