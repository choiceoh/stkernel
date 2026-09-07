#!/usr/bin/env bash
# Idle-serving probe: admission needs 16 GiB; the monitor enforces 12 GiB.
set -euo pipefail
cd /home/choiceoh/stkernel-input-cta-next-20260908
exec python3 probes/run_input_cta_guarded.py \
  --out "${INPUT_NEXT_OUT:-/home/choiceoh/glm53-logs/INPUTNEXT0908_GUARDED}"
