#!/usr/bin/env bash
# chain.sh LEVER adapter: freeze B1's effective memory/scheduling controls.
set -euo pipefail
: "${PREFILL_SERVING_RUNNER:?}"
exec python3 "$PREFILL_SERVING_RUNNER" boot "$@"
