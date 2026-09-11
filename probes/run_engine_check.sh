#!/usr/bin/env bash
# Real-weight GLM check on the standalone ST image.
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
exec bash "$repo/probes/run_engine_probe.sh" engine/profiles/glm53/check.py --lanes served "$@"
