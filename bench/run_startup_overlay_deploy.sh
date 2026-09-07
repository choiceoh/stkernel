#!/usr/bin/env bash
# fleet.sh run --gpu SESSION 40 "identical-overlay redeploy B/A/A/B" -- bash bench/run_startup_overlay_deploy.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export REPO=$PWD
: "${FLEET_SESSION:?run through fleet.sh run --gpu}"
export STARTUP_CACHE_EVIDENCE=${STARTUP_CACHE_EVIDENCE:-/home/choiceoh/glm53-logs/overlay-deploy-$(date +%Y%m%d-%H%M%S)}
export STARTUP_CACHE_PREFIX=DEPLOYCACHE
export STARTUP_CACHE_MODE=overlay-deploy
[ "$(cut -d'|' -f1 /home/choiceoh/glm53-logs/fleet/holder)" = "$FLEET_SESSION" ] || exit 2
[ -z "$(git status --porcelain --untracked-files=normal)" ] || exit 2
mkdir -p "$STARTUP_CACHE_EVIDENCE"
python3 -m unittest discover -s tests -p test_glm53_overlay_sync.py > "$STARTUP_CACHE_EVIDENCE/sync-tests.log" 2>&1
python3 -m unittest discover -s tests -p test_startup_cache_receipts.py > "$STARTUP_CACHE_EVIDENCE/receipt-tests.log" 2>&1
python3 bench/startup_host_memory.py "$STARTUP_CACHE_EVIDENCE" > "$STARTUP_CACHE_EVIDENCE/memwatch.out" 2>&1 &
sampler_pid=$!
trap 'kill "$sampler_pid" 2>/dev/null || true; wait "$sampler_pid" 2>/dev/null || true' EXIT
bash bench/startup_cache_boots.sh
