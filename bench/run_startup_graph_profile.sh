#!/usr/bin/env bash
# fleet.sh run --gpu SESSION 35 "graph-profile B/A/A/B" -- bash bench/run_startup_graph_profile.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export REPO=$PWD
: "${FLEET_SESSION:?run through fleet.sh run --gpu}"
export STARTUP_CACHE_EVIDENCE=${STARTUP_CACHE_EVIDENCE:-/home/choiceoh/glm53-logs/graph-profile-$(date +%Y%m%d-%H%M%S)}
export STARTUP_CACHE_PREFIX=GRAPHMEM
export STARTUP_CACHE_MODE=graph-profile
[ "$(cut -d'|' -f1 /home/choiceoh/glm53-logs/fleet/holder)" = "$FLEET_SESSION" ] || exit 2
[ -z "$(git status --porcelain --untracked-files=normal)" ] || exit 2
mkdir -p "$STARTUP_CACHE_EVIDENCE"
python3 -m unittest discover -s tests -p test_glm53_graph_profile.py > "$STARTUP_CACHE_EVIDENCE/worker-tests.log" 2>&1
python3 -m unittest discover -s tests -p test_startup_cache_receipts.py > "$STARTUP_CACHE_EVIDENCE/receipt-tests.log" 2>&1
bash launchers/deploy-overlays.sh glm53 > "$STARTUP_CACHE_EVIDENCE/deploy.log" 2>&1
cp /home/choiceoh/overlays/glm53/manifest.tsv "$STARTUP_CACHE_EVIDENCE/deployed-manifest.tsv"
python3 bench/startup_host_memory.py "$STARTUP_CACHE_EVIDENCE" > "$STARTUP_CACHE_EVIDENCE/memwatch.out" 2>&1 &
sampler_pid=$!
trap 'kill "$sampler_pid" 2>/dev/null || true; wait "$sampler_pid" 2>/dev/null || true' EXIT
bash bench/startup_cache_boots.sh
