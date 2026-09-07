#!/usr/bin/env bash
set -euo pipefail
cd /home/choiceoh/glm53-early-mm-20260908
export REPO=$PWD
export STARTUP_CACHE_EVIDENCE=/home/choiceoh/glm53-logs/early-mm-20260908-v2
export STARTUP_CACHE_PREFIX=EARLYMM
export STARTUP_CACHE_MODE=renderer-warmup
[ "$(cut -d'|' -f1 /home/choiceoh/glm53-logs/fleet/holder)" = "$FLEET_SESSION" ] || exit 2
git fetch -q origin main
git rebase origin/main
mkdir -p "$STARTUP_CACHE_EVIDENCE"
git rev-parse HEAD > "$STARTUP_CACHE_EVIDENCE/screen-source-commit.txt"
python3 tests/test_logic.py > "$STARTUP_CACHE_EVIDENCE/logic.log" 2>&1
bash launchers/compose-overlays.sh glm53
[ -z "$(git status --porcelain --untracked-files=normal)" ] || exit 2
screen_image=$(docker image inspect --format '{{.Id}}' glm53:v13-b12x-it)
printf '%s\n' "$screen_image" > "$STARTUP_CACHE_EVIDENCE/screen-image-id.txt"
docker run --rm --gpus all --network none --ipc host --volumes-from glm53:ro \
  -v "$REPO":/candidate:ro -v "$STARTUP_CACHE_EVIDENCE":/proof \
  -v "$REPO/build/glm53/glm53_renderer_warmup.py":/usr/local/lib/python3.12/dist-packages/vllm/renderers/glm53_renderer_warmup.py:ro \
  --entrypoint python3 "$screen_image" /candidate/probes/glm53_renderer_warmup_check.py \
  --template /candidate/launchers/chat_template_mm_v2.jinja --out /proof/screen.json \
  > "$STARTUP_CACHE_EVIDENCE/screen.log" 2>&1
python3 - "$STARTUP_CACHE_EVIDENCE/screen.json" <<'GATE'
import json, sys
result = json.load(open(sys.argv[1]))
assert result['ok'] and result['checks'] == 6, result
GATE
bash launchers/deploy-overlays.sh glm53
head -1 /home/choiceoh/overlays/glm53/manifest.tsv > "$STARTUP_CACHE_EVIDENCE/runtime-source-commit.txt"
python3 /home/choiceoh/glm53-logs/pack-io-memwatch-20260907.py "$STARTUP_CACHE_EVIDENCE" > "$STARTUP_CACHE_EVIDENCE/memwatch.out" 2>&1 &
sampler_pid=$!
trap 'kill "$sampler_pid" 2>/dev/null || true; wait "$sampler_pid" 2>/dev/null || true' EXIT
bash bench/startup_cache_boots.sh
