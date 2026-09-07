#!/usr/bin/env bash
# One production restore owned by the boot supervisor. Never inherit candidate knobs.
set -euo pipefail
session=${FLEET_SESSION:?}
[[ $(cut -d'|' -f1 "${FLEET_DIR:?}/holder") == "$session" ]] || exit 2
repo=${FLEET_PRODUCTION_REPO:-/home/choiceoh/stkernel}
cd "$repo"
[[ -z $(git status --porcelain) ]] || { echo 'production checkout is dirty'; exit 2; }
git fetch origin
git merge --ff-only origin/main
[[ $(git rev-parse HEAD) == $(git rev-parse origin/main) ]] || { echo 'production checkout is not approved main'; exit 2; }
# A completed public defaults arm of this approved build needs no second boot.
if python3 "${FLEET_RUNNER_REPO:?}/bench/fleet_entry.py" production-current "$repo"; then
  echo 'approved public defaults already healthy; no restore boot'
  exit 0
fi
bash launchers/deploy-overlays.sh glm53
python3 - "$repo" "$session" <<'PY'
import os, subprocess, sys
env = {k:v for k,v in os.environ.items() if not k.startswith(('VLLM_', 'ONEPASS_', 'MK_', 'STARTUP_CACHE_'))}
for key in ('IMAGE', 'SPEC_K', 'SPEC', 'LEVER', 'SKIP_BOOT', 'HEAD', 'GLM53_API_HOST', 'GLM53_API_PORT'):
    env.pop(key, None)
env.update(REPO=sys.argv[1], GLM53_API_HOST='0.0.0.0', GLM53_API_PORT='8000', HEAD='10.10.10.2',
           PREFILL_WARMUP='1', LEGS='none', SKIP_BOOT='0')
raise SystemExit(subprocess.call(['bash', 'bench/ab-lever.sh', sys.argv[2]+'RESTORE', ''], env=env))
PY
