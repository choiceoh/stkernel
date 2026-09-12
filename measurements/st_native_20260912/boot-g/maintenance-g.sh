#!/bin/bash
# The caller holds the canonical bench queue lock for the whole transition.
set -euo pipefail
ROOT=/home/choiceoh/st-native-f4d7-20260912g
LEASE=/home/choiceoh/st-releases/perf-f4d7-20260912g/engine/base/fleet_lease.py
python3 "$LEASE" owner --container st-glm53 --path /home/choiceoh/st-fleet.lock > /dev/null
[ ! -s /home/choiceoh/glm53-logs/fleet/holder ]
if ! curl -fsS --max-time 4 http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
 bash /home/choiceoh/st-native-f4d7-20260912f/restore-now.sh > "$ROOT/restore-before-retry.log" 2>&1
fi
cd "$ROOT"
mkdir -p attempt1
for f in run.log restore.log restore-reclaim.log restored-status.json restored-service.txt exclusive-exit-code.txt baseline-container.json; do
 [ ! -f "$f" ] || mv "$f" attempt1/
done
bash run-exclusive.sh > run.log 2>&1
