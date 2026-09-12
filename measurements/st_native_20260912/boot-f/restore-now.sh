#!/bin/bash
set -euo pipefail
node_sh() { local ip=$1; shift; if [ "$ip" = 10.10.10.2 ]; then bash -c "$*"; else ssh -o BatchMode=yes "$ip" "$@"; fi; }
set -a
source /home/choiceoh/.config/st-glm53.env
set +a
bash /home/choiceoh/st-releases/5734b29fde84/launchers/start-st-glm53.sh stop
for ip in 10.10.10.2 10.10.10.1 10.10.10.3 10.10.10.4; do
 node_sh "$ip" "PYTHONPATH=/home/choiceoh/st-releases/perf-f4d7-20260912e python3 -c 'from engine.base.arena import release_model_cache; print(release_model_cache([\"/home/choiceoh/models\"]))'"
done
node_sh 10.10.10.4 'python3 /home/choiceoh/st-perf-f4d7/st-restore-cache-return.py' > /home/choiceoh/st-native-f4d7-20260912f/restore-reclaim2.log 2>&1 &
bash /home/choiceoh/st-releases/5734b29fde84/launchers/start-st-glm53.sh
for i in $(seq 1 60); do
 if curl -fsS --max-time 4 http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
  systemctl --user start st-glm53.service
  exit 0
 fi
 sleep 5
done
exit 1
