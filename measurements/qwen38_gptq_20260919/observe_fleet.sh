#!/usr/bin/env bash
# Read-only GPU occupancy samples. Run on srv2, while collect_window.sh owns
# the fleet. No serving requests, container mutations or GPU kernels are issued.
set -euo pipefail
TREE=${1:?frozen source tree}
OUT=${2:?output directory}
OWNER=session/q38gptq-0919
LOCK=/home/choiceoh/glm53-logs/st-fleet.lock
NODES=(10.10.10.2 10.10.10.1 10.10.10.3 10.10.10.4)
mkdir -p "$OUT"
while python3 "$TREE/engine/base/fleet_lease.py" verify --owner "$OWNER" --path "$LOCK" >/dev/null 2>&1; do
  stamp=$(date -u +%Y%m%dT%H%M%SZ)
  jobs=()
  for r in 0 1 2 3; do
    ip=${NODES[$r]}
    cmd="date -Is; hostname; docker ps --format '{{.Names}} {{.Image}}'; nvidia-smi pmon -c 1"
    if [ "$ip" = 10.10.10.2 ]; then
      bash -c "$cmd" > "$OUT/$stamp-rank$r.log" 2>&1 &
    else
      ssh -n -o BatchMode=yes -o ConnectTimeout=8 "choiceoh@$ip" "$cmd" > "$OUT/$stamp-rank$r.log" 2>&1 &
    fi
    jobs+=("$!")
  done
  for pid in "${jobs[@]}"; do wait "$pid" || true; done
  sleep 15
done
