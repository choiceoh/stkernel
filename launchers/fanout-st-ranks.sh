#!/usr/bin/env bash
# Put rank r's file on node r (base/comm.NODES order): the preshard wrote all
# four on srv4. srv3 is reached through srv2 (fleet ops memory). Idempotent
# (rsync); ~44.5 GiB per file.
#   bash launchers/fanout-st-ranks.sh [ranks...]      default: 0 1 2
set -euo pipefail
SRC=${RANKS_DIR:-/home/choiceoh/models/glm53-redhat-nvfp4-tp4}
NODES=(10.10.10.1 10.10.10.2 10.10.10.3 10.10.10.4)
for r in "${@:-0 1 2}"; do
  ip=${NODES[$r]}; f="rank${r}of4.safetensors"
  jump=""; [ "$ip" = 10.10.10.3 ] && jump="-J srv2"
  echo "== $f -> $ip"
  ssh -o BatchMode=yes $jump "choiceoh@$ip" "mkdir -p $SRC"
  rsync -a --partial --inplace --whole-file -e "ssh -o BatchMode=yes $jump" "$SRC/$f" "choiceoh@$ip:$SRC/$f"
done
