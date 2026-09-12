#!/usr/bin/env bash
# Run on the node holding the completed preshard. Put rank r and the vision
# tower on node r (base/comm.NODES order), along with checkpoint metadata.
# Transfers use partial files and are SHA-256 checked before returning.
#   RANKS_DIR=/path/to/completed bash launchers/fanout-st-ranks.sh [ranks...]
set -euo pipefail
SRC=${RANKS_DIR:-/home/choiceoh/models/st-glm53-nvidia-tp4-9391}
NODES=(10.10.10.2 10.10.10.1 10.10.10.3 10.10.10.4)
SELF_IPS=" $(hostname -I 2>/dev/null) "
if [ "$#" -eq 0 ]; then set -- 0 1 2 3; fi
metadata=()
for name in config.json tokenizer.json tokenizer_config.json generation_config.json processor_config.json \
  hf_quant_config.json preshard-manifest.json completion-verification.json SHA256SUMS; do
  [ ! -f "$SRC/$name" ] || metadata+=("$SRC/$name")
done
for path in "$SRC"/chat_template*.jinja; do
  [ ! -f "$path" ] || metadata+=("$path")
done
for r in "$@"; do
  [[ "$r" =~ ^[0-3]$ ]] || { echo "invalid rank: $r" >&2; exit 2; }
  ip=${NODES[$r]}; f="rank${r}of4.safetensors"
  [ -s "$SRC/$f" ] && [ -s "$SRC/vision.safetensors" ] || { echo "missing $f or vision.safetensors in $SRC" >&2; exit 1; }
  if [[ "$SELF_IPS" == *" $ip "* ]]; then
    echo "rank $r: source node already holds $f and vision.safetensors"
    continue
  fi
  printf -v quoted_src '%q' "$SRC"
  ssh_command="ssh -o BatchMode=yes -o ConnectTimeout=10"
  [ -z "${FANOUT_JUMP:-}" ] || ssh_command+=" -J $FANOUT_JUMP"
  echo "rank $r: $f + vision + metadata -> $ip"
  $ssh_command "choiceoh@$ip" "mkdir -p -- $quoted_src"
  # No in-place writes: an interrupted transfer cannot publish a partial rank.
  rsync -a --partial-dir=.rsync-partial --whole-file --fsync \
    -e "$ssh_command" "$SRC/$f" "$SRC/vision.safetensors" "${metadata[@]}" "choiceoh@$ip:$SRC/"
  digest=$(cd "$SRC" && sha256sum "$f" vision.safetensors)
  $ssh_command "choiceoh@$ip" "cd -- $quoted_src && sha256sum --check --strict" <<< "$digest"
done
