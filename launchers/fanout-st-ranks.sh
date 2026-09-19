#!/usr/bin/env bash
# Run on the node holding the completed preshard. Put rank r and the vision
# tower on node r (base/comm.NODES order), along with checkpoint metadata.
# Transfers use partial files and are SHA-256 checked before returning.
#   RANKS_DIR=/path/to/completed bash launchers/fanout-st-ranks.sh [ranks...]
# VISION=0 for a text-only preshard: the ranks and metadata only (a Qwen3.8 fleet without the tower serves text).
set -euo pipefail
SRC=${RANKS_DIR:-/home/choiceoh/models/st-glm53-nvidia-tp4-9391}
VISION=${VISION:-1}
case "$VISION" in 0|1) ;; *) echo "VISION must be 0 or 1" >&2; exit 2 ;; esac
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
  files=("$f")
  [ "$VISION" = 0 ] || files+=(vision.safetensors)
  for name in "${files[@]}"; do
    [ -s "$SRC/$name" ] || { echo "missing $name in $SRC" >&2; exit 1; }
  done
  if [[ "$SELF_IPS" == *" $ip "* ]]; then
    echo "rank $r: source node already holds ${files[*]}"
    continue
  fi
  printf -v quoted_src '%q' "$SRC"
  ssh_command="ssh -o BatchMode=yes -o ConnectTimeout=10"
  [ -z "${FANOUT_JUMP:-}" ] || ssh_command+=" -J $FANOUT_JUMP"
  echo "rank $r: ${files[*]} + metadata -> $ip"
  $ssh_command "choiceoh@$ip" "mkdir -p -- $quoted_src"
  # No in-place writes: an interrupted transfer cannot publish a partial rank.
  rsync -a --partial-dir=.rsync-partial --whole-file --fsync \
    -e "$ssh_command" "${files[@]/#/$SRC/}" "${metadata[@]}" "choiceoh@$ip:$SRC/"
  digest=$(cd "$SRC" && sha256sum "${files[@]}")
  $ssh_command "choiceoh@$ip" "cd -- $quoted_src && sha256sum --check --strict" <<< "$digest"
done
