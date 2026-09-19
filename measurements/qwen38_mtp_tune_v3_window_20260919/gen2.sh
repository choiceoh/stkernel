#!/usr/bin/env bash
# datagen2.py through the OpenRouter key Deneb holds (read here, never printed): gen2.sh STAGE OUT_DIR [N]
cd ~/q38mtp-gen2
OPENROUTER_API_KEY=$(grep -E "^OPENROUTER_API_KEY=" ~/.deneb/.env | head -1 | cut -d= -f2- | tr -d "\"'") \
  exec python3 datagen2.py "$@"
