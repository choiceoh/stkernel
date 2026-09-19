#!/usr/bin/env bash
# selfgen3.py with OpenRouter's key read from Deneb's env at run time (never printed, never stored elsewhere).
cd ~/q38mtp-gen2
OPENROUTER_API_KEY=$(grep -E "^OPENROUTER_API_KEY=" ~/.deneb/.env | head -1 | cut -d= -f2- | tr -d "\"'") \
  exec python3 selfgen3.py "$@"
