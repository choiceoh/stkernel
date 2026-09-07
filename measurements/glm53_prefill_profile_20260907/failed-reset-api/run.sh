#!/usr/bin/env bash
set -euo pipefail
# Owned serving profile; run.py boots with bench/ab-lever.sh and restores it.
cd /home/choiceoh/stkernel-prefill-fused-serving
exec python3 /tmp/glm53-prefill-profile.JjntmWN5/run.py
