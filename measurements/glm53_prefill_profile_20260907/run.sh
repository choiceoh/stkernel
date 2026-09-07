#!/usr/bin/env bash
set -euo pipefail
# run.py owns the bench/ab-lever.sh boots and full restore.
cd /home/choiceoh/stkernel-prefill-fused-serving
exec python3 /tmp/glm53-prefill-profile2.yfJv5Fe8/run.py
