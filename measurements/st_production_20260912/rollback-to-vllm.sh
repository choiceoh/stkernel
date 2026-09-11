#!/usr/bin/env bash
# Run on srv2. Restores the existing approved vLLM fleet through its idle controller.
set -euo pipefail
systemctl --user disable --now st-glm53.service
set -a
source /home/choiceoh/.config/st-glm53.env
set +a
bash "$ST_REPO/launchers/start-st-glm53.sh" stop
systemctl --user enable --now fleet-idle-recovery.timer
systemctl --user show fleet-idle-recovery.timer -p ActiveState -p UnitFileState
printf '%s\n' 'vLLM recovery re-enabled; the controller waits for five idle minutes before restoring the approved production checkout.'
