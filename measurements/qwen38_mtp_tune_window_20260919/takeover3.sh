#!/usr/bin/env bash
# takeover3.sh -- the last arm of mtp_window3c.sh (tuned3-off) serving its requests, or 3c saying the third training
# served nothing: start mtp_window3d.sh (the door held for a peer's profile, then the release) and kill 3c and its
# heartbeat (-9: no EXIT trap, so no release).
OUT=/home/choiceoh/glm53-logs/q38mtp-window-0919
start() {
  PENDING_ARM="$1" setsid nohup bash "$OUT/mtp_window3d.sh" >> "$OUT/window.log" 2>&1 < /dev/null &
  sleep 2
  pkill -9 -f mtp_window3c.sh
  echo "[$(date +%H:%M:%S)] takeover3: mtp_window3c.sh killed (${1:-no arm pending}); mtp_window3d.sh holds the door, then releases"
  exit 0
}
for i in $(seq 1 7200); do
  pgrep -f mtp_window3c.sh >/dev/null || { echo "[$(date +%H:%M:%S)] takeover3: mtp_window3c.sh ended first -- nothing taken over"; exit 1; }
  pgrep -f "mtp_requests.py eval http://10.10.10.2:8000 tuned3-off" >/dev/null && start tuned3-off
  tail -3 "$OUT/window.log" | grep -q "the third training did not beat\|no second head or no new shards\|tuned3-off did not serve" && start ""
  sleep 0.5
done
echo "[$(date +%H:%M:%S)] takeover3: nothing to take over in an hour"
