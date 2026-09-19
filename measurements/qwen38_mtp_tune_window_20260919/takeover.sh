#!/usr/bin/env bash
# takeover.sh -- waits for mtp_window3.sh's last arm (tuned-cut) to be serving its requests, starts mtp_window3b.sh,
# then kills the first script (-9: no EXIT trap, so no release). The operator asked for a second training in the
# same window (2026-09-19); the first script would have released right after that arm.
OUT=/home/choiceoh/glm53-logs/q38mtp-window-0919
FIRST=874613
for i in $(seq 1 900); do
  if ! kill -0 "$FIRST" 2>/dev/null; then
    echo "[$(date +%H:%M:%S)] takeover: mtp_window3.sh ended before tuned-cut's requests -- nothing taken over"; exit 1
  fi
  if pgrep -f "mtp_requests.py eval http://10.10.10.2:8000 tuned-cut" >/dev/null; then
    setsid nohup bash "$OUT/mtp_window3b.sh" >> "$OUT/window.log" 2>&1 < /dev/null &
    sleep 3
    kill -9 "$FIRST"
    echo "[$(date +%H:%M:%S)] takeover: mtp_window3.sh ($FIRST) killed during tuned-cut's requests; mtp_window3b.sh runs the rest"
    exit 0
  fi
  sleep 2
done
echo "[$(date +%H:%M:%S)] takeover: no tuned-cut requests in 30 min -- nothing taken over"
