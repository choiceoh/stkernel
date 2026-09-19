#!/usr/bin/env bash
# takeover2.sh -- waits for mtp_window3b.sh's last arm (tuned2-off) to be serving its requests, starts mtp_window3c.sh,
# then kills 3b and its heartbeat subshell (-9: no EXIT trap, so no release). The operator asked for more data and a
# third training in the same window (2026-09-19).
OUT=/home/choiceoh/glm53-logs/q38mtp-window-0919
for i in $(seq 1 900); do
  if ! pgrep -f mtp_window3b.sh >/dev/null; then
    echo "[$(date +%H:%M:%S)] takeover2: mtp_window3b.sh ended before tuned2-off's requests -- nothing taken over"; exit 1
  fi
  if pgrep -f "mtp_requests.py eval http://10.10.10.2:8000 tuned2-off" >/dev/null; then
    setsid nohup bash "$OUT/mtp_window3c.sh" >> "$OUT/window.log" 2>&1 < /dev/null &
    sleep 3
    pkill -9 -f mtp_window3b.sh
    echo "[$(date +%H:%M:%S)] takeover2: mtp_window3b.sh killed during tuned2-off's requests; mtp_window3c.sh runs the rest"
    exit 0
  fi
  sleep 2
done
echo "[$(date +%H:%M:%S)] takeover2: no tuned2-off requests in 30 min -- nothing taken over"
