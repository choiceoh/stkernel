#!/usr/bin/env bash
# srv2: queue an st-hold of the ACE quality arm, feed the corpus five times (one routed-slot skip per pass, set by the
# capture from the document number), end the hold, fetch head.jsonl from the capture rank (srv4).
#   setsid nohup bash run_ace_quality.sh <session> <full sha> <arm worktree> <est_min> > ~/expert-capture/ace-quality/<session>.runner.log 2>&1 &
set -u
S=$1; SHA=$2; WT=$3; EST=$4
OUT=~/expert-capture/ace-quality/$S
mkdir -p "$OUT"
Q=~/st-worktrees/queue-main-$(git -C ~/stkernel rev-parse --short=8 origin/main)
TICKET_LOG=$OUT/ticket.log
log() { echo "$(date '+%F %T') $S: $*"; }
test -d "$Q" || git -C ~/stkernel worktree add -q --detach "$Q" origin/main

log "queue checkout $Q, sha $SHA, arm tree $WT, est $EST"
( cd "$Q" && exec bash bench/fleet.sh st-hold "$S" "$SHA" "$EST" "ACE quality: prefill head NLL over five corpus passes (skip off/10%/15%/per-layer 10%/off) in one boot (debug, never merge)" ) > "$TICKET_LOG" 2>&1 &
TICKET=$!
deadline=$(( $(date +%s) + 28800 ))
while :; do
  if grep -q "door up:" "$TICKET_LOG" 2>/dev/null && grep -q "queue/$S" ~/glm53-logs/st-fleet.lock 2>/dev/null; then break; fi
  if ! kill -0 "$TICKET" 2>/dev/null; then log "ticket exited before the door came up"; tail -25 "$TICKET_LOG"; exit 1; fi
  if [ "$(date +%s)" -gt "$deadline" ]; then log "gave up waiting"; ( cd "$Q" && bash bench/fleet.sh cancel "$S" ); exit 1; fi
  sleep 20
done
log "door up"
for k in 0 1 2 3 4; do
  python3 "$WT/probes/expert_capture_feed.py" --corpus ~/expert-capture/corpus.jsonl --log "$OUT/feed-pass$k.jsonl" \
    --release "$SHA" --port 8001 --wait-minutes 30 --cache-salt "acepass$k-$S" > "$OUT/feed-pass$k.log" 2>&1
  log "pass $k exit $? ($(grep -c '"status": "ok"' "$OUT/feed-pass$k.jsonl" 2>/dev/null) ok rows)"
done
touch ~/glm53-logs/st-bracket/"$S"/stop 2>/dev/null || ( cd "$Q" && bash bench/fleet.sh cancel "$S" )
log "stop requested"
wait "$TICKET"
log "hold ended"
CAP=$(ssh -o BatchMode=yes 10.10.10.4 "ls -dt ~/glm53-logs/st-bracket-dumps/$S-hold-*/expert-capture 2>/dev/null | head -1")
log "capture dir on srv4: $CAP"
scp -q "10.10.10.4:$CAP/head.jsonl" "10.10.10.4:$CAP/stats.jsonl" "$OUT/" && log "fetched head.jsonl ($(wc -l < "$OUT/head.jsonl") rows)"
log "ACE-QUALITY done"
