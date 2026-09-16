#!/usr/bin/env bash
# srv2: queue an st-hold of the route-dump debug commit, run the ACE route probe once the door is up, end the hold.
#   setsid nohup bash run_ace_hold.sh <session> <full sha> <est_min> > ~/expert-capture/ace-routes/<session>.runner.log 2>&1 &
set -u
S=$1; SHA=$2; EST=$3
OUT=~/expert-capture/ace-routes
mkdir -p "$OUT"
Q=~/st-worktrees/queue-main-$(git -C ~/stkernel rev-parse --short=8 origin/main)
TICKET_LOG=$OUT/$S.ticket.log
log() { echo "$(date '+%F %T') $S: $*"; }

log "queue checkout $Q, sha $SHA, est $EST"
( cd "$Q" && exec bash bench/fleet.sh st-hold "$S" "$SHA" "$EST" "ACE byte-reduction: raw routes of C=1 verify steps, 12 onepass JSON questions (debug, never merge)" ) > "$TICKET_LOG" 2>&1 &
TICKET=$!
log "ticket waiter pid $TICKET"

# Wait for the door (up to 6 h in the queue), or for the ticket to exit.
deadline=$(( $(date +%s) + 21600 ))
while :; do
  if grep -q "door up:" "$TICKET_LOG" 2>/dev/null && grep -q "queue/$S" ~/glm53-logs/st-fleet.lock 2>/dev/null; then
    break
  fi
  if ! kill -0 "$TICKET" 2>/dev/null; then
    log "ticket exited before the door came up"; tail -20 "$TICKET_LOG"; exit 1
  fi
  if [ "$(date +%s)" -gt "$deadline" ]; then
    log "gave up waiting"; ( cd "$Q" && bash bench/fleet.sh cancel "$S" ); exit 1
  fi
  sleep 20
done
log "door up"
sleep 20
python3 ~/expert-capture/ace_route_probe.py --prompts ~/expert-capture/style-prompts.jsonl --out "$OUT/$S.rows.jsonl" > "$OUT/$S.client.log" 2>&1
log "probe exit $?"
touch ~/glm53-logs/st-bracket/"$S"/stop 2>/dev/null || ( cd "$Q" && bash bench/fleet.sh cancel "$S" )
log "stop requested"
wait "$TICKET"
log "hold ended"
log "ACE-HOLD done"
