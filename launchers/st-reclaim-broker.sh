#!/usr/bin/env bash
# The host's half of admission's cache reclaim: one short-lived broker per rank, on that rank's node.
#
# The engine's admission (engine/base/arena.prepare_allocation) and its warmup checkpoints need immediately
# free memory, and the page cache in the way is not the engine's to drop -- its container sees only its own
# checkpoint. Its own fallback faults the shortfall as anonymous memory, and a strict-overcommit node refuses
# that mapping however clean the cache under it: srv2 runs overcommit_memory=2 at the default ratio 50, so
# CommitLimit is 75.8 GiB and a 73-76 GiB reclaim never maps (2026-09-13). The kernel setting is the
# operator's; this makes it irrelevant to a boot. The host drops the cache on request, which costs no
# allocation and no commit charge, at the moment the engine is short -- not only before the container starts
# (launchers/st-return-file-cache.sh), because what the boot reads refills it.
#
# The protocol is files in a directory both sides see (glm53-logs is mounted at the same path):
#   heartbeat   touched every poll while a broker serves; the engine asks only when it is fresh
#   request     the engine's id, written whole (rename); the broker takes it and removes it
#   done        three lines: that id, the return script's exit code, its one-line report
#
#   st-reclaim-broker.sh start <dir> <container>   replace this rank's broker and serve in the background
#   st-reclaim-broker.sh serve <dir> <container>   the loop (what start runs)
#   st-reclaim-broker.sh stop <dir>                end this rank's broker
#   st-reclaim-broker.sh stop-all <root>           end every rank's broker under root
#
# A broker ends by itself: when its container has run and is gone, when the container never ran, or after
# ST_RECLAIM_MAX_S (two hours) -- a boot's requests all come in its first minutes.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
SELF="$HERE/$(basename "$0")"

alive() { [ -n "${1:-}" ] && kill -0 "$1" 2>/dev/null; }

stop_dir() {
  local dir=$1 pid
  pid=$(cat "$dir/broker.pid" 2>/dev/null || true)
  if alive "$pid"; then
    kill "$pid" 2>/dev/null || true
    # The broker may be inside a bounded `timeout ... sync` in
    # st-return-file-cache.sh, which delays trap delivery past a second. Wait
    # longer, then escalate, so a fresh broker cannot start while this one is
    # still alive.
    for _ in $(seq 1 100); do alive "$pid" || break; sleep 0.1; done
    if alive "$pid"; then
      kill -9 "$pid" 2>/dev/null || true
      for _ in $(seq 1 20); do alive "$pid" || break; sleep 0.1; done
    fi
  fi
  rm -f "$dir/broker.pid" "$dir/heartbeat"
}

case "${1:-}" in
  start)
    dir=${2:?dir} name=${3:?container}
    mkdir -p "$dir" || exit 1
    stop_dir "$dir"
    rm -f "$dir/request" "$dir/done" "$dir"/request.*.tmp
    # its own session: the ssh that started it returns, and a hang-up of that session does not reach it
    detach=""; command -v setsid >/dev/null 2>&1 && detach=setsid
    $detach nohup bash "$SELF" serve "$dir" "$name" </dev/null >>"$dir/broker.log" 2>&1 &
    for _ in $(seq 1 50); do [ -s "$dir/broker.pid" ] && [ -e "$dir/heartbeat" ] && break; sleep 0.1; done
    pid=$(cat "$dir/broker.pid" 2>/dev/null || true)
    alive "$pid" || { echo "reclaim broker did not start in $dir"; exit 1; }
    echo "reclaim broker serving $dir (pid $pid)"
    ;;
  serve)
    dir=${2:?dir} name=${3:?container}
    echo $$ > "$dir/broker.pid"
    poll=${ST_RECLAIM_POLL_S:-0.5} max=${ST_RECLAIM_MAX_S:-7200} unseen=${ST_RECLAIM_UNSEEN_S:-300}
    started=$(date +%s) checked=0 seen=0
    # Remove the pid only if it still names this process: a stale broker whose
    # TERM arrives after a replacement started must not delete the new broker's
    # pid (which would make stop/stop-all miss a live broker).
    trap 'if [ "$(cat "$dir/broker.pid" 2>/dev/null)" = "$$" ]; then rm -f "$dir/broker.pid" "$dir/heartbeat"; fi; exit 0' TERM INT
    echo "$(date '+%F %T') serving $dir for $name"
    while :; do
      now=$(date +%s)
      touch "$dir/heartbeat"
      if [ -f "$dir/request" ]; then
        id=$(head -c 128 "$dir/request" 2>/dev/null | tr -dc 'A-Za-z0-9._-')
        rm -f "$dir/request"                               # taken: a request written while this one runs waits its turn
        line=$(bash "$HERE/st-return-file-cache.sh" 2>&1)
        rc=$?
        printf '%s\n%s\n%s\n' "$id" "$rc" "$line" > "$dir/done.tmp" && mv -f "$dir/done.tmp" "$dir/done"
        echo "$(date '+%F %T') request $id: exit $rc: $line"
      fi
      if [ $((now - checked)) -ge "${ST_RECLAIM_CHECK_S:-5}" ]; then
        checked=$now
        running=$(docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null || true)
        if [ "$running" = true ]; then
          seen=1
        elif [ "$seen" = 1 ]; then
          echo "$(date '+%F %T') $name is gone: done"; break
        elif [ $((now - started)) -ge "$unseen" ]; then
          echo "$(date '+%F %T') $name never ran: done"; break
        fi
      fi
      [ $((now - started)) -lt "$max" ] || { echo "$(date '+%F %T') lifetime over: done"; break; }
      sleep "$poll"
    done
    rm -f "$dir/heartbeat" "$dir/broker.pid"
    ;;
  stop)
    stop_dir "${2:?dir}"
    ;;
  stop-all)
    root=${2:?root}
    for dir in "$root"/rank*; do [ -d "$dir" ] && stop_dir "$dir"; done
    exit 0
    ;;
  *)
    echo "usage: $0 start|serve <dir> <container> | stop <dir> | stop-all <root>" >&2
    exit 2
    ;;
esac
