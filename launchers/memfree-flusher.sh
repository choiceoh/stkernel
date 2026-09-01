#!/bin/bash
# Unconditional page-cache flusher for the boot window.
#
# WHY THIS EXISTS
#
# NVRM allocates GPU context from MemFree, not MemAvailable. vLLM sizes its KV
# pool from psutil's "available", which counts reclaimable page cache the driver
# cannot actually use. On a unified-memory box the gap between those two numbers
# is the whole problem, and `docker run` opens it wide: pulling a ~31 GB image
# through the page cache costs ~11 GiB of MemFree between the moment the
# preflight measures and the moment the engine allocates. That number is not a
# guess -- memfree-preflight.sh carries it as BOOT_COST=11, measured three times
# on this fleet (101.6 -> 90.19, 102.6 -> 90.49, 113 -> 90.5).
#
# Until now this repo CONCEDED those 11 GiB: the preflight subtracts them from
# the budget and the KV pool is smaller for it. This script reclaims them
# instead, by dropping caches on every node for the whole boot rather than once
# before it.
#
# UNCONDITIONAL, NOT THRESHOLD-TRIGGERED
#
# The obvious design -- flush only when cached memory crosses some line -- fails,
# and it fails silently: the threshold can sit unmet while the allocator is
# already starved, so the same boot command succeeds or dies depending on when
# the kernel happened to reclaim. Flushing unconditionally on a fixed interval
# removes the race instead of racing it. (This matches what the adjacent
# 4x-Spark GLM deployment reports, where making it unconditional was the
# difference between 24 GiB/rank working and failing.)
#
# COST -- read this before running it fleet-wide
#
# drop_caches is global. It evicts every tenant's hot pages, not just ours:
# srv4 runs nemotron-vllm and the solarflow stack alongside this fleet, and they
# will re-read from disk after each flush. It is bounded in time for that
# reason, and it is not wired into any launcher by default -- an operator turns
# it on for a boot and stops it once the engine serves.
#
# USAGE
#
#   memfree-flusher.sh --start [seconds]   # fan out to every node, detached
#   memfree-flusher.sh --status            # who is flushing, and MemFree/Cached
#   memfree-flusher.sh --stop              # kill it everywhere
#   memfree-flusher.sh --run [seconds]     # the loop itself (runs on one node)
#
# Then, and only while it is confirmed running on ALL nodes, the preflight can
# stop conceding the boot cost:
#
#   BOOT_COST=2 launchers/memfree-preflight.sh
#
# The reclaimed budget is only real if the reclaim is actually running, so keep
# the default BOOT_COST=11 unless --status says every node is up.
set -uo pipefail

INTERVAL=${INTERVAL:-60}     # seconds between flushes; the adjacent fleet's
                             # proven value. Tighter reclaims more but evicts
                             # co-tenants more often.
DURATION_DEFAULT=5400        # 90 min: longer than any boot here, short enough
                             # that a forgotten flusher expires on its own.
LOG=${LOG:-/tmp/memfree-flusher.log}
# A pid file, not a pgrep pattern. This is a memory-safety gate: a FALSE
# POSITIVE here tells the operator "every node is flushing, lower BOOT_COST"
# and the next boot OOMs. pgrep -f matches any process whose cmdline merely
# CONTAINS the pattern -- a wrapper shell, an editor, this repo's own tooling --
# so liveness is proven from /proc instead.
PIDF=${PIDF:-/tmp/memfree-flusher.pid}
SSHOPT="-o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=no"
NODES_DEFAULT=(10.10.10.2 10.10.10.1 10.10.10.3 10.10.10.4)

mem_line() {
  awk '/^MemFree:/{f=$2} /^MemAvailable:/{a=$2} /^Cached:/{c=$2}
       END{printf "%.1f %.1f %.1f\n", f/1048576, a/1048576, c/1048576}' /proc/meminfo
}

# --- the loop, run on one node -------------------------------------------
do_run() {
  local dur=${1:-$DURATION_DEFAULT}
  if ! sudo -n true 2>/dev/null; then
    echo "ABORT: passwordless sudo is required to drop caches" >&2
    exit 2
  fi
  local end=$((SECONDS + dur))
  echo $$ > "$PIDF"
  trap 'rm -f "$PIDF"' EXIT INT TERM
  echo "[flusher] pid=$$ interval=${INTERVAL}s duration=${dur}s host=$(hostname)"
  while [ "$SECONDS" -lt "$end" ]; do
    read -r before _ cached < <(mem_line)
    sync
    sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null
    read -r after _ _ < <(mem_line)
    # Log what each flush actually returned. Summing this column after a boot
    # is how we find out whether the 11 GiB is really recoverable here, rather
    # than trusting the neighbour's number.
    printf '[flusher] %s MemFree %.1f -> %.1f (+%.1f GiB, cached was %.1f)\n' \
      "$(date +%H:%M:%S)" "$before" "$after" \
      "$(awk "BEGIN{print $after-$before}")" "$cached"
    sleep "$INTERVAL"
  done
  echo "[flusher] window expired after ${dur}s -- exiting"
}

# --- fleet control --------------------------------------------------------
nodes() { if [ ${#NODES[@]} -eq 0 ]; then printf '%s\n' "${NODES_DEFAULT[@]}"; else printf '%s\n' "${NODES[@]}"; fi; }

# 10.10.10.3 is not reachable directly from every node here -- from srv4 it
# only answers through the head. Try direct, then hop, so this works wherever
# it is invoked instead of silently reporting a node UNREACHABLE (which the
# status check would then read as "not flushing" and, worse, a start would
# leave genuinely unflushed).
HOP=${HOP:-10.10.10.2}
on_node() {  # on_node <ip> <command...>
  local ip="$1"; shift
  if hostname -I 2>/dev/null | tr ' ' '\n' | grep -qx "$ip"; then
    bash -c "$*"
  elif ssh $SSHOPT "choiceoh@$ip" true 2>/dev/null; then
    ssh $SSHOPT "choiceoh@$ip" "$*"
  else
    ssh $SSHOPT "choiceoh@$HOP" "ssh $SSHOPT choiceoh@$ip $(printf %q "$*")"
  fi
}

copy_to_node() {  # copy_to_node <ip> <src>
  local ip="$1" src="$2"
  if hostname -I 2>/dev/null | tr ' ' '\n' | grep -qx "$ip"; then
    cp "$src" /tmp/memfree-flusher.sh
  elif ssh $SSHOPT "choiceoh@$ip" true 2>/dev/null; then
    scp $SSHOPT -q "$src" "choiceoh@$ip:/tmp/memfree-flusher.sh"
  else
    scp $SSHOPT -q "$src" "choiceoh@$HOP:/tmp/memfree-flusher.sh" \
      && ssh $SSHOPT "choiceoh@$HOP" "scp $SSHOPT -q /tmp/memfree-flusher.sh choiceoh@$ip:/tmp/memfree-flusher.sh"
  fi
}

# True only if the pid file names a live process that is actually our loop.
ALIVE_TEST='p=$(cat '"$PIDF"' 2>/dev/null); \
  [ -n "$p" ] && [ -d /proc/$p ] && tr "\\0" " " </proc/$p/cmdline 2>/dev/null | grep -q -- "--run"'

do_start() {
  local dur=${1:-$DURATION_DEFAULT} rc=0
  for ip in $(nodes); do
    printf '  %-14s ' "$ip"
    if ! on_node "$ip" "sudo -n true" 2>/dev/null; then
      echo "no passwordless sudo -- NOT started"; rc=1; continue
    fi
    # Ship the script rather than assume a checkout on every node.
    copy_to_node "$ip" "$0" || { echo "copy failed"; rc=1; continue; }
    on_node "$ip" "chmod +x /tmp/memfree-flusher.sh; \
      p=\$(cat $PIDF 2>/dev/null); [ -n \"\$p\" ] && kill \$p 2>/dev/null; \
      setsid nohup /tmp/memfree-flusher.sh --run $dur >$LOG 2>&1 </dev/null & \
      sleep 2; if $ALIVE_TEST; then echo started; else echo 'FAILED to start'; fi"
  done
  echo
  echo "  stop with: $0 --stop   (do it once the engine serves)"
  return $rc
}

do_stop() {
  for ip in $(nodes); do
    printf '  %-14s ' "$ip"
    on_node "$ip" "p=\$(cat $PIDF 2>/dev/null); \
      if [ -n \"\$p\" ] && kill \$p 2>/dev/null; then rm -f $PIDF; echo stopped; else echo 'not running'; fi"
  done
}

do_status() {
  local all=1
  printf '  %-14s %-12s %8s %8s %8s\n' node flusher MemFree MemAvail Cached
  for ip in $(nodes); do
    local out
    out=$(on_node "$ip" "if $ALIVE_TEST; then echo RUNNING; else echo stopped; fi; \
      awk '/^MemFree:/{f=\$2} /^MemAvailable:/{a=\$2} /^Cached:/{c=\$2} END{printf \"%.1f %.1f %.1f\\n\", f/1048576, a/1048576, c/1048576}' /proc/meminfo" 2>/dev/null)
    local state free avail cached
    state=$(echo "$out" | sed -n 1p); read -r free avail cached < <(echo "$out" | sed -n 2p)
    [ "$state" = "RUNNING" ] || all=0
    [ -n "${state:-}" ] || state=UNREACHABLE
    printf '  %-14s %-12s %8s %8s %8s\n' "$ip" "${state:-UNREACHABLE}" "${free:--}" "${avail:--}" "${cached:--}"
  done
  echo
  if [ "$all" = 1 ]; then
    echo "  all nodes flushing -- BOOT_COST may be lowered for this boot"
  else
    echo "  NOT every node is flushing -- keep BOOT_COST at its default (11)"
  fi
  return $((1 - all))
}

MODE=${1:---status}; shift 2>/dev/null || true
ARG=""
case "${1:-}" in ''|-*) ;; *) ARG="$1"; shift ;; esac
NODES=("$@")

case "$MODE" in
  --run)    do_run "${ARG:-$DURATION_DEFAULT}" ;;
  --start)  do_start "${ARG:-$DURATION_DEFAULT}" ;;
  --stop)   do_stop ;;
  --status) do_status ;;
  *) echo "usage: $0 --start [seconds] | --status | --stop | --run [seconds]" >&2; exit 2 ;;
esac
