#!/usr/bin/env bash
# Return THIS node's clean file cache, from the host, the moment before an ST container starts on it.
#
# GB10 has one pool: the page cache and the engine's arena are the same DRAM, and NVRM allocates
# device memory against MemFree, not MemAvailable (launchers/memfree-preflight.sh). The engine's
# admission (engine/base/arena.prepare_allocation) can return only what it sees from inside its
# container -- the pages of its own checkpoint, which its loader reads O_DIRECT anyway -- and
# otherwise evicts cache by faulting its whole allocation plus headroom as anonymous memory. On
# 2026-09-13 that path refused all nine production boots whose rank logs survive (19:29 to 19:48):
#
#   srv2, rank 0   overcommit_memory=2 with CommitLimit 75.8 GiB: the 76.47 GiB mapping that was to
#                  evict the cache is refused outright ([Errno 12]), however clean the 75 GiB of
#                  cache under it
#   srv4, rank 3   MemAvailable 83 GiB, 12 GiB of it cache: faulting 76.47 GiB would take the box
#                  under its SIGTERM line plus margin (8 GiB), so admission refused
#
# The cache was not the engine's: its loader and its tiers read and write O_DIRECT. Dropped from the host
# it costs no allocation and no commit charge. Only clean pages go; nothing any process holds changes.
#
# Prints one line. Exit 0 when the cache was returned, 3 when it could not be (no passwordless sudo):
# the launcher says so and starts the container anyway -- the engine's admission still decides.
set -u
MEMINFO=${ST_MEMINFO:-/proc/meminfo}

mem() {
  awk '/^(MemFree|MemAvailable|Cached):/ {
         printf "%s%s %.1f GiB", sep, substr($1, 1, length($1) - 1), $2 / 1048576; sep = ", "
       }' "$MEMINFO" 2>/dev/null
}

before=$(mem)
# What is dirty cannot be dropped: flush it first, bounded -- a heavy writer can hold sync for minutes.
timeout "${ST_SYNC_TIMEOUT_S:-60}" sync 2>/dev/null || true
if sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null; then
  echo "file cache returned: ${before:-meminfo unreadable} -> $(mem)"
  exit 0
fi
echo "file cache NOT returned (sudo -n refused): ${before:-meminfo unreadable}"
exit 3
