#!/usr/bin/env bash
# Size --gpu-memory-utilization from measurement, and account for what the boot
# itself will take before anyone checks.
#
# GB10 has no separate VRAM: NVRM allocates against MemFree, not MemAvailable.
# Three unrelated-looking failures all come from ignoring that:
#
#   Free memory on device cuda:0 (90.19/119.69 GiB) on startup is less than
#     desired GPU memory utilization (0.83, 99.35 GiB)
#   NVRM: Out of memory [NV_ERR_NO_MEMORY] ... kgrctxAllocMainCtxBuffer
#     -- and the rank dies as "Connection closed by peer"
#   Available KV cache memory: -5.67 GiB
#
# Two facts this encodes, both measured on this fleet rather than assumed:
#
# 1. BOOT_COST. Measuring free memory and then launching is not enough: `docker
#    run` pulls a ~31 GB image through the page cache before the engine's
#    startup check runs, and roughly 11 GiB of what you just measured is gone by
#    then. Measured three times at 101.6 -> 90.19, 102.6 -> 90.49, 113 -> 90.5.
#    So the budget is (measured - BOOT_COST - margin), not (measured - margin).
#
# 2. Idle build daemons. Gradle keeps ~10 GiB of JVM heap on whichever node last
#    built the Android client, and respawns after being killed. It is invisible
#    to `docker stats` and to a container-only audit -- the tell is AnonPages,
#    which read 22.6 GiB on that node against 2.2 GiB on its peers. This kills
#    them and reports the spread so a new tenant is visible rather than absorbed.
#
# Raising GMU is usually the fix, not lowering it. A model whose weights plus
# activations need ~78 GiB per rank has no KV at 0.65 and less at 0.60 -- the
# negative number gets worse as you back off, which reads like the opposite of
# what it is.
#
# Usage:
#   memfree-preflight.sh [margin_gib] [node ...]      -> prints a safe GMU
#   eval "$(memfree-preflight.sh --export)"           -> sets GMU_SAFE
set -uo pipefail

BOOT_COST=${BOOT_COST:-11}      # GiB the boot consumes before the engine checks
TOTAL_GIB=${TOTAL_GIB:-119.69}  # per-node MemTotal as vLLM reports it

EXPORT=0
if [ "${1:-}" = "--export" ]; then EXPORT=1; shift; fi
# Default 10, not 3. Both launchers pass 10 explicitly, so this default is only
# reached by a direct call -- and 3 is the exact margin whose boot left the
# fleet at ~3 GiB free against a 4.0 GiB kernel watermark and wedged three
# nodes on 2026-09-04. A stale default is a trap for the next caller.
#
# The margin is only consumed when it LOOKS like one. The usage line calls it
# optional, but `MARGIN=${1:-10}; shift` took the first argument unconditionally,
# so `memfree-preflight.sh 10.10.10.2 10.10.10.1` used "10.10.10.2" AS THE
# MARGIN -- awk read it as 10.10 -- and dropped that node from the list it was
# meant to measure. Both launchers pass the margin first so they never hit it;
# a human following the usage line does. A bare number is a margin, anything
# else (an address) is a node.
if [[ "${1:-}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  MARGIN=$1
  shift
else
  MARGIN=10
fi
NODES=("$@")
[ ${#NODES[@]} -eq 0 ] && NODES=(10.10.10.2 10.10.10.1 10.10.10.3 10.10.10.4)
SSHOPT="-o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=no"
IS_SELF=$(hostname -I 2>/dev/null | tr " " "\n" | grep -cx "10.10.10.2" || true)

probe() {
  cat <<'RE'
# Build daemons hold heap NVRM then cannot allocate around; they rebuild on the
# next build, so reclaiming them costs a warm start and nothing else.
pkill -9 -f '[j]ava.*gradle' 2>/dev/null
pkill -9 -f '[j]ava.*add-opens' 2>/dev/null
pkill -9 -f '[k]otlin-daemon' 2>/dev/null
sync; sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null
sleep 2
awk '/^MemFree:/{f=$2} /^AnonPages:/{a=$2} END{printf "%.1f %.1f\n", f/1048576, a/1048576}' /proc/meminfo
RE
}

echo "# node          MemFree   AnonPages   (GiB, after reclaim)" >&2
MIN=""; ANONS=()
for n in "${NODES[@]}"; do
  if [ "$n" = "10.10.10.2" ] && [ "$IS_SELF" = "1" ]; then
    read -r free anon < <(bash -c "$(probe)")
  else
    read -r free anon < <(ssh $SSHOPT "choiceoh@$n" "bash -s" <<< "$(probe)" 2>/dev/null)
  fi
  [ -z "${free:-}" ] && { echo "  $n  UNREACHABLE -- refusing to size against a node we cannot see" >&2; exit 1; }
  printf "  %-12s %7s   %7s\n" "$n" "$free" "$anon" >&2
  ANONS+=("$anon:$n")
  if [ -z "$MIN" ] || awk "BEGIN{exit !($free < $MIN)}"; then MIN=$free; fi
done

# An outlier in AnonPages is a tenant nobody meant to leave running.
HI=$(printf '%s\n' "${ANONS[@]}" | sort -t: -k1 -rn | head -1)
LO=$(printf '%s\n' "${ANONS[@]}" | sort -t: -k1 -n  | head -1)
if awk "BEGIN{exit !(${HI%%:*} - ${LO%%:*} > 5)}"; then
  echo "  ! ${HI##*:} holds $(printf '%.1f' "${HI%%:*}") GiB anon against ${LO##*:}'s ${LO%%:*} -- an unexpected tenant" >&2
fi

USABLE=$(awk "BEGIN{printf \"%.1f\", $MIN - $BOOT_COST - $MARGIN}")

# This number is an UPPER BOUND from free memory, never a floor from what the
# model needs -- weights and activations come out of the same budget, so KV is
# (GMU x total - overhead) and a small GMU drives it negative. The old code
# clamped a too-small result up to 0.40 and returned it as if it were measured.
# That is the worst of both: 0.40 x 119.69 = 47.9 GiB against ~78 GiB of
# weights and activations per rank, i.e. KV about -30 GiB and a boot that dies
# on "No available memory for the cache blocks".
#
# hy4 never adopts a lower value so the clamp was inert there, but glm53 adopts
# in BOTH directions, so on a node with an unexpected tenant it would have
# taken the fabricated 0.40. Refusing instead makes every caller fall back to
# its configured value, which is what both already do on an unreachable node.
GMU=$(awk "BEGIN{printf \"%.2f\", $USABLE/$TOTAL_GIB}")
if awk "BEGIN{exit !($GMU < 0.40)}"; then
  echo "  ! usable $USABLE GiB gives GMU $GMU -- refusing to size this low." >&2
  echo "    This is an upper bound from free memory, not a floor from what the" >&2
  echo "    model needs: at this budget KV goes negative and the boot dies on" >&2
  echo "    'No available memory for the cache blocks'. Free memory on the" >&2
  echo "    tightest node, or pin GMU deliberately." >&2
  exit 1
fi
echo "  min=$MIN  -boot=$BOOT_COST  -margin=$MARGIN  => usable $USABLE GiB => GMU $GMU" >&2
if [ "$EXPORT" = 1 ]; then echo "export GMU_SAFE=$GMU"; else echo "$GMU"; fi
