#!/bin/bash
# Shared RoCE GID selection for the TP=4 lane. Sourced, never executed.
#
# This file began (2026-09-04) as the shared machinery of the two vLLM lane
# launchers. The vLLM overlay lanes were decommissioned (2026-09-18) and the
# profile/EXTRA_ENV/overlay-target guards retired with them; what remains is
# the one piece the ST boot payload and the fleet audit still share.
#
# Consumers: launchers/start-st-glm53.sh and launchers/start-st-qwen38.sh
# source this inside each rank container and eval $CT_GID_PRELUDE before the
# engine starts; launchers/fleet-audit.sh runs the same prelude on every node
# so its GID verdict is, by construction, the value the boot exports.

# --- RoCE GID index ----------------------------------------------------------
# The RoCE-v2 IPv4 GID index is a PER-NODE, PER-BOOT property: enabling IPv6 on
# the fabric NIC changes how the GID table is laid out, so the index that means
# "RoCE v2 over IPv4" differs between machines and moves across reboots. The
# runbook records srv1 at 3 while srv2 and srv4 read 4.
#
# A single -e from the head cannot carry four different values, so the
# detection runs per rank, inside the container, and this is the one copy of
# it. It is a literal string rather than a function: the boot payload is
# delivered as text, so what is shared is source, not a call. Defined through
# a quoted heredoc so the single quotes in `tr ',' ' '` survive.
CT_GID_PRELUDE=$(cat <<'GIDEOF'
# Auto-detect the RoCE-v2 IPv4 GID index (per node, re-numbers across reboots).
for HCA in $(echo "${NCCL_IB_HCA}" | tr ',' ' '); do
  for i in $(seq 0 15); do
    t=$(cat /sys/class/infiniband/$HCA/ports/1/gid_attrs/types/$i 2>/dev/null || true)
    g=$(cat /sys/class/infiniband/$HCA/ports/1/gids/$i 2>/dev/null || true)
    case "$t" in *"RoCE v2"*) case "$g" in *0000:0000:0000:0000:0000:ffff:*) export NCCL_IB_GID_INDEX=$i; break 2;; esac;; esac
  done
done
GIDEOF
)
