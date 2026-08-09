#!/usr/bin/env bash
# fabric-200g.sh — force the CRS812 fabric ports to 200G.
#
# Why this exists: the CRS812's QSFP56/QSFP56-DD ports do NOT auto-negotiate
# above 100G. Left on auto-negotiation the four node links come up at 100Gbps,
# and the DGX Spark's two 100G MACs (one physical QSFP, PCIe Gen5 x4 each) then
# SHARE that single 100G link — measured 49+49 = 98 Gb/s aggregate, exactly half
# the fabric's design. Forcing the speed restores 169.9 Gb/s and buys +9.1%
# prefill (MEASUREMENTS.md).
#
# The setting is persistent in RouterOS config; this script is for rebuild /
# disaster recovery. Links bounce, so stop the engine first.
set -uo pipefail
SW="${CRS812_HOST:-admin@192.168.88.1}"
VIA="${CRS812_VIA:-srv2}"
PORTS="qsfp56-1-1 qsfp56-2-1 qsfp56-dd-1-1 qsfp56-dd-2-1"

for p in $PORTS; do
    ssh -o BatchMode=yes "$VIA" \
        "ssh -o BatchMode=yes $SW '/interface ethernet set $p auto-negotiation=no speed=200G-baseCR4'" \
        || echo "fabric-200g: $p set failed" >&2
done
sleep 40
for p in $PORTS; do
    printf '%-16s ' "$p"
    ssh -o BatchMode=yes "$VIA" "ssh -o BatchMode=yes $SW '/interface ethernet monitor $p once'" 2>/dev/null \
        | grep -E "^ *(status|rate)" | tr -d ' \n'; echo
done
