#!/bin/bash
# Read-only fleet audit for the maintenance-window items the 2026-08-11
# ledger left "operator decision pending" (MEASUREMENTS.md ★★UMA 챕터 +
# 플릿 위생 관찰): UMA compaction sysctl + persistence, THP defrag policy,
# high-order page health, RoCE GID/IPv6 layout, fabric link speed, GPU
# clock-cap service, zombie vllm-tp2 containers, bluetoothd runaway.
# Changes NOTHING — apply commands live in RUNBOOK_MAINTENANCE.md.
# RUN ON srv2 (workers audited over ssh).
set -euo pipefail
SSHOPT="-o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=no"
NODES="srv2:local srv3:10.10.10.3 srv1:10.10.10.1 srv4:10.10.10.4"

AUDIT=$(cat <<'AUDITEOF'
echo "-- UMA/컴팩션 --"
echo "  compaction_proactiveness=$(cat /proc/sys/vm/compaction_proactiveness 2>/dev/null || echo n/a)"
persist=$(grep -rhs "compaction_proactiveness" /etc/sysctl.d /etc/sysctl.conf 2>/dev/null | tr -d "[:space:]" | paste -sd, -)
echo "  persisted: ${persist:-NONE}"
echo "  THP enabled=[$(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null | grep -o "\[.*\]" | tr -d "[]")] defrag=[$(cat /sys/kernel/mm/transparent_hugepage/defrag 2>/dev/null | grep -o "\[.*\]" | tr -d "[]")]"
echo "  kcompactd0 cputime=$(ps -o cputime= -C kcompactd0 2>/dev/null | tr -d " " || true)"
awk '$1 ~ /^(thp_fault_alloc|pgmigrate_success|compact_stall)$/ {print "  vmstat "$0}' /proc/vmstat
awk '/Normal/ {print "  buddy zone="$4" order9="$(NF-1)" order10="$NF}' /proc/buddyinfo | head -2
echo "-- RoCE GID / IPv6 / 링크 --"
for H in /sys/class/infiniband/*; do
  [ -d "$H/ports/1" ] || continue
  hn=$(basename "$H")
  for i in 0 1 2 3 4 5 6 7 8 9; do
    g=$(cat "$H/ports/1/gids/$i" 2>/dev/null) || continue
    case "$g" in 0000:0000:0000:0000:0000:0000:0000:0000) continue;; esac
    t=$(cat "$H/ports/1/gid_attrs/types/$i" 2>/dev/null || echo "?")
    echo "  $hn gid[$i] [$t] $g"
  done
done
for IF in /sys/class/net/en*; do
  ifn=$(basename "$IF")
  case "$ifn" in *np0) ;; *) continue;; esac
  d=$(cat /proc/sys/net/ipv6/conf/$ifn/disable_ipv6 2>/dev/null || echo n/a)
  n6=$(ip -6 addr show dev "$ifn" 2>/dev/null | grep -c inet6 || true)
  sp=$(cat "$IF/speed" 2>/dev/null || echo n/a)
  echo "  $ifn speed=${sp}Mb ipv6_disabled=$d inet6_addrs=$n6"
done
echo "-- GPU 클럭 --"
echo "  gpu-clock-cap.service: $(systemctl is-enabled gpu-clock-cap.service 2>/dev/null || echo ABSENT)"
nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,temperature.gpu --format=csv,noheader 2>/dev/null | sed "s/^/  sm,max,temp: /" || echo "  nvidia-smi n/a"
echo "-- 잔존물 --"
z=$(docker ps -a --format "{{.Names}}\t{{.Status}}" 2>/dev/null | grep -i "vllm-tp2" || true)
if [ -n "$z" ]; then printf "%s\n" "$z" | sed "s/^/  zombie: /"; else echo "  vllm-tp2: 없음"; fi
bt=$(ps -o cputime= -C bluetoothd 2>/dev/null | tr -d " ")
echo "  bluetoothd cputime=${bt:-미가동}"
AUDITEOF
)

for spec in $NODES; do
  name=${spec%%:*}; ip=${spec##*:}
  echo "===== $name ($ip) ====="
  if [ "$ip" = "local" ]; then
    bash -c "$AUDIT" || echo "  (audit failed)"
  else
    ssh $SSHOPT "choiceoh@$ip" "$AUDIT" || echo "  (unreachable)"
  fi
done
echo
echo "합격 기준 (RUNBOOK_MAINTENANCE.md): proactiveness=0 ×4 + persisted != NONE,"
echo "GID 인덱스 4노드 동일, np0 speed=200000Mb, clock-cap 4노드 enabled+~2000MHz,"
echo "buddy order10 > 0 (재부팅 직후), 잔존물 0."
