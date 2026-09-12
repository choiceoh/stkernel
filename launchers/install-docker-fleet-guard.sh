#!/usr/bin/env bash
# Put launchers/docker-fleet-guard.sh in front of docker on every node, and prove it still
# passes everything else through before leaving it there.
#
#   bash launchers/install-docker-fleet-guard.sh            # install on all four Sparks
#   bash launchers/install-docker-fleet-guard.sh --check    # say what is installed, change nothing
#   bash launchers/install-docker-fleet-guard.sh --uninstall
#
# /usr/local/bin precedes /usr/bin on the login PATH and on the PATH `ssh host "..."` runs
# with, so the guard is what `ssh srv1 "docker rm -f st-glm53"` finds. `/usr/bin/docker` is
# untouched and one path away -- this stops a reflex, not a decision.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
NODES=(10.10.10.2 10.10.10.1 10.10.10.3 10.10.10.4)
GUARD=$REPO/launchers/docker-fleet-guard.sh
TARGET=/usr/local/bin/docker
SSHOPT="-o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new"
SELF_IPS=" $(hostname -I 2>/dev/null) "
node_sh() { local ip=$1; shift; if [[ "$SELF_IPS" == *" $ip "* ]]; then bash -c "$*"; else ssh $SSHOPT "choiceoh@$ip" "$@"; fi; }

bash -n "$GUARD"
want=$(sha256sum < "$GUARD" | cut -c1-12)

case "${1:-install}" in
  --check)
    for ip in "${NODES[@]}"; do
      have=$(node_sh "$ip" "[ -f $TARGET ] && sha256sum < $TARGET | cut -c1-12 || true" 2>/dev/null || true)
      printf '%-13s %s\n' "$ip" "${have:-none}$([ "$have" = "$want" ] && echo '  (current)')"
    done; exit 0 ;;
  --uninstall)
    for ip in "${NODES[@]}"; do
      node_sh "$ip" "sudo rm -f $TARGET && command -v docker" | sed "s|^|$ip: |"
    done; exit 0 ;;
  install) ;;
  *) echo "usage: $0 [install|--check|--uninstall]" >&2; exit 2 ;;
esac

for ip in "${NODES[@]}"; do
  # staged, tested, and only then moved into place: a broken shim here breaks every docker
  # command on a node that serves, so it never becomes $TARGET until it has answered `ps`.
  if [[ "$SELF_IPS" == *" $ip "* ]]; then cp "$GUARD" /tmp/.docker-guard.$$
  else scp -q $SSHOPT "$GUARD" "choiceoh@$ip:/tmp/.docker-guard.$$"; fi
  node_sh "$ip" "set -e
    bash -n /tmp/.docker-guard.$$
    ST_DOCKER_REAL=/usr/bin/docker bash /tmp/.docker-guard.$$ ps -q >/dev/null
    sudo install -m 0755 /tmp/.docker-guard.$$ $TARGET
    rm -f /tmp/.docker-guard.$$
    command -v docker | grep -q '^/usr/local/bin/docker\$'
    docker ps -q >/dev/null
    echo 'installed, docker still answers'" | sed "s|^|$ip: |"
done
