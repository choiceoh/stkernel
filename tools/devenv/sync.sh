#!/bin/bash
# srv1..srv4 and ost-97x (the RTX 5050 PC, WSL2) brought to tools/devenv/versions.env: node.sh on every node at once, each
# node's log under ~/.local/state/devenv-sync and its verdict printed. Run from a node that reaches the others (this one
# without ssh); srv4's devenv-sync timer runs main's copy every morning through the devenv-sync bootstrap. A node that
# does not answer is reported; only an optional one -- ost-97x, whose Windows host sleeps -- is skipped without failing
# the run, while a server that does not answer fails it (after the others are synced), so a stale server is never a
# quiet success.
#
#   bash tools/devenv/sync.sh             apply
#   bash tools/devenv/sync.sh --verify    apply, then a few CPU test files on every node
#   bash tools/devenv/sync.sh --install   on srv4: the bootstrap into ~/.local/bin and the daily user timer
set -euo pipefail
DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NODES=${DEVENV_NODES:-"srv1 srv2 srv3 srv4 ost-97x"}
OPTIONAL=${DEVENV_OPTIONAL-ost-97x}
LOGS=${DEVENV_LOGS:-$HOME/.local/state/devenv-sync}
mkdir -p "$LOGS"

if [ "${1:-}" = --install ]; then
  mkdir -p "$HOME/.local/bin" "$HOME/.config/systemd/user"
  install -m 0755 "$DIR/devenv-sync" "$HOME/.local/bin/devenv-sync"
  install -m 0644 "$DIR/devenv-sync.service" "$DIR/devenv-sync.timer" "$HOME/.config/systemd/user/"
  systemctl --user daemon-reload
  systemctl --user enable --now devenv-sync.timer
  systemctl --user list-timers devenv-sync.timer --no-pager
  exit 0
fi

run_on() {        # host [args]: node.sh there, after the manifest -- this node itself without ssh
  local h=$1
  shift
  if [ "$h" = "$(hostname -s)" ]; then
    cat "$DIR/versions.env" "$DIR/node.sh" | bash -s -- "$@"
  else
    cat "$DIR/versions.env" "$DIR/node.sh" | ssh -o BatchMode=yes -o ConnectTimeout=10 "choiceoh@$h" bash -s -- "$@"
  fi
}

stamp=$(date +%Y%m%d-%H%M%S)
up=()
failed=0
for h in $NODES; do
  if [ "$h" = "$(hostname -s)" ] || ssh -o BatchMode=yes -o ConnectTimeout=10 "choiceoh@$h" true 2>/dev/null; then
    up+=("$h")
  elif [[ " $OPTIONAL " == *" $h "* ]]; then
    echo "== $h: unreachable -- skipped (optional)"
  else
    echo "== $h: UNREACHABLE -- not synced"
    failed=1
  fi
done
[ "${#up[@]}" -gt 0 ] || exit "$failed"
pids=()
for h in "${up[@]}"; do
  run_on "$h" "$@" > "$LOGS/$stamp-$h.log" 2>&1 &
  pids+=("$!")
done
i=0
for h in "${up[@]}"; do
  if wait "${pids[$i]}"; then status=ok; else status=FAILED; failed=1; fi
  echo "== $h: $status ($LOGS/$stamp-$h.log)"
  grep -E '\] (tools|tests):|^  (Ran |OK|FAILED)|^  \[(FAIL|WARN)' "$LOGS/$stamp-$h.log" | grep -v 'WARN\] cuda' || true
  [ "$status" = ok ] || tail -5 "$LOGS/$stamp-$h.log"
  i=$((i + 1))
done
exit "$failed"
