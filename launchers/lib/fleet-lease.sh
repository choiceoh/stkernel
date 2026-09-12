# The fleet lease, for shell callers. Source this; do not run it.
#
# ONE file, on the head node, because that is where the containers that are its evidence
# run and where every session can reach it -- homes are NOT shared between the Sparks, so
# a lease written "here" is a lease nobody else can see. That was a real hole: the probe
# runner took $HOME/st-fleet.lock on whatever node it ran on while a boot took the head
# node's, and the two never met (2026-09-12, found auditing §28).
FLEET_HEAD=${FLEET_HEAD:-10.10.10.2}                    # rank 0 = srv2 (base/comm.NODES)
# Under glm53-logs, which every ST container already bind-mounts at the SAME path: the
# engine has to READ this file to notice a yield request, and ~/st-fleet.lock was not
# mounted, so the whole handover was inert on the fleet (found reviewing §28/§29).
# _write replaces the inode, so a single-file bind mount would go stale anyway.
FLEET_LEASE_PATH=${FLEET_LEASE_PATH:-/home/choiceoh/glm53-logs/st-fleet.lock}
FLEET_LEASE_LEGACY=${FLEET_LEASE_LEGACY:-/home/choiceoh/st-fleet.lock}   # older launchers wrote here
FLEET_LEASE_SSH=${FLEET_LEASE_SSH:--o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new}

# fleet_lease <action> [args...] -- runs engine/base/fleet_lease.py on the head node by
# piping it there, so taking a lease never needs a tree rsynced onto a node first.
# A node cannot ssh to itself here (srv2 refuses its own key), and the head node is where
# the queue's controller runs -- so the most important caller was the one that could not ask.
_fleet_lease_is_head() {
  case " $(hostname -I 2>/dev/null) $(hostname -s) " in *" $FLEET_HEAD "*) return 0 ;; esac
  return 1
}

fleet_lease() {
  local module="${FLEET_REPO:?FLEET_REPO must name this checkout}/engine/base/fleet_lease.py"
  if _fleet_lease_is_head; then
    python3 "$module" "$@" --path "$FLEET_LEASE_PATH"
  else
    local quoted
    printf -v quoted '%q ' "$@" --path "$FLEET_LEASE_PATH"
    ssh $FLEET_LEASE_SSH "choiceoh@$FLEET_HEAD" "python3 - $quoted" < "$module"
  fi
}

# fleet_lease_beat <owner> -- keep a long hold fresh while its evidence is on another node.
# The head node's docker cannot see a probe container running on srv4, so without this the
# lease would look stale after GRACE_S and another session could take the fleet underneath it.
fleet_lease_beat() {
  local owner=$1
  ( while sleep 120; do fleet_lease renew --owner "$owner" >/dev/null 2>&1 || exit 0; done ) &
  echo $!
}
