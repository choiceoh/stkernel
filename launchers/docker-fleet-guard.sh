#!/bin/bash
# The `docker` the shells find first, so a leased rank cannot be removed by hand.
#
# Every rank of this fleet is load-bearing: take one away and the other three die at the
# rendezvous. On 2026-09-12 a session cleaning up after its own crashed boot ran
# `docker rm -f st-glm53` across the four nodes and took down ANOTHER session's fleet 95
# seconds after it started. The lease named the owner the whole time (engine/base/fleet_lease.py)
# and the raw command never read it -- the launcher's `stop` does, but nothing makes you use it.
#
# So the lease is read where the command actually goes through. /usr/local/bin precedes
# /usr/bin on the login PATH *and* on the PATH `ssh host "..."` runs with, which is the shape
# the incident had: `ssh srv1 "docker rm -f st-glm53"` lands here.
#
# What it judges, and nothing else:
#   kill | stop | rm | restart | pause, aimed at a container that is RUNNING and carries an
#   ST_LEASE_OWNER. A stopped container is forensics, not a fleet -- removing it is allowed,
#   which is what `start-st-glm53.sh` does before every boot. Everything else is exec'd
#   straight through, and so is anything this script cannot judge.
#
# It is a guard rail, not a boundary. It prints the sanctioned command, and it prints the
# override -- you name the owner you are evicting, so it cannot happen by reflex.
set -u
REAL=${ST_DOCKER_REAL:-/usr/bin/docker}
[ -x "$REAL" ] || exec docker "$@"                     # nothing to guard: do not become the bug

# Docker's global options, so `docker -H x rm c` finds `rm` and `docker run img rm -rf /` does not.
takes_value=" -H --host -c --context --config -l --log-level --tlscacert --tlscert --tlskey "
argv=("$@"); i=0; verb=""
while [ $i -lt ${#argv[@]} ]; do
  a=${argv[$i]}
  case "$a" in
    -*) case "$takes_value" in *" $a "*) i=$((i + 2));; *) i=$((i + 1));; esac;;
    *)  verb=$a; i=$((i + 1)); break;;
  esac
done
case "$verb" in kill|stop|rm|restart|pause) ;; *) exec "$REAL" "$@";; esac

# The subcommand's own value-taking flags; everything else bare is a target.
sub_value=" -t --time -s --signal --filter -f "                 # `rm -f` is boolean, handled below
targets=()
while [ $i -lt ${#argv[@]} ]; do
  a=${argv[$i]}
  case "$a" in
    -f|--force|--link|--volumes|-v) i=$((i + 1));;              # rm's booleans
    -t|--time|-s|--signal) i=$((i + 2));;
    --filter) i=$((i + 2));;
    -*) i=$((i + 1));;
    *) targets+=("$a"); i=$((i + 1));;
  esac
done
[ ${#targets[@]} -gt 0 ] || exec "$REAL" "$@"

for t in "${targets[@]}"; do
  info=$("$REAL" inspect --format '{{.State.Running}}{{println}}{{range .Config.Env}}{{println .}}{{end}}' "$t" 2>/dev/null) || continue
  [ "$(printf '%s\n' "$info" | head -1)" = true ] || continue   # stopped: forensics, not a fleet
  owner=$(printf '%s\n' "$info" | sed -n 's/^ST_LEASE_OWNER=//p' | head -1)
  [ -n "$owner" ] || continue                                   # not a leased rank
  [ "${ST_FLEET_OK:-}" = "$owner" ] && continue                 # the owner, or someone who named it
  cat >&2 <<EOF
REFUSED: $t on $(hostname -s) is a running fleet rank leased by
         $owner
         Removing one rank takes all four down -- that is how one session's cleanup killed
         another session's fleet on 2026-09-12, 95 seconds into its boot.

  stop the fleet:   bash launchers/start-st-glm53.sh stop      # checks the lease, all four nodes
  ask for it:       bash launchers/start-st-glm53.sh yield "why"   # it parks its conversations first
  see the holder:   bash launchers/start-st-glm53.sh held

  if you really mean it, name who you are evicting:
      ST_FLEET_OK='$owner' docker ${argv[*]}
EOF
  exit 125
done
exec "$REAL" "$@"
