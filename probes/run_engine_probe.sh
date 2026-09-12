#!/usr/bin/env bash
# Run an ST probe in a fresh container with no overlay mounts.
# Build the runtime first: bash engine/runtime/build.sh
#
# ST_PROBE_HOST=[user@]host runs the container on THAT host's GPU instead of this node's.
# That is the queue's single-GPU lane (bench/fleet.sh): a check that needs one GPU, not
# four, goes to ONE Spark beside production (srv4 by default) and leaves the fleet alone.
# engine/ and probes/ are rsynced under ~/$ST_PROBE_TREE on that host, the container runs
# there on the image production runs there (or $ST_IMAGE), with the same mounts and env,
# and NO fleet lease is taken -- the queue's holder-single is the reservation. Beside
# production the guard is ROOM, the same rule as a --test boot: that box's MemAvailable
# less this check's budget (ST_PROBE_GIB, 8 GiB by default) must clear the 16 GiB floor,
# and no other probe may be there; this waits for it, bounded, and never guesses -- on
# 2026-09-11 a smoke test beside production killed the fleet's worker, not itself. A check
# that needs every rank file cannot run on one node (each Spark holds its own rank): keep
# those on the fleet with `fleet.sh run --gpu --fleet`. A box of its own (ost-97x, the
# operator's Windows PC on the tailnet, once it has sshd in WSL2, docker with the NVIDIA
# runtime and an x86_64 image) works the same way through an ssh alias in the controller's
# ~/.ssh/config, which owns address, user and port; a verdict from another card (an RTX
# 5050 is sm_120, the Sparks are sm_121a) is that card's.
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
probe=${1:?usage: run_engine_probe.sh probes/engine_kernel_check.py [args...]}
shift
[ -f "$repo/$probe" ] || { echo "missing probe: $repo/$probe" >&2; exit 1; }
image=${ST_IMAGE:-st-engine:glm53}
cache=${ST_CACHE:-$HOME/.cache/st}
models=${MODELS:-/home/choiceoh/models}
envs=(-e PYTHONPATH=/repo -e "MAX_JOBS=${MAX_JOBS:-2}")
# ST_* (cache paths, probe switches) and STK_* (the profile's declared D11 knobs) reach the container
for name in $(compgen -v ST_ || true) $(compgen -v STK_ || true); do envs+=(-e "$name=${!name}"); done
gpu=(--gpus all)
if [ "${ST_PROBE_NO_GPU:-0}" = 1 ]; then
  gpu=()
  envs+=(-e CUTE_DSL_ARCH=sm_121a)
fi

# ---- elsewhere: the single-GPU lane's host. Nothing below this block runs for it -- no
# local mounts, no lease -- because none of that is about the GPU it uses.
probe_host=${ST_PROBE_HOST:-}
if [ -n "$probe_host" ] && [ "${probe_host#*@}" != "$(hostname -s)" ] && [ "${probe_host#*@}" != "$(hostname)" ]; then
  # The host is used exactly as given: the controller's ~/.ssh/config (Host ost-97x) names
  # the address, user and port, because that box is not a Spark and not choiceoh's.
  SSHOPT="-o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=30 -o StrictHostKeyChecking=accept-new"
  tree=${ST_PROBE_TREE:-st-probe-tree}          # under that host's home
  home=$(ssh $SSHOPT "$probe_host" "mkdir -p '$tree' .cache/st && printf %s \"\$HOME\"") \
    || { echo "ABORT: $probe_host is unreachable; the single-GPU lane cannot run $probe" >&2; exit 1; }
  rsync -a --delete --exclude __pycache__ -e "ssh $SSHOPT" "$repo/engine" "$repo/probes" "$probe_host:$tree/" \
    || { echo "ABORT: could not push engine/ and probes/ to $probe_host:$tree" >&2; exit 1; }
  mounts=(--mount "type=bind,src=$home/$tree,dst=/repo,readonly"
          --mount "type=bind,src=$home/.cache/st,dst=/cache")
  if ssh $SSHOPT "$probe_host" "test -d '$models'"; then
    mounts+=(--mount "type=bind,src=$models,dst=$models,readonly")
  fi
  # The image production runs on that box, unless the caller named one: the check then
  # judges the deployed build, and a box that serves has it by construction.
  if [ -z "${ST_IMAGE:-}" ]; then
    image=$(ssh $SSHOPT "$probe_host" "docker inspect st-glm53 --format '{{.Config.Image}}' 2>/dev/null \
              || docker image inspect st-engine:glm53 --format '{{index .RepoTags 0}}' 2>/dev/null" | tail -1)
    [ -n "$image" ] || { echo "ABORT: no ST image on $probe_host (no st-glm53 container, no st-engine:glm53); name one with ST_IMAGE" >&2; exit 1; }
  fi
  # Room beside production, right before taking it: bounded wait, then refuse (D3).
  deadline=$(( $(date +%s) + 60 * ${ST_PROBE_WAIT_MINUTES:-10} ))
  until room=$(python3 "$repo/bench/fleet_single.py" evidence --host "$probe_host" --gib "${ST_PROBE_GIB:-8}"); do
    [ "$(date +%s)" -lt "$deadline" ] \
      || { echo "ABORT: no room on $probe_host for ${ST_PROBE_WAIT_MINUTES:-10} min: $room" >&2; exit 1; }
    echo "  waiting for room on $probe_host: $room" >&2
    sleep "${ST_PROBE_POLL_S:-10}"
  done
  NAME=st-probe-$(hostname -s)-$$
  # Killed here (the queue's supervisor stops its process group), the container there must
  # not outlive us: remove it however this ends.
  trap 'ssh $SSHOPT "$probe_host" "docker rm -f $NAME" >/dev/null 2>&1 || true' EXIT INT TERM
  printf -v remote '%q ' docker run --rm --name "$NAME" "${gpu[@]}" "${mounts[@]}" "${envs[@]}" \
    --entrypoint python3 "$image" -u "/repo/$probe" "$@"
  echo "  single GPU: $probe on $probe_host (tree ~/$tree, image $image, budget ${ST_PROBE_GIB:-8} GiB beside production, no fleet lease)" >&2
  rc=0; ssh $SSHOPT "$probe_host" "$remote" || rc=$?
  exit $rc
fi

mkdir -p "$cache"
mounts=(--mount "type=bind,src=$repo,dst=/repo,readonly"
        --mount "type=bind,src=$cache,dst=/cache")
if [ -d "$models" ]; then
  mounts+=(--mount "type=bind,src=$models,dst=$models,readonly")
fi
# A probe takes the same GPUs as a boot, so it runs under the same lease -- the QUEUE's.
# `fleet.sh run --gpu <s> -- bash probes/run_engine_probe.sh <probe>` takes the lease as
# queue/<session> at GO and hands ST_LEASE_OWNER here; this runner only VERIFIES it (one
# record, one owner), and the queue passes the lease on or lets it go when the ticket ends.
# A bare run with no ticket is refused: that was the last way onto the GPUs that no launcher
# and no queue could see, and beside production the OOM killer takes production first.
NAME=st-probe-$$
# A probe that asks for no GPU (ST_PROBE_NO_GPU=1: import and source checks) reserves
# nothing -- taking four Sparks for a CPU check is the opposite of smooth.
if [ "${ST_PROBE_NO_GPU:-0}" != 1 ]; then
  [ -n "${ST_LEASE_OWNER:-}" ] || { cat >&2 <<EOF
ABORT: this probe holds no fleet reservation. Take a ticket -- the queue grants the lease and hands it here:
  bash bench/fleet.sh run --gpu <session> [est] [note] -- bash probes/run_engine_probe.sh $probe
(ST_PROBE_NO_GPU=1 for an import/source check that needs no GPU.)
EOF
    exit 1; }
  # ST_PROBE_NO_LEASE=1: a nested run under the same ticket, already verified by its caller.
  if [ "${ST_PROBE_NO_LEASE:-0}" != 1 ]; then
    # Set, then source: an assignment prefix on `.` is temporary in bash, so the helper's
    # own defaults would be discarded the moment the builtin returns.
    FLEET_REPO=$repo
    . "$repo/launchers/lib/fleet-lease.sh"
    # Verify, do not take. Wait a little rather than abort: the previous holder's handover
    # lands as a transfer to this ticket, and a probe the queue just sent in should not lose
    # its turn to that last second (2026-09-12: three reservations died in two seconds each).
    until fleet_lease verify --owner "$ST_LEASE_OWNER" >/dev/null 2>&1; do
      [ "$(date +%s)" -lt "$(( ${LEASE_DEADLINE:=$(( $(date +%s) + 60 * ${ST_PROBE_WAIT_MINUTES:-10} ))} ))" ] \
        || { echo "ABORT: the lease is not this ticket's after ${ST_PROBE_WAIT_MINUTES:-10} min: $(fleet_lease read 2>/dev/null || echo unreachable)" >&2; exit 1; }
      echo "  waiting for the ticket's lease: $(fleet_lease read 2>/dev/null || echo unreachable)" >&2
      sleep "${ST_PROBE_POLL_S:-10}"
    done
  fi
  # No heartbeat and no release here: the ticket's supervisor on the head node is the lease's
  # evidence (its pid), and the queue is who lets the lease go. A probe killed outright leaks
  # nothing -- the supervisor notices the payload's end and releases or passes the lease on.
fi
docker run --rm --name "$NAME" "${gpu[@]}" "${mounts[@]}" "${envs[@]}" \
  --entrypoint python3 "$image" -u "/repo/$probe" "$@"
