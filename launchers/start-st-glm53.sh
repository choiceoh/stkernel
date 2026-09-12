#!/usr/bin/env bash
# Boot the ST engine's GLM-5.3 on the four Sparks: one container per node,
# inside the standalone ST image, the engine tree mounted at /repo,
# this node's rank file, /cache for the JIT builds, and the production
# launcher's NCCL/RoCE environment (start-glm53-nvfp4-tp4.sh 431-445).
# The image is built on each node from the rsynced tree (engine/runtime/build.sh) before the container starts.
#
#   bash launchers/start-st-glm53.sh            # start all four (rank 0=srv2, rank 1=srv1, then srv3/srv4)
#   bash launchers/start-st-glm53.sh stop       # docker rm -f st-glm53 on every node
#
# Every boot holds the fleet LEASE (engine/base/fleet_lease.py), and who may start one is decided
# by how it got the lease:
#   ST_LEASE_OWNER=queue/<s>  a ticket's boot: bench/fleet.sh took the lease at GO and hands the
#                             owner here; this script only VERIFIES it. Sessions get a window with
#                             `bench/fleet.sh st-hold <session> <sha>`, not by running this by hand.
#   ST_LEASE_KIND=production  the supervisor's / deploy-watch's own boot: acquires a `production`
#                             lease, the fleet's default state, which the queue asks to hand over
#                             only through the quiet gate.
#   ST_LEASE_KIND=session     a session's own boot by hand: a `session` lease the queue never asks
#                             to hand over (a ticket behind it waits). Kept until the queue can boot
#                             for sessions (`fleet.sh st-hold`); say it, it is not the default.
# Anything else is refused: a boot nobody reserved is the collision of 09-11 19:42 waiting to happen.
#
# `stop` is the only way to take these containers down, and the nodes enforce it: a running
# rank carries its lease owner, and launchers/docker-fleet-guard.sh (installed in front of
# docker) refuses `docker rm|kill|stop` on one unless you name the owner you are evicting.
# And only the holder stops its own boot: a ticket by its exact owner, production by its kind,
# a person with STOP_FORCE=1 (the operator's word).
#   bash launchers/start-st-glm53.sh logs [r]   # tail rank r's container log
#
# Rank order is base/comm.NODES (srv2, srv1, srv3, srv4): rank 0 hosts the rendezvous store, so it is the head.
# Never beside a serving vLLM or a q38 stack: check `docker ps` on every node
# first -- this script refuses if a glm53*/q38* container is up.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
NODES=(10.10.10.2 10.10.10.1 10.10.10.3 10.10.10.4)
IMAGE=${ST_IMAGE:-${IMAGE:-st-engine:glm53}}
PORT=${PORT:-8000}
KV_ARG=""
if [ -n "${ST_KV_GIB:-}" ]; then
  [[ "$ST_KV_GIB" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "ST_KV_GIB must be a positive GiB byte budget" >&2; exit 2; }
  KV_ARG="--kv-gib $ST_KV_GIB"
fi
PRODUCTION_ARG=""
case "${ST_PRODUCTION:-0}" in
  0) ;;
  1) PRODUCTION_ARG="--production" ;;
  *) echo "ST_PRODUCTION must be 0 or 1" >&2; exit 2 ;;
esac
RANKS_DIR=${RANKS_DIR:-/home/choiceoh/models/st-glm53-nvidia-tp4-9391}
CKPT=${CKPT:-/home/choiceoh/models/st-glm53-nvidia-tp4-9391}
DRAFTER=${DRAFTER:-/home/choiceoh/models/GLM-5.3-Flash-DFlash2}
ENGINE_DIR=${ST_ENGINE_DIR:-/home/choiceoh/st-engine}    # production can pin a release directory on every node
CACHE_DIR=${CACHE_DIR:-/home/choiceoh/glm53-cache}
TIER_DIR=${ST_TIER_DIR:-/home/choiceoh/glm53-logs/st-tier}
DUMP_DIR=${ST_DUMP_DIR:-/home/choiceoh/glm53-logs/st-dumps}
SSHOPT="-o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new"
NAME=st-glm53
LEASE_OWNER=${LEASE_OWNER:-$(whoami)@$(hostname -s)/$$}   # who holds the fleet, for the lease record
# Under glm53-logs, the one directory every ST container already mounts at this path --
# the engine must READ its lease to notice a yield request, and ~/st-fleet.lock was not
# mounted into any container, so the handover was inert on the fleet.
LOCK=${FLEET_LEASE_PATH:-/home/choiceoh/glm53-logs/st-fleet.lock}
LEGACY_LOCK=/home/choiceoh/st-fleet.lock                   # older launchers still write here

# A node cannot ssh to itself (srv2 refuses its own key), and the head runs this script: run its own
# commands in a local shell instead. Same for the tree push -- and if this checkout *is* the node's
# engine tree there is nothing to push.
SELF_IPS=" $(hostname -I 2>/dev/null) "
is_self() { [[ "$SELF_IPS" == *" $1 "* ]]; }
node_sh() { local ip=$1; shift; if is_self "$ip"; then bash -c "$*"; else ssh $SSHOPT "choiceoh@$ip" "$@"; fi; }
push_tree() {
  local ip=$1
  if is_self "$ip"; then
    [ "$(readlink -f "$REPO")" = "$(readlink -f "$ENGINE_DIR")" ] && return 0
    mkdir -p "$ENGINE_DIR"
    rsync -a --delete --exclude __pycache__ "$REPO/engine" "$REPO/launchers" "$META" "$ENGINE_DIR/"
  else
    rsync -a --delete -e "ssh $SSHOPT" --exclude __pycache__ "$REPO/engine" "$REPO/launchers" "$META" "choiceoh@$ip:$ENGINE_DIR/"
  fi
}

# Assignment prefixes on `.` are temporary in bash -- the helper's own defaults would be
# discarded when the builtin returns. Set, then source.
use_lease() {
  FLEET_REPO=$REPO; FLEET_HEAD=${NODES[0]}; FLEET_LEASE_PATH=$LOCK; FLEET_LEASE_SSH=$SSHOPT
  . "$REPO/launchers/lib/fleet-lease.sh"
}

# Preserve argument boundaries when notes contain spaces; the lease lives at
# the canonical, container-mounted path selected above.
lease() {
  if is_self "${NODES[0]}"; then
    python3 "$REPO/engine/base/fleet_lease.py" "$@" --path "$LOCK"
  else
    local quoted
    printf -v quoted '%q ' "$@"
    ssh $SSHOPT "choiceoh@${NODES[0]}" "python3 - $quoted --path $LOCK" < "$REPO/engine/base/fleet_lease.py"
  fi
}

case "${1:-start}" in
  stop)
    held_owner=$(lease owner --container "$NAME") || {
      echo "ABORT: refusing to stop another fleet owner's lease" >&2; exit 1;
    }
    legacy_owner=$(LOCK=$LEGACY_LOCK lease owner --container "$NAME") || {
      echo "ABORT: refusing to stop another owner's legacy lease" >&2; exit 1;
    }
    # Only the holder stops its own boot. Every boot is named st-glm53, so a `stop` resolved by
    # container name alone would let the production supervisor's crash recovery evict a ticket's
    # boot 90 s in (its door is elsewhere, so its health checks fail). A ticket names its owner
    # exactly (ST_LEASE_OWNER), production names its kind (ST_LEASE_KIND=production -- whichever
    # supervisor pid took the lease, the loop restarts and adopts), a person says STOP_FORCE=1.
    # A record from before kinds (a plain-text lock, or a JSON lease naming no kind -- the
    # production lease of 2026-09-12 is one) is judged as it always was: by the container name
    # `owner` resolved above, and a stop is allowed.
    held_origin=$(lease origin 2>/dev/null || echo unreadable)
    if [ -n "$held_owner" ] && [ "$held_origin" = explicit ]; then
      if [ -n "${ST_LEASE_OWNER:-}" ]; then
        [ "$held_owner" = "$ST_LEASE_OWNER" ] || { echo "ABORT: the fleet is held by $held_owner, not by this ticket ($ST_LEASE_OWNER)" >&2; exit 1; }
      elif [ "${ST_LEASE_KIND:-}" = production ] || [ "${ST_LEASE_KIND:-}" = session ]; then
        held_kind=$(lease kind 2>/dev/null || echo unreadable)
        case "$held_kind" in "$ST_LEASE_KIND"|free) ;; *) echo "ABORT: the fleet is held by $held_owner ($held_kind), not by a $ST_LEASE_KIND boot; the queue hands it back when its tickets are done" >&2; exit 1 ;; esac
      elif [ "${STOP_FORCE:-0}" != 1 ]; then
        echo "ABORT: the fleet is held by $held_owner. Stop it from its own side (bench/fleet.sh cancel or release; the supervisor for production). STOP_FORCE=1 is the operator's word." >&2; exit 1
      fi
    fi
    # `stop` has already resolved the lease above -- owner_for RAISES when another workload
    # holds it -- so this is the deliberate path, and it names the owner it is evicting for
    # the node's docker guard (launchers/docker-fleet-guard.sh). Read the owner off the
    # container rather than from $held_owner: a rank orphaned by a lost lease file must still
    # be stoppable, and the lease check that guards this line already happened.
    for ip in "${NODES[@]}"; do node_sh "$ip" "ST_FLEET_OK=\$(docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' $NAME 2>/dev/null | sed -n 's/^ST_LEASE_OWNER=//p' | head -1) docker rm -f $NAME >/dev/null 2>&1 && echo '$ip: stopped' || echo '$ip: none'"; done
    use_lease
    # A ticket's lease is the queue's to pass on or let go at the ticket's end, not this
    # script's to release: released here, the next waiting ticket could not be handed it and
    # the production supervisor would relaunch into the window. Say the boot is down instead.
    if [ -n "$held_owner" ] && [ "${ST_LEASE_OWNER:-}" = "$held_owner" ]; then
      lease publish --owner "$held_owner" --state phase=stopped >/dev/null 2>&1 || true
    elif [ -n "$held_owner" ]; then
      lease release --owner "$held_owner"
    fi
    node_sh "${NODES[0]}" "rm -f $LEGACY_LOCK" >/dev/null 2>&1 || true
    exit 0 ;;
  yield)
    # Ask whoever holds the fleet to finish, park its conversations and let go, then WAIT
    # for that to happen -- asking and leaving the caller to poll is not a handover.
    # An operator's tool: the queue asks production by itself (through the quiet gate) and
    # never asks a session's boot, so this is for a person who has decided. The handover is a
    # TRANSFER to the requester named here, so the fleet is this owner's the moment the holder
    # lets go -- start with that owner in ST_LEASE_OWNER.
    use_lease
    asked=$(fleet_lease yield --requester "'$LEASE_OWNER'" --kind session --pid $$ --host "$(hostname -s)" \
              --note "'${2:-another session needs the fleet}'") || exit 1
    case "$asked" in free) echo "the fleet is already free"; exit 0 ;; esac
    echo "asked: $asked"
    deadline=$(( $(date +%s) + 60 * ${YIELD_WAIT_MINUTES:-30} ))
    while [ "$(date +%s)" -lt "$deadline" ]; do
      held=$(fleet_lease read 2>/dev/null || echo unreachable)
      case "$held" in
        free|free\ *) echo "the fleet is free: start when ready"; exit 0 ;;
        "session $LEASE_OWNER "*) echo "the fleet is yours -- handed to $LEASE_OWNER. Start with: ST_LEASE_OWNER='$LEASE_OWNER' $0"; exit 0 ;;
        unreachable) ;;
        *) echo "  waiting: $held" ;;
      esac
      sleep "${YIELD_POLL_S:-10}"
    done
    echo "ABORT: the holder did not let go within ${YIELD_WAIT_MINUTES:-30} min: $(fleet_lease read)" >&2
    exit 1 ;;
  held)
    use_lease
    fleet_lease read; exit 0 ;;
  logs)
    r=${2:-0}; node_sh "${NODES[$r]}" "docker logs --tail 60 $NAME"; exit 0 ;;
  start) ;;
  *) echo "usage: $0 [start|stop|yield [reason]|held|logs r]" >&2; exit 2 ;;
esac

# refuse to share the fleet: a serving/other container on any node, or another runner's lock on the head
# (the lock is a file on rank 0's node; `stop` removes it; every fleet runner -- every session -- honours it)
for ip in "${NODES[@]}"; do
  busy=$(node_sh "$ip" "docker ps --format '{{.Names}}' | grep -E '^(glm53|q38|vllm|st-)' || true")
  [ -z "$busy" ] || { echo "ABORT: $ip runs $busy -- the fleet is taken (hand off the queue, do not squat)" >&2; exit 1; }
done
# The lease is the engine's own (engine/base/fleet_lease.py): an owner, the container that
# is its evidence, an estimate and a reason -- so a crashed boot goes stale on its own
# instead of needing a human to delete a file, and a live one names who to ask.
# Piped, not rsynced: taking the lease must not touch $ENGINE_DIR, which a live session
# may have mounted into its containers. The module is stdlib-only, so `python3 -` is enough.
use_lease

# The lease is ONE record and bench/fleet.sh is its authority for tickets: a ticket's boot arrives
# with the owner the queue took at GO and only verifies it, production takes a lease of its own
# kind, and nothing else boots. (Before this the queue's holder file and this lock were two records
# that refused each other in both directions, and a ticket's own boot was refused by the holder
# file that was its -- so no ST boot could run under the queue at all, 2026-09-12.)
legacy=$(node_sh "${NODES[0]}" "cat $LEGACY_LOCK 2>/dev/null || true")
[ -z "$legacy" ] || { echo "ABORT: a session on the older lock path holds the fleet: $legacy ($LEGACY_LOCK on ${NODES[0]}); 'stop' from that side" >&2; exit 1; }
LEASE_MODE=""
if [ -n "${ST_LEASE_OWNER:-}" ]; then
  LEASE_OWNER=$ST_LEASE_OWNER
  held=$(lease verify --owner "$LEASE_OWNER" 2>&1) \
    || { echo "ABORT: this boot's ticket does not hold the fleet: $held" >&2; exit 1; }
  echo "lease: verified, $held"
  LEASE_MODE=ticket
elif [ "${ST_LEASE_KIND:-}" = production ]; then
  LEASE_OWNER=${LEASE_OWNER_PRODUCTION:-production/$(hostname -s)/$$}
  lease acquire --owner "$LEASE_OWNER" --kind production --container "$NAME" --est-minutes "${LEASE_MINUTES:-0}" \
        --note "${LEASE_NOTE:-production st-glm53 on four Sparks}" \
    || { echo "ABORT: $(lease read 2>/dev/null || echo 'the fleet lease refused')" >&2; exit 1; }
  LEASE_MODE=production
elif [ "${ST_LEASE_KIND:-}" = session ]; then
  # A session's own boot, by hand, said so: a `session` lease the queue never asks to hand over
  # (45차 §91) -- a ticket behind it waits. This is the path sessions used until now, kept until
  # the queue can boot for them (`fleet.sh st-hold`, the ST bracket runner); it is not the default,
  # because a boot that did not say what it is was the collision of 09-11 19:42.
  lease acquire --owner "$LEASE_OWNER" --kind session --container "$NAME" --est-minutes "${LEASE_MINUTES:-45}" \
        --note "${LEASE_NOTE:-st-glm53 on four Sparks}" \
    || { echo "ABORT: $(lease read 2>/dev/null || echo 'the fleet lease refused')" >&2; exit 1; }
  LEASE_MODE=session
else
  cat >&2 <<EOF
ABORT: this boot holds no reservation, and a boot nobody reserved is not started. Say what it is:
  ST_LEASE_OWNER=queue/<session>   a ticket's boot -- the queue takes the lease at GO and hands the owner here
  ST_LEASE_KIND=production $0       the supervisor's / deploy-watch's own production boot
  ST_LEASE_KIND=session $0          a session's own boot by hand (never asked to hand over; tickets wait behind it)
EOF
  exit 1
fi
stage=$(mktemp -d)
launched=0
cleanup() {
  rm -rf "$stage"
  if [ "$launched" = 0 ]; then
    # a boot of our own that did not come up gives its lease back; a ticket's lease is the
    # queue's to pass on or let go -- say what happened on it instead
    case "$LEASE_MODE" in
      production|session) lease release --owner "$LEASE_OWNER" >/dev/null || true ;;
      ticket) lease publish --owner "$LEASE_OWNER" --state phase=boot-failed >/dev/null 2>&1 || true ;;
    esac
  fi
}
trap cleanup EXIT


NCCL_ENV="-e NCCL_P2P_LEVEL=SYS -e TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200 \
-e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0 \
-e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e GLOO_SOCKET_IFNAME=${GLOO_IFNAME:-enP2p1s0f0np0} \
-e NCCL_CROSS_NIC=1 -e NCCL_PROTO=LL,LL128,Simple -e NCCL_CUMEM_ENABLE=0 \
-e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET \
-e NCCL_NVLS_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN \
-e NCCL_MIN_NCHANNELS=16 -e NCCL_MAX_NCHANNELS=16 -e NCCL_NCHANNELS_PER_NET_PEER=4 \
-e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
-e TRITON_CACHE_DIR=/cache/triton -e TILELANG_CACHE_DIR=/cache/tilelang \
-e DG_JIT_CACHE_DIR=/cache/deep_gemm -e ST_MLA_BUILD_ROOT=/cache/mla -e FLASHINFER_WORKSPACE_BASE=/cache \
-e ST_DENSE_BUILD_ROOT=/cache/st-dense -e ST_ONESHOT_BUILD_ROOT=/cache/st-oneshot -e MAX_JOBS=2"
# the profile's declared D11 knobs (STK_*, boot.declared) travel from this shell into every rank; an undeclared one kills the boot
for v in $(compgen -v STK_ || true); do NCCL_ENV="$NCCL_ENV -e $v=${!v}"; done

# the checkpoint's metadata travels with the engine tree: a node needs its rank file, the drafter and these few files,
# not the full HF checkpoint
META="$REPO/build/st-glm53-meta"; mkdir -p "$META"
cp "$CKPT"/config.json "$CKPT"/tokenizer.json "$CKPT"/tokenizer_config.json "$CKPT"/generation_config.json "$CKPT"/processor_config.json "$META"/ 2>/dev/null
cp "$CKPT"/chat_template*.jinja "$META"/ 2>/dev/null || true
# ST owns this template (thinking, tools and multimodal request semantics).
# NVIDIA's source does not include it; never inherit a stale previous staging.
cp "$REPO/launchers/chat_template_mm_v2.jinja" "$META/"
# When the supervisor launches from the installed tree, refresh its own metadata too.
if [ "$(readlink -f "$REPO")" = "$(readlink -f "$ENGINE_DIR")" ]; then
  rsync -a --delete "$META/" "$ENGINE_DIR/st-glm53-meta/"
fi
# Every node prepares and starts in PARALLEL. A node's rsync, image build and container start depend
# on no other node's, but rank 0 waits at the rendezvous for the last node to arrive -- so a sequential
# loop put its own stagger straight into rank 0's boot: measured 9.0 s of a 90.2 s boot (2026-09-11,
# srv2 :39.3 / srv1 :42.9 / srv3 :46.1 / srv4 :48.3). Each node's output is buffered and printed in rank
# order so the log stays readable, and one node's failure stops the whole fleet rather than leaving a
# partial one behind.
start_rank() {
  local r=$1 ip=${NODES[$1]}
  echo "== rank $r on $ip"
  push_tree "$ip" || { echo "ABORT: $ip could not receive the engine tree (rsync)" >&2; return 1; }
  # the ST image is built on the node from the tree just rsynced: seconds (two thin layers on the seed every node has); the seed ID is pinned in build.sh
  node_sh "$ip" "ST_IMAGE=$IMAGE bash $ENGINE_DIR/engine/runtime/build.sh" \
    || { echo "ABORT: $ip could not build $IMAGE (engine/runtime/build.sh)" >&2; return 1; }
  node_sh "$ip" "test -s $RANKS_DIR/rank${r}of4.safetensors" || { echo "ABORT: $ip lacks rank${r}of4.safetensors (fanout-st-ranks.sh)" >&2; return 1; }
  node_sh "$ip" "test -s $RANKS_DIR/vision.safetensors" || { echo "ABORT: $ip lacks vision.safetensors (preshard.py --vision --out $RANKS_DIR, once per node)" >&2; return 1; }
  node_sh "$ip" "test -s $DRAFTER/model.safetensors" || { echo "ABORT: $ip lacks the DFlash2 drafter at $DRAFTER" >&2; return 1; }
  node_sh "$ip" "docker rm -f $NAME >/dev/null 2>&1 || true; docker run -d --name $NAME --gpus all --restart no \
    --network host --ipc host --shm-size 32g --ulimit memlock=-1:-1 --ulimit nofile=524288:524288 --cap-add IPC_LOCK \
    --device /dev/infiniband:/dev/infiniband \
    -e RANK=$r -e WORLD_SIZE=4 -e MASTER_ADDR=10.10.10.2 -e MASTER_PORT=29555 -e LOCAL_RANK=0 $NCCL_ENV \
    -v $ENGINE_DIR:/repo:ro -v $RANKS_DIR:$RANKS_DIR:ro -v $DRAFTER:$DRAFTER:ro -v $CACHE_DIR:/cache \
    -v /home/choiceoh/glm53-logs:/home/choiceoh/glm53-logs \
    -e ST_LEASE_OWNER="$LEASE_OWNER" -e ST_LEASE_PATH="$LOCK" \
    --entrypoint /bin/bash $IMAGE -lc 'source /repo/launchers/lib/common-tp4.sh; eval \"\$CT_GID_PRELUDE\"; cd /repo && PYTHONPATH=/repo exec python3 -u engine/profiles/glm53/boot.py $PRODUCTION_ARG $KV_ARG --port $PORT --ranks $RANKS_DIR --ckpt-meta /repo/st-glm53-meta --drafter-dir $DRAFTER --tier-dir $TIER_DIR --dump-dir $DUMP_DIR' >/dev/null && echo '$ip: started'"
}

pids=()
for r in "${!NODES[@]}"; do
  start_rank "$r" >"$stage/rank$r.log" 2>&1 &
  pids[$r]=$!
done
failed=""
for r in "${!NODES[@]}"; do
  wait "${pids[$r]}" || failed="$failed $r"
  cat "$stage/rank$r.log"
done
if [ -n "$failed" ]; then
  echo "ABORT: rank(s)$failed did not start; stopping the rest so no partial fleet is left" >&2
  bash "$0" stop >/dev/null 2>&1 || true
  exit 1
fi
launched=1
# a ticket's lease was taken without a container; now that rank 0 is up, that container is its
# evidence (the head's docker answers for it) -- attach it, best effort
[ "$LEASE_MODE" != ticket ] || lease attach --owner "$LEASE_OWNER" --container "$NAME" >/dev/null 2>&1 || true
echo "head: http://10.10.10.2:$PORT/v1/chat/completions (OpenAI), /v1/engine/completions (engine dialect), GET / for status"
