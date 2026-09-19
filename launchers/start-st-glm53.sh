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
# Each node returns its clean file cache from the host the moment before its container starts
# (launchers/st-return-file-cache.sh): on this UMA box the engine's admission cannot evict another
# workload's cache itself (2026-09-13). A broker per rank then serves the boot's own requests
# (launchers/st-reclaim-broker.sh, ST_RECLAIM_DIR). ST_RECLAIM_FILE_CACHE=0 skips both.
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
# The paged KV budget this fleet serves on. boot.py's own default is 24.0 -- vLLM parity for a
# single-box comparison (28th) -- and production had been running 7.0 from a hand-edited env file
# with no ledger entry behind it. 14.0 is the value measured on 2026-09-16: 2,987 blocks, declared
# paged KV 13.16 GiB, unassigned +19.59 GiB, booted in 135 s on the first attempt. It is here and
# not in that env file because production shape that lives only on the box does not survive --
# the deploy relaunches from the tree, and the tree is this.
KV_GIB=${ST_KV_GIB:-14.0}
[[ "$KV_GIB" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "ST_KV_GIB must be a positive GiB byte budget" >&2; exit 2; }
KV_ARG="--kv-gib $KV_GIB"
# The runtime workspace ceiling (engine/profiles/glm53/budget.WORKSPACE_GIB) is what admission asks each node for on
# top of the arena. A shape that spends more than the profile's ceiling raises it here, and says so in its ledger.
WORKSPACE_ARG=""
if [ -n "${ST_WORKSPACE_GIB:-}" ]; then
  [[ "$ST_WORKSPACE_GIB" =~ ^[0-9]+([.][0-9]+)?$ ]] && awk -v w="$ST_WORKSPACE_GIB" 'BEGIN { exit !(w > 0) }' \
    || { echo "ST_WORKSPACE_GIB must be a positive GiB ceiling" >&2; exit 2; }
  WORKSPACE_ARG="--workspace-gib $ST_WORKSPACE_GIB"
fi
PRODUCTION_ARG=""
# A same-build restart reuses each node's passed far prefill memory pass (~/glm53-cache/st-gate, base/prefill_record).
# ST_FULL_MEMORY_GATE=1 runs it anyway and rewrites the record; so does removing that directory.
GATE_ARG=""
case "${ST_FULL_MEMORY_GATE:-0}" in
  0) ;;
  1) GATE_ARG="--full-memory-gate" ;;
  *) echo "ST_FULL_MEMORY_GATE must be 0 or 1" >&2; exit 2 ;;
esac
RECLAIM_FILE_CACHE=${ST_RECLAIM_FILE_CACHE:-1}
RECLAIM_ROOT=/home/choiceoh/glm53-logs/st-reclaim           # one broker directory per rank, on that rank's node
case "$RECLAIM_FILE_CACHE" in
  0|1) ;;
  *) echo "ST_RECLAIM_FILE_CACHE must be 0 or 1" >&2; exit 2 ;;
esac
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
# The NVMe tier (parked conversations and prefix boundaries). It was off from 2026-09-15, when the
# ranks' tiers diverged and production could not boot; the cause's fix -- key-level reconciliation,
# `Server._reconcile_parked` (#837) -- is in, the four ranks' tier directories were left empty, and
# production booted on it again 2026-09-16 (135 s, first attempt). ST_TIER_DIR=off turns it back off
# for a boot; any other value is the tier root, under which boot.py claims one `rank<N>` per rank.
# start-st-qwen38.sh uses the same root under the same caps (base/tiered_kv.TIER_ROOT): the two never
# serve at once, and each layout's files are foreign to the other -- counted, forgotten first, replaced.
#
# Off is not free: with no tier a finished turn is never registered as a conversation
# (base/serve, `if self.runner.tiered is None`), so its boundaries are dropped when its row is
# reclaimed. Production measured 17 prefix hits in 100 queries against 1,144 evictions -- with 97%
# of the blocks free. That is not capacity, it is having nowhere to keep what was just computed.
#
# The path must be under MOUNTED_ROOT. That is the only host directory the rank containers bind
# (see `docker run` below), so a tier anywhere else is written into the container's own writable
# layer: the boot *succeeds*, the tier reports itself live, reuse inside that boot works -- and
# every parked conversation and boundary is discarded with the container. It cost a production
# boot on 2026-09-16 (~/st-tier, off the mount, looked healthy for 3 minutes). The fleet lease
# learned the same lesson at ~/st-fleet.lock; refuse it here instead of learning it a third time.
MOUNTED_ROOT=/home/choiceoh/glm53-logs
TIER_DIR=${ST_TIER_DIR:-$MOUNTED_ROOT/st-tier}
case $TIER_DIR in
  off) TIER_ARG="--tier-dir=" ;;              # boot.py builds no tier from an empty directory
  "$MOUNTED_ROOT"/?*) TIER_ARG="--tier-dir $TIER_DIR" ;;
  *) echo "ST_TIER_DIR must be 'off' or a path under $MOUNTED_ROOT (the only host directory the rank containers mount); got: $TIER_DIR" >&2; exit 2 ;;
esac
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
# a script from this tree, run on the node through stdin: it needs nothing rsynced first
node_script() { local ip=$1 script=$2; if is_self "$ip"; then bash "$script"; else ssh $SSHOPT "choiceoh@$ip" "bash -s" < "$script"; fi; }
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
    # the brokers end on their own once their container is gone; this ends them now (any release's script reads the same pids)
    for ip in "${NODES[@]}"; do node_sh "$ip" "bash $ENGINE_DIR/launchers/st-reclaim-broker.sh stop-all $RECLAIM_ROOT >/dev/null 2>&1 || true"; done
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
    asked=$(fleet_lease yield --requester "$LEASE_OWNER" --kind session --pid $$ --host "$(hostname -s)" \
              --note "${2:-another session needs the fleet}") || exit 1
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


# TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC bounds a process whose NCCL watchdog thread stopped answering
# (stuck in ncclCommAbort or a wedged CUDA call): the monitor thread kills it after this many
# seconds. 7200 was copied from the vLLM launcher; with it a fleet whose four ranks sat in
# mismatched collectives (2026-09-12 22:31, GPUs at 0%, requests queued) would have stood for two
# hours. Collectives captured into a CUDA graph are never watched, so boot-time capture cannot trip
# it; a serving step is milliseconds, a prefill chunk seconds. ST_NCCL_HEARTBEAT_S overrides per boot.
NCCL_ENV="-e NCCL_P2P_LEVEL=SYS -e TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${ST_NCCL_HEARTBEAT_S:-300} \
-e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0 \
-e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e GLOO_SOCKET_IFNAME=${GLOO_IFNAME:-enP2p1s0f0np0} \
-e NCCL_CROSS_NIC=1 -e NCCL_PROTO=LL,LL128,Simple -e NCCL_CUMEM_ENABLE=0 \
-e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET \
-e NCCL_NVLS_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN \
-e NCCL_MIN_NCHANNELS=16 -e NCCL_MAX_NCHANNELS=16 -e NCCL_NCHANNELS_PER_NET_PEER=4 \
-e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
-e TRITON_CACHE_DIR=/cache/cu132/triton -e TILELANG_CACHE_DIR=/cache/cu132/tilelang \
-e DG_JIT_CACHE_DIR=/cache/cu132/deep_gemm -e ST_MLA_BUILD_ROOT=/cache/cu132/mla -e FLASHINFER_WORKSPACE_BASE=/cache/cu132 -e CUDA_CACHE_PATH=/cache/cu132/driver \
-e ST_DENSE_BUILD_ROOT=/cache/cu132/st-dense -e ST_ONESHOT_BUILD_ROOT=/cache/cu132/st-oneshot \
-e ST_NATIVE_BUILD_ROOT=/cache/cu132/st-native -e MAX_JOBS=2"
# the profile's declared D11 knobs (STK_*, boot.declared) travel from this shell into every rank; an undeclared one kills the boot
for v in $(compgen -v STK_ || true); do NCCL_ENV="$NCCL_ENV -e $v=${!v}"; done
# CUDA_MODULE_LOADING, when this shell names it. The 2026-09-16 boot put 14.3 s of one-time cost inside
# the memory gate's FIRST forward -- not compile (the walk saw no artifact), not the fleet vote (1.8 ms),
# not reclaim (58 ms) -- and the container leaves this unset, which is LAZY on CUDA 13.2. So the leading
# candidate is a cuModuleLoad per first launch, and EAGER is the one boot that decides it. It is not a
# declared knob: nothing reads it but the driver, and the prefill gate record keys on it so an EAGER
# boot cannot reuse a LAZY one's record.
# An `[ -n ... ] && ...` one-liner would be the last command of a `set -e` script's line and take the
# boot down every time the variable is NOT set, which is every production launch. `if`, then.
if [ -n "${CUDA_MODULE_LOADING:-}" ]; then
  NCCL_ENV="$NCCL_ENV -e CUDA_MODULE_LOADING=$CUDA_MODULE_LOADING"
fi

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
  node_sh "$ip" "docker rm -f $NAME >/dev/null 2>&1 || true"
  # The node's clean file cache goes back from the host now -- after the rsync and the image build, the
  # moment before the container starts, so nothing refills it before the engine's admission. It runs
  # only here, with the lease held and every node checked idle. The engine cannot do this itself: from
  # its container it sees only its own checkpoint, and the anonymous fault it falls back on was refused
  # by srv2's strict overcommit and by srv4's SIGTERM line in every boot of 2026-09-13 19:29-19:48.
  # A node that cannot (no passwordless sudo) says so and starts anyway: admission still decides.
  local reclaim_env=""
  if [ "$RECLAIM_FILE_CACHE" = 1 ]; then
    local returned
    if returned=$(node_script "$ip" "$REPO/launchers/st-return-file-cache.sh" 2>&1); then
      echo "$ip: $returned"
    else
      echo "$ip: ${returned:-the file cache return did not answer} -- starting anyway, the engine's admission decides"
    fi
    # And for the rest of the boot a broker on the host: what the boot reads refills the cache, and admission and
    # warmup ask the broker the moment they are short. The host's drop needs no commit room, which srv2's strict
    # overcommit (CommitLimit 75.8 GiB) never gave the engine's own anonymous reclaim.
    if node_sh "$ip" "bash $ENGINE_DIR/launchers/st-reclaim-broker.sh start $RECLAIM_ROOT/rank$r $NAME"; then
      reclaim_env="-e ST_RECLAIM_DIR=$RECLAIM_ROOT/rank$r"
    else
      echo "$ip: the reclaim broker did not start -- admission falls back to its own reclaim"
    fi
  fi
  node_sh "$ip" "docker run -d --name $NAME --gpus all --restart no \
    --network host --ipc host --shm-size 32g --ulimit memlock=-1:-1 --ulimit nofile=524288:524288 --cap-add IPC_LOCK \
    --device /dev/infiniband:/dev/infiniband \
    -e RANK=$r -e WORLD_SIZE=4 -e MASTER_ADDR=10.10.10.2 -e MASTER_PORT=29555 -e LOCAL_RANK=0 $NCCL_ENV \
    -v $ENGINE_DIR:/repo:ro -v $RANKS_DIR:$RANKS_DIR:ro -v $DRAFTER:$DRAFTER:ro -v $CACHE_DIR:/cache \
    -v /home/choiceoh/glm53-logs:/home/choiceoh/glm53-logs \
    -e ST_LEASE_OWNER="$LEASE_OWNER" -e ST_LEASE_PATH="$LOCK" -e ST_RELEASE="$(basename "$ENGINE_DIR")" $reclaim_env \
    --entrypoint /bin/bash $IMAGE -lc 'source /repo/launchers/lib/common-tp4.sh; eval \"\$CT_GID_PRELUDE\"; cd /repo && PYTHONPATH=/repo exec python3 -u engine/profiles/glm53/boot.py $PRODUCTION_ARG $KV_ARG $WORKSPACE_ARG --port $PORT --ranks $RANKS_DIR --ckpt-meta /repo/st-glm53-meta --drafter-dir $DRAFTER $TIER_ARG --dump-dir $DUMP_DIR $GATE_ARG' >/dev/null && echo '$ip: started'"
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
# A ticket's lease deliberately names NO container. Its evidence is the queue's supervisor process
# on the head, which is conclusive; and a supervisor from before kinds resolves `stop` by container
# name alone, so a lease naming st-glm53 would let that supervisor's crash recovery evict the
# ticket's boot 90 s in. Nameless, the older `stop` refuses it (2026-09-13).
echo "head: http://10.10.10.2:$PORT/v1/chat/completions (OpenAI), /v1/engine/completions (engine dialect), GET / for status"
