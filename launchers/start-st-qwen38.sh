#!/usr/bin/env bash
# Boot the ST engine's Qwen3.8-Flash-Next on the four Sparks (engine/profiles/qwen38/fleet.py): one container per node,
# inside the ST image built from this tree, the tree mounted at /repo, this node's TEP=4 rank file and its metadata,
# /cache for the JIT builds, and the fleet's NCCL/RoCE environment (start-st-glm53.sh's, which this script follows).
#
#   bash launchers/start-st-qwen38.sh            # start all four (rank 0=srv2, rank 1=srv1, then srv3/srv4)
#   bash launchers/start-st-qwen38.sh stop       # docker rm -f st-qwen38 on every node
#   bash launchers/start-st-qwen38.sh logs [r]   # tail rank r's container log
#   bash launchers/start-st-qwen38.sh prebuild   # this tree's b12x MoE kernels, compiled on every node's CPU while
#                                                # production still serves: run it BEFORE taking the window
#                                                # (launchers/b12x-prebuild.sh), so the window does not compile them
#
# Before the first boot, once: the preshard on the node holding the checkpoint, then the fan-out --
#   python3 -m engine.profiles.qwen38.preshard --ckpt /home/choiceoh/models/qwen38-flash-next-nvfp4 \
#       --out /home/choiceoh/models/st-qwen38-tep4 --source-revision <revision>
#   VISION=0 RANKS_DIR=/home/choiceoh/models/st-qwen38-tep4 bash launchers/fanout-st-ranks.sh
# The rank directory carries the checkpoint's config, tokenizer, generation config and chat template beside the
# ranks (preshard.py copies them), so a node needs nothing else of the checkpoint.
#
# NOT BESIDE PRODUCTION. The ST fleet serves GLM-5.3 (st-glm53): this script refuses while any glm53*/q38*/vllm*/st-*
# container is up on any node, and it takes the fleet lease exactly as start-st-glm53.sh does -- a ticket's owner
# (ST_LEASE_OWNER, verified), or ST_LEASE_KIND=session for a session's own window by hand. A window here is production
# downtime: plan it, announce it, and hand the fleet back (`stop`) when done.
#
# Its own engine directory and image tag: this tree is rsynced to ST_ENGINE_DIR (default ~/st-engine-qwen38) and
# built as st-engine:qwen38, so a Qwen3.8 window never overwrites production's release tree or its st-engine:glm53.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
NODES=(10.10.10.2 10.10.10.1 10.10.10.3 10.10.10.4)
IMAGE=${ST_IMAGE:-st-engine:qwen38}
PORT=${PORT:-8000}
KV_ARG=""
if [ -n "${ST_KV_GIB:-}" ]; then
  [[ "$ST_KV_GIB" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "ST_KV_GIB must be a positive GiB byte budget" >&2; exit 2; }
  KV_ARG="--kv-gib $ST_KV_GIB"
fi
SEQS_ARG=""
if [ -n "${ST_MAX_SEQS:-}" ]; then
  [[ "$ST_MAX_SEQS" =~ ^[1-9][0-9]*$ ]] || { echo "ST_MAX_SEQS must be a positive row count" >&2; exit 2; }
  SEQS_ARG="--max-seqs $ST_MAX_SEQS"
fi
DRAFTER_ARG=""
case "${ST_DRAFTER:-1}" in
  1) ;;
  0) DRAFTER_ARG="--no-drafter" ;;
  *) echo "ST_DRAFTER must be 0 or 1" >&2; exit 2 ;;
esac
SHARDS_ARG=""                                                 # ST_QUERY_SHARDS=0: every rank scores every index query (carry Q11's rollback)
case "${ST_QUERY_SHARDS:-1}" in
  1) ;;
  0) SHARDS_ARG="--no-query-shards" ;;
  *) echo "ST_QUERY_SHARDS must be 0 or 1" >&2; exit 2 ;;
esac
OVERLAP_ARG=""                                                # ST_SHARED_OVERLAP=one|all: the shared expert beside the routed ones (carry M5; off by default)
case "${ST_SHARED_OVERLAP:-off}" in
  off) ;;
  one|all) OVERLAP_ARG="--shared-overlap $ST_SHARED_OVERLAP" ;;
  *) echo "ST_SHARED_OVERLAP must be off, one or all" >&2; exit 2 ;;
esac
HC_ARG=""                                                     # ST_HC_FP8=1: the mixers on FP8 (a quality bracket judges it)
case "${ST_HC_FP8:-0}" in
  0) ;;
  1) HC_ARG="--hc-fp8" ;;
  *) echo "ST_HC_FP8 must be 0 or 1" >&2; exit 2 ;;
esac
MTP_ARG=""                                                    # ST_MTP_PRECISION=bf16|fp8|w4: the MTP head's dense projections (fleet default bf16)
if [ -n "${ST_MTP_PRECISION:-}" ]; then
  case "$ST_MTP_PRECISION" in
    bf16|fp8|w4) MTP_ARG="--mtp-precision $ST_MTP_PRECISION" ;;
    *) echo "ST_MTP_PRECISION must be bf16, fp8 or w4" >&2; exit 2 ;;
  esac
fi
INDEX_ARG=""                                                  # ST_DRAFT_INDEX=CLUSTERS/PROBES: the drafter's argmax from an index over the head
if [ -n "${ST_DRAFT_INDEX:-}" ]; then
  [[ "$ST_DRAFT_INDEX" =~ ^[1-9][0-9]*/[1-9][0-9]*$ ]] || { echo "ST_DRAFT_INDEX must be CLUSTERS/PROBES" >&2; exit 2; }
  INDEX_ARG="--draft-index $ST_DRAFT_INDEX"
fi
EXPERTS_ARG="" EXPERTS_MOUNT=""                               # ST_MTP_EXPERTS_DIR=DIR: the MTP head's experts in the export's FP8 (mtp_fp8.py)
if [ -n "${ST_MTP_EXPERTS_DIR:-}" ]; then
  EXPERTS_ARG="--mtp-experts-dir $ST_MTP_EXPERTS_DIR"
  EXPERTS_MOUNT="-v $ST_MTP_EXPERTS_DIR:$ST_MTP_EXPERTS_DIR:ro"
fi
TAP_ARG=""                                                    # ST_TAP_DRAFT_QUERIES=ROWS: rank 0 records the draft queries (the IVF head's recall)
if [ -n "${ST_TAP_DRAFT_QUERIES:-}" ]; then
  [[ "$ST_TAP_DRAFT_QUERIES" =~ ^[1-9][0-9]*$ ]] || { echo "ST_TAP_DRAFT_QUERIES must be a row count" >&2; exit 2; }
  TAP_ARG="--tap-draft-queries $ST_TAP_DRAFT_QUERIES"
fi
ONESHOT_ARG=""                                                # ST_ONESHOT=0: every collective on NCCL (the one-shot cell at hidden 2560 is unmeasured)
case "${ST_ONESHOT:-1}" in
  1) ;;
  0) ONESHOT_ARG="--no-oneshot" ;;
  *) echo "ST_ONESHOT must be 0 or 1" >&2; exit 2 ;;
esac
SPEC_ARG=""                                                   # ST_SPEC_K=K: K drafts a step from the MTP head (the checkpoint's 1; K > 1 chains it)
if [ -n "${ST_SPEC_K:-}" ]; then
  [[ "$ST_SPEC_K" =~ ^[1-9][0-9]*$ ]] || { echo "ST_SPEC_K must be a positive draft count" >&2; exit 2; }
  SPEC_ARG="--spec-k $ST_SPEC_K"
fi
RECLAIM_FILE_CACHE=${ST_RECLAIM_FILE_CACHE:-1}
RECLAIM_ROOT=/home/choiceoh/glm53-logs/st-reclaim           # one broker directory per rank, on that rank's node
case "$RECLAIM_FILE_CACHE" in
  0|1) ;;
  *) echo "ST_RECLAIM_FILE_CACHE must be 0 or 1" >&2; exit 2 ;;
esac
RANKS_DIR=${RANKS_DIR:-/home/choiceoh/models/st-qwen38-tep4}
ENGINE_DIR=${ST_ENGINE_DIR:-/home/choiceoh/st-engine-qwen38}
# A shell carrying production's environment must not rsync --delete this tree over a release or retag its image:
# production runs ~/st-engine or a pinned ~/st-releases/<commit> as st-engine:glm53 or st-engine:prod-<commit>.
case "$(readlink -m "$ENGINE_DIR")" in
  */st-engine|*/st-engine/|/home/choiceoh/st-releases|/home/choiceoh/st-releases/*)
    echo "ABORT: $ENGINE_DIR is production's release tree; a Qwen3.8 window uses its own (ST_ENGINE_DIR)" >&2; exit 2 ;;
esac
case "$IMAGE" in
  st-engine:glm53|st-engine:prod-*)
    echo "ABORT: $IMAGE is production's image tag; a Qwen3.8 window builds its own (ST_IMAGE)" >&2; exit 2 ;;
esac
CACHE_DIR=${CACHE_DIR:-/home/choiceoh/glm53-cache}
SSHOPT="-o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new"
NAME=st-qwen38
LEASE_OWNER=${LEASE_OWNER:-$(whoami)@$(hostname -s)/$$}
LOCK=${FLEET_LEASE_PATH:-/home/choiceoh/glm53-logs/st-fleet.lock}
LEGACY_LOCK=/home/choiceoh/st-fleet.lock

SELF_IPS=" $(hostname -I 2>/dev/null) "
is_self() { [[ "$SELF_IPS" == *" $1 "* ]]; }
node_sh() { local ip=$1; shift; if is_self "$ip"; then bash -c "$*"; else ssh $SSHOPT "choiceoh@$ip" "$@"; fi; }
node_script() { local ip=$1 script=$2; if is_self "$ip"; then bash "$script"; else ssh $SSHOPT "choiceoh@$ip" "bash -s" < "$script"; fi; }
push_tree() {
  local ip=$1
  if is_self "$ip"; then
    [ "$(readlink -f "$REPO")" = "$(readlink -f "$ENGINE_DIR")" ] && return 0
    mkdir -p "$ENGINE_DIR"
    rsync -a --delete --exclude __pycache__ "$REPO/engine" "$REPO/launchers" "$ENGINE_DIR/"
  else
    rsync -a --delete -e "ssh $SSHOPT" --exclude __pycache__ "$REPO/engine" "$REPO/launchers" "choiceoh@$ip:$ENGINE_DIR/"
  fi
}

use_lease() {
  FLEET_REPO=$REPO; FLEET_HEAD=${NODES[0]}; FLEET_LEASE_PATH=$LOCK; FLEET_LEASE_SSH=$SSHOPT
  . "$REPO/launchers/lib/fleet-lease.sh"
}

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
    # Only the holder stops its own boot: a ticket by its exact owner, a session by its kind, a person with STOP_FORCE=1.
    held_origin=$(lease origin 2>/dev/null || echo unreadable)
    if [ -n "$held_owner" ] && [ "$held_origin" = explicit ]; then
      if [ -n "${ST_LEASE_OWNER:-}" ]; then
        [ "$held_owner" = "$ST_LEASE_OWNER" ] || { echo "ABORT: the fleet is held by $held_owner, not by this ticket ($ST_LEASE_OWNER)" >&2; exit 1; }
      elif [ "${ST_LEASE_KIND:-}" = session ]; then
        held_kind=$(lease kind 2>/dev/null || echo unreadable)
        case "$held_kind" in session|free) ;; *) echo "ABORT: the fleet is held by $held_owner ($held_kind), not by a session boot" >&2; exit 1 ;; esac
      elif [ "${STOP_FORCE:-0}" != 1 ]; then
        echo "ABORT: the fleet is held by $held_owner. Stop it from its own side. STOP_FORCE=1 is the operator's word." >&2; exit 1
      fi
    fi
    # the owner the node's docker guard asks for is read off the container (launchers/docker-fleet-guard.sh)
    for ip in "${NODES[@]}"; do node_sh "$ip" "ST_FLEET_OK=\$(docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' $NAME 2>/dev/null | sed -n 's/^ST_LEASE_OWNER=//p' | head -1) docker rm -f $NAME >/dev/null 2>&1 && echo '$ip: stopped' || echo '$ip: none'"; done
    for ip in "${NODES[@]}"; do node_sh "$ip" "bash $ENGINE_DIR/launchers/st-reclaim-broker.sh stop-all $RECLAIM_ROOT >/dev/null 2>&1 || true"; done
    use_lease
    if [ -n "$held_owner" ] && [ "${ST_LEASE_OWNER:-}" = "$held_owner" ]; then
      lease publish --owner "$held_owner" --state phase=stopped >/dev/null 2>&1 || true
    elif [ -n "$held_owner" ]; then
      lease release --owner "$held_owner"
    fi
    exit 0 ;;
  prebuild)
    # the kernels this tree's boot will ask for, from the requests the last Qwen3.8 boots recorded on each node
    exec bash "$REPO/launchers/b12x-prebuild.sh" --tree "$REPO" --profile qwen38 --cache "$CACHE_DIR" --nodes "${NODES[*]}" ;;
  logs)
    r=${2:-0}; node_sh "${NODES[$r]}" "docker logs --tail 60 $NAME"; exit 0 ;;
  start) ;;
  *) echo "usage: $0 [start|stop|logs r|prebuild]" >&2; exit 2 ;;
esac

for ip in "${NODES[@]}"; do
  busy=$(node_sh "$ip" "docker ps --format '{{.Names}}' | grep -E '^(glm53|q38|vllm|st-)' || true")
  [ -z "$busy" ] || { echo "ABORT: $ip runs $busy -- the fleet is taken (production is st-glm53: hand off, do not squat)" >&2; exit 1; }
done
use_lease
legacy=$(node_sh "${NODES[0]}" "cat $LEGACY_LOCK 2>/dev/null || true")
[ -z "$legacy" ] || { echo "ABORT: a session on the older lock path holds the fleet: $legacy ($LEGACY_LOCK on ${NODES[0]})" >&2; exit 1; }
LEASE_MODE=""
if [ -n "${ST_LEASE_OWNER:-}" ]; then
  LEASE_OWNER=$ST_LEASE_OWNER
  held=$(lease verify --owner "$LEASE_OWNER" 2>&1) \
    || { echo "ABORT: this boot's ticket does not hold the fleet: $held" >&2; exit 1; }
  echo "lease: verified, $held"
  LEASE_MODE=ticket
elif [ "${ST_LEASE_KIND:-}" = session ]; then
  lease acquire --owner "$LEASE_OWNER" --kind session --container "$NAME" --est-minutes "${LEASE_MINUTES:-45}" \
        --note "${LEASE_NOTE:-st-qwen38 on four Sparks}" \
    || { echo "ABORT: $(lease read 2>/dev/null || echo 'the fleet lease refused')" >&2; exit 1; }
  LEASE_MODE=session
else
  cat >&2 <<EOF
ABORT: this boot holds no reservation, and a boot nobody reserved is not started. Say what it is:
  ST_LEASE_OWNER=queue/<session>   a ticket's boot -- the queue takes the lease at GO and hands the owner here
  ST_LEASE_KIND=session $0          a session's own window by hand (production is down for its length)
EOF
  exit 1
fi
stage=$(mktemp -d)
launched=0
cleanup() {
  rm -rf "$stage"
  if [ "$launched" = 0 ]; then
    case "$LEASE_MODE" in
      session) lease release --owner "$LEASE_OWNER" >/dev/null || true ;;
      ticket) lease publish --owner "$LEASE_OWNER" --state phase=boot-failed >/dev/null 2>&1 || true ;;
    esac
  fi
}
trap cleanup EXIT

# start-st-glm53.sh's NCCL environment (its comments say why each is there)
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
-e ST_DENSE_BUILD_ROOT=/cache/cu132/st-dense -e ST_ONESHOT_BUILD_ROOT=/cache/cu132/st-oneshot -e MAX_JOBS=2"

start_rank() {
  local r=$1 ip=${NODES[$1]}
  echo "== rank $r on $ip"
  push_tree "$ip" || { echo "ABORT: $ip could not receive the engine tree (rsync)" >&2; return 1; }
  node_sh "$ip" "ST_IMAGE=$IMAGE bash $ENGINE_DIR/engine/runtime/build.sh" \
    || { echo "ABORT: $ip could not build $IMAGE (engine/runtime/build.sh)" >&2; return 1; }
  node_sh "$ip" "test -s $RANKS_DIR/rank${r}of4.safetensors && test -s $RANKS_DIR/config.json && test -s $RANKS_DIR/tokenizer.json" \
    || { echo "ABORT: $ip lacks rank${r}of4.safetensors or its metadata in $RANKS_DIR (VISION=0 fanout-st-ranks.sh)" >&2; return 1; }
  node_sh "$ip" "docker rm -f $NAME >/dev/null 2>&1 || true"
  local reclaim_env=""
  if [ "$RECLAIM_FILE_CACHE" = 1 ]; then
    local returned
    if returned=$(node_script "$ip" "$REPO/launchers/st-return-file-cache.sh" 2>&1); then
      echo "$ip: $returned"
    else
      echo "$ip: ${returned:-the file cache return did not answer} -- starting anyway, the engine's admission decides"
    fi
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
    -v $ENGINE_DIR:/repo:ro -v $RANKS_DIR:$RANKS_DIR:ro $EXPERTS_MOUNT -v $CACHE_DIR:/cache \
    -v /home/choiceoh/glm53-logs:/home/choiceoh/glm53-logs \
    -e ST_LEASE_OWNER=\"$LEASE_OWNER\" -e ST_LEASE_PATH=\"$LOCK\" -e ST_RELEASE=\"$(basename "$ENGINE_DIR")\" $reclaim_env \
    --entrypoint /bin/bash $IMAGE -lc 'source /repo/launchers/lib/common-tp4.sh; eval \"\$CT_GID_PRELUDE\"; cd /repo && PYTHONPATH=/repo exec python3 -u -m engine.profiles.qwen38.fleet $KV_ARG $SEQS_ARG $DRAFTER_ARG $HC_ARG $SPEC_ARG $MTP_ARG $INDEX_ARG $EXPERTS_ARG $OVERLAP_ARG $TAP_ARG $ONESHOT_ARG $SHARDS_ARG --port $PORT --ranks $RANKS_DIR --ckpt-meta $RANKS_DIR' >/dev/null && echo '$ip: started'"
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
echo "head: http://10.10.10.2:$PORT/v1/chat/completions (OpenAI), GET / for status"
