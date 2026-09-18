#!/usr/bin/env bash
# Compile the b12x MoE kernels a tree's boot will ask for, on every node's CPU and into that node's /cache, while the
# fleet still serves -- so the window that boots the tree (production's downtime) reads them instead of compiling them
# (engine/kernels/b12x_requests.py; measurements/qwen38_boot_20260918: six recompiled kernels put production's door at
# 150 s instead of 105 s, and Qwen3.8's first boot was 107.4 s cold against 40.1 s warm).
#
#   bash launchers/b12x-prebuild.sh --tree DIR --profile qwen38|glm53 [--cache DIR] [--nodes "IP ..."]
#
# On each node, in parallel:
#   * the tree's engine/ is rsynced to ~/st-prebuild/<profile>, never over a directory a container serves from;
#   * the tree's pinned seed image (engine/runtime/dependencies.json: the libraries its boot links) runs the prebuild
#     with that copy at /repo, the path every boot mounts its tree at and the objects' source hash includes;
#   * it replays the requests this node's last boots of <profile> recorded (<cache>/cu132/st-b12x-requests/).
# No GPU, no network, nice 19, two CPUs, a memory cap, and only on a node with the cap and earlyoom's floor to spare
# (earlyoom kills production first: launchers/earlyoom.default). A node that is short is skipped and said. Nothing here
# fails a launch: a kernel that is not prebuilt is compiled by the boot, as before. The replay itself is
# engine/runtime/b12x_prebuild.py.
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
TREE="" PROFILE="" CACHE_DIR=/home/choiceoh/glm53-cache
NODES_LIST="10.10.10.2 10.10.10.1 10.10.10.3 10.10.10.4"
CPUS=${ST_PREBUILD_CPUS:-2}
MEMORY_GIB=${ST_PREBUILD_MEMORY_GIB:-3}      # three kernels peaked at 0.56 GiB of cgroup memory (1.0 GiB RSS) on 2026-09-18
FLOOR_GIB=6                                  # earlyoom's absolute floor on these nodes
MARGIN_GIB=3
while [ $# -gt 0 ]; do
  case "$1" in
    --tree) TREE=$2; shift 2 ;;
    --profile) PROFILE=$2; shift 2 ;;
    --cache) CACHE_DIR=$2; shift 2 ;;
    --nodes) NODES_LIST=$2; shift 2 ;;
    *) echo "usage: $0 --tree DIR --profile NAME [--cache DIR] [--nodes \"IP ...\"]" >&2; exit 2 ;;
  esac
done
[ -d "$TREE/engine" ] || { echo "prebuild: $TREE has no engine/" >&2; exit 2; }
[[ "$PROFILE" =~ ^[a-z0-9_]+$ ]] || { echo "prebuild: --profile must be a profile name (qwen38, glm53)" >&2; exit 2; }
[[ "$MEMORY_GIB" =~ ^[1-9][0-9]*$ && "$CPUS" =~ ^[1-9][0-9]*$ ]] || { echo "prebuild: CPUs and memory must be whole numbers" >&2; exit 2; }
IMAGE=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["seed_image_id"])' "$TREE/engine/runtime/dependencies.json") \
  || { echo "prebuild: $TREE pins no seed image" >&2; exit 2; }
STAGE=/home/choiceoh/st-prebuild/$PROFILE
REQUESTS=/cache/cu132/st-b12x-requests/$PROFILE.jsonl
NEED_GIB=$((MEMORY_GIB + FLOOR_GIB + MARGIN_GIB))
SSHOPT="-o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new"
SELF_IPS=" $(hostname -I 2>/dev/null) "
is_self() { [[ "$SELF_IPS" == *" $1 "* ]]; }
node_sh() { local ip=$1; shift; if is_self "$ip"; then bash -c "$*"; else ssh $SSHOPT "choiceoh@$ip" "$@"; fi; }

prebuild_node() {
  local ip=$1 avail
  avail=$(node_sh "$ip" "awk '/^MemAvailable:/ {print int(\$2 / 1048576)}' /proc/meminfo") || { echo "$ip: unreachable -- skipped"; return 0; }
  if [ "${avail:-0}" -lt "$NEED_GIB" ]; then
    echo "$ip: ${avail} GiB available, ${NEED_GIB} needed (cap ${MEMORY_GIB} + earlyoom floor ${FLOOR_GIB} + ${MARGIN_GIB}) -- skipped; its boot compiles"
    return 0
  fi
  if is_self "$ip"; then
    mkdir -p "$STAGE" && rsync -a --delete --exclude __pycache__ "$TREE/engine" "$STAGE/"
  else
    node_sh "$ip" "mkdir -p $STAGE" && rsync -a --delete -e "ssh $SSHOPT" --exclude __pycache__ "$TREE/engine" "choiceoh@$ip:$STAGE/"
  fi || { echo "$ip: the tree did not arrive (rsync) -- skipped"; return 0; }
  local out
  out=$(node_sh "$ip" "timeout 1500 docker run --rm --name b12x-prebuild-$PROFILE --pull never --network none \
      --cpus $CPUS --memory ${MEMORY_GIB}g -e CUDA_VISIBLE_DEVICES= -e NVIDIA_VISIBLE_DEVICES=void \
      -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/repo -e FLASHINFER_WORKSPACE_BASE=/cache/cu132 \
      -e CUTE_DSL_ARCH=sm_121a -e FLASHINFER_CUDA_ARCH_LIST=12.1 \
      -v $STAGE:/repo:ro -v $CACHE_DIR:/cache -w /repo --entrypoint nice $IMAGE \
      -n 19 python3 -m engine.runtime.b12x_prebuild prebuild --requests $REQUESTS 2>&1")
  local code=$?
  local summary
  summary=$(printf '%s\n' "$out" | grep '"summary"' | tail -1)
  if [ -n "$summary" ]; then
    echo "$ip: $summary"
    printf '%s\n' "$out" | grep '"status": "failed"' | head -5 | sed "s/^/$ip:   /"
  else
    echo "$ip: prebuild rc=$code: $(printf '%s\n' "$out" | tail -1 | cut -c1-200)"
  fi
}

stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
read -r -a NODES <<<"$NODES_LIST"
for ip in "${NODES[@]}"; do
  prebuild_node "$ip" >"$stage/$ip.log" 2>&1 &
done
wait
for ip in "${NODES[@]}"; do cat "$stage/$ip.log"; done
exit 0
