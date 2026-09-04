#!/bin/bash
# Shared machinery for the TP=4 lane launchers. Sourced, never executed.
#
# Why this file exists (2026-09-04): start-hy4-tp4.sh (685 lines) and
# start-glm53-nvfp4-tp4.sh (707) implement the same four concerns twice, and
# the copies drifted. The drift is not hypothetical -- it has already cost:
#
#   memfree margin      hy4 sat at 3 while #270 moved glm53 to 10. The same
#                       bug had to be fixed twice, and between the two fixes
#                       the margin-3 boot wedged three nodes (09-04).
#   NCCL_IB_GID_INDEX   hy4 auto-detects the RoCE-v2 IPv4 index at boot;
#                       glm53 hardcoded 3, which the runbook records as wrong
#                       on srv2 and srv4 (they read 4).
#   TORCH_NCCL_ASYNC_   hy4 ran 0, glm53 ran 1, neither with a reason. See
#     ERROR_HANDLING    CT_NCCL_ASYNC_ERR below -- 0 is why the 09-04 wedge
#                       could not clear itself.
#   EXTRA_ENV guard     glm53 rejects a comma-joined list; hy4 did not, so
#                       EXTRA_ENV="A=1,B=2" landed as one -e and every knob
#                       read as off while the log said they were set.
#
# The rule this encodes: when the two lanes implement one concern, the better
# implementation wins and lives here, so the next fix lands once.

# --- profile loading ---------------------------------------------------------
# ct_load_profile <default_profile_path> <caller-precedence name>...
#
# Sets PROFILE_ENV, exports _vllm_keys for the EXTRA_ENV guard and the caller's
# own -e loop, sources the profile, then re-applies any of the named variables
# that the CALLER had set -- caller environment beats the profile, which is what
# makes `KNOB=1 bash launchers/start-...sh` a one-off rather than a silent no-op.
#
# The names are lane-specific (the two lanes share almost none), so they are
# arguments rather than a list here. $_vllm_keys is appended automatically: a
# knob added to a profile reaches the container without editing any launcher.
ct_load_profile() {
  local _default="$1"; shift
  PROFILE_ENV="${PROFILE_ENV:-$_default}"
  if [ ! -f "$PROFILE_ENV" ]; then
    # Running a copy of a launcher from outside the checkout resolves
    # PROFILE_ENV to a path that does not exist, and every profile value --
    # the module list's env knobs included -- silently falls back to the
    # literals in the launcher. That failure is why this says so out loud.
    echo "WARNING: profile not found at $PROFILE_ENV -- using built-in defaults."
    echo "         Run the launcher from the checkout, or set PROFILE_ENV explicitly."
    return 0
  fi
  # A profile with no VLLM_* key at all is legitimate, and an empty grep exits
  # 1 -- which under `set -euo pipefail` would end the script silently.
  _vllm_keys=$(grep -oE '^VLLM_[A-Z0-9_]+' "$PROFILE_ENV" 2>/dev/null | sort -u || true)
  local _caller="" _v
  for _v in "$@" $_vllm_keys; do
    if [ -n "${!_v:-}" ]; then _caller="$_caller $_v=$(printf %q "${!_v}")"; fi
  done
  # shellcheck disable=SC1090
  . "$PROFILE_ENV"
  [ -n "$_caller" ] && eval "$_caller"
  return 0
}

# --- EXTRA_ENV ---------------------------------------------------------------
# ct_extra_env_flags <launcher-path-for-the-message>
#
# Validates EXTRA_ENV and sets EXTRA_ENV_FLAGS. Both guards below exist because
# each failure mode produced a boot that measured the baseline while the log
# claimed the knobs were set.
ct_extra_env_flags() {
  local _launcher="$1" _kv
  EXTRA_ENV_FLAGS=""
  # _vllm_keys is newline-separated (grep | sort -u); flatten it so the
  # membership test below can use a space-delimited case pattern.
  local _vllm_keys_sp=" $(printf '%s ' ${_vllm_keys:-})"
  for _kv in ${EXTRA_ENV:-}; do
    case "$_kv" in
      # A comma-joined list is the natural mistake and it passes the KEY=VALUE
      # shape: the whole string lands in ONE -e whose value is
      # "1,NEXT_KEY=1,..." -- so every knob reads as off.
      [A-Za-z_]*=*,[A-Za-z_]*=*)
        echo "ABORT: EXTRA_ENV is space-separated, not comma-separated: $_kv" >&2
        echo "       use EXTRA_ENV=\"A=1 B=2\"" >&2
        exit 2 ;;
      [A-Za-z_]*=*)
        # A profile-declared VLLM_* key is emitted as its own -e further down,
        # and docker takes the LAST -e for a name. So EXTRA_ENV silently loses
        # to the profile for exactly the knobs a sweep wants to move. Pass
        # those as caller environment instead: ct_load_profile re-applies them
        # after the profile is sourced.
        case "$_vllm_keys_sp" in
          *" ${_kv%%=*} "*)
            echo "ABORT: ${_kv%%=*} is declared in the profile, so EXTRA_ENV cannot" >&2
            echo "       override it (docker takes the last -e). Pass it directly:" >&2
            echo "         ${_kv%%=*}=${_kv#*=} bash $_launcher" >&2
            exit 2 ;;
        esac
        EXTRA_ENV_FLAGS="$EXTRA_ENV_FLAGS -e $_kv" ;;
      *) echo "ABORT: EXTRA_ENV entry is not KEY=VALUE: $_kv" >&2; exit 2 ;;
    esac
  done
  [ -n "$EXTRA_ENV_FLAGS" ] && echo "extra env:$EXTRA_ENV_FLAGS"
  return 0
}

# --- load format -------------------------------------------------------------
# The instanttensor import check stays lane-side: it needs $IMAGE and a docker
# run, and only the glm53 lane has ever set it.
ct_check_load_format() {
  case "${LOAD_FORMAT}" in
    auto|safetensors|instanttensor) ;;
    *) echo "ABORT: LOAD_FORMAT must be auto, safetensors or instanttensor (got $LOAD_FORMAT)" >&2
       exit 2 ;;
  esac
}

# --- torch/NCCL error handling -----------------------------------------------
# TORCH_NCCL_ASYNC_ERROR_HANDLING. 1 = TearDown: the watchdog aborts the process
# when a collective errors or times out. 0 = the process hangs instead.
#
# Unified to 1 (2026-09-04). hy4 ran 0 and glm53 ran 1, neither with a stated
# reason, and 0 is the value that made the 09-04 wedge unrecoverable: the ring
# broke at 00:29 when srv4's worker was killed, and srv2/srv3's workers were
# still holding ~55 GiB twelve hours later because nothing ever tore them down.
# TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200 was set alongside it and is inert while
# this is 0. With 1 a broken ring frees the node instead of stranding it.
#
# The risk 0 was presumably buying is a spurious teardown on a legitimately slow
# collective; the watchdog fires on the process-group timeout (minutes), and the
# longest real prefill measured here is ~170 s at 430K, so the margin is wide.
# Roll back per-boot with NCCL_ASYNC_ERR=0, or per-lane in the profile.
CT_NCCL_ASYNC_ERR="${NCCL_ASYNC_ERR:-1}"

# --- foreign stacks ----------------------------------------------------------
# ct_refuse_foreign_stacks <foreign-name-regex> <lane label> <ssh opts> \
#                          <head ip> <worker ip>...
#
# A bring-up stack from another lane keeps tens of GiB of weights resident under
# its own container names. Sizing a second engine's UMA/KV pool while one is
# alive double-books the same memory, which on these unified-memory boxes is not
# a slow degradation -- it is the 09-04 wedge: the host crosses the 4 GiB kernel
# watermark, sshd stops forking, and the node needs a power cycle.
#
# Correction (2026-09-04): #277 introduced this saying glm53 had no such guard.
# That was wrong -- glm53 had an equivalent check written differently
# (`run 'docker ps ... grep -qE "^(hy4|q38)"'`) inside its per-node loop, and
# the survey behind #277 grepped for hy4's function name and missed it. What
# was actually true is that the two were written twice, ran at different points
# and used different anchors (glm53's lacked the trailing (-|$), so it also
# matched an unrelated container merely starting with the name). Both call this
# one now, each naming the OTHER lanes as foreign, and glm53's inline copy is
# gone.
#
# DRY_RUN skips it: a dry run creates nothing, so a live stack is not a conflict.
ct_refuse_foreign_stacks() {
  local _re="$1" _lane="$2" _sshopt="$3" _head="$4"; shift 4
  [ "${DRY_RUN:-0}" = 1 ] && return 0
  local _ip _hits
  _hits=$(docker ps --format '{{.Names}}' | grep -E "$_re" | tr '\n' ' ' || true)
  if [ -n "$_hits" ]; then
    echo "ABORT: $_head runs a foreign stack ($_hits) -- stop it before starting $_lane" >&2
    exit 1
  fi
  for _ip in "$@"; do
    _hits=$(ssh $_sshopt "choiceoh@$_ip" \
              "docker ps --format '{{.Names}}' | grep -E '$_re' | tr '\n' ' '" 2>/dev/null || true)
    if [ -n "$_hits" ]; then
      echo "ABORT: $_ip runs a foreign stack ($_hits) -- stop it before starting $_lane" >&2
      exit 1
    fi
  done
  return 0
}
# --- RoCE GID index ----------------------------------------------------------
# The RoCE-v2 IPv4 GID index is a PER-NODE, PER-BOOT property: enabling IPv6 on
# the fabric NIC changes how the GID table is laid out, so the index that means
# "RoCE v2 over IPv4" differs between machines and moves across reboots. The
# runbook records srv1 at 3 while srv2 and srv4 read 4.
#
# hy4 detected it inside the container, where each rank sees its own node's
# table -- correct. glm53 shipped `-e NCCL_IB_GID_INDEX=3` from the head, one
# value for all four nodes, which is wrong on three of them: NCCL then falls
# back to a GID that is not RoCE-v2/IPv4 and the fabric is not what the boot
# says it is. A launcher cannot fix this from the head at all, because a single
# -e cannot carry four different values.
#
# So the detection has to run per rank, inside the container, and this is the
# one copy of it. It is a literal string rather than a function: both lanes
# deliver their serve command as text (hy4 through a quoted heredoc, glm53
# through a base64 payload), so what they need to share is source, not a call.
# Defined through a quoted heredoc so the single quotes in `tr ',' ' '` survive.
CT_GID_PRELUDE=$(cat <<'GIDEOF'
# Auto-detect the RoCE-v2 IPv4 GID index (per node, re-numbers across reboots).
for HCA in $(echo "${NCCL_IB_HCA}" | tr ',' ' '); do
  for i in $(seq 0 15); do
    t=$(cat /sys/class/infiniband/$HCA/ports/1/gid_attrs/types/$i 2>/dev/null || true)
    g=$(cat /sys/class/infiniband/$HCA/ports/1/gids/$i 2>/dev/null || true)
    case "$t" in *"RoCE v2"*) case "$g" in *0000:0000:0000:0000:0000:ffff:*) export NCCL_IB_GID_INDEX=$i; break 2;; esac;; esac
  done
done
GIDEOF
)

# --- image uniformity --------------------------------------------------------
# ct_verify_image_uniform <ssh opts> <image> <expected id | ""> <head ip> <worker ip>...
#
# Sets CT_IMAGE_ID to the head's image ID once every node agrees on it.
#
# hy4 compared each worker's image ID against the head's and refused on skew.
# glm53 only asked whether the tag RESOLVED on each node -- and its tags are
# locally built (glm53:v13-b12x-it), not registry digests, so the same tag can
# name a different build per node. Nothing else covered the gap: the overlay
# attestation compares HOST files, and the manifest preimage check runs a
# container on the HEAD only. A node carrying a different base image passed
# every one of those and then ran different kernels inside a TP=4 collective,
# with the boot log saying nothing.
#
# expected id is hy4's EXPECTED_IMAGE_ID pin; glm53 passes "" because its tag
# moves by design, so there it enforces agreement without pinning a value.
ct_verify_image_uniform() {
  local _sshopt="$1" _image="$2" _expect="$3" _head="$4"; shift 4
  local _hid _wid _ip
  _hid=$(docker image inspect "$_image" --format '{{.Id}}' 2>/dev/null || true)
  [ -n "$_hid" ] || { echo "ABORT: $_head has no image $_image" >&2; exit 1; }
  if [ -n "$_expect" ] && [ "$_hid" != "$_expect" ]; then
    echo "ABORT: $_image ID drifted on $_head" >&2
    echo "  expected: $_expect" >&2
    echo "  actual:   $_hid" >&2
    exit 1
  fi
  for _ip in "$@"; do
    _wid=$(ssh $_sshopt "choiceoh@$_ip" \
             "docker image inspect $_image --format '{{.Id}}'" 2>/dev/null || true)
    [ -n "$_wid" ] || { echo "ABORT: $_ip has no image $_image" >&2; exit 1; }
    if [ "$_wid" != "$_hid" ]; then
      echo "ABORT: $_image differs on $_ip -- the ranks would run different builds" >&2
      echo "  $_head: $_hid" >&2
      echo "  $_ip: $_wid" >&2
      exit 1
    fi
  done
  CT_IMAGE_ID="$_hid"
}

# --- custom_ops axis ---------------------------------------------------------
# ct_apply_custom_ops_axis <axis> <allow-empty 0|1>
#
# Rewrites COMPILE_CFG's custom_ops list to the requested axis. Both lanes did
# this with sed; glm53's was unguarded and unanchored, and both gaps produce a
# boot that measures the CONTROL arm while the caller believes the axis is set:
#
#   silent no-op    `sed 's/"all"/.../'` against a caller-supplied COMPILE_CFG
#                   with no custom_ops matches nothing, and the launcher went
#                   on to boot. Reproduced: the config comes out byte-identical.
#   wrong field     the pattern was bare "all", and sed replaces the FIRST
#                   match. With the built-in configs custom_ops IS first, so
#                   they are safe; a caller config carrying "all" earlier gets
#                   that other field rewritten instead. Reproduced.
#
# hy4 already refused both. This is that refusal, shared.
#
# allow-empty is a real lane difference, not an oversight: glm53 uses
# CUSTOM_OPS_AXIS="" as its fusion arm (custom_ops:[""] removes the inductor
# walls), so empty is a value there and merely unset in hy4. Anchored and bare
# substitutions agree on that case -- verified -- so the arm is unchanged.
ct_apply_custom_ops_axis() {
  local _axis="$1" _allow_empty="${2:-0}"
  if [ -z "$_axis" ]; then
    if [ "$_allow_empty" != 1 ]; then
      echo "ABORT: CUSTOM_OPS_AXIS must not be empty on this lane" >&2
      exit 2
    fi
  elif [[ ! "$_axis" =~ ^(all|none|[+-][A-Za-z_][A-Za-z0-9_]*)$ ]]; then
    echo "ABORT: CUSTOM_OPS_AXIS must be all, none, +op or -op (got '$_axis')" >&2
    exit 2
  fi
  case "$COMPILE_CFG" in
    *'"custom_ops":["all"]'*) ;;
    *) echo "ABORT: CUSTOM_OPS_AXIS is set but COMPILE_CFG has no" >&2
       echo "       \"custom_ops\":[\"all\"] to replace -- the substitution would be a" >&2
       echo "       silent no-op and the boot would measure the control arm." >&2
       exit 2 ;;
  esac
  COMPILE_CFG=$(printf '%s' "$COMPILE_CFG" | sed \
    's/"custom_ops":\["all"\]/"custom_ops":["'"$_axis"'"]/')
}

# --- overlay target validation -----------------------------------------------
# ct_check_overlay_target <target> <required prefix>
#
# The manifest's target becomes a docker bind-mount destination
# (`-v <host file>:<target>:ro`), so "inside the package root" is the property
# the check exists to enforce. Both lanes enforced it with a PREFIX test plus a
# character class that allows "." -- and a prefix test is not containment:
#
#   /opt/venv/lib/python3.12/site-packages/vllm/../../../../etc/cron.d/evil
#
# starts with the required prefix and uses only allowed characters, so BOTH
# lanes accepted it and would have mounted over a path outside the package
# root. Verified by running each lane's own case statements against it.
#
# Neither lane had this, so it is not drift -- the guard simply did not cover
# the classic escape. The manifest is a repo file rather than untrusted input,
# so this is a malformed-manifest guard, not a security boundary; the value is
# that a bad row fails loudly here instead of silently shadowing a container
# path. The prefix stays an argument because the lanes require different roots
# (hy4 demands .../site-packages/vllm/, glm53 the profile's TARGET_PREFIX).
ct_check_overlay_target() {
  local _t="$1" _prefix="$2"
  case "$_t" in
    "$_prefix"*) ;;
    *) echo "ABORT: overlay target outside the package root: $_t" >&2; exit 1 ;;
  esac
  case "$_t" in
    *[!A-Za-z0-9_./-]*)
      echo "ABORT: unsafe character in overlay target: $_t" >&2; exit 1 ;;
  esac
  case "/$_t/" in
    */../*)
      echo "ABORT: overlay target escapes the package root with .. : $_t" >&2
      exit 1 ;;
  esac
}
