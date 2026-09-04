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
# hy4 carried this guard and glm53 did not, so the asymmetry only protected the
# lane that was less likely to be started second. Both call it now, each naming
# the OTHER lanes as foreign.
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
