#!/usr/bin/env bash
# Render a profile's modules into the flat overlay directory + single manifest
# the rest of the stack expects.
#
# Collisions abort rather than resolve: two modules claiming the same source
# filename, or two rows binding the same container path. A module that collides
# with another was not as independent as its README claims, and letting one win
# would bury that.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
PROFILE=${1:?usage: compose-overlays.sh <profile>}
ENVFILE="$REPO/profiles/$PROFILE.env"
[ -f "$ENVFILE" ] || { echo "ABORT: no such profile: $ENVFILE"; exit 1; }
# shellcheck disable=SC1090
. "$ENVFILE"
[ -n "${MODULES:-}" ] || { echo "ABORT: $PROFILE.env names no MODULES"; exit 1; }
# Where this image keeps its packages. Profile-owned: the DSV4 image uses the
# venv site-packages, GLM's uses dist-packages, and a module may target
# flashinfer rather than vllm.
TARGET_PREFIX=${TARGET_PREFIX:-/opt/venv/lib/python3.12/site-packages/vllm/}

OUT="$REPO/build/$PROFILE"
rm -rf "$OUT"
mkdir -p "$OUT"

declare -A SRC_OWNER=() TGT_OWNER=()
: > "$OUT/manifest.tsv"
count=0
for mod in $MODULES; do
  case "$mod" in
    *[!A-Za-z0-9_-]*) echo "ABORT: unsafe module name: $mod"; exit 1 ;;
  esac
  m="$REPO/overlay/modules/$mod/manifest.tsv"
  [ -f "$m" ] || { echo "ABORT: no such module: $mod"; exit 1; }
  while IFS=$'\t' read -r source target contract || [ -n "${source:-}" ]; do
    [[ -z "$source" || "$source" == \#* ]] && continue
    [ -n "$target" ] && [ -n "$contract" ] \
      || { echo "ABORT: malformed row in module $mod: $source"; exit 1; }
    [ -z "${SRC_OWNER[$source]:-}" ] \
      || { echo "ABORT: $source claimed by ${SRC_OWNER[$source]} and $mod"; exit 1; }
    [ -z "${TGT_OWNER[$target]:-}" ] \
      || { echo "ABORT: $target bound by ${TGT_OWNER[$target]} and $mod"; exit 1; }
    SRC_OWNER[$source]=$mod
    TGT_OWNER[$target]=$mod
    case "$target" in
      "$TARGET_PREFIX"*) ;;
      *) echo "ABORT: $mod binds $target, outside $PROFILE's TARGET_PREFIX ($TARGET_PREFIX)"; exit 1 ;;
    esac
    case "$target" in
      *[!A-Za-z0-9_./-]*) echo "ABORT: unsafe character in target: $target"; exit 1 ;;
    esac
    src="$REPO/overlay/modules/$mod/$source"
    [ -f "$src" ] || { echo "ABORT: $src missing"; exit 1; }
    install -m 0644 "$src" "$OUT/$source"
    printf '%s\t%s\t%s\n' "$source" "$target" "$contract" >> "$OUT/manifest.tsv"
    count=$((count + 1))
  done < "$m"
done

# A module may name modules it cannot run without (dsv4_model imports the
# gate class from moe_gate_sm121). Unmet requirements abort rather than fail
# later as an ImportError inside a rank.
for mod in $MODULES; do
  req="$REPO/overlay/modules/$mod/requires"
  [ -f "$req" ] || continue
  while read -r need || [ -n "${need:-}" ]; do
    [[ -z "$need" || "$need" == \#* ]] && continue
    case " $MODULES " in
      *" $need "*) ;;
      *) echo "ABORT: module $mod requires $need, which $PROFILE does not load"; exit 1 ;;
    esac
  done < "$req"
done

((count > 0)) || { echo "ABORT: composed manifest is empty"; exit 1; }
echo "$OUT  ($count overlays from $(echo "$MODULES" | wc -w) modules)"
