#!/usr/bin/env bash
# Render a profile's modules into the flat overlay directory + single manifest
# the rest of the stack expects.
#
# Collisions abort rather than resolve: two modules claiming the same source
# filename, or two rows binding the same container path. A module that collides
# with another was not as independent as its README claims, and letting one win
# would bury that.
set -euo pipefail
# Shared manifest/target validation -- one implementation for the launchers,
# the composer and the deployer.
# shellcheck source=lib/common-tp4.sh
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common-tp4.sh"

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
TARGET_PREFIX=${TARGET_PREFIX:-/opt/venv/lib/python3.12/site-packages/}

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
    # deneb fork: same CRLF checkout hazard as the requires loop -- a \r on
    # the contract hash would land in the composed manifest and fail the
    # 4-node SHA-256 verification with a hash that never matches.
    contract=${contract%$'\r'} target=${target%$'\r'}
    [[ -z "$source" || "$source" == \#* ]] && continue
    [ -n "$target" ] && [ -n "$contract" ] \
      || { echo "ABORT: malformed row in module $mod: $source"; exit 1; }
    [ -z "${SRC_OWNER[$source]:-}" ] \
      || { echo "ABORT: $source claimed by ${SRC_OWNER[$source]} and $mod"; exit 1; }
    [ -z "${TGT_OWNER[$target]:-}" ] \
      || { echo "ABORT: $target bound by ${TGT_OWNER[$target]} and $mod"; exit 1; }
    SRC_OWNER[$source]=$mod
    TGT_OWNER[$target]=$mod
    # A target may be written relative to the package root, which is what lets
    # one module serve images that install to different places. Absolute targets
    # are left alone.
    case "$target" in
      /*) ;;
      *) target="${TARGET_PREFIX}${target}" ;;
    esac
    # Validated AFTER the relative-target expansion above, which is what makes
    # the .. case reachable here: a module writing "../../x" gets the prefix
    # glued on, so it satisfies the prefix test while pointing outside the root.
    ct_check_overlay_target "$target" "$TARGET_PREFIX" "$mod in $PROFILE"
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
    # deneb fork: a CRLF checkout (Windows clones) leaves \r on the module
    # name, which silently fails EVERY requires match and aborts compose
    # with a false "does not load" -- strip it before matching (#191 class).
    need=${need%$'\r'}
    [[ -z "$need" || "$need" == \#* ]] && continue
    case " $MODULES " in
      *" $need "*) ;;
      *) echo "ABORT: module $mod requires $need, which $PROFILE does not load"; exit 1 ;;
    esac
  done < "$req"
done

((count > 0)) || { echo "ABORT: composed manifest is empty"; exit 1; }
echo "$OUT  ($count overlays from $(echo "$MODULES" | wc -w) modules)"
