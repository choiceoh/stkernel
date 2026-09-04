#!/bin/bash
# Deploy the canonical overlay manifest and every listed source to the mounted
# -b12x dirs on all 4 nodes, then verify SHA-256 parity. RUN ON srv2 from a repo
# checkout; restart afterwards (systemctl --user restart dsv4-tp4, or run the
# launcher directly).
set -euo pipefail
# Shared manifest/target validation -- one implementation for the launchers,
# the composer and the deployer.
# shellcheck source=lib/common-tp4.sh
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common-tp4.sh"

REPO=$(cd "$(dirname "$0")/.." && pwd)

require_deployable_checkout() {
  command -v git >/dev/null 2>&1 \
    || { echo "ABORT: git is required to attest the deploy source"; exit 1; }
  git -C "$REPO" rev-parse --is-inside-work-tree >/dev/null 2>&1 \
    || { echo "ABORT: deploy source is not a git checkout ($REPO)"; exit 1; }
  if [ -n "$(git -C "$REPO" status --porcelain --untracked-files=normal)" ]; then
    echo "ABORT: deploy checkout is dirty; commit or remove local changes first"
    git -C "$REPO" status --short
    exit 1
  fi
  git -C "$REPO" fetch --quiet origin main \
    || { echo "ABORT: could not refresh origin/main"; exit 1; }
  git -C "$REPO" merge-base --is-ancestor origin/main HEAD \
    || {
      echo "ABORT: HEAD is not based on current origin/main; refusing a stale overlay rollback"
      echo "  HEAD        $(git -C "$REPO" rev-parse --short HEAD)"
      echo "  origin/main $(git -C "$REPO" rev-parse --short origin/main)"
      exit 1
    }
  SOURCE_COMMIT=$(git -C "$REPO" rev-parse --verify HEAD)
}

require_deployable_checkout
# Which model this deploys. The profile names its modules; the composer renders
# them into the flat directory + single manifest this script has always shipped.
PROFILE=${PROFILE:-${1:-dsv4}}
bash "$REPO/launchers/compose-overlays.sh" "$PROFILE" >&2
BUILD="$REPO/build/$PROFILE"
# The profile owns its package root and, if it has one, its overlay
# directory. dsv4 predates the split and keeps its per-node paths.
# shellcheck disable=SC1090
. "$REPO/profiles/$PROFILE.env"
# Deploy a staged manifest so the live artifact records the exact source
# revision without creating an impossible self-referential hash in the tracked
# build manifest. Parsers already ignore comment rows.
DEPLOY_STAGE=$(mktemp -d "${TMPDIR:-/tmp}/stkernel-deploy.XXXXXX")
trap 'rm -rf "$DEPLOY_STAGE"' EXIT
MANIFEST="$DEPLOY_STAGE/manifest.tsv"
{
  printf '# source_commit=%s\n' "$SOURCE_COMMIT"
  cat "$BUILD/manifest.tsv"
} > "$MANIFEST"
MANIFEST_NAME=${MANIFEST##*/}
SSHOPT="-o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=no"
WORKERS="10.10.10.3 10.10.10.1 10.10.10.4"
if [ -n "${PROFILE_OVERLAY_DIR:-}" ]; then
  HEAD_OV="$PROFILE_OVERLAY_DIR"
  OV_SUFFIX=""
  overlay_dir() { echo "$PROFILE_OVERLAY_DIR"; }
else
  HEAD_OV=/home/choiceoh/hybrid-stack/overlay-b12x
  OV_SUFFIX="-b12x"
  overlay_dir() { case "$1" in 10.10.10.3) echo /home/choiceoh/hybrid-stack/overlay;; *) echo /home/choiceoh/hybrid-stack-port/overlay;; esac; }
fi

load_overlay_manifest() {
  local source target base_contract extra seen
  [ -f "$MANIFEST" ] || { echo "ABORT: overlay manifest missing ($MANIFEST)"; exit 1; }
  OVFILES=()
  OVTARGETS=()
  OVBASES=()
  while IFS=$'\t' read -r source target base_contract extra \
      || [ -n "${source:-}${target:-}${base_contract:-}${extra:-}" ]; do
    [[ -z "$source" || "$source" == \#* ]] && continue
    [ -n "$target" ] && [ -n "$base_contract" ] && [ -z "${extra:-}" ] \
      || { echo "ABORT: malformed overlay manifest row: $source $target $base_contract ${extra:-}"; exit 1; }
    case "$source" in
      *[!A-Za-z0-9._-]*|.*)
        echo "ABORT: unsafe overlay source in manifest: $source"; exit 1 ;;
    esac
    ct_check_overlay_target "$target" \
      "${TARGET_PREFIX:-/opt/venv/lib/python3.12/site-packages/}" "$MANIFEST"
    if [ "$base_contract" != "absent" ] \
        && [[ ! "$base_contract" =~ ^[0-9a-f]{64}$ ]]; then
      echo "ABORT: invalid base preimage contract for $source: $base_contract"
      exit 1
    fi
    for seen in "${OVFILES[@]}"; do
      [ "$seen" != "$source" ] \
        || { echo "ABORT: duplicate overlay source in manifest: $source"; exit 1; }
    done
    for seen in "${OVTARGETS[@]}"; do
      [ "$seen" != "$target" ] \
        || { echo "ABORT: duplicate overlay target in manifest: $target"; exit 1; }
    done
    OVFILES+=("$source")
    OVTARGETS+=("$target")
    OVBASES+=("$base_contract")
  done < "$MANIFEST"
  ((${#OVFILES[@]} > 0)) || { echo "ABORT: overlay manifest is empty"; exit 1; }
}
load_overlay_manifest

PYFILES=()
SOURCE_PATHS=("$MANIFEST")
for f in "${OVFILES[@]}"; do
  source_path="$BUILD/$f"
  [ -f "$source_path" ] || { echo "ABORT: $source_path missing"; exit 1; }
  SOURCE_PATHS+=("$source_path")
  case "$f" in *.py) PYFILES+=("$source_path");; esac
done
if ((${#PYFILES[@]} > 0)); then
  python3 -m py_compile "${PYFILES[@]}" \
    || { echo "ABORT: overlay does not compile"; exit 1; }
fi
rm -rf "$BUILD/__pycache__"
bash -n "$REPO/launchers/${PROFILE_LAUNCHER:-start-hy4-tp4.sh}" "$REPO/launchers/deploy-overlays.sh"
python3 "$REPO/launchers/audit-runtime-guards.py" --self-test
if [ -f "$REPO/tests/test_logic.py" ]; then
  python3 "$REPO/tests/test_logic.py" || { echo "ABORT: tests/test_logic.py failed"; exit 1; }
fi

echo "=== head ($HEAD_OV) ==="
mkdir -p "$HEAD_OV"
install -m 0644 "$MANIFEST" "$HEAD_OV/$MANIFEST_NAME"
for f in "${OVFILES[@]}"; do
  install -m 0644 "$BUILD/$f" "$HEAD_OV/$f"
done
SUM=$(cd "$HEAD_OV" && sha256sum "$MANIFEST_NAME" "${OVFILES[@]}")
echo "$SUM" | sed 's/^/  /'

REMOTE_FILES="$MANIFEST_NAME ${OVFILES[*]}"
for ip in $WORKERS; do
  dir=$(overlay_dir "$ip")$OV_SUFFIX
  echo "=== $ip ($dir) ==="
  ssh $SSHOPT "choiceoh@$ip" "mkdir -p $dir"
  scp $SSHOPT -q "${SOURCE_PATHS[@]}" "choiceoh@$ip:$dir/"
  W=$(ssh $SSHOPT "choiceoh@$ip" "cd $dir && sha256sum $REMOTE_FILES")
  [ "$W" = "$SUM" ] || { echo "ABORT: verify failed on $ip"; exit 1; }
  echo "  verified"
done
echo "${#OVFILES[@]} overlays + manifest deployed and SHA-256 verified on head + 3 workers."
echo "restart to apply: launchers/${PROFILE_LAUNCHER:-start-hy4-tp4.sh}   (expect a long"
echo "recompile warmup on the first boot after an overlay-source change)"
