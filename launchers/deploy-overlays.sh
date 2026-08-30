#!/bin/bash
# Deploy the canonical overlay manifest and every listed source to the mounted
# -b12x dirs on all 4 nodes, then verify SHA-256 parity. RUN ON srv2 from a repo
# checkout; restart afterwards (systemctl --user restart dsv4-tp4, or run the
# launcher directly).
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
# Which model this deploys. The profile names its modules; the composer renders
# them into the flat directory + single manifest this script has always shipped.
PROFILE=${PROFILE:-${1:-dsv4}}
bash "$REPO/launchers/compose-overlays.sh" "$PROFILE" >&2
BUILD="$REPO/build/$PROFILE"
MANIFEST="$BUILD/manifest.tsv"
MANIFEST_NAME=${MANIFEST##*/}
SSHOPT="-o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=no"
HEAD_OV=/home/choiceoh/hybrid-stack/overlay-b12x
WORKERS="10.10.10.3 10.10.10.1 10.10.10.4"
overlay_dir() { case "$1" in 10.10.10.3) echo /home/choiceoh/hybrid-stack/overlay;; *) echo /home/choiceoh/hybrid-stack-port/overlay;; esac; }

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
    case "$target" in
      /opt/venv/lib/python3.12/site-packages/vllm/*) ;;
      *) echo "ABORT: unsafe overlay target in manifest: $target"; exit 1 ;;
    esac
    case "$target" in
      *[!A-Za-z0-9_./-]*)
        echo "ABORT: unsafe character in overlay target: $target"; exit 1 ;;
    esac
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
bash -n "$REPO/launchers/start-hy4-tp4.sh" "$REPO/launchers/deploy-overlays.sh"
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
  dir=$(overlay_dir "$ip")-b12x
  echo "=== $ip ($dir) ==="
  ssh $SSHOPT "choiceoh@$ip" "mkdir -p $dir"
  scp $SSHOPT -q "${SOURCE_PATHS[@]}" "choiceoh@$ip:$dir/"
  W=$(ssh $SSHOPT "choiceoh@$ip" "cd $dir && sha256sum $REMOTE_FILES")
  [ "$W" = "$SUM" ] || { echo "ABORT: verify failed on $ip"; exit 1; }
  echo "  verified"
done
echo "${#OVFILES[@]} overlays + manifest deployed and SHA-256 verified on head + 3 workers."
echo "restart to apply: systemctl --user restart dsv4-tp4   (expect a long"
echo "recompile warmup on the first boot after an overlay-source change)"
