#!/bin/bash
# Deploy the repo's overlay/*.py to the mounted -b12x dirs on all 4 nodes and
# verify md5 parity — the write side of the skew check start-hy4-tp4.sh's
# preflight enforces. RUN ON srv2 from a repo checkout; restart afterwards
# (systemctl --user restart dsv4-tp4, or bash start-hy4-tp4.sh).
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
OVFILES="attention.py flashinfer_sparse.py indexer.py sparse_swa_dsv4.py"
SSHOPT="-o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=no"
HEAD_OV=/home/choiceoh/hybrid-stack/overlay-b12x
WORKERS="10.10.10.3 10.10.10.1 10.10.10.4"
overlay_dir() { case "$1" in 10.10.10.3) echo /home/choiceoh/hybrid-stack/overlay;; *) echo /home/choiceoh/hybrid-stack-port/overlay;; esac; }

for f in $OVFILES; do test -f "$REPO/overlay/$f" || { echo "ABORT: $REPO/overlay/$f missing"; exit 1; }; done
python3 -m py_compile $(for f in $OVFILES; do echo "$REPO/overlay/$f"; done) \
  || { echo "ABORT: overlay does not compile"; exit 1; }
rm -rf "$REPO/overlay/__pycache__"
if [ -f "$REPO/tests/test_logic.py" ]; then
  python3 "$REPO/tests/test_logic.py" || { echo "ABORT: tests/test_logic.py failed"; exit 1; }
fi

echo "=== head ($HEAD_OV) ==="
mkdir -p "$HEAD_OV"
for f in $OVFILES; do install -m 0644 "$REPO/overlay/$f" "$HEAD_OV/$f"; done
SUM=$(cd "$HEAD_OV" && md5sum $OVFILES)
echo "$SUM" | sed 's/^/  /'

for ip in $WORKERS; do
  dir=$(overlay_dir "$ip")-b12x
  echo "=== $ip ($dir) ==="
  ssh $SSHOPT "choiceoh@$ip" "mkdir -p $dir"
  for f in $OVFILES; do scp $SSHOPT -q "$REPO/overlay/$f" "choiceoh@$ip:$dir/$f"; done
  W=$(ssh $SSHOPT "choiceoh@$ip" "cd $dir && md5sum $OVFILES")
  [ "$W" = "$SUM" ] || { echo "ABORT: verify failed on $ip"; exit 1; }
  echo "  verified"
done
echo "overlays deployed + md5-verified on head + 3 workers."
echo "restart to apply: systemctl --user restart dsv4-tp4   (expect a long"
echo "recompile warmup on the first boot after an overlay-source change)"
