# Preserve the inode and mtime of identical overlay sources. Ninja sees their
# bind-mounted mtimes, so rewriting identical .cu files recompiles warm kernels.
# No --times/-a: a changed file must get a fresh mtime even when the checkout's
# source timestamp is older than its previously compiled object.
glm53_sync_overlays() {
  rsync --checksum --perms --chmod=u=rw,go=r "$@"
}

# rsync chooses its own transfer checksum. Attest the final head against the
# canonical sources with SHA-256 before its digest becomes the worker oracle.
glm53_verify_overlay_sources() {
  python3 - "$@" <<'PY'
import hashlib
from pathlib import Path
import sys
target = Path(sys.argv[1])
for name in sys.argv[2:]:
    source = Path(name)
    try:
        same = hashlib.sha256(source.read_bytes()).digest() == hashlib.sha256((target/source.name).read_bytes()).digest()
    except OSError:
        same = False
    if not same:
        raise SystemExit(f"ABORT: deployed {source.name} does not match the canonical source SHA256")
PY
}
