# Preserve the inode and mtime of identical overlay sources. Ninja sees their
# bind-mounted mtimes, so rewriting identical .cu files recompiles warm kernels.
# No --times/-a: a changed file must get a fresh mtime even when the checkout's
# source timestamp is older than its previously compiled object.
glm53_sync_overlays() {
  rsync --checksum --perms --chmod=u=rw,go=r "$@"
}
