"""Shared-memory plan for st_dsa_select, mirrored by the Python wrapper."""
STATIC_SMEM = 8192          # hist + replicas + selected + scalars, rounded up
MIN_STASH = 1024
MAX_STASH = 8192


def plan(n_cand, smem_limit):
    avail = smem_limit - STATIC_SMEM
    bin_bytes = (n_cand + 3) // 4 * 4
    if bin_bytes + 16 * MIN_STASH <= avail:
        return bin_bytes, min((avail - bin_bytes) // 16, MAX_STASH)
    return 0, min(avail // 16, MAX_STASH)


def block_threads(n_cand):
    threads = 1 << max(0, (max(1, n_cand // 4) - 1).bit_length())
    return max(256, min(threads, 1024))
