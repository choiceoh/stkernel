# Follow-up: override precedes maximum-length admission

The original GMU review is preserved byte-for-byte under `gmu-original/`, including its then-unresolved statement about the missing pinned KV source. This follow-up resolves **only that source-code ordering question**.

The captured image is `sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211`; the original receipt records a never-started container. Its `kv_cache_utils.py` SHA is `624ea7b0244972cb6c53044588912dfea54f3d6a91661cbc423af27e3b5c4b86`. In `get_kv_cache_configs`, lines2597–2617 read `num_gpu_blocks_override` and replace each nonempty worker group's profiled available memory with `override * bytes_per_block`. This occurs before optional auto-fit at2619 and `_check_enough_kv_cache_memory` at2628. The explicit1056 override is therefore the effective capacity used at this admission point. `core.py:305` passes profiling results into this function; the adjustment occurs inside it.

This removes the earlier possibility that the projected head4.86GiB raw profiling budget necessarily reaches maximum-length admission unchanged. It does **not** prove physical headroom, a successful allocation, a fresh preflight/boot, or any v9 result. GMU0.60 does not reduce the fixed1056-block KV allocation. The original normal safety checks and fresh all-rank capacity/readiness evidence remain authoritative.

Only existing local source and receipts were read. No runtime import, tests, remote operation, GPU request or service change was performed.
