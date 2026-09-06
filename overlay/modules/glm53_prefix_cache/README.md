# glm53_prefix_cache — hybrid-APC coordinator fix (39차)

`kv_cache_coordinator.py` replaces `vllm/v1/core/kv_cache_coordinator.py`
(image build `0.1.dev20051+g487ecf187`, base sha in `manifest.tsv`). Three
hunks, all marked `[glm53-hybrid-apc]`:

1. `eagle_group_ids`: `dflash` counts as `use_eagle()`, but GLM-5.3 never sets
   `is_eagle_group` on any KV group (that annotator is DeepSeek-V4-only), so
   stock vLLM's fallback flags EVERY group as EAGLE and MLA drops its last
   scheduler-aligned block on each hit. Now only the drafter's
   `SlidingWindowSpec` group is flagged.
2. A boot-log line `hybrid APC groups: [...]; eagle_group_ids=[...]` -- the
   expected healthy shape is one flagged group, the drafter's SWA.
3. `find_longest_cache_hit`: the drafter SWA group no longer `min()`s the
   hybrid hit; if its cached window does not cover the MLA/mamba hit, its
   blocks are left empty so a fresh window is allocated (the indexer's
   `KpoolTail` group already opted out of prefix caching).

Origin: MiaAI-Lab `overlay/patch_hybrid_prefix_hit.py` (EXL3 stack), carried
to NVFP4 by gorbatjovy as `kv_cache_coordinator.hybrid-apc.py`; their
measurement on 2x GB10: 100K session 96.7 % reuse, needle test passing.

Arming: the module is always composed (it is inert with prefix caching off);
`PREFIX_CACHE=1` (launcher `--enable-prefix-caching`) turns the cache on.

What it does NOT fix: upstream vLLM #47926 (draft, needs rebase) -- tokens
restored from the prefix cache never flow through the target, so the DFlash2
drafter's context KV for them is unwritten unless the drafter's own cached
window blocks are reused. `probes/apc_hit_test.py` measures acceptance on
warm hits for exactly this reason; a hit counter that rises while acceptance
collapses to position 0 is worse than no cache.
