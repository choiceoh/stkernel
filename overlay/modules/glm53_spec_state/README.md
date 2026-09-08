# glm53_spec_state — stale spec row must not touch recurrent state (vLLM #51508)

Three files replace their `0.1.dev20051+g487ecf187` originals (base shas in
`manifest.tsv`). A fourth hunk of the same fix lives in `glm53_kernels`'s
`fused_recurrent.py`, which owns that FLA op file already.

## The defect

`update_num_computed_tokens_for_batch_change`
(`vllm/v1/spec_decode/utils.py:566` in this image) corrects the async
spec-decode drift by writing `valid_sampled_token_count` into
`num_accepted_tokens`:

```python
num_accepted_tokens.copy_(
    torch.where(participating, valid_counts, num_accepted_tokens)
)
```

`valid_counts` is **0** for a row whose sampled tokens were discarded — a stale
async-scheduling step whose drafts were still scheduled. Every consumer assumes
the count is in `1..1+num_speculative_tokens` and indexes the request's state
slots with `num_accepted_tokens - 1`, so a 0 becomes **-1**:

- Triton (`fused_recurrent.py:116`, `causal_conv1d.py:874` in this image) —
  an out-of-bounds load. `causal_conv1d_update` usually faults first.
- The CPU align-mode copy specs (`mamba_utils.py:338,361`) — `-1` is a *valid*
  Python index, so the copy silently reads from the end of the state and
  corrupts the output with no error at all.

Upstream's note is the important part for us: under
`compute-sanitizer --tool memcheck` the pre-fix kernel reports **zero errors**
at these shapes, because -1 lands inside the allocation on the neighbouring
row. What it produces there is NaN output (75.9 % of elements mismatched), not
a fault. **In the wild this presents as bad output, not a clean crash.**

## Why this profile is exposed

`ASYNC_SCHED` defaults to 1 (`start-glm53-nvfp4-tp4.sh:588`), `DFLASH2=1`,
`SPEC_K=7`, and GLM-5.3-Flash's 34 `linear_attention` layers run KDA through
`fused_recurrent_kda` → `fused_recurrent_gated_delta_rule_fwd_kernel`
(`kda.py:26,384`) with `num_accepted_tokens` and through
`causal_conv1d_update` (`glm5next_kda.py:627,661`). The profile also runs the
mamba `align` cache mode, which is where `mamba_utils.py`'s copy specs live.

## The fix

Two layers, as upstream:

1. `gdn_attn.py` — the builder treats 0 as the staleness signal it is: null the
   row's state slots (`NULL_BLOCK_ID`, already imported at `:19`) so the kernels
   skip both the initial-state read and the final-state write, and clamp the
   count to 1 so the slot index stays in bounds. In-place `masked_fill_` is
   safe here because `spec_state_indices_tensor` comes from **boolean** mask
   indexing of `block_table_tensor`, which always copies — unlike Kimi-K3's
   basic slice, where upstream had to go out-of-place. The FULL-cudagraph
   branch below copies the already-nulled tensor into its persistent buffer, so
   the guard survives capture.
2. `causal_conv1d.py`, `mamba_utils.py` (and `fused_recurrent.py` in
   `glm53_kernels`) — clamp the index. Defense in depth: a raw 0 that reaches a
   kernel without the builder having nulled the row is bounded rather than
   read at -1.

Not gated. This is a correctness fix whose failure mode is silent, so an
opt-in arm would only mean shipping the bug by default.
