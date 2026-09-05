# glm53_drafter_prep

The DFlash drafter's per-step host build of its attention metadata, skipped
when the drafter step is a FULL cudagraph replay. Knob
`VLLM_GLM53_DRAFTER_PREP` = `0` (default, stock) | `time` | `shadow` | `1`.

## What the trace showed (34차, 2026-09-05, production boot, rank 0 and rank 3)

Between the last `reshape_and_cache` of `precompute_and_store_context_kv`
and the first kernel of the drafter graph the GPU idles **420 us on rank 0
and 640 us on rank 3** every decode step (profiler-inflated; the un-profiled
number is what the `time` mode logs). `tools/trace_step_nodes.py --gap`
lists what the host does in that window:

| rel us (rank 0) | event | what |
|---|---|---|
| 52 492 | `Memcpy DtoH (Device -> Pageable)` 3.6 us | `seq_lens.cpu()` |
| 52 502 | `cudaStreamSynchronize` | the same `.cpu()` |
| 52 505 -> 52 793 | (nothing on the GPU) | ~290 us of Python |
| 52 793 | `cudaGraphLaunch` 192 us | the drafter graph |
| 52 923 | `k_oneshot` **245 us** | rank 0 waiting for rank 3's longer gap |

The Python is `DFlashSpeculator.propose` -> `_build_draft_attn_metadata` ->
`build_attn_metadata` -> the FlashInfer builder. The drafter's block
attention is non-causal, so the builder's `all_uses_trtllm` is False and it
reads `common_attn_metadata.seq_lens_cpu` -- a property that copies the GPU
`seq_lens` to pageable host memory (the DtoH + sync above) -- and then
builds metadata objects. `propose` then takes the FULL branch:

```python
if batch_desc.cg_mode == CUDAGraphMode.FULL:
    self.query_cudagraph_manager.run_fullgraph(batch_desc)   # replays; never reads draft_attn_metadata
else:
    self._generate_draft(num_reqs, num_tokens_padded, draft_attn_metadata, ...)
```

The replayed graph reads the GPU buffers `_prepare_dflash_inputs_kernel`
wrote (input ids, positions, seq_lens, query_start_loc) and the runner's
block-table / slot-mapping gathers; the metadata dict is consumed only by
the eager branch. So on the FULL branch the build is pure host cost, and
the sync in front of it turns the Python into GPU idle.

## What this module does

`install_glm53_drafter_prep()` (called from the GLM model wiring, like
`glm53_prep_fused` and `glm53_dflash_early_fc`) wraps
`DFlashSpeculator._build_draft_attn_metadata` once, after verifying the
sha256 of the three image files whose internals it relies on (the DFlash
speculator, its base, the cudagraph manager -- drift means DISARM, stock).

The wrapper decides FULL-ness the way `propose` does -- it asks the
drafter's own cudagraph manager to dispatch `(num_reqs, num_reqs x
num_query_per_req)` and requires a FULL descriptor of exactly the padded
token count it was handed -- and then, by mode:

| mode | build | serves | logs |
|---|---|---|---|
| `time` | stock, every step | stock | `[drafter-prep] stock build host us: n= median= p90= max=` every 256 calls |
| `shadow` | stock, every step | stock | the dict cached for this shape is compared field by field: GPU tensors by identity/shape/dtype/stride, host values by equality with the per-step scalars (`max_seq_len`, `seq_lens_cpu_upper_bound`, ...) counted apart; `[drafter-prep] shadow: full= drift= ...` every 256 |
| `1` | once per shape `(num_reqs, num_reqs_padded, num_tokens_padded, step, causal)` | the cached dict on FULL replays | `[drafter-prep] serving: FULL replay -> draft metadata cached for key=...` once, `tally: served= stock=` every 1024 |

Anything the FULL test rejects -- eager batches, a `query_start_loc_np`
override, DP > 1, profile runs (the manager answers NONE before capture)
-- runs the stock build. Any exception inside the wrapper disables it for
the boot, loudly, and returns the stock build.

Nothing numeric changes: the same captured graph replays the same buffers.
The step gets shorter only by the host time removed, so the judgement is
the C=1 step/s bracket with prefill alongside (RUNBOOK EXP-24); `time`
mode on a production boot gives the honest ceiling first.

## Verification ladder

1. `tests/test_logic.py` -- contracts (knob, preimages, wiring, the FULL
   test and the compare on fakes).
2. A `time` boot: the median build time is the ceiling.
3. A `shadow` boot: `drift=0` over a bench run (per-step scalars may
   differ; anything else is a bug).
4. `1` as one arm of the fusion bundle (EXP-24).
