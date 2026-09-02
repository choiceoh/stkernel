# glm53_prep_fused

The decode step's host-side input preparation, replaced by one Triton launch.

## What the trace showed (2026-09-01, rank 3, 229 steps)

The V2 runner prepares each step between the drafter graph and the target
graph: `prepare_inputs`, `prepare_attn`, and `model_state.prepare_attn` over
the seven KV-cache groups (MLA+indexer, kpool tail, four mamba groups, the
drafter). Per step that is ~1,000 aten calls, ~45 memcpys and ~100 kernel
launches of 1-3 us each, and with DFlash the scheduler is synchronous, so the
GPU idles through all of it.

| profiled step (72.1 ms) | idle |
|---|---|
| eager prep between the two graphs (6.6-12.3 ms) | ~5.7 ms |
| `cudaGraphLaunch` of the 1,640-node target graph | 1.43 ms |
| DtoH wait before the drafter graph, step turnaround | ~1.4 ms |
| total | 8.9 ms (12%) |

nvidia-smi without the profiler puts decode at 93% busy, i.e. 4-5 ms of a
66 ms step. The prep region is the largest part of it. Ceiling by the ledger's
rule: the region itself, 4-6% of the step; nothing in it reads bytes.

## What this module does

For the steady-state decode step -- every request in spec-verify with the
full draft, FULL cudagraph dispatched, no request padding -- the patched
`prepare_inputs` issues:

1. one pinned H2D copy of `idx_mapping`,
2. one Triton launch (`_glm53_prep_fused_kernel`, grid = requests + 1 + groups)
   that writes every persistent buffer the captured graph reads:
   input ids (last sampled + drafts), positions, seq_lens, query_start_loc,
   expanded idx mapping, block-table gathers and slot mappings for all groups,
   the four GDN builders' FULL-graph buffers, the sparse-MLA req ids, and the
   indexer's expanded block table / compressed context lengths / translated
   table / compressed slot mapping, plus every tail the stock path re-pads,
3. the deep_gemm `get_paged_mqa_logits_metadata` call and its copy.

`prepare_attn` then returns the same views the stock path returns without
launching, and `model_state.prepare_attn` serves the attention-metadata dict
from a per-shape cache (it is only handed to the speculator, which ignores it;
under FULL replay the graph reads buffers, not Python objects). Anything
outside that shape -- prefill, partial drafts, padded graphs, adaptive
verification, DCP/PCP/PP/LoRA -- takes the stock path untouched.

## Contract: bit-exact with the stock buffers

`VLLM_GLM53_PREP_FUSED=shadow` runs the fused path first, then the stock path
over the same buffers, and diffs every buffer above plus the InputBatch
fields; stock stays the truth. `[prep-fused] shadow ... drift=0` over a full
bench boot is the gate before `=1`.

Offline first (srv4, fresh container, never the serving one):

```bash
bash probes/run_prep_fused_check.sh --trials 60
```

runs the stock building blocks (input_batch kernels, BlockTables gather /
slot kernels, the GDN copies, the indexer's uniform-decode kernel and
compressed slot mapping) and the fused kernel on randomized batches at the
production geometry and requires every tensor bit-exact.

2026-09-02, srv4, `glm53:sm121-fi618`: 60/60 randomized batches (C=1..4), 46
tensors bit-exact. Per C=1 step the stock building blocks take 2,556 us of
wall time (~30 kernels + ~8 memcpys, host-bound); the fused path 201 us
(3 kernels + 1 memcpy), of which the kernel is 88 us and the deep_gemm
schedule 64 us on the GPU. The stock column excludes the runner's ~1,000
aten calls around those blocks, so serving saves more than the difference;
the ceiling is the trace's prep region. Serving numbers: none yet (EXP-7).

## Guards

The installer pins the sha256 of every runner and builder file whose control
flow it bypasses (as shipped in `glm53:v13-b12x`; `mla/indexer.py` is the
`glm53_tail_slot_persistent` copy). Any drift logs `preimage drift -> DISARM`
and leaves the runner stock. The plan is built on the first eligible decode
step from the live runner and refuses (stock for the boot, logged) on any
geometry it was not read against: mamba cache mode other than `none`, an
indexer builder off the flattened uniform-decode path, unknown builders, a
draft width that is not `decode_query_len - 1`.

## What was noticed but not changed

The runner's `MambaHybridModelState.prepare_attn` never passes `positions`
to the builders, so `KpoolTailMetadataBuilder`'s circular tail mapping (and
`glm53_tail_slot_persistent`'s persistent buffer for it) is dormant: the tail
group serves the generic per-group slot mapping, which its own docstring
says collapses concurrent requests onto tail block 0 for `pos >= kpool`.
This module reproduces the generic mapping bit for bit. Enabling the circular
mapping is a numerics change for C >= 2 and belongs to its own bracket.

## Arming

```
VLLM_GLM53_PREP_FUSED=0        # default: inert
VLLM_GLM53_PREP_FUSED=shadow   # bench boot: fused + stock, diff logged
VLLM_GLM53_PREP_FUSED=1        # armed after a clean shadow + EXP-7 bracket
VLLM_GLM53_PREP_FUSED_SHADOW_EVERY=16   # compare every Nth fused step
```

Rollback is the env line. Base contract from `glm53:v13-b12x`.
