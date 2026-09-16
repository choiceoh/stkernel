# Classification rubric — ST engine / ST kernel optimizations by model dependence

Context. The repo (stkernel2) serves LLMs on 4x DGX Spark GB10 (SM121, unified memory) at TP=4 over RoCE.
`engine/` is the in-house "ST engine" (base/ = model-free runtime; modules/ = feature references; profiles/<model>/ =
per-model; kernels/ = the ST kernels: Triton/TileLang/CuTe DSL/CUDA). The ONLY production model is **GLM-5.3-Flash**:
45 layers = 34 KDA linear-attention layers (Kimi Delta Attention, short conv 4, per-channel decay) + 11 DSA sparse
attention layers (MLA 16 heads x 512 latent per rank, kpool indexer with Hadamard-128 FP8 keys, no attention sink),
mHC hyper-connections (hc=4, hidden 4096) on every layer, 3 dense MLPs then 42 MoE layers (288 routed experts, top-8,
1 shared, NVFP4 via the b12x kernels), DFlash2 drafter for speculative decoding (K=7 drafts -> 8 rows at C=1, 16 at C=2).
Other models the engine knows: **Qwen3.8-Flash-Next** (GDN linear attention with per-head decay, GQA+QSA attention,
512-expert NVFP4 MoE top-10, gated-residual hyper-connections, PLE n-gram embedding, hidden 2560, MTP head) and
**DeepSeek-V4.1-Flash** (MLA with an attention sink, CED indexer, split-sinkhorn mHC, MXFP4 experts, engram, hidden 5120).

Your job: for each ITEM (one merged PR or direct commit that touched engine/), decide whether it is an optimization,
and if so how model-dependent it is. Judge from the commit message and file list. If an item is ambiguous you may run
single plain read-only commands such as `git show --stat <hash>` or `git log -1 --format=%B <hash>` or read a file —
NO pipes, NO `&&`, NO `cd`, never modify the repo.

## Fields (one JSON object per item)

- `item` (int, from the header), `pr` (int or null), `title` (<= 90 chars, the gist)
- `opt`: "yes" | "no"
  - yes = its primary purpose (or a substantial part) makes serving faster or cheaper: decode/prefill throughput or
    latency, fewer kernel launches/copies/host syncs, fusions, faster GEMM/comm, lower memory footprint or more cache
    capacity, faster boot/load/compile, better prefix-cache reuse, higher speculative acceptance / tokens-per-step,
    lower-bit lanes whose point is speed/memory (W4A8, FP8 dispatch, GPTQ packs that enable a low-bit lane).
    An optimization that was merged but later rejected/reverted/default-off is still opt=yes (see status).
  - no = features (HTTP door/API, tools, grammar, reasoning/prompt text), correctness/stability fixes, numerics
    guards, docs, tests, measurement/ledger records, probes/instrumentation/profilers/gauges, fleet/queue/lease/launcher
    ops, model bring-up or portability frames (composition, families, kernel-shape descriptor, wizard, glue adapters),
    admission/budget bookkeeping that doesn't change speed/memory use.
- `status` (opt=yes only): "default" (on in production / default path) | "optin" (merged but off by default,
  experimental, probe-only, or awaiting GPU validation and not default) | "rejected" (measured and rejected, reverted,
  or superseded/removed later — only if the message or an obvious later item says so) | "unknown"
- `scope` (opt=yes only) — the key field:
  - "U" model-agnostic: nothing depends on the model's architecture or tensor shapes, or the kernel is purely
    argument-shaped with no GLM-measured dispatch choice. E.g. loader read width, boot/compile caching & parallel
    builds, scheduler/chunking mechanism, KV/prefix cache & snapshot machinery, NVMe tiers, memory admission/page-cache
    reclaim, CUDA-graph capture/replay mechanics, async decode readback/host staging, decode queue, sampler/draws,
    generic block verification, embedding lookup, logits buffers.
  - "S" generic op, GLM-tuned: an op essentially every model has (dense linear GEMM / W4A8 / FP8 projections, norms,
    RoPE, all-reduce / one-shot comm, prefill collectives, chunk sizing), whose adopted choice (tiles, row counts like
    8/16 rows, CTA counts, split points, precision thresholds, compiled hidden=4096 instance) was measured or compiled
    at GLM-5.3's cell. Another model runs it but should re-measure (or needs padding glue).
  - "F" architecture family: only helps models that have that feature: NVFP4 MoE via b12x (routing, scatter, SF6
    scales, expert tiles/streaming), DSA/MLA sparse attention and its indexer (kpool/top-k), KDA/GDN linear attention
    (recurrent state rings, chunk, conv, state precision), mHC hyper-connections, DFlash/MTP drafter kernels and draft
    selection. (Even if it only matches GLM's exact variant today, use F unless it is truly GLM-only per below.)
  - "G" GLM-5.3-only: tied to GLM-5.3-Flash's checkpoint or exact math in a way another model of the same family could
    not reuse without re-deriving it (GLM router FP32 sigmoid-288 specifics, GLM DFlash2 drafter's trained target
    layers, GLM's first-3-dense-MLP guard, GLM-specific calibration data, a megakernel segment hard-compiled for GLM's
    exact combined layer layout).
  - "O" another model only (Qwen3.8 / DeepSeek-V4.1 lanes).
  - If an item mixes scopes, pick the dominant one and list the others in `mixed` (e.g. ["U"]).
- `feature`: one of moe | mla_dsa | indexer | kda_linear | mhc | drafter | dense_gemm | comm | sampler |
  prefix_kv_cache | boot_load_compile | memory | scheduler_runtime | embed_logits | graphs_host | other
- `hw`: true if the gain is specific to GB10/SM121 unified memory/TP4 RoCE (e.g. CTA/TMA tiling, PDL, SM121 byte
  lanes, RDMA one-shot), false if it would help on any GPU box.
- `measured`: "e2e" (fleet/GPU tok/s, step ms, prefill s, boot s measured end to end) | "component" (kernel
  microbench/probe/launch counts only) | "none" | "unknown" — from the message.
- `why`: <= 25 words justifying opt and scope.

Be consistent and conservative: when an item is mostly a feature/fix with a small perf side effect, opt=no.
