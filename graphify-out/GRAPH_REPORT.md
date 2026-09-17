# Graph Report - engine/base  (2026-09-18)

## Corpus Check
- 50 files · ~103,656 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1501 nodes · 4082 edges · 31 communities detected
- Extraction: 56% EXTRACTED · 44% INFERRED · 0% AMBIGUOUS · INFERRED: 1785 edges (avg confidence: 0.62)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Memory Plan|Memory Plan]]
- [[_COMMUNITY_Serving Diagnostics|Serving Diagnostics]]
- [[_COMMUNITY_Device Memory Arena|Device Memory Arena]]
- [[_COMMUNITY_Fleet Lease & Latency|Fleet Lease & Latency]]
- [[_COMMUNITY_Distributed Communication|Distributed Communication]]
- [[_COMMUNITY_Prefix Cache & Tenancy|Prefix Cache & Tenancy]]
- [[_COMMUNITY_Scheduler & KV Blocks|Scheduler & KV Blocks]]
- [[_COMMUNITY_Memory Budget Gates|Memory Budget Gates]]
- [[_COMMUNITY_Request Cache|Request Cache]]
- [[_COMMUNITY_Sampling Constants|Sampling Constants]]
- [[_COMMUNITY_Runtime Introspection|Runtime Introspection]]
- [[_COMMUNITY_Composed Model Lifecycle|Composed Model Lifecycle]]
- [[_COMMUNITY_Stateless Randomness|Stateless Randomness]]
- [[_COMMUNITY_Model & Kernel Shapes|Model & Kernel Shapes]]
- [[_COMMUNITY_Snapshot Publication|Snapshot Publication]]
- [[_COMMUNITY_Stage Timing|Stage Timing]]
- [[_COMMUNITY_Instrumentation|Instrumentation]]
- [[_COMMUNITY_KV Cache Sizing|KV Cache Sizing]]
- [[_COMMUNITY_Common Kernel Lanes|Common Kernel Lanes]]
- [[_COMMUNITY_Stall Watchdog|Stall Watchdog]]
- [[_COMMUNITY_Process Topology|Process Topology]]
- [[_COMMUNITY_Tensor Layout|Tensor Layout]]
- [[_COMMUNITY_Package Init|Package Init]]
- [[_COMMUNITY_Free Block State|Free Block State]]
- [[_COMMUNITY_Block Reservation|Block Reservation]]
- [[_COMMUNITY_Boundary Cache Release|Boundary Cache Release]]
- [[_COMMUNITY_Snapshot Cache Release|Snapshot Cache Release]]
- [[_COMMUNITY_Phase Measurement|Phase Measurement]]
- [[_COMMUNITY_Death Notes|Death Notes]]
- [[_COMMUNITY_Distributed Gate Diagnostics|Distributed Gate Diagnostics]]
- [[_COMMUNITY_Community 30|Community 30]]

## God Nodes (most connected - your core abstractions)
1. `TierFull` - 140 edges
2. `BlockPool` - 116 edges
3. `Tripwire` - 105 edges
4. `StepWatch` - 103 edges
5. `Recorder` - 101 edges
6. `DiagnosticMetrics` - 100 edges
7. `SlotPool` - 85 edges
8. `PagedSpec` - 59 edges
9. `SlotSpec` - 59 edges
10. `Runner` - 58 edges

## Surprising Connections (you probably didn't know these)
- `The shard's RankLoader, header parsed once (148,498 tensors across eleven).` --uses--> `RankLoader`  [INFERRED]
  engine/base/checkpoint.py → engine/base/loader.py
- `The disk cannot take this conversation; forget one or drop this one.` --uses--> `CompressedSnapshots`  [INFERRED]
  engine/base/kv_tier.py → engine/base/compressed_snapshots.py
- `Sampling policy and committed token counts carried alongside device decode rows.` --uses--> `History`  [INFERRED]
  engine/base/sampling_options.py → engine/base/sampler.py
- `Only the clipped, accepted output updates history. Rejected drafts never do.` --uses--> `History`  [INFERRED]
  engine/base/sampling_options.py → engine/base/sampler.py
- `Compact device results; host publication waits for the ordinary outcome event.` --uses--> `History`  [INFERRED]
  engine/base/sampling_options.py → engine/base/sampler.py

## Communities

### Community 0 - "Memory Plan"
Cohesion: 0.03
Nodes (93): _dev_free_bytes(), phase(), process_seconds(), One tree per run. Not thread-safe by design: a step is one thread, and a     loc, A phase that was timed somewhere this recorder could not reach -- the boot's `fr, Seconds since THIS PROCESS started, or None where /proc does not say.      A rec, Free device memory, or None before CUDA is up (see the module note)., Recorder (+85 more)

### Community 1 - "Serving Diagnostics"
Cohesion: 0.04
Nodes (73): _layout(), PagedSpec, Plan, What each layer kind needs cached, declared once, sized from the budget (base)., A spec that names a key also names a dtype and a shape whose bytes are its decla, Split a KV budget: slots first (fixed by max_seqs), blocks with the rest.      B, _selfcheck(), SlotSpec (+65 more)

### Community 2 - "Device Memory Arena"
Cohesion: 0.04
Nodes (135): build_identity(), context_band(), DiagnosticMetrics, Bounded host-side serving diagnostics; never synchronizes a device.  Completion, Exception, The disk cannot take this conversation; forget one or drop this one., TierFull, Recorder (+127 more)

### Community 3 - "Fleet Lease & Latency"
Cohesion: 0.03
Nodes (63): Arena, expandable_segments(), host_reclaim(), _meminfo(), prepare_allocation(), One allocation for persistent weights and caches (base).  D16 in code: weights (, Ask this node's host to return its clean file cache now, and wait for it to say, Drop clean pages of the supplied weight files, reclaim the rest of the     short (+55 more)

### Community 4 - "Distributed Communication"
Cohesion: 0.04
Nodes (81): Budget, A box, the claims on it, and whatever is left for KV., A budget can gate a boot only when nothing in it is a guess., The shard's RankLoader, header parsed once (148,498 tensors across eleven)., acquire(), alive(), attach(), clear_yield() (+73 more)

### Community 5 - "Prefix Cache & Tenancy"
Cohesion: 0.02
Nodes (83): host_box(), Line, probe_box(), rank_files(), rank_weights(), The box, declared -- not discovered (framework).  A budget is a list of claims o, (total device GiB, free GiB after our own context).      Asking costs a CUDA con, (MemTotal GiB, MemAvailable GiB) -- what earlyoom actually counts. (+75 more)

### Community 6 - "Scheduler & KV Blocks"
Cohesion: 0.04
Nodes (42): Comm, fleet_env(), init(), _LocalRank, _LocalRun, LocalTP, pick_gid_index(), RankLeft (+34 more)

### Community 7 - "Memory Budget Gates"
Cohesion: 0.04
Nodes (64): Attention, bind(), bind_recorded(), bound(), Comm, config_sha256(), derive_for(), Device (+56 more)

### Community 8 - "Request Cache"
Cohesion: 0.04
Nodes (67): forget(), fresh(), iota(), Index constants a captured step reads, and may not create.  A decode step asks f, 0..n-1 on `device`, built once and kept. Never write to what this returns., The same values, not kept: for lengths that follow the request., n zeros on `device`, built once and kept, under the same rule as `iota`: the ind, Drop every kept constant. Tests only -- a live capture recorded their addresses. (+59 more)

### Community 9 - "Sampling Constants"
Cohesion: 0.04
Nodes (38): Release NCCL graph references before destroying its process group., Drop that owner; blocks nobody else holds go back to the free list. Returns how, Entry, is_tenant_salt(), PrefixCache, Prefix reuse: a prompt that begins the way an earlier one did is not prefilled a, `chain` continued over `ids[start:]` -- the tokens after the last boundary it kn, `lookup` over a chain computed once by the caller. (+30 more)

### Community 10 - "Runtime Introspection"
Cohesion: 0.06
Nodes (8): boundary tokens -> hash of the prompt up to there, for every whole BLOCK (45차 §2, The tokens `lookup` would reuse, without counting a query (admission asks before, context_overflow(), _media_after(), Server, The loop is about to do `what`; the clock starts (or restarts) now., The thread; idempotent, a no-op when both thresholds are 0., The parked conversation is over: its disk copy goes.

### Community 11 - "Composed Model Lifecycle"
Cohesion: 0.06
Nodes (27): Builder, CompressedSnapshots, Bounded, lossless RAM copies of snapshots whose durable copy is on NVMe.  Only t, Decode through shared staging; `write(offset, n)` must fence its read., Snapshot, NvmeTier, The NVMe tier for cold KV (base). Contiguous per sequence, O_DIRECT, survives bo, Every parked conversation this layout can promote. (+19 more)

### Community 12 - "Stateless Randomness"
Cohesion: 0.07
Nodes (22): available(), for_checkpoint(), Grammars, Matcher, Structured output (base): a grammar per request, a matcher per row, one bitmask, The compile itself. Everything it can fail at is the request's schema, so those, A handle for `spec`: the compiled grammar, or the thread compiling it. Shared by, The compiled grammar, waiting for its thread if it is still running (and raising (+14 more)

### Community 13 - "Model & Kernel Shapes"
Cohesion: 0.09
Nodes (32): _arguments_json(), detect(), grammar_arg_pairs(), grammar_function_xml(), _is_text(), parse_arg_pairs(), parse_function_xml(), partial_arg_pairs() (+24 more)

### Community 14 - "Snapshot Publication"
Cohesion: 0.09
Nodes (16): cleanup_after_error(), DecodeGraphs, frozen_gc(), Decode steps as captured graphs, one per shape (base). I1's mechanism.  I1 says, fill(inputs) copies this step's data into the static buffers; then replay., Keep the first capture failure; attach secondary teardown failures., No Python garbage collection while a graph is recording.      This is a correctn, step_fn(inputs) runs one decode step over static `inputs`;         make_inputs(n (+8 more)

### Community 15 - "Stage Timing"
Cohesion: 0.14
Nodes (15): Config, ConfigError, Fact, Knob, Facts in, knobs that expire, nothing else (base).  D11's evidence was 133 `VLLM_, Boot dies. Loudly, with the reason as the exit message (D3)., _selfcheck(), Lane (+7 more)

### Community 16 - "Instrumentation"
Cohesion: 0.22
Nodes (8): Which window of a boot compiled something -- one walk of the JIT caches, after t, One log line. It says "no JIT writes" out loud, because that is the answer that, The declared cache directories that exist, with nested ones folded into their pa, Named wall-clock windows over a boot, and the cache artifacts that landed inside, Walk once; count the artifacts newer than the first window, by the window they f, roots(), _selfcheck(), Windows

### Community 17 - "KV Cache Sizing"
Cohesion: 0.53
Nodes (5): Gate, judge(), Bands and repeat counts, or it is not a gate (base).  D14 accepted that quiet qu, _selfcheck(), Verdict

### Community 18 - "Common Kernel Lanes"
Cohesion: 0.33
Nodes (4): Cache, max_seq(), What a model caches per sequence and per token, and what a budget buys (framewor, Longest context that fits, at this concurrency.

### Community 19 - "Stall Watchdog"
Cohesion: 0.5
Nodes (4): CommonLanes, The engine's default lanes (base): the model-free kernels every profile inherits, The common kernels, bound once. Importing them imports Triton. The norms and Swi, served()

### Community 20 - "Process Topology"
Cohesion: 0.4
Nodes (1): A step that never ends, bounded (45차, 2026-09-12 22:31).  The fleet's worst fail

### Community 21 - "Tensor Layout"
Cohesion: 1.0
Nodes (1): The tensor as a lane decision may read it: contiguous and 16-byte aligned.

### Community 22 - "Package Init"
Cohesion: 1.0
Nodes (1): One process per node. World 1 needs no process group at all.

### Community 23 - "Free Block State"
Cohesion: 1.0
Nodes (0):

### Community 24 - "Block Reservation"
Cohesion: 1.0
Nodes (1): Measure one phase, optionally accumulating repeated calls in one span.

### Community 25 - "Boundary Cache Release"
Cohesion: 1.0
Nodes (1): Blocks no row holds: free now, counting the ones a boundary would give up.

### Community 26 - "Snapshot Cache Release"
Cohesion: 1.0
Nodes (1): Free blocks nothing remembers: what a reservation spends before any boundary pay

### Community 27 - "Phase Measurement"
Cohesion: 1.0
Nodes (1): Free blocks a cached boundary still holds -- the reuse a reservation spends afte

### Community 28 - "Death Notes"
Cohesion: 1.0
Nodes (1): Free blocks held by a boundary whose snapshot is gone (its state lives on the pr

### Community 29 - "Distributed Gate Diagnostics"
Cohesion: 1.0
Nodes (1): Never raises. A key that cannot be computed is a full gate, and this rank still

### Community 30 - "Community 30"
Cohesion: 1.0
Nodes (1): What the loop is doing now, for a death note.

## Knowledge Gaps
- **321 isolated node(s):** `The box, declared -- not discovered (framework).  A budget is a list of claims o`, `(total device GiB, free GiB after our own context).      Asking costs a CUDA con`, `(MemTotal GiB, MemAvailable GiB) -- what earlyoom actually counts.`, `(GiB, tensor count) read out of a safetensors header.      The header is a const`, `One claim on the box.` (+316 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Tensor Layout`** (1 nodes): `The tensor as a lane decision may read it: contiguous and 16-byte aligned.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Package Init`** (1 nodes): `One process per node. World 1 needs no process group at all.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Free Block State`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Block Reservation`** (1 nodes): `Measure one phase, optionally accumulating repeated calls in one span.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Boundary Cache Release`** (1 nodes): `Blocks no row holds: free now, counting the ones a boundary would give up.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Snapshot Cache Release`** (1 nodes): `Free blocks nothing remembers: what a reservation spends before any boundary pay`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Phase Measurement`** (1 nodes): `Free blocks a cached boundary still holds -- the reuse a reservation spends afte`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Death Notes`** (1 nodes): `Free blocks held by a boundary whose snapshot is gone (its state lives on the pr`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Distributed Gate Diagnostics`** (1 nodes): `Never raises. A key that cannot be computed is a full gate, and this rank still`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 30`** (1 nodes): `What the loop is doing now, for a death note.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `BlockPool` connect `Memory Plan` to `Serving Diagnostics`, `Runtime Introspection`, `Fleet Lease & Latency`, `Sampling Constants`?**
  _High betweenness centrality (0.098) - this node is a cross-community bridge._
- **Why does `TierFull` connect `Device Memory Arena` to `Memory Plan`, `Runtime Introspection`, `Composed Model Lifecycle`?**
  _High betweenness centrality (0.074) - this node is a cross-community bridge._
- **Why does `History` connect `Serving Diagnostics` to `Request Cache`, `Memory Plan`?**
  _High betweenness centrality (0.049) - this node is a cross-community bridge._
- **Are the 136 inferred relationships involving `TierFull` (e.g. with `CompressedSnapshots` and `Model`) actually correct?**
  _`TierFull` has 136 INFERRED edges - model-reasoned connections that need verification._
- **Are the 91 inferred relationships involving `BlockPool` (e.g. with `ComposedModel` and `Drafter`) actually correct?**
  _`BlockPool` has 91 INFERRED edges - model-reasoned connections that need verification._
- **Are the 92 inferred relationships involving `Tripwire` (e.g. with `_Choice` and `_Histogram`) actually correct?**
  _`Tripwire` has 92 INFERRED edges - model-reasoned connections that need verification._
- **Are the 93 inferred relationships involving `StepWatch` (e.g. with `_Choice` and `_Histogram`) actually correct?**
  _`StepWatch` has 93 INFERRED edges - model-reasoned connections that need verification._
