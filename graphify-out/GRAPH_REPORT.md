# Graph Report - engine/base  (2026-09-19)

## Corpus Check
- 54 files · ~108,955 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1574 nodes · 4303 edges · 32 communities detected
- Extraction: 55% EXTRACTED · 45% INFERRED · 0% AMBIGUOUS · INFERRED: 1924 edges (avg confidence: 0.63)
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
- [[_COMMUNITY_Community 31|Community 31]]

## God Nodes (most connected - your core abstractions)
1. `TierFull` - 140 edges
2. `BlockPool` - 116 edges
3. `Tripwire` - 105 edges
4. `StepWatch` - 103 edges
5. `Recorder` - 101 edges
6. `DiagnosticMetrics` - 100 edges
7. `SlotPool` - 85 edges
8. `PagedSpec` - 60 edges
9. `SlotSpec` - 60 edges
10. `Runner` - 58 edges

## Surprising Connections (you probably didn't know these)
- `Free the one allocation now, whatever still points at it, and say how many bytes` --uses--> `RankLoader`  [INFERRED]
  engine/base/arena.py → engine/base/loader.py
- `step()` --calls--> `profiling()`  [INFERRED]
  engine/base/latency.py → engine/base/graph_labels.py
- `Sampling policy and committed token counts carried alongside device decode rows.` --uses--> `History`  [INFERRED]
  engine/base/sampling_options.py → engine/base/sampler.py
- `Only the clipped, accepted output updates history. Rejected drafts never do.` --uses--> `History`  [INFERRED]
  engine/base/sampling_options.py → engine/base/sampler.py
- `Compact device results; host publication waits for the ordinary outcome event.` --uses--> `History`  [INFERRED]
  engine/base/sampling_options.py → engine/base/sampler.py

## Communities

### Community 0 - "Memory Plan"
Cohesion: 0.03
Nodes (128): _layout(), PagedSpec, Plan, What each layer kind needs cached, declared once, sized from the budget (base)., A spec that names a key also names a dtype and a shape whose bytes are its decla, Split a KV budget: slots first (fixed by max_seqs), blocks with the rest.      B, _selfcheck(), SlotSpec (+120 more)

### Community 1 - "Serving Diagnostics"
Cohesion: 0.04
Nodes (135): DiagnosticMetrics, Exception, TierFull, preparation(), Recorder, _write(), answer_budget(), _byte_level_chars() (+127 more)

### Community 2 - "Device Memory Arena"
Cohesion: 0.04
Nodes (40): The next block to hand out: anonymous first, a whole boundary's last., Atomically cover absolute write ends, reusing rejected draft space., MemoryError, The pool this cache claims blocks in; the pool tells it when a claimed block has, No cached boundary one block longer extends this one. Kept as a count at insert/, Up to `count` boundaries in eviction order that have no copy on the tier yet: wh, A slot whose contents belong to nobody (an aborted checkpoint, a boundary that i, Which boundary the tier's slot `key` belongs to, or None.          The tier inde (+32 more)

### Community 3 - "Fleet Lease & Latency"
Cohesion: 0.04
Nodes (28): dict, decode_host_state(), install_preparation_observers(), Bounded, request-scoped onepass recording, on each serving rank.  Normal request, Already committed host bookkeeping only; never query device tensors.      Keep t, Observe new specializations, including disk-cache loads, without changing result, step(), _arg() (+20 more)

### Community 4 - "Distributed Communication"
Cohesion: 0.04
Nodes (51): expandable_segments(), host_reclaim(), _meminfo(), prepare_allocation(), One allocation for persistent weights and caches (base).  D16 in code: weights (, Ask this node's host to return its clean file cache now, and wait for it to say, Drop clean pages of the supplied weight files, reclaim the rest of the     short, Back every new caching-allocator segment with 20 MiB physical chunks     under o (+43 more)

### Community 5 - "Prefix Cache & Tenancy"
Cohesion: 0.03
Nodes (48): Background, Host work a boot runs where it is already waiting (base): a thread started early, Host work started where the boot is already waiting, and joined where its result, check_box(), The box every profile is written for, and its assertion (base).  The charter's D, The node the engine is written for, asserted (D3): one GB10, unified memory., Budget, host_box() (+40 more)

### Community 6 - "Scheduler & KV Blocks"
Cohesion: 0.06
Nodes (64): acquire(), alive(), attach(), clear_yield(), describe(), docker_evidence(), door_load(), door_unsupported() (+56 more)

### Community 7 - "Memory Budget Gates"
Cohesion: 0.04
Nodes (62): rank_weights(), (GiB, tensor count) read out of a safetensors header.      The header is a const, _float32(), _lsr(), mix(), mix_tensor(), Stateless uniforms: every draw is a function of what it is for, never of what ca, Logical shift right of an int64 tensor: torch shifts arithmetically, the mask dr (+54 more)

### Community 8 - "Request Cache"
Cohesion: 0.04
Nodes (42): Put `block` where its owners say it belongs, at the back of that grade., A boundary remembers these blocks: they stay where they are, behind everything o, That boundary is gone (or changed grade): the blocks fall back to what is left h, Seat a cached prefix -- `tokens` whole blocks that are complete and shared -- at, One more owner for each block, for an owner that is not a row: bytes a transfer, Drop that owner; blocks nobody else holds go back to the free list. Returns how, `block` joins the back of `grade`; the front is what leaves first., Entry (+34 more)

### Community 9 - "Sampling Constants"
Cohesion: 0.05
Nodes (34): Comm, fleet_env(), init(), _LocalRank, _LocalRun, LocalTP, pick_gid_index(), RankLeft (+26 more)

### Community 10 - "Runtime Introspection"
Cohesion: 0.06
Nodes (62): Attention, bind(), bind_recorded(), bound(), claims(), Comm, config_sha256(), derive_for() (+54 more)

### Community 11 - "Composed Model Lifecycle"
Cohesion: 0.04
Nodes (66): forget(), fresh(), iota(), Index constants a captured step reads, and may not create.  A decode step asks f, 0..n-1 on `device`, built once and kept. Never write to what this returns., The same values, not kept: for lengths that follow the request., n zeros on `device`, built once and kept, under the same rule as `iota`: the ind, Drop every kept constant. Tests only -- a live capture recorded their addresses. (+58 more)

### Community 12 - "Stateless Randomness"
Cohesion: 0.07
Nodes (21): Builder, CompressedSnapshots, Bounded, lossless RAM copies of snapshots whose durable copy is on NVMe.  Only t, Decode through shared staging; `write(offset, n)` must fence its read., Snapshot, NvmeTier, The NVMe tier for cold KV (base). Contiguous per sequence, O_DIRECT, survives bo, Every parked conversation this layout can promote. (+13 more)

### Community 13 - "Model & Kernel Shapes"
Cohesion: 0.04
Nodes (34): cleanup_after_error(), DecodeGraphs, frozen_gc(), Decode steps as captured graphs, one per shape (base). I1's mechanism.  I1 says, fill(inputs) copies this step's data into the static buffers; then replay., Release NCCL graph references before destroying its process group., Keep the first capture failure; attach secondary teardown failures., No Python garbage collection while a graph is recording.      This is a correctn (+26 more)

### Community 14 - "Snapshot Publication"
Cohesion: 0.06
Nodes (24): Arena, Free the one allocation now, whatever still points at it, and say how many bytes, carve(), Bind the plan to the arena: one KV region, one slot region, two pools., Block tables and state slots, as flat arrays (base).  Two kinds of per-sequence, Every free block in the order it would be handed out -- anonymous first, an oper, Bind the pool to `num_blocks * block_bytes` of arena (D16)., A sequence's blocks in order, as views (what a tier demotes). (+16 more)

### Community 15 - "Stage Timing"
Cohesion: 0.07
Nodes (22): available(), for_checkpoint(), Grammars, Matcher, Structured output (base): a grammar per request, a matcher per row, one bitmask, The compile itself. Everything it can fail at is the request's schema, so those, A handle for `spec`: the compiled grammar, or the thread compiling it. Shared by, The compiled grammar, waiting for its thread if it is still running (and raising (+14 more)

### Community 16 - "Instrumentation"
Cohesion: 0.07
Nodes (22): changed(), digest_tree(), file_identity(), key_of(), Limits, node_identity(), optional_identity(), PrefillRecord (+14 more)

### Community 17 - "KV Cache Sizing"
Cohesion: 0.09
Nodes (32): _arguments_json(), detect(), grammar_arg_pairs(), grammar_function_xml(), _is_text(), parse_arg_pairs(), parse_function_xml(), partial_arg_pairs() (+24 more)

### Community 18 - "Common Kernel Lanes"
Cohesion: 0.2
Nodes (7): _dev_free_bytes(), phase(), process_seconds(), A phase that was timed somewhere this recorder could not reach -- the boot's `fr, Seconds since THIS PROCESS started, or None where /proc does not say.      A rec, Free device memory, or None before CUDA is up (see the module note)., Span

### Community 19 - "Stall Watchdog"
Cohesion: 0.33
Nodes (4): Cache, max_seq(), What a model caches per sequence and per token, and what a budget buys (framewor, Longest context that fits, at this concurrency.

### Community 20 - "Process Topology"
Cohesion: 0.5
Nodes (4): CommonLanes, The engine's default lanes (base): the model-free kernels every profile inherits, The common kernels, bound once. Importing them imports Triton. The norms and Swi, served()

### Community 21 - "Tensor Layout"
Cohesion: 0.4
Nodes (1): A step that never ends, bounded (45차, 2026-09-12 22:31).  The fleet's worst fail

### Community 22 - "Package Init"
Cohesion: 1.0
Nodes (1): The tensor as a lane decision may read it: contiguous and 16-byte aligned.

### Community 23 - "Free Block State"
Cohesion: 1.0
Nodes (1): One process per node. World 1 needs no process group at all.

### Community 24 - "Block Reservation"
Cohesion: 1.0
Nodes (0):

### Community 25 - "Boundary Cache Release"
Cohesion: 1.0
Nodes (1): Measure one phase, optionally accumulating repeated calls in one span.

### Community 26 - "Snapshot Cache Release"
Cohesion: 1.0
Nodes (1): Blocks no row holds: free now, counting the ones a boundary would give up.

### Community 27 - "Phase Measurement"
Cohesion: 1.0
Nodes (1): Free blocks nothing remembers: what a reservation spends before any boundary pay

### Community 28 - "Death Notes"
Cohesion: 1.0
Nodes (1): Free blocks a cached boundary still holds -- the reuse a reservation spends afte

### Community 29 - "Distributed Gate Diagnostics"
Cohesion: 1.0
Nodes (1): Free blocks held by a boundary whose snapshot is gone (its state lives on the pr

### Community 30 - "Community 30"
Cohesion: 1.0
Nodes (1): Never raises. A key that cannot be computed is a full gate, and this rank still

### Community 31 - "Community 31"
Cohesion: 1.0
Nodes (1): What the loop is doing now, for a death note.

## Knowledge Gaps
- **340 isolated node(s):** `Host work a boot runs where it is already waiting (base): a thread started early`, `Host work started where the boot is already waiting, and joined where its result`, `The box every profile is written for, and its assertion (base).  The charter's D`, `The node the engine is written for, asserted (D3): one GB10, unified memory.`, `The box, declared -- not discovered (framework).  A budget is a list of claims o` (+335 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Package Init`** (1 nodes): `The tensor as a lane decision may read it: contiguous and 16-byte aligned.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Free Block State`** (1 nodes): `One process per node. World 1 needs no process group at all.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Block Reservation`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Boundary Cache Release`** (1 nodes): `Measure one phase, optionally accumulating repeated calls in one span.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Snapshot Cache Release`** (1 nodes): `Blocks no row holds: free now, counting the ones a boundary would give up.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Phase Measurement`** (1 nodes): `Free blocks nothing remembers: what a reservation spends before any boundary pay`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Death Notes`** (1 nodes): `Free blocks a cached boundary still holds -- the reuse a reservation spends afte`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Distributed Gate Diagnostics`** (1 nodes): `Free blocks held by a boundary whose snapshot is gone (its state lives on the pr`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 30`** (1 nodes): `Never raises. A key that cannot be computed is a full gate, and this rank still`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 31`** (1 nodes): `What the loop is doing now, for a death note.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `BlockPool` connect `Memory Plan` to `Request Cache`, `Device Memory Arena`, `Fleet Lease & Latency`, `Snapshot Publication`?**
  _High betweenness centrality (0.109) - this node is a cross-community bridge._
- **Why does `TierFull` connect `Serving Diagnostics` to `Memory Plan`, `Device Memory Arena`, `Fleet Lease & Latency`, `Stateless Randomness`?**
  _High betweenness centrality (0.081) - this node is a cross-community bridge._
- **Why does `numel()` connect `Stateless Randomness` to `Memory Plan`, `Fleet Lease & Latency`, `Distributed Communication`, `Memory Budget Gates`, `Sampling Constants`, `Composed Model Lifecycle`, `Snapshot Publication`?**
  _High betweenness centrality (0.056) - this node is a cross-community bridge._
- **Are the 136 inferred relationships involving `TierFull` (e.g. with `CompressedSnapshots` and `Model`) actually correct?**
  _`TierFull` has 136 INFERRED edges - model-reasoned connections that need verification._
- **Are the 91 inferred relationships involving `BlockPool` (e.g. with `ComposedModel` and `Drafter`) actually correct?**
  _`BlockPool` has 91 INFERRED edges - model-reasoned connections that need verification._
- **Are the 92 inferred relationships involving `Tripwire` (e.g. with `_Choice` and `_Histogram`) actually correct?**
  _`Tripwire` has 92 INFERRED edges - model-reasoned connections that need verification._
- **Are the 93 inferred relationships involving `StepWatch` (e.g. with `_Choice` and `_Histogram`) actually correct?**
  _`StepWatch` has 93 INFERRED edges - model-reasoned connections that need verification._
