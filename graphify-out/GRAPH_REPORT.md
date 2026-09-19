# Graph Report - engine/base  (2026-09-19)

## Corpus Check
- 54 files · ~107,202 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1564 nodes · 4272 edges · 29 communities detected
- Extraction: 55% EXTRACTED · 45% INFERRED · 0% AMBIGUOUS · INFERRED: 1906 edges (avg confidence: 0.63)
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
- `Off-thread I/O. Poll `.done()`, then `.result()` to surface failures.          T` --uses--> `CompressedSnapshots`  [INFERRED]
  engine/base/kv_tier.py → engine/base/compressed_snapshots.py
- `The disk cannot take this conversation; forget one or drop this one.` --uses--> `CompressedSnapshots`  [INFERRED]
  engine/base/kv_tier.py → engine/base/compressed_snapshots.py
- `step()` --calls--> `profiling()`  [INFERRED]
  engine/base/latency.py → engine/base/graph_labels.py
- `Sampling policy and committed token counts carried alongside device decode rows.` --uses--> `History`  [INFERRED]
  engine/base/sampling_options.py → engine/base/sampler.py

## Communities

### Community 0 - "Memory Plan"
Cohesion: 0.03
Nodes (142): carve(), _layout(), PagedSpec, Plan, What each layer kind needs cached, declared once, sized from the budget (base)., Bind the plan to the arena: one KV region, one slot region, two pools., A spec that names a key also names a dtype and a shape whose bytes are its decla, Split a KV budget: slots first (fixed by max_seqs), blocks with the rest.      B (+134 more)

### Community 1 - "Serving Diagnostics"
Cohesion: 0.02
Nodes (58): Builder, CompressedSnapshots, Bounded, lossless RAM copies of snapshots whose durable copy is on NVMe.  Only t, Decode through shared staging; `write(offset, n)` must fence its read., Snapshot, Drop that owner; blocks nobody else holds go back to the free list. Returns how, NvmeTier, The NVMe tier for cold KV (base). Contiguous per sequence, O_DIRECT, survives bo (+50 more)

### Community 2 - "Device Memory Arena"
Cohesion: 0.05
Nodes (138): DiagnosticMetrics, Exception, The disk cannot take this conversation; forget one or drop this one., TierFull, Recorder, answer_budget(), _byte_level_chars(), cache_key() (+130 more)

### Community 3 - "Fleet Lease & Latency"
Cohesion: 0.02
Nodes (94): expandable_segments(), host_reclaim(), _meminfo(), prepare_allocation(), One allocation for persistent weights and caches (base).  D16 in code: weights (, Ask this node's host to return its clean file cache now, and wait for it to say, Drop clean pages of the supplied weight files, reclaim the rest of the     short, Back every new caching-allocator segment with 20 MiB physical chunks     under o (+86 more)

### Community 4 - "Distributed Communication"
Cohesion: 0.04
Nodes (41): Atomically cover absolute write ends, reusing rejected draft space., Entry, is_tenant_salt(), Prefix reuse: a prompt that begins the way an earlier one did is not prefilled a, The pool this cache claims blocks in; the pool tells it when a claimed block has, Who gives up its snapshot when one is needed: among the boundaries nobody adopte, A free snapshot slot, taking one from a boundary if none is free (`_victim`). Th, A slot whose contents belong to nobody (an aborted checkpoint, a boundary that i (+33 more)

### Community 5 - "Prefix Cache & Tenancy"
Cohesion: 0.04
Nodes (62): rank_weights(), (GiB, tensor count) read out of a safetensors header.      The header is a const, _float32(), _lsr(), mix(), mix_tensor(), Stateless uniforms: every draw is a function of what it is for, never of what ca, Logical shift right of an int64 tensor: torch shifts arithmetically, the mask dr (+54 more)

### Community 6 - "Scheduler & KV Blocks"
Cohesion: 0.05
Nodes (24): ComposedModel, available(), for_checkpoint(), Grammars, Matcher, Structured output (base): a grammar per request, a matcher per row, one bitmask, The compile itself. Everything it can fail at is the request's schema, so those, A handle for `spec`: the compiled grammar, or the thread compiling it. Shared by (+16 more)

### Community 7 - "Memory Budget Gates"
Cohesion: 0.03
Nodes (45): Background, Host work a boot runs where it is already waiting (base): a thread started early, Host work started where the boot is already waiting, and joined where its result, Budget, host_box(), Line, probe_box(), rank_files() (+37 more)

### Community 8 - "Request Cache"
Cohesion: 0.04
Nodes (35): Arena, Free the one allocation now, whatever still points at it, and say how many bytes, Release NCCL graph references before destroying its process group., Grow `seq` to hold `tokens` more. Returns blocks newly taken., Give every block of `seq` back, TAIL FIRST -- the head of a prompt is what the n, Off-thread I/O. Poll `.done()`, then `.result()` to surface failures.          T, The blocks of a boundary memory still holds, entry or faded: where a tier read m, No cached boundary one block longer extends this one. Kept as a count at insert/ (+27 more)

### Community 9 - "Sampling Constants"
Cohesion: 0.05
Nodes (33): Comm, fleet_env(), init(), _LocalRank, _LocalRun, LocalTP, pick_gid_index(), RankLeft (+25 more)

### Community 10 - "Runtime Introspection"
Cohesion: 0.04
Nodes (65): forget(), fresh(), iota(), Index constants a captured step reads, and may not create.  A decode step asks f, 0..n-1 on `device`, built once and kept. Never write to what this returns., The same values, not kept: for lengths that follow the request., n zeros on `device`, built once and kept, under the same rule as `iota`: the ind, Drop every kept constant. Tests only -- a live capture recorded their addresses. (+57 more)

### Community 11 - "Composed Model Lifecycle"
Cohesion: 0.06
Nodes (59): Attention, bind(), bind_recorded(), bound(), claims(), Comm, config_sha256(), derive_for() (+51 more)

### Community 12 - "Stateless Randomness"
Cohesion: 0.09
Nodes (49): acquire(), alive(), attach(), clear_yield(), describe(), door_load(), door_unsupported(), _here() (+41 more)

### Community 13 - "Model & Kernel Shapes"
Cohesion: 0.1
Nodes (30): _arguments_json(), detect(), grammar_arg_pairs(), grammar_function_xml(), _is_text(), parse_arg_pairs(), parse_function_xml(), partial_arg_pairs() (+22 more)

### Community 14 - "Snapshot Publication"
Cohesion: 0.07
Nodes (21): changed(), digest_tree(), file_identity(), key_of(), Limits, node_identity(), optional_identity(), PrefillRecord (+13 more)

### Community 15 - "Stage Timing"
Cohesion: 0.08
Nodes (17): cleanup_after_error(), DecodeGraphs, frozen_gc(), Decode steps as captured graphs, one per shape (base). I1's mechanism.  I1 says, fill(inputs) copies this step's data into the static buffers; then replay., Keep the first capture failure; attach secondary teardown failures., No Python garbage collection while a graph is recording.      This is a correctn, step_fn(inputs) runs one decode step over static `inputs`;         make_inputs(n (+9 more)

### Community 16 - "Instrumentation"
Cohesion: 0.33
Nodes (4): Cache, max_seq(), What a model caches per sequence and per token, and what a budget buys (framewor, Longest context that fits, at this concurrency.

### Community 17 - "KV Cache Sizing"
Cohesion: 0.5
Nodes (4): CommonLanes, The engine's default lanes (base): the model-free kernels every profile inherits, The common kernels, bound once. Importing them imports Triton. The norms and Swi, served()

### Community 18 - "Common Kernel Lanes"
Cohesion: 0.4
Nodes (1): A step that never ends, bounded (45차, 2026-09-12 22:31).  The fleet's worst fail

### Community 19 - "Stall Watchdog"
Cohesion: 1.0
Nodes (1): The tensor as a lane decision may read it: contiguous and 16-byte aligned.

### Community 20 - "Process Topology"
Cohesion: 1.0
Nodes (1): One process per node. World 1 needs no process group at all.

### Community 21 - "Tensor Layout"
Cohesion: 1.0
Nodes (0):

### Community 22 - "Package Init"
Cohesion: 1.0
Nodes (1): Measure one phase, optionally accumulating repeated calls in one span.

### Community 23 - "Free Block State"
Cohesion: 1.0
Nodes (1): Blocks no row holds: free now, counting the ones a boundary would give up.

### Community 24 - "Block Reservation"
Cohesion: 1.0
Nodes (1): Free blocks nothing remembers: what a reservation spends before any boundary pay

### Community 25 - "Boundary Cache Release"
Cohesion: 1.0
Nodes (1): Free blocks a cached boundary still holds -- the reuse a reservation spends afte

### Community 26 - "Snapshot Cache Release"
Cohesion: 1.0
Nodes (1): Free blocks held by a boundary whose snapshot is gone (its state lives on the pr

### Community 27 - "Phase Measurement"
Cohesion: 1.0
Nodes (1): Never raises. A key that cannot be computed is a full gate, and this rank still

### Community 28 - "Death Notes"
Cohesion: 1.0
Nodes (1): What the loop is doing now, for a death note.

## Knowledge Gaps
- **337 isolated node(s):** `Host work a boot runs where it is already waiting (base): a thread started early`, `Host work started where the boot is already waiting, and joined where its result`, `The box every profile is written for, and its assertion (base).  The charter's D`, `The node the engine is written for, asserted (D3): one GB10, unified memory.`, `The box, declared -- not discovered (framework).  A budget is a list of claims o` (+332 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Stall Watchdog`** (1 nodes): `The tensor as a lane decision may read it: contiguous and 16-byte aligned.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Process Topology`** (1 nodes): `One process per node. World 1 needs no process group at all.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Tensor Layout`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Package Init`** (1 nodes): `Measure one phase, optionally accumulating repeated calls in one span.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Free Block State`** (1 nodes): `Blocks no row holds: free now, counting the ones a boundary would give up.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Block Reservation`** (1 nodes): `Free blocks nothing remembers: what a reservation spends before any boundary pay`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Boundary Cache Release`** (1 nodes): `Free blocks a cached boundary still holds -- the reuse a reservation spends afte`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Snapshot Cache Release`** (1 nodes): `Free blocks held by a boundary whose snapshot is gone (its state lives on the pr`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Phase Measurement`** (1 nodes): `Never raises. A key that cannot be computed is a full gate, and this rank still`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Death Notes`** (1 nodes): `What the loop is doing now, for a death note.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `BlockPool` connect `Memory Plan` to `Request Cache`, `Serving Diagnostics`, `Distributed Communication`, `Scheduler & KV Blocks`?**
  _High betweenness centrality (0.097) - this node is a cross-community bridge._
- **Why does `TierFull` connect `Device Memory Arena` to `Memory Plan`, `Serving Diagnostics`, `Distributed Communication`?**
  _High betweenness centrality (0.076) - this node is a cross-community bridge._
- **Why does `numel()` connect `Serving Diagnostics` to `Memory Plan`, `Fleet Lease & Latency`, `Prefix Cache & Tenancy`, `Sampling Constants`, `Runtime Introspection`?**
  _High betweenness centrality (0.064) - this node is a cross-community bridge._
- **Are the 136 inferred relationships involving `TierFull` (e.g. with `CompressedSnapshots` and `Model`) actually correct?**
  _`TierFull` has 136 INFERRED edges - model-reasoned connections that need verification._
- **Are the 91 inferred relationships involving `BlockPool` (e.g. with `ComposedModel` and `Drafter`) actually correct?**
  _`BlockPool` has 91 INFERRED edges - model-reasoned connections that need verification._
- **Are the 92 inferred relationships involving `Tripwire` (e.g. with `_Choice` and `_Histogram`) actually correct?**
  _`Tripwire` has 92 INFERRED edges - model-reasoned connections that need verification._
- **Are the 93 inferred relationships involving `StepWatch` (e.g. with `_Choice` and `_Histogram`) actually correct?**
  _`StepWatch` has 93 INFERRED edges - model-reasoned connections that need verification._
