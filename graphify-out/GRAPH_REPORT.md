# Graph Report - engine/base  (2026-09-19)

## Corpus Check
- 52 files · ~106,112 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1538 nodes · 4239 edges · 30 communities detected
- Extraction: 55% EXTRACTED · 45% INFERRED · 0% AMBIGUOUS · INFERRED: 1900 edges (avg confidence: 0.63)
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
- `One allocation for persistent weights and caches (base).  D16 in code: weights (` --uses--> `RankLoader`  [INFERRED]
  engine/base/arena.py → engine/base/loader.py
- `Ask this node's host to return its clean file cache now, and wait for it to say` --uses--> `RankLoader`  [INFERRED]
  engine/base/arena.py → engine/base/loader.py
- `Drop clean pages of the supplied weight files, reclaim the rest of the     short` --uses--> `RankLoader`  [INFERRED]
  engine/base/arena.py → engine/base/loader.py
- `Hold `nbytes` of anonymous memory for an instant, then give it back.      MemAva` --uses--> `RankLoader`  [INFERRED]
  engine/base/arena.py → engine/base/loader.py
- `Return clean model-download pages without allocating or changing files.      The` --uses--> `RankLoader`  [INFERRED]
  engine/base/arena.py → engine/base/loader.py

## Communities

### Community 0 - "Memory Plan"
Cohesion: 0.03
Nodes (142): carve(), _layout(), PagedSpec, Plan, What each layer kind needs cached, declared once, sized from the budget (base)., Bind the plan to the arena: one KV region, one slot region, two pools., A spec that names a key also names a dtype and a shape whose bytes are its decla, Split a KV budget: slots first (fixed by max_seqs), blocks with the rest.      B (+134 more)

### Community 1 - "Serving Diagnostics"
Cohesion: 0.03
Nodes (140): DiagnosticMetrics, Exception, The disk cannot take this conversation; forget one or drop this one., TierFull, Recorder, answer_budget(), _byte_level_chars(), cache_key() (+132 more)

### Community 2 - "Device Memory Arena"
Cohesion: 0.03
Nodes (52): Arena, Grow `seq` to hold `tokens` more. Returns blocks newly taken., Atomically cover absolute write ends, reusing rejected draft space., Give every block of `seq` back, TAIL FIRST -- the head of a prompt is what the n, Off-thread I/O. Poll `.done()`, then `.result()` to surface failures.          T, Entry, is_tenant_salt(), PrefixCache (+44 more)

### Community 3 - "Fleet Lease & Latency"
Cohesion: 0.03
Nodes (65): host_reclaim(), _meminfo(), prepare_allocation(), One allocation for persistent weights and caches (base).  D16 in code: weights (, Ask this node's host to return its clean file cache now, and wait for it to say, Drop clean pages of the supplied weight files, reclaim the rest of the     short, Hold `nbytes` of anonymous memory for an instant, then give it back.      MemAva, Return clean model-download pages without allocating or changing files.      The (+57 more)

### Community 4 - "Distributed Communication"
Cohesion: 0.04
Nodes (72): Background, Host work a boot runs where it is already waiting (base): a thread started early, Host work started where the boot is already waiting, and joined where its result, Budget, host_box(), Line, probe_box(), rank_files() (+64 more)

### Community 5 - "Prefix Cache & Tenancy"
Cohesion: 0.04
Nodes (54): expandable_segments(), Back every new caching-allocator segment with 20 MiB physical chunks     under o, Free the one allocation now, whatever still points at it, and say how many bytes, A uint8 view of `nbytes`, or MemoryError -- never a second allocation (D3)., Region, Checkpoint, _LazyDtypes, Read exactly these tensors as coalesced ranges, one shard at a time.          Ea (+46 more)

### Community 6 - "Scheduler & KV Blocks"
Cohesion: 0.05
Nodes (24): ComposedModel, available(), for_checkpoint(), Grammars, Matcher, Structured output (base): a grammar per request, a matcher per row, one bitmask, The compile itself. Everything it can fail at is the request's schema, so those, A handle for `spec`: the compiled grammar, or the thread compiling it. Shared by (+16 more)

### Community 7 - "Memory Budget Gates"
Cohesion: 0.05
Nodes (33): Comm, fleet_env(), init(), _LocalRank, _LocalRun, LocalTP, pick_gid_index(), RankLeft (+25 more)

### Community 8 - "Request Cache"
Cohesion: 0.04
Nodes (67): forget(), fresh(), iota(), Index constants a captured step reads, and may not create.  A decode step asks f, 0..n-1 on `device`, built once and kept. Never write to what this returns., The same values, not kept: for lengths that follow the request., n zeros on `device`, built once and kept, under the same rule as `iota`: the ind, Drop every kept constant. Tests only -- a live capture recorded their addresses. (+59 more)

### Community 9 - "Sampling Constants"
Cohesion: 0.06
Nodes (59): Attention, bind(), bind_recorded(), bound(), claims(), Comm, config_sha256(), derive_for() (+51 more)

### Community 10 - "Runtime Introspection"
Cohesion: 0.05
Nodes (50): _float32(), _lsr(), mix(), mix_tensor(), Stateless uniforms: every draw is a function of what it is for, never of what ca, Logical shift right of an int64 tensor: torch shifts arithmetically, the mask dr, `mix` over an int64 tensor, bit for bit., `row_key` for every row: `nonces` and `generations` are int64 tensors [n]. The s (+42 more)

### Community 11 - "Composed Model Lifecycle"
Cohesion: 0.05
Nodes (32): Builder, Bounded, lossless RAM copies of snapshots whose durable copy is on NVMe.  Only t, Decode through shared staging; `write(offset, n)` must fence its read., Snapshot, docker_evidence(), Is that container up HERE? None when docker cannot answer -- the caller keeps, _dev_free_bytes(), phase() (+24 more)

### Community 12 - "Stateless Randomness"
Cohesion: 0.09
Nodes (32): _arguments_json(), detect(), grammar_arg_pairs(), grammar_function_xml(), _is_text(), parse_arg_pairs(), parse_function_xml(), partial_arg_pairs() (+24 more)

### Community 13 - "Model & Kernel Shapes"
Cohesion: 0.07
Nodes (21): changed(), digest_tree(), file_identity(), key_of(), Limits, node_identity(), optional_identity(), PrefillRecord (+13 more)

### Community 14 - "Snapshot Publication"
Cohesion: 0.09
Nodes (26): The pool this cache claims blocks in; the pool tells it when a claimed block has, advance(), arrive(), Contract, _decode(), finish(), plan(), _prefill() (+18 more)

### Community 15 - "Stage Timing"
Cohesion: 0.08
Nodes (17): cleanup_after_error(), DecodeGraphs, frozen_gc(), Decode steps as captured graphs, one per shape (base). I1's mechanism.  I1 says, fill(inputs) copies this step's data into the static buffers; then replay., Keep the first capture failure; attach secondary teardown failures., No Python garbage collection while a graph is recording.      This is a correctn, step_fn(inputs) runs one decode step over static `inputs`;         make_inputs(n (+9 more)

### Community 16 - "Instrumentation"
Cohesion: 0.13
Nodes (11): build_identity(), context_band(), Bounded host-side serving diagnostics; never synchronizes a device.  Completion, Which window of a boot compiled something -- one walk of the JIT caches, after t, One log line. It says "no JIT writes" out loud, because that is the answer that, The declared cache directories that exist, with nested ones folded into their pa, Named wall-clock windows over a boot, and the cache artifacts that landed inside, Walk once; count the artifacts newer than the first window, by the window they f (+3 more)

### Community 17 - "KV Cache Sizing"
Cohesion: 0.33
Nodes (4): Cache, max_seq(), What a model caches per sequence and per token, and what a budget buys (framewor, Longest context that fits, at this concurrency.

### Community 18 - "Common Kernel Lanes"
Cohesion: 0.5
Nodes (4): CommonLanes, The engine's default lanes (base): the model-free kernels every profile inherits, The common kernels, bound once. Importing them imports Triton. The norms and Swi, served()

### Community 19 - "Stall Watchdog"
Cohesion: 0.4
Nodes (1): A step that never ends, bounded (45차, 2026-09-12 22:31).  The fleet's worst fail

### Community 20 - "Process Topology"
Cohesion: 1.0
Nodes (1): The tensor as a lane decision may read it: contiguous and 16-byte aligned.

### Community 21 - "Tensor Layout"
Cohesion: 1.0
Nodes (1): One process per node. World 1 needs no process group at all.

### Community 22 - "Package Init"
Cohesion: 1.0
Nodes (0):

### Community 23 - "Free Block State"
Cohesion: 1.0
Nodes (1): Measure one phase, optionally accumulating repeated calls in one span.

### Community 24 - "Block Reservation"
Cohesion: 1.0
Nodes (1): Blocks no row holds: free now, counting the ones a boundary would give up.

### Community 25 - "Boundary Cache Release"
Cohesion: 1.0
Nodes (1): Free blocks nothing remembers: what a reservation spends before any boundary pay

### Community 26 - "Snapshot Cache Release"
Cohesion: 1.0
Nodes (1): Free blocks a cached boundary still holds -- the reuse a reservation spends afte

### Community 27 - "Phase Measurement"
Cohesion: 1.0
Nodes (1): Free blocks held by a boundary whose snapshot is gone (its state lives on the pr

### Community 28 - "Death Notes"
Cohesion: 1.0
Nodes (1): Never raises. A key that cannot be computed is a full gate, and this rank still

### Community 29 - "Distributed Gate Diagnostics"
Cohesion: 1.0
Nodes (1): What the loop is doing now, for a death note.

## Knowledge Gaps
- **325 isolated node(s):** `Host work a boot runs where it is already waiting (base): a thread started early`, `Host work started where the boot is already waiting, and joined where its result`, `The box, declared -- not discovered (framework).  A budget is a list of claims o`, `(total device GiB, free GiB after our own context).      Asking costs a CUDA con`, `(MemTotal GiB, MemAvailable GiB) -- what earlyoom actually counts.` (+320 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Process Topology`** (1 nodes): `The tensor as a lane decision may read it: contiguous and 16-byte aligned.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Tensor Layout`** (1 nodes): `One process per node. World 1 needs no process group at all.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Package Init`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Free Block State`** (1 nodes): `Measure one phase, optionally accumulating repeated calls in one span.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Block Reservation`** (1 nodes): `Blocks no row holds: free now, counting the ones a boundary would give up.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Boundary Cache Release`** (1 nodes): `Free blocks nothing remembers: what a reservation spends before any boundary pay`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Snapshot Cache Release`** (1 nodes): `Free blocks a cached boundary still holds -- the reuse a reservation spends afte`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Phase Measurement`** (1 nodes): `Free blocks held by a boundary whose snapshot is gone (its state lives on the pr`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Death Notes`** (1 nodes): `Never raises. A key that cannot be computed is a full gate, and this rank still`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Distributed Gate Diagnostics`** (1 nodes): `What the loop is doing now, for a death note.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `BlockPool` connect `Memory Plan` to `Device Memory Arena`, `Fleet Lease & Latency`, `Scheduler & KV Blocks`, `Snapshot Publication`?**
  _High betweenness centrality (0.131) - this node is a cross-community bridge._
- **Why does `TierFull` connect `Serving Diagnostics` to `Memory Plan`, `Device Memory Arena`, `Fleet Lease & Latency`?**
  _High betweenness centrality (0.076) - this node is a cross-community bridge._
- **Why does `numel()` connect `Fleet Lease & Latency` to `Memory Plan`, `Device Memory Arena`, `Prefix Cache & Tenancy`, `Memory Budget Gates`, `Request Cache`, `Runtime Introspection`?**
  _High betweenness centrality (0.056) - this node is a cross-community bridge._
- **Are the 136 inferred relationships involving `TierFull` (e.g. with `CompressedSnapshots` and `Model`) actually correct?**
  _`TierFull` has 136 INFERRED edges - model-reasoned connections that need verification._
- **Are the 91 inferred relationships involving `BlockPool` (e.g. with `ComposedModel` and `Drafter`) actually correct?**
  _`BlockPool` has 91 INFERRED edges - model-reasoned connections that need verification._
- **Are the 92 inferred relationships involving `Tripwire` (e.g. with `_Choice` and `_Histogram`) actually correct?**
  _`Tripwire` has 92 INFERRED edges - model-reasoned connections that need verification._
- **Are the 93 inferred relationships involving `StepWatch` (e.g. with `_Choice` and `_Histogram`) actually correct?**
  _`StepWatch` has 93 INFERRED edges - model-reasoned connections that need verification._
