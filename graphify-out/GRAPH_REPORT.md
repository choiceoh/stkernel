# Graph Report - engine/base  (2026-09-19)

## Corpus Check
- 54 files · ~114,408 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1619 nodes · 4517 edges · 36 communities detected
- Extraction: 54% EXTRACTED · 46% INFERRED · 0% AMBIGUOUS · INFERRED: 2066 edges (avg confidence: 0.62)
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
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]

## God Nodes (most connected - your core abstractions)
1. `TierFull` - 151 edges
2. `BlockPool` - 131 edges
3. `Tripwire` - 108 edges
4. `StepWatch` - 106 edges
5. `Recorder` - 104 edges
6. `DiagnosticMetrics` - 103 edges
7. `SlotPool` - 95 edges
8. `PagedSpec` - 62 edges
9. `SlotSpec` - 62 edges
10. `Runner` - 62 edges

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
Nodes (160): DiagnosticMetrics, Exception, The disk cannot take this conversation; forget one or drop this one., TierFull, Recorder, boundary tokens -> hash of the prompt up to there, for every whole BLOCK (45차 §2, The tokens `lookup` would reuse, without counting a query (admission asks before, answer_budget() (+152 more)

### Community 1 - "Serving Diagnostics"
Cohesion: 0.02
Nodes (116): Arena, carve(), Bind the plan to the arena: one KV region, one slot region, two pools., One tree per run. Not thread-safe by design: a step is one thread, and a     loc, Recorder, BlockPool, Block tables and state slots, as flat arrays (base).  Two kinds of per-sequence, Put `block` where its owners say it belongs, at the back of that grade. (+108 more)

### Community 2 - "Device Memory Arena"
Cohesion: 0.03
Nodes (77): _layout(), PagedSpec, Plan, What each layer kind needs cached, declared once, sized from the budget (base)., A spec that names a key also names a dtype and a shape whose bytes are its decla, Split a KV budget: slots first (fixed by max_seqs), blocks with the rest.      B, _selfcheck(), SlotSpec (+69 more)

### Community 3 - "Fleet Lease & Latency"
Cohesion: 0.03
Nodes (83): host_reclaim(), _meminfo(), prepare_allocation(), One allocation for persistent weights and caches (base).  D16 in code: weights (, Ask this node's host to return its clean file cache now, and wait for it to say, Drop clean pages of the supplied weight files, reclaim the rest of the     short, Hold `nbytes` of anonymous memory for an instant, then give it back.      MemAva, Return clean model-download pages without allocating or changing files.      The (+75 more)

### Community 4 - "Distributed Communication"
Cohesion: 0.05
Nodes (82): assign(), Capture, check(), CUDA, profiling(), Name captured graph nodes without adding timing nodes or changing their work.  A, scope(), Attention (+74 more)

### Community 5 - "Prefix Cache & Tenancy"
Cohesion: 0.06
Nodes (64): acquire(), alive(), attach(), clear_yield(), describe(), docker_evidence(), door_load(), door_unsupported() (+56 more)

### Community 6 - "Scheduler & KV Blocks"
Cohesion: 0.04
Nodes (67): forget(), fresh(), iota(), Index constants a captured step reads, and may not create.  A decode step asks f, 0..n-1 on `device`, built once and kept. Never write to what this returns., The same values, not kept: for lengths that follow the request., n zeros on `device`, built once and kept, under the same rule as `iota`: the ind, Drop every kept constant. Tests only -- a live capture recorded their addresses. (+59 more)

### Community 7 - "Memory Budget Gates"
Cohesion: 0.05
Nodes (39): expandable_segments(), Back every new caching-allocator segment with 20 MiB physical chunks     under o, Free the one allocation now, whatever still points at it, and say how many bytes, A uint8 view of `nbytes`, or MemoryError -- never a second allocation (D3)., Region, _selfcheck(), Checkpoint, _LazyDtypes (+31 more)

### Community 8 - "Request Cache"
Cohesion: 0.05
Nodes (55): rank_weights(), (GiB, tensor count) read out of a safetensors header.      The header is a const, _float32(), _lsr(), mix(), mix_tensor(), Stateless uniforms: every draw is a function of what it is for, never of what ca, Logical shift right of an int64 tensor: torch shifts arithmetically, the mask dr (+47 more)

### Community 9 - "Sampling Constants"
Cohesion: 0.04
Nodes (43): cleanup_after_error(), DecodeGraphs, frozen_gc(), Decode steps as captured graphs, one per shape (base). I1's mechanism.  I1 says, fill(inputs) copies this step's data into the static buffers; then replay., Keep the first capture failure; attach secondary teardown failures., No Python garbage collection while a graph is recording.      This is a correctn, step_fn(inputs) runs one decode step over static `inputs`;         make_inputs(n (+35 more)

### Community 10 - "Runtime Introspection"
Cohesion: 0.06
Nodes (24): Comm, fleet_env(), init(), _LocalRank, _LocalRun, LocalTP, pick_gid_index(), RankLeft (+16 more)

### Community 11 - "Composed Model Lifecycle"
Cohesion: 0.07
Nodes (22): available(), for_checkpoint(), Grammars, Matcher, Structured output (base): a grammar per request, a matcher per row, one bitmask, The compile itself. Everything it can fail at is the request's schema, so those, A handle for `spec`: the compiled grammar, or the thread compiling it. Shared by, The compiled grammar, waiting for its thread if it is still running (and raising (+14 more)

### Community 12 - "Stateless Randomness"
Cohesion: 0.07
Nodes (21): changed(), digest_tree(), file_identity(), key_of(), Limits, node_identity(), optional_identity(), PrefillRecord (+13 more)

### Community 13 - "Model & Kernel Shapes"
Cohesion: 0.1
Nodes (30): _arguments_json(), detect(), grammar_arg_pairs(), grammar_function_xml(), _is_text(), parse_arg_pairs(), parse_function_xml(), partial_arg_pairs() (+22 more)

### Community 14 - "Snapshot Publication"
Cohesion: 0.09
Nodes (17): Release NCCL graph references before destroying its process group., field_dtype(), A served model's caches in one arena: paged blocks, per-sequence state slots, pr, Publish changed block mappings before a step, after its reservation.          Se, Upload through a pinned ring; fence reuse after the copy that reads the slot., The `count` ring cells before `position` in a ring `width` long, oldest first, o, What the arena carves for the paged blocks, the state slots (one per request id,, How many prefix snapshots a budget holds, and never fewer than MIN_SNAPSHOTS. (+9 more)

### Community 15 - "Stage Timing"
Cohesion: 0.12
Nodes (18): check_box(), The box every profile is written for, and its assertion (base).  The charter's D, The node the engine is written for, asserted (D3): one GB10, unified memory., Config, ConfigError, Fact, Knob, Facts in, knobs that expire, nothing else (base).  D11's evidence was 133 `VLLM_ (+10 more)

### Community 16 - "Instrumentation"
Cohesion: 0.1
Nodes (11): Budget, host_box(), Line, probe_box(), rank_files(), The box, declared -- not discovered (framework).  A budget is a list of claims o, (total device GiB, free GiB after our own context).      Asking costs a CUDA con, (MemTotal GiB, MemAvailable GiB) -- what earlyoom actually counts. (+3 more)

### Community 17 - "KV Cache Sizing"
Cohesion: 0.22
Nodes (8): Which window of a boot compiled something -- one walk of the JIT caches, after t, One log line. It says "no JIT writes" out loud, because that is the answer that, The declared cache directories that exist, with nested ones folded into their pa, Named wall-clock windows over a boot, and the cache artifacts that landed inside, Walk once; count the artifacts newer than the first window, by the window they f, roots(), _selfcheck(), Windows

### Community 18 - "Common Kernel Lanes"
Cohesion: 0.24
Nodes (7): _dev_free_bytes(), phase(), process_seconds(), A phase that was timed somewhere this recorder could not reach -- the boot's `fr, Seconds since THIS PROCESS started, or None where /proc does not say.      A rec, Free device memory, or None before CUDA is up (see the module note)., Span

### Community 19 - "Stall Watchdog"
Cohesion: 0.25
Nodes (4): Builder, Bounded, lossless RAM copies of snapshots whose durable copy is on NVMe.  Only t, Decode through shared staging; `write(offset, n)` must fence its read., Snapshot

### Community 20 - "Process Topology"
Cohesion: 0.25
Nodes (3): Background, Host work a boot runs where it is already waiting (base): a thread started early, Host work started where the boot is already waiting, and joined where its result

### Community 21 - "Tensor Layout"
Cohesion: 0.53
Nodes (5): Gate, judge(), Bands and repeat counts, or it is not a gate (base).  D14 accepted that quiet qu, _selfcheck(), Verdict

### Community 22 - "Package Init"
Cohesion: 0.33
Nodes (3): build_identity(), context_band(), Bounded host-side serving diagnostics; never synchronizes a device.  Completion

### Community 23 - "Free Block State"
Cohesion: 0.33
Nodes (4): Cache, max_seq(), What a model caches per sequence and per token, and what a budget buys (framewor, Longest context that fits, at this concurrency.

### Community 24 - "Block Reservation"
Cohesion: 0.4
Nodes (1): A step that never ends, bounded (45차, 2026-09-12 22:31).  The fleet's worst fail

### Community 25 - "Boundary Cache Release"
Cohesion: 0.5
Nodes (4): CommonLanes, The engine's default lanes (base): the model-free kernels every profile inherits, The common kernels, bound once. Importing them imports Triton. The norms and Swi, served()

### Community 26 - "Snapshot Cache Release"
Cohesion: 1.0
Nodes (1): The tensor as a lane decision may read it: contiguous and 16-byte aligned.

### Community 27 - "Phase Measurement"
Cohesion: 1.0
Nodes (1): One process per node. World 1 needs no process group at all.

### Community 28 - "Death Notes"
Cohesion: 1.0
Nodes (0):

### Community 29 - "Distributed Gate Diagnostics"
Cohesion: 1.0
Nodes (1): Measure one phase, optionally accumulating repeated calls in one span.

### Community 30 - "Community 30"
Cohesion: 1.0
Nodes (1): Blocks no row holds: free now, counting the ones a boundary would give up.

### Community 31 - "Community 31"
Cohesion: 1.0
Nodes (1): Free blocks nothing remembers: what a reservation spends before any boundary pay

### Community 32 - "Community 32"
Cohesion: 1.0
Nodes (1): Free blocks a cached boundary still holds -- the reuse a reservation spends afte

### Community 33 - "Community 33"
Cohesion: 1.0
Nodes (1): Free blocks held by a boundary whose snapshot is gone (its state lives on the pr

### Community 34 - "Community 34"
Cohesion: 1.0
Nodes (1): Never raises. A key that cannot be computed is a full gate, and this rank still

### Community 35 - "Community 35"
Cohesion: 1.0
Nodes (1): What the loop is doing now, for a death note.

## Knowledge Gaps
- **342 isolated node(s):** `Host work a boot runs where it is already waiting (base): a thread started early`, `Host work started where the boot is already waiting, and joined where its result`, `The box every profile is written for, and its assertion (base).  The charter's D`, `The node the engine is written for, asserted (D3): one GB10, unified memory.`, `The box, declared -- not discovered (framework).  A budget is a list of claims o` (+337 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Snapshot Cache Release`** (1 nodes): `The tensor as a lane decision may read it: contiguous and 16-byte aligned.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Phase Measurement`** (1 nodes): `One process per node. World 1 needs no process group at all.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Death Notes`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Distributed Gate Diagnostics`** (1 nodes): `Measure one phase, optionally accumulating repeated calls in one span.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 30`** (1 nodes): `Blocks no row holds: free now, counting the ones a boundary would give up.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 31`** (1 nodes): `Free blocks nothing remembers: what a reservation spends before any boundary pay`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 32`** (1 nodes): `Free blocks a cached boundary still holds -- the reuse a reservation spends afte`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 33`** (1 nodes): `Free blocks held by a boundary whose snapshot is gone (its state lives on the pr`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 34`** (1 nodes): `Never raises. A key that cannot be computed is a full gate, and this rank still`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 35`** (1 nodes): `What the loop is doing now, for a death note.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `BlockPool` connect `Serving Diagnostics` to `Sampling Constants`, `Device Memory Arena`, `Fleet Lease & Latency`?**
  _High betweenness centrality (0.100) - this node is a cross-community bridge._
- **Why does `TierFull` connect `Memory Plan` to `Serving Diagnostics`, `Fleet Lease & Latency`?**
  _High betweenness centrality (0.070) - this node is a cross-community bridge._
- **Why does `numel()` connect `Fleet Lease & Latency` to `Memory Plan`, `Serving Diagnostics`, `Device Memory Arena`, `Scheduler & KV Blocks`, `Memory Budget Gates`, `Request Cache`, `Runtime Introspection`?**
  _High betweenness centrality (0.060) - this node is a cross-community bridge._
- **Are the 147 inferred relationships involving `TierFull` (e.g. with `CompressedSnapshots` and `Model`) actually correct?**
  _`TierFull` has 147 INFERRED edges - model-reasoned connections that need verification._
- **Are the 106 inferred relationships involving `BlockPool` (e.g. with `ComposedModel` and `Drafter`) actually correct?**
  _`BlockPool` has 106 INFERRED edges - model-reasoned connections that need verification._
- **Are the 95 inferred relationships involving `Tripwire` (e.g. with `_Choice` and `_Histogram`) actually correct?**
  _`Tripwire` has 95 INFERRED edges - model-reasoned connections that need verification._
- **Are the 96 inferred relationships involving `StepWatch` (e.g. with `_Choice` and `_Histogram`) actually correct?**
  _`StepWatch` has 96 INFERRED edges - model-reasoned connections that need verification._
