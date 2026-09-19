# Graph Report - engine/base  (2026-09-19)

## Corpus Check
- 54 files · ~113,449 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1613 nodes · 4492 edges · 33 communities detected
- Extraction: 54% EXTRACTED · 46% INFERRED · 0% AMBIGUOUS · INFERRED: 2049 edges (avg confidence: 0.62)
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

## God Nodes (most connected - your core abstractions)
1. `TierFull` - 151 edges
2. `BlockPool` - 128 edges
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
Nodes (147): DiagnosticMetrics, Exception, The disk cannot take this conversation; forget one or drop this one., TierFull, Recorder, answer_budget(), _byte_level_chars(), cache_key() (+139 more)

### Community 1 - "Serving Diagnostics"
Cohesion: 0.03
Nodes (138): carve(), _layout(), PagedSpec, Plan, What each layer kind needs cached, declared once, sized from the budget (base)., Bind the plan to the arena: one KV region, one slot region, two pools., A spec that names a key also names a dtype and a shape whose bytes are its decla, Split a KV budget: slots first (fixed by max_seqs), blocks with the rest.      B (+130 more)

### Community 2 - "Device Memory Arena"
Cohesion: 0.03
Nodes (52): Arena, Every free block in the order it would be handed out -- anonymous first, an oper, A sequence's blocks in order, as views (what a tier demotes)., Read-only block ids; only reserve/release may change the mapping., Grow `seq` to hold `tokens` more. Returns blocks newly taken., Atomically cover absolute write ends, reusing rejected draft space., Give every block of `seq` back, TAIL FIRST -- the head of a prompt is what the n, _selfcheck() (+44 more)

### Community 3 - "Fleet Lease & Latency"
Cohesion: 0.03
Nodes (69): host_reclaim(), _meminfo(), prepare_allocation(), One allocation for persistent weights and caches (base).  D16 in code: weights (, Ask this node's host to return its clean file cache now, and wait for it to say, Drop clean pages of the supplied weight files, reclaim the rest of the     short, Hold `nbytes` of anonymous memory for an instant, then give it back.      MemAva, Return clean model-download pages without allocating or changing files.      The (+61 more)

### Community 4 - "Distributed Communication"
Cohesion: 0.06
Nodes (74): Attention, bind(), bind_drafter(), bind_recorded(), bound(), claims(), Comm, config_sha256() (+66 more)

### Community 5 - "Prefix Cache & Tenancy"
Cohesion: 0.04
Nodes (63): host_box(), Line, probe_box(), rank_files(), rank_weights(), The box, declared -- not discovered (framework).  A budget is a list of claims o, (total device GiB, free GiB after our own context).      Asking costs a CUDA con, (MemTotal GiB, MemAvailable GiB) -- what earlyoom actually counts. (+55 more)

### Community 6 - "Scheduler & KV Blocks"
Cohesion: 0.04
Nodes (41): Put `block` where its owners say it belongs, at the back of that grade., A boundary remembers these blocks: they stay where they are, behind everything o, That boundary is gone (or changed grade): the blocks fall back to what is left h, Seat a cached prefix -- `tokens` whole blocks that are complete and shared -- at, One more owner for each block, for an owner that is not a row: bytes a transfer, Drop that owner; blocks nobody else holds go back to the free list. Returns how, `block` joins the back of `grade`; the front is what leaves first., Entry (+33 more)

### Community 7 - "Memory Budget Gates"
Cohesion: 0.06
Nodes (60): Background, Host work a boot runs where it is already waiting (base): a thread started early, Host work started where the boot is already waiting, and joined where its result, Budget, A box, the claims on it, and whatever is left for KV., A budget can gate a boot only when nothing in it is a guess., Gate, judge() (+52 more)

### Community 8 - "Request Cache"
Cohesion: 0.05
Nodes (40): expandable_segments(), Back every new caching-allocator segment with 20 MiB physical chunks     under o, Free the one allocation now, whatever still points at it, and say how many bytes, A uint8 view of `nbytes`, or MemoryError -- never a second allocation (D3)., Region, _selfcheck(), Checkpoint, _LazyDtypes (+32 more)

### Community 9 - "Sampling Constants"
Cohesion: 0.06
Nodes (27): Comm, fleet_env(), init(), _LocalRank, _LocalRun, LocalTP, pick_gid_index(), RankLeft (+19 more)

### Community 10 - "Runtime Introspection"
Cohesion: 0.04
Nodes (42): cleanup_after_error(), DecodeGraphs, frozen_gc(), Decode steps as captured graphs, one per shape (base). I1's mechanism.  I1 says, fill(inputs) copies this step's data into the static buffers; then replay., Keep the first capture failure; attach secondary teardown failures., No Python garbage collection while a graph is recording.      This is a correctn, step_fn(inputs) runs one decode step over static `inputs`;         make_inputs(n (+34 more)

### Community 11 - "Composed Model Lifecycle"
Cohesion: 0.05
Nodes (56): forget(), fresh(), iota(), Index constants a captured step reads, and may not create.  A decode step asks f, 0..n-1 on `device`, built once and kept. Never write to what this returns., The same values, not kept: for lengths that follow the request., n zeros on `device`, built once and kept, under the same rule as `iota`: the ind, Drop every kept constant. Tests only -- a live capture recorded their addresses. (+48 more)

### Community 12 - "Stateless Randomness"
Cohesion: 0.07
Nodes (14): ComposedModel, process_logits(), One row's raw logits [V] -> the logits the pick is made from: logit_bias, repeti, from_histories(), needs_device_policy(), Sampling policy and committed token counts carried alongside device decode rows., Only the clipped, accepted output updates history. Rejected drafts never do., Compact device results; host publication waits for the ordinary outcome event. (+6 more)

### Community 13 - "Model & Kernel Shapes"
Cohesion: 0.04
Nodes (31): Builder, Bounded, lossless RAM copies of snapshots whose durable copy is on NVMe.  Only t, Decode through shared staging; `write(offset, n)` must fence its read., Snapshot, docker_evidence(), Is that container up HERE? None when docker cannot answer -- the caller keeps, _dev_free_bytes(), phase() (+23 more)

### Community 14 - "Snapshot Publication"
Cohesion: 0.07
Nodes (22): available(), for_checkpoint(), Grammars, Matcher, Structured output (base): a grammar per request, a matcher per row, one bitmask, The compile itself. Everything it can fail at is the request's schema, so those, A handle for `spec`: the compiled grammar, or the thread compiling it. Shared by, The compiled grammar, waiting for its thread if it is still running (and raising (+14 more)

### Community 15 - "Stage Timing"
Cohesion: 0.09
Nodes (32): _arguments_json(), detect(), grammar_arg_pairs(), grammar_function_xml(), _is_text(), parse_arg_pairs(), parse_function_xml(), partial_arg_pairs() (+24 more)

### Community 16 - "Instrumentation"
Cohesion: 0.12
Nodes (18): check_box(), The box every profile is written for, and its assertion (base).  The charter's D, The node the engine is written for, asserted (D3): one GB10, unified memory., Config, ConfigError, Fact, Knob, Facts in, knobs that expire, nothing else (base).  D11's evidence was 133 `VLLM_ (+10 more)

### Community 17 - "KV Cache Sizing"
Cohesion: 0.1
Nodes (15): Release NCCL graph references before destroying its process group., field_dtype(), A served model's caches in one arena: paged blocks, per-sequence state slots, pr, The `count` ring cells before `position` in a ring `width` long, oldest first, o, What the arena carves for the paged blocks, the state slots (one per request id,, How many prefix snapshots a budget holds, and never fewer than MIN_SNAPSHOTS., `f` in each of `count` records of `stride_bytes` in `storage` (uint8): [count, *, The regions' shared moves. A profile's caches subclass it and add their fields, (+7 more)

### Community 18 - "Common Kernel Lanes"
Cohesion: 0.13
Nodes (11): build_identity(), context_band(), Bounded host-side serving diagnostics; never synchronizes a device.  Completion, Which window of a boot compiled something -- one walk of the JIT caches, after t, One log line. It says "no JIT writes" out loud, because that is the answer that, The declared cache directories that exist, with nested ones folded into their pa, Named wall-clock windows over a boot, and the cache artifacts that landed inside, Walk once; count the artifacts newer than the first window, by the window they f (+3 more)

### Community 19 - "Stall Watchdog"
Cohesion: 0.33
Nodes (4): Cache, max_seq(), What a model caches per sequence and per token, and what a budget buys (framewor, Longest context that fits, at this concurrency.

### Community 20 - "Process Topology"
Cohesion: 0.33
Nodes (1): Block tables and state slots, as flat arrays (base).  Two kinds of per-sequence

### Community 21 - "Tensor Layout"
Cohesion: 0.5
Nodes (4): CommonLanes, The engine's default lanes (base): the model-free kernels every profile inherits, The common kernels, bound once. Importing them imports Triton. The norms and Swi, served()

### Community 22 - "Package Init"
Cohesion: 0.4
Nodes (1): A step that never ends, bounded (45차, 2026-09-12 22:31).  The fleet's worst fail

### Community 23 - "Free Block State"
Cohesion: 1.0
Nodes (1): The tensor as a lane decision may read it: contiguous and 16-byte aligned.

### Community 24 - "Block Reservation"
Cohesion: 1.0
Nodes (1): One process per node. World 1 needs no process group at all.

### Community 25 - "Boundary Cache Release"
Cohesion: 1.0
Nodes (0):

### Community 26 - "Snapshot Cache Release"
Cohesion: 1.0
Nodes (1): Measure one phase, optionally accumulating repeated calls in one span.

### Community 27 - "Phase Measurement"
Cohesion: 1.0
Nodes (1): Blocks no row holds: free now, counting the ones a boundary would give up.

### Community 28 - "Death Notes"
Cohesion: 1.0
Nodes (1): Free blocks nothing remembers: what a reservation spends before any boundary pay

### Community 29 - "Distributed Gate Diagnostics"
Cohesion: 1.0
Nodes (1): Free blocks a cached boundary still holds -- the reuse a reservation spends afte

### Community 30 - "Community 30"
Cohesion: 1.0
Nodes (1): Free blocks held by a boundary whose snapshot is gone (its state lives on the pr

### Community 31 - "Community 31"
Cohesion: 1.0
Nodes (1): Never raises. A key that cannot be computed is a full gate, and this rank still

### Community 32 - "Community 32"
Cohesion: 1.0
Nodes (1): What the loop is doing now, for a death note.

## Knowledge Gaps
- **342 isolated node(s):** `Host work a boot runs where it is already waiting (base): a thread started early`, `Host work started where the boot is already waiting, and joined where its result`, `The box every profile is written for, and its assertion (base).  The charter's D`, `The node the engine is written for, asserted (D3): one GB10, unified memory.`, `The box, declared -- not discovered (framework).  A budget is a list of claims o` (+337 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Free Block State`** (1 nodes): `The tensor as a lane decision may read it: contiguous and 16-byte aligned.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Block Reservation`** (1 nodes): `One process per node. World 1 needs no process group at all.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Boundary Cache Release`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Snapshot Cache Release`** (1 nodes): `Measure one phase, optionally accumulating repeated calls in one span.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Phase Measurement`** (1 nodes): `Blocks no row holds: free now, counting the ones a boundary would give up.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Death Notes`** (1 nodes): `Free blocks nothing remembers: what a reservation spends before any boundary pay`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Distributed Gate Diagnostics`** (1 nodes): `Free blocks a cached boundary still holds -- the reuse a reservation spends afte`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 30`** (1 nodes): `Free blocks held by a boundary whose snapshot is gone (its state lives on the pr`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 31`** (1 nodes): `Never raises. A key that cannot be computed is a full gate, and this rank still`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 32`** (1 nodes): `What the loop is doing now, for a death note.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `BlockPool` connect `Serving Diagnostics` to `Device Memory Arena`, `Scheduler & KV Blocks`, `Runtime Introspection`, `Stateless Randomness`, `Model & Kernel Shapes`, `Process Topology`?**
  _High betweenness centrality (0.099) - this node is a cross-community bridge._
- **Why does `TierFull` connect `Memory Plan` to `Serving Diagnostics`, `Device Memory Arena`, `Fleet Lease & Latency`, `Model & Kernel Shapes`?**
  _High betweenness centrality (0.070) - this node is a cross-community bridge._
- **Why does `numel()` connect `Composed Model Lifecycle` to `Memory Plan`, `Serving Diagnostics`, `Device Memory Arena`, `Fleet Lease & Latency`, `Prefix Cache & Tenancy`, `Request Cache`, `Sampling Constants`?**
  _High betweenness centrality (0.060) - this node is a cross-community bridge._
- **Are the 147 inferred relationships involving `TierFull` (e.g. with `CompressedSnapshots` and `Model`) actually correct?**
  _`TierFull` has 147 INFERRED edges - model-reasoned connections that need verification._
- **Are the 103 inferred relationships involving `BlockPool` (e.g. with `ComposedModel` and `Drafter`) actually correct?**
  _`BlockPool` has 103 INFERRED edges - model-reasoned connections that need verification._
- **Are the 95 inferred relationships involving `Tripwire` (e.g. with `_Choice` and `_Histogram`) actually correct?**
  _`Tripwire` has 95 INFERRED edges - model-reasoned connections that need verification._
- **Are the 96 inferred relationships involving `StepWatch` (e.g. with `_Choice` and `_Histogram`) actually correct?**
  _`StepWatch` has 96 INFERRED edges - model-reasoned connections that need verification._
