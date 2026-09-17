# Graph Report - engine/base  (2026-09-18)

## Corpus Check
- 50 files · ~103,656 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1501 nodes · 4082 edges · 30 communities detected
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

## God Nodes (most connected - your core abstractions)
1. `TierFull` - 140 edges
2. `BlockPool` - 116 edges
3. `Tripwire` - 105 edges
4. `StepWatch` - 103 edges
5. `DiagnosticMetrics` - 100 edges
6. `Recorder` - 98 edges
7. `SlotPool` - 85 edges
8. `PagedSpec` - 59 edges
9. `SlotSpec` - 59 edges
10. `Runner` - 58 edges

## Surprising Connections (you probably didn't know these)
- `Off-thread I/O. Poll `.done()`, then `.result()` to surface failures.          T` --uses--> `CompressedSnapshots`  [INFERRED]
  engine/base/kv_tier.py → engine/base/compressed_snapshots.py
- `step()` --calls--> `profiling()`  [INFERRED]
  engine/base/latency.py → engine/base/graph_labels.py
- `Sampling policy and committed token counts carried alongside device decode rows.` --uses--> `History`  [INFERRED]
  engine/base/sampling_options.py → engine/base/sampler.py
- `Admission must not synchronize a surviving row's queued device work.` --uses--> `History`  [INFERRED]
  engine/base/sampling_options.py → engine/base/sampler.py
- `Only the clipped, accepted output updates history. Rejected drafts never do.` --uses--> `History`  [INFERRED]
  engine/base/sampling_options.py → engine/base/sampler.py

## Communities

### Community 0 - "Memory Plan"
Cohesion: 0.03
Nodes (139): carve(), _layout(), PagedSpec, Plan, What each layer kind needs cached, declared once, sized from the budget (base)., Bind the plan to the arena: one KV region, one slot region, two pools., A spec that names a key also names a dtype and a shape whose bytes are its decla, Split a KV budget: slots first (fixed by max_seqs), blocks with the rest.      B (+131 more)

### Community 1 - "Serving Diagnostics"
Cohesion: 0.04
Nodes (135): DiagnosticMetrics, Exception, TierFull, Recorder, answer_budget(), _byte_level_chars(), cache_key(), _Choice (+127 more)

### Community 2 - "Device Memory Arena"
Cohesion: 0.03
Nodes (61): Arena, expandable_segments(), host_reclaim(), _meminfo(), prepare_allocation(), One allocation for persistent weights and caches (base).  D16 in code: weights (, Ask this node's host to return its clean file cache now, and wait for it to say, Drop clean pages of the supplied weight files, reclaim the rest of the     short (+53 more)

### Community 3 - "Fleet Lease & Latency"
Cohesion: 0.04
Nodes (86): acquire(), alive(), attach(), clear_yield(), describe(), door_load(), door_unsupported(), _here() (+78 more)

### Community 4 - "Distributed Communication"
Cohesion: 0.04
Nodes (47): Comm, fleet_env(), init(), _LocalRank, _LocalRun, LocalTP, pick_gid_index(), RankLeft (+39 more)

### Community 5 - "Prefix Cache & Tenancy"
Cohesion: 0.03
Nodes (48): Release NCCL graph references before destroying its process group., That boundary is gone (or changed grade): the blocks fall back to what is left h, Drop that owner; blocks nobody else holds go back to the free list. Returns how, Off-thread I/O. Poll `.done()`, then `.result()` to surface failures.          T, Entry, is_tenant_salt(), PrefixCache, Prefix reuse: a prompt that begins the way an earlier one did is not prefilled a (+40 more)

### Community 6 - "Scheduler & KV Blocks"
Cohesion: 0.05
Nodes (31): Read-only block ids; only reserve/release may change the mapping., Atomically cover absolute write ends, reusing rejected draft space., Seat a cached prefix -- `tokens` whole blocks that are complete and shared -- at, A slot whose contents belong to nobody (an aborted checkpoint, a boundary that i, _launch(), Model, _run(), Runner (+23 more)

### Community 7 - "Memory Budget Gates"
Cohesion: 0.04
Nodes (63): Budget, host_box(), Line, probe_box(), rank_files(), The box, declared -- not discovered (framework).  A budget is a list of claims o, (total device GiB, free GiB after our own context).      Asking costs a CUDA con, (MemTotal GiB, MemAvailable GiB) -- what earlyoom actually counts. (+55 more)

### Community 8 - "Request Cache"
Cohesion: 0.05
Nodes (12): _media_after(), Server, The thread; idempotent, a no-op when both thresholds are 0., CollectiveDivergence, How many ranks say yes to each value (the sums), the site agreed first., Every rank's values, by rank, the site and the count agreed first., The values every rank holds; raises when any rank holds different ones., Meet the same fixed all-reduce as a peer's vote BEFORE entering a broadcast. (+4 more)

### Community 9 - "Sampling Constants"
Cohesion: 0.04
Nodes (67): forget(), fresh(), iota(), Index constants a captured step reads, and may not create.  A decode step asks f, 0..n-1 on `device`, built once and kept. Never write to what this returns., The same values, not kept: for lengths that follow the request., n zeros on `device`, built once and kept, under the same rule as `iota`: the ind, Drop every kept constant. Tests only -- a live capture recorded their addresses. (+59 more)

### Community 10 - "Runtime Introspection"
Cohesion: 0.04
Nodes (44): build_identity(), context_band(), Bounded host-side serving diagnostics; never synchronizes a device.  Completion, docker_evidence(), Is that container up HERE? None when docker cannot answer -- the caller keeps, available(), for_checkpoint(), Grammars (+36 more)

### Community 11 - "Composed Model Lifecycle"
Cohesion: 0.06
Nodes (3): ComposedModel, PositionStore, (seen, counts) for `tokens`, without walking them again when nothing but the end

### Community 12 - "Stateless Randomness"
Cohesion: 0.05
Nodes (53): rank_weights(), (GiB, tensor count) read out of a safetensors header.      The header is a const, _float32(), _lsr(), mix(), mix_tensor(), Stateless uniforms: every draw is a function of what it is for, never of what ca, Logical shift right of an int64 tensor: torch shifts arithmetically, the mask dr (+45 more)

### Community 13 - "Model & Kernel Shapes"
Cohesion: 0.06
Nodes (48): Plan, Attention, bind(), bind_drafter(), bind_recorded(), bound(), Comm, config_sha256() (+40 more)

### Community 14 - "Snapshot Publication"
Cohesion: 0.08
Nodes (20): Builder, CompressedSnapshots, Bounded, lossless RAM copies of snapshots whose durable copy is on NVMe.  Only t, Decode through shared staging; `write(offset, n)` must fence its read., Snapshot, NvmeTier, The NVMe tier for cold KV (base). Contiguous per sequence, O_DIRECT, survives bo, Every parked conversation this layout can promote. (+12 more)

### Community 15 - "Stage Timing"
Cohesion: 0.12
Nodes (8): _Nothing, Where a decode step's device time actually goes, without a synchronisation in th, Call once at the top of a step. True when this one is being measured., `with clock.mark("forward"): ...` -- a no-op except on a sampled step., Read the previous round's events. A whole sampling interval has passed, so they, Each stage's fraction of the measured total, for a reader who wants the shape no, _Span, StageClock

### Community 16 - "Instrumentation"
Cohesion: 0.2
Nodes (7): _dev_free_bytes(), phase(), process_seconds(), A phase that was timed somewhere this recorder could not reach -- the boot's `fr, Seconds since THIS PROCESS started, or None where /proc does not say.      A rec, Free device memory, or None before CUDA is up (see the module note)., Span

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
Nodes (1): One process per node. World 1 needs no process group at all.

### Community 21 - "Tensor Layout"
Cohesion: 1.0
Nodes (1): The tensor as a lane decision may read it: contiguous and 16-byte aligned.

### Community 22 - "Package Init"
Cohesion: 1.0
Nodes (0):

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
Nodes (1): Measure one phase, optionally accumulating repeated calls in one span.

### Community 28 - "Death Notes"
Cohesion: 1.0
Nodes (1): What the loop is doing now, for a death note.

### Community 29 - "Distributed Gate Diagnostics"
Cohesion: 1.0
Nodes (1): Never raises. A key that cannot be computed is a full gate, and this rank still

## Knowledge Gaps
- **321 isolated node(s):** `A byte ceiling for arena + workspaces, with a measured boot ledger.  The allocat`, `MemAvailable, which is the number earlyoom decides on -- not MemFree.      `host`, `(SIGTERM bytes, SIGKILL bytes) -- the lines at which this box kills the engine.`, `The sizes of device blocks still ALLOCATED, largest first.      After a release,`, `Make a boot's remaining byte budget available despite UMA file cache.      Anony` (+316 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Process Topology`** (1 nodes): `One process per node. World 1 needs no process group at all.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Tensor Layout`** (1 nodes): `The tensor as a lane decision may read it: contiguous and 16-byte aligned.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Package Init`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Free Block State`** (1 nodes): `Blocks no row holds: free now, counting the ones a boundary would give up.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Block Reservation`** (1 nodes): `Free blocks nothing remembers: what a reservation spends before any boundary pay`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Boundary Cache Release`** (1 nodes): `Free blocks a cached boundary still holds -- the reuse a reservation spends afte`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Snapshot Cache Release`** (1 nodes): `Free blocks held by a boundary whose snapshot is gone (its state lives on the pr`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Phase Measurement`** (1 nodes): `Measure one phase, optionally accumulating repeated calls in one span.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Death Notes`** (1 nodes): `What the loop is doing now, for a death note.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Distributed Gate Diagnostics`** (1 nodes): `Never raises. A key that cannot be computed is a full gate, and this rank still`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `BlockPool` connect `Memory Plan` to `Device Memory Arena`, `Prefix Cache & Tenancy`, `Scheduler & KV Blocks`, `Request Cache`, `Composed Model Lifecycle`?**
  _High betweenness centrality (0.079) - this node is a cross-community bridge._
- **Why does `numel()` connect `Snapshot Publication` to `Memory Plan`, `Device Memory Arena`, `Distributed Communication`, `Sampling Constants`, `Composed Model Lifecycle`, `Stateless Randomness`?**
  _High betweenness centrality (0.065) - this node is a cross-community bridge._
- **Why does `TierFull` connect `Serving Diagnostics` to `Memory Plan`, `Request Cache`, `Snapshot Publication`, `Scheduler & KV Blocks`?**
  _High betweenness centrality (0.052) - this node is a cross-community bridge._
- **Are the 136 inferred relationships involving `TierFull` (e.g. with `Pending` and `Model`) actually correct?**
  _`TierFull` has 136 INFERRED edges - model-reasoned connections that need verification._
- **Are the 91 inferred relationships involving `BlockPool` (e.g. with `Pending` and `Model`) actually correct?**
  _`BlockPool` has 91 INFERRED edges - model-reasoned connections that need verification._
- **Are the 92 inferred relationships involving `Tripwire` (e.g. with `RequestError` and `PromptTokens`) actually correct?**
  _`Tripwire` has 92 INFERRED edges - model-reasoned connections that need verification._
- **Are the 93 inferred relationships involving `StepWatch` (e.g. with `RequestError` and `PromptTokens`) actually correct?**
  _`StepWatch` has 93 INFERRED edges - model-reasoned connections that need verification._
