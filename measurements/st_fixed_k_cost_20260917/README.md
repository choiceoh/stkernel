# Fixed K7 mHC input packing

**Verdict: component cost reduction and target execution are proven; a whole-engine
speed improvement is not established.** Canonical `st_judge.py` returns
`NO EVIDENCE: the candidate has no valid warm sample (its records failed gates)`.
PR #1068 remains a draft, with adoption pending. K stays 7, verification stays
8 rows per request, and KDA state stays FP32.

Three requested directions were evaluated across seven GPU revisions. Resident
MoE wave scheduling had no clear gain. Adjacent-query sparse MLA sharing passed
numerics but remained slower. Both were removed from the final source. The
retained candidate is mHC producer packing; it is not a claimed serving win.

## Retained implementation and numerical proof

The direct mHC consumer packs its once-rounded BF16 layer input using warp-local
128-column FP8 packing. The next bound KDA input projection consumes that exact
invocation-owned buffer. Observers and unsupported cells use the existing path.
Pack-writing template instantiations and occupancy entries are separate from
ordinary consumers. `net.mhc_input_packs=False` disables this new boundary only;
existing KDA output producer packs stay enabled in both arms.

All 90 real mHC coefficient sets pass bitwise equality of all four outputs and
FP8 pack bytes at magnitudes 0, .001, 1 and 30, with and without changing TP4
packet descriptors and repeated graph replay. The final native implementation
was qualified in revisions 3, 4, 6 and 7. The final focused Linux CPU gate ran
55 tests (52 passed, 3 GPU-only skips); the source-lifetime repair below ran
40 tests (39 passed, one GPU-only skip). Full CI passed on the measured engine.

Two graphs containing all 90 distinct real coefficient sets, including the
separate pack in the control and no pack clone, were timed B/A/A/B:

| Revision | Packet control → fused | Nonpacket control → fused |
|---|---:|---:|
| v6 | 1592.05 → 1287.60 us (-19.12%) | 1522.76 → 1223.76 us (-19.64%) |
| v7 | 1601.07 → 1290.03 us (-19.43%) | 1520.75 → 1226.71 us (-19.34%) |

Single-coefficient packet replay is near parity (v7: 19.57 → 19.72 us).
Serving fuses only 30 KDA boundaries, so the chain reduction is not an engine
throughput estimate. L0 and the boundaries following auxiliary features
(L6/L25/L34) retain their ordinary path.

## Matched TP4 consumer result

Control `ffd19b44630b905f0c68db5e8f060bc78680012e` and candidate
`69c5420cf497992184842308990a1f6d90613a5b` differ by one boolean assignment in
`net.py`. The source-lifetime repair is identical in both. CUDA source hash is
`8d3bd8a91b1d81b3de92c00ef9d250e4a0e24915f3ff5602df0ae13ebda71154` on all four
nodes. Each node reuses its native cache key `350283036157c48725e8dc7c` across
both boots; per-node binary hashes and mtimes are in `consumer-runtime.json`.
Binaries are built per node, not asserted byte-identical across nodes.

Each arm ran the canonical extended onepass profile: 2K/32K/128K, C1 twice,
current serving capacity C2 once (without 128K C2), with exclusive traffic.
C4 was not measured. Compilation/capture precede measurement; diagnostic
profiling follows it. All listed C1 prefill samples have zero prefix reuse.

The following are **raw observations from failed-quality runs**, not eligible
speed verdicts. Request tok/s is sum(completion tokens - 1) / sum(decode seconds),
not step/s multiplied by acceptance. The harness correctly clears its primary
step-rate field when a gate fails; the table shows the retained raw median.

| Arm / run | Actual request tok/s | Acceptance | Raw median step/s | C1 quality |
|---|---:|---:|---:|---:|
| B / 1 | 87.3047 | 53.6728% | 19.887089 | 6/9 |
| B / 2 | 83.3485 | 50.4309% | 19.891678 | 7/9 |
| A / 1 | 85.2527 | 52.0330% | 19.885538 | 7/9 |
| A / 2 | 86.1621 | 53.3209% | 19.882239 | 7/9 |

C1 Korean corruption is zero in all four runs. C2 quality is B 8/12, A 10/12.
Cold request-rate change is -2.35%, warm +3.38%; warm acceptance also rises
2.89 percentage points while raw median step/s changes -0.047%. Baseline
repetitions already differ substantially in output length and acceptance.
None of the five full C1 response hashes match between arms in either run.
Greedy text differences alone are not a rejection (ledger rule 8); the quality
gates and lack of repeated valid brackets prevent an adoption claim.

The first run also measures four fixed 1024-token C1 requests. Two output hashes
match exactly; their observed decode rates are 106.4239 → 106.2806 and
107.6071 → 107.4849 tok/s (about -0.1%). This single pair does not establish
neutrality or regression. The fixed C2 observation is invalid in both arms
because its preparation identity changed/was unknown; it is not adoption proof.

`consumer.jsonl` retains all four completed records with per-request hashes,
TTFT, generation rates, quality, acceptance, serving shape and runtime identity.
`consumer-summary.json` is reproducible with:

```
python3 measurements/st_fixed_k_cost_20260917/summarize.py \
  measurements/st_fixed_k_cost_20260917/consumer.jsonl
python3 bench/st_judge.py judge --cand 69c5420c --base ffd19b44 \
  --jsonl measurements/st_fixed_k_cost_20260917/consumer.jsonl
```

The baseline ticket `fixedkfull3-0917` returns its quality failure after retaining
both C1 passes, before running A. A was therefore measured in the separately
queued `fixedkcandidate-0917` ticket with the same profile, avoiding a third
baseline boot. Both tickets stopped their four containers, dropped their owned
tiers and released the fleet reservation.

## Actual target execution

Startup proves 30 eight-row KDA input-pack consumers on every candidate rank
and zero in the control (`consumer-native-A/B.jsonl`). A separate 2K diagnostic
confirms the GPU specialization ran: across four traced decode steps, each rank
changes 336 ordinary packet mHC launches into 216 ordinary + 120 pack-writing
launches. Standalone input-pack launches fall **298 → 178**, or 30 removed per
traced step (`consumer-launch-proof.json`). CUPTI launch counts are evidence;
overlapping PDL kernel durations are not added as latency.

The optimization selects the eight-row graph, not the HTTP group's label.
Two concurrently decoding requests use the sixteen-row path, but one request
finishing early can leave an eight-row tail that uses the fusion. C2 group-level
results are therefore not a strict no-treatment control.

## Seven component revisions

All revisions passed their numerical/replay gates. Earlier records also cover
42 final MLA geometry/selection fixtures (including duplicates, permutations,
asymmetric lengths and empty rows) against an independent FP32 reference, and
real L3 MoE weights with changed routing and zero-output replay. Rejections below
are performance findings, not numerical failures.

| Rev / commit | Packet mHC single-boundary us | MLA C1 identical-selection us | Change / outcome |
|---|---:|---:|---|
| 1 / `18e14162` | 20.19 → 20.61 | 43.11 → 142.99 | Initial fusion/union sharing rejected; MoE U40 evicted 679.47 → 679.21 us has no clear gain. |
| 2 / `a236b5fa` | 19.40 → 20.50 | 43.17 → 85.49 | Reuse rounded mHC values; MLA shared-Q/matrix loads, still slower. |
| 3 / `03cb3652` | 19.63 → 20.17 | 43.07 → 87.40 | Warp-local mHC packing; parallel query groups. |
| 4 / `8f3fe48a` | 19.68 → 20.01 | 43.00 → 84.08 | Compact union reservations and shared BF16 KV ring. |
| 5 / `2715eda1` | 19.56 → 22.65 | 43.05 → 92.33 | Ready-row mHC helpers and radix union both rejected and removed. |
| 6 / `b631febd` | 19.33 → 19.58 | 42.97 → 61.52 | Retain warp-local mHC; matching tiles without union or separate merge. Add distinct-coefficient chain. |
| 7 / `1f049629` | 19.57 → 19.72 | 43.20 → 47.50 | Remove tile search and empty-query work; MLA still +9.96% C1 / +19.39% C2, removed. |

The final probe is narrowed to retained mHC (`fixed_k_compile` builds dense
without GPUs; `fixed_k_cost` checks real coefficients and chains under a fleet
reservation). Rejected implementations remain in the recorded commits. Runtime,
source, binary and coefficient hashes are in `compile*.jsonl` / `gpu-v*.jsonl`.

## Boot repair encountered during validation

The initial baseline stopped before serving on all ranks: automatic draft FC
collection inherited from main was armed after compact storage had retired its
BF16 reference. Both arms now retain an independent host copy before compaction
and move it to the reader device only for the shutdown pair bundle. The live
reader and compact arena are unchanged. A regression overwrites the retired
storage and checks identical reference/actual outputs and reader identity.

`boot-failure.json` pins the initial source-lifetime failure and one retry that
caught a script-mode relative import. The measured commit uses an absolute
import. Neither failed boot contributes a consumer speed claim.
