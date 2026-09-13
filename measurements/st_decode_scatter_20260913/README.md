# Decode output and projection candidates, 2026-09-13

The first GPU hold started at 11:59:59 KST and completed at 12:01:00, shortly
after admission. `gpu-v1-summary.json` and `gpu-v1-records.jsonl.gz` retain the
results on `fe6e8275`. Both projection pairs passed their real-weight numerical
checks and reduced component latency: KDA M7/M28 by 37.65%/25.13%, indexer by
40.40%/23.69%. These are small component chains, not whole-engine speedups.
The serial shared MLP was 6.42–9.24% slower; its probe lane was removed.

Direct MoE scatter passed every M7 routing case exactly, then a second
normalization of M14 incorrectly rejected its already-selected M32 geometry.
Both route-owned variants failed at M7 because they omitted the existing
saturated BF16 cast AFTER FP32 route weighting (maximum relative error
0.0051282). This was an implementation error, not a reason to relax the gate.
The repair copies the served cast/expand instructions before storing each
contribution, retains FP32 RED's FTZ addition semantics in the reduction, and
makes shape normalization idempotent. The next hold reruns only the three
repaired MoE variants. Projection and shared timing are not repeated.

Nothing here establishes 22 step/s or an acceptance improvement. The completed
six-lane capacity experiment supplied no new serving winner; its results are in
`../st_decode_capacity_20260913/gpu-summary.json`. Losing private branches and
the earlier losing dense probes are removed; exact tested commits remain.


| Candidate | Work removed or changed | Contract and cost |
|---|---|---|
| `moe_route_scatter` | Replace contended FP32 output atomics with uniquely owned route/partial stores and one reduction | Preserve BF16 per-part rounding, FP32 route multiplication and saturated BF16 contribution rounding; change only FP32 summation order. Extra scratch: 3.5 MiB at M7, 14 MiB at M28. Reduction and final BF16 copy are timed. |
| `moe_direct_scatter` | Scatter the actual register pairs without the FC2 output shared-memory round trip and publication barrier | Compile-time enumeration proves exact coordinate coverage and pair alignment for both geometries. Retain the final barrier protecting metadata and input lifetimes. |
| `moe_route_direct` | Combine the two output changes | Independently compiled and timed; no assumption that gains add. |
| `paired_projection` | One KDA kernel replaces two BF16 projections; one indexer projection replaces two calls | Separate KDA inputs, FP32 accumulation and BF16 output. Indexer weights are joined once after smoothing and before capture, adding 22 MiB for 11 layers. The FP32 head gate and recurrent state are unchanged. |
| `shared_serial` (removed) | Existing fused shared MLP in the serial C4 path | All 42 real layer packs passed numerics but latency regressed 6.42–9.24%; retain the served serial path. |

The MoE shape contract is M7/14/21/28, hidden 4096, intermediate 512, E288,
top8, TP4 and the served SF6 packs. M7 keeps the served M16 reform; wider
rows retain M32. The larger output ABI requires a prewarmed owner and
reduction adapter; an accidental direct launch is explicitly refused. Owners
and captured inputs survive all replays. The probe poisons scratch and output,
changes route reuse from 8 through 224 experts, and tests zero route weights.
No numerical tolerance is loosened to obtain timing results.

Projection qualification uses all 34 KDA and 11 indexer weight pairs from the
exact rank file. It covers M1/6/7/14/21/28, parent stride 6416, mixed KDA input
strides, changed inputs, zero inputs and reverse replay order. The shared lane
uses all 42 distinct packs, with M14/21/28. Both output ownership and retained
weight memory are explicit. These component tests do not measure consumer
acceptance, RDMA overlap or total decode step time.

## CPU preparation

`cpu-initial/` records 16 actual CuTe/SM121 compiled handles on `0576272b`,
13 Triton/PTXAS projection/reduction combinations, a full native CUDA/Torch
extension build, and CPU package/import checks on `e40b0662`. All passed,
with CUDA devices hidden and no CUDA context initialized. The only hardware
queries overridden for CuTe compilation were capability and occupancy facts;
the real compiler and output-coordinate validator ran.

After removing the rejected epilogue alias and other old hooks, the final
CuTe build and focused tests were rerun on `8a2aa95b` in `cpu-final/`: all
16 CuTe combinations, all 13 Triton combinations, 15 CPU tests and all 88
module imports passed. The recorded source hashes match the final kernel files. The unchanged dense
translation unit retains the initial full-build evidence. CPU success does
not qualify GPU arithmetic or speed.

## One bounded GPU hold

Run the three repaired MoE lanes in one admitted hold with `probes/engine_kernel_check.py
--lanes scatter_bundle --ranks st-glm53-9391-up-gate-full` through the official
`bench/fleet.sh run --gpu --fleet` and `probes/run_engine_probe.sh` entry.
Each lane runs in a separate child so a failed numerical gate does not erase
other results. MoE timeouts are 240 seconds each, within one 15-minute maximum reservation.
The two completed projection/shared lanes are excluded from this recovery hold. Successful completion
releases the hold immediately. The private campaign expires September 16 UTC.

Retain only measured component winners for a candidate consumer boot. That
consumer remains C1 twice and C4 once, including 32K/128K, decode step/s,
committed token rate, decode row width and acceptance. PR760's `step_peek.py`
can report the live counters before the complete one-pass report. There is
no new baseline model boot, and answer grades are not the decision gate.
