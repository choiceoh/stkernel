# Decode output and projection candidates, 2026-09-13

The next candidates are implemented; GPU numerics and timing are pending.
Nothing here establishes 22 step/s or an acceptance improvement. The completed
six-lane capacity experiment produced no new serving winner; its results are in
`../st_decode_capacity_20260913/gpu-summary.json`. Its losing private branches
and the earlier losing dense probes are removed. Reproduce those campaigns at
their recorded commits, not through the current dispatcher.

| Candidate | Work removed or changed | Contract and cost |
|---|---|---|
| `moe_route_scatter` | Replace contended FP32 output atomics with uniquely owned route/partial stores and one reduction | Preserve BF16 per-part rounding and FP32 route multiplication; change only FP32 summation order. Extra scratch: 3.5 MiB at M7, 14 MiB at M28. Reduction and final BF16 copy are timed. |
| `moe_direct_scatter` | Scatter the actual register pairs without the FC2 output shared-memory round trip and publication barrier | Compile-time enumeration proves exact coordinate coverage and pair alignment for both geometries. Retain the final barrier protecting metadata and input lifetimes. |
| `moe_route_direct` | Combine the two output changes | Independently compiled and timed; no assumption that gains add. |
| `paired_projection` | One KDA kernel replaces two BF16 projections; one indexer projection replaces two calls | Separate KDA inputs, FP32 accumulation and BF16 output. Indexer weights are joined once after smoothing and before capture, adding 22 MiB for 11 layers. The FP32 head gate and recurrent state are unchanged. |
| `shared_serial` | Qualify the existing fused shared MLP in the serial C4 path | Actual weights from all 42 layers. Both arms use the same RTN W4 packs. This is new qualification, not a new kernel; no C4 overlap candidate. |

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

Run the five lanes in one admitted hold with `probes/engine_kernel_check.py
--lanes scatter_bundle --ranks st-glm53-9391-up-gate-full` through the official
`bench/fleet.sh run --gpu --fleet` and `probes/run_engine_probe.sh` entry.
Each lane runs in a separate child so a failed numerical gate does not erase
other results. MoE timeouts are 240 seconds each; the two other lanes have
180 seconds each, within one 20-minute reservation. Successful completion
releases the hold immediately. The private campaign expires September 16 UTC.

Retain only measured component winners for a candidate consumer boot. That
consumer remains C1 twice and C4 once, including 32K/128K, decode step/s,
committed token rate, decode row width and acceptance. PR760's `step_peek.py`
can report the live counters before the complete one-pass report. There is
no new baseline model boot, and answer grades are not the decision gate.
