# C=1 input-pack launch geometry (2026-09-15)

## Change under measurement

The C=1 dense input pack has eight independent warp rows per 128-column K block.
Its previous 256-thread CTA kept all eight rows together: K=1536/2048/3072/4096
launched 12/16/24/32 CTAs on GB10's 48 SMs. The candidate splits these rows across
CTAs with 1, 2 or 4 warps. Two warps per CTA launch 48/64/96/128 CTAs respectively.

Each warp retains the original load, amax reduction, FP32 scale and reciprocal,
FP8 conversion and output offsets. No inter-warp reduction, pack-layout change,
scratch growth or extra kernel launch is introduced. The PDL dependency remains.
Seven-row packing, wide-row packing and existing producer-owned packs retain
their paths. The provisional default selects two rows per CTA only for eight-row
inputs of at most 4096 columns; the other sizes are measurement controls.

## Reproduction and gate

The named canonical GPU probe is:

```sh
bash probes/run_engine_probe.sh probes/engine_kernel_check.py \
  --lanes input_pack --samples 2 \
  --ranks /home/choiceoh/models/st-glm53-9391-up-gate-full/rank3of4.safetensors \
  --output /cache/c1-pack-a55c.jsonl
```

- Queue: `c1pack-grid-a55c`, ticket `178944615793684`, revision 4.
- Frozen source: `090beab7`, on current base `5871c559`; native source SHA-256
  is in `compile.json`. The CUDA source, flags and compiler cache identity match
  the original candidate; current native prebuild integration is included.
- Runtime: `sha256:848e493f37af252865deea2fe6169916f6bac727343b5ab592cd74fcf3639544`,
  Torch 2.13.0+cu132, CUDA 13.2. The probe reserves one GB10 through the fleet queue.
- Geometry controls are native probe arguments, with the prior eight-warp layout
  as the same-build control. They are not request options or environment knobs.
- Exact pack gate: six widths (128 through 20480), six magnitudes, strided inputs,
  poisoned guarded destinations, both replay orders and invalid geometry refusal.
- Consumer gate: six C=1 dense/query cells, single layers and distinct-layer
  chains, the same 34 real BF16 weight tensors and RTN W4 packs in every arm.
  These are not the serving GPTQ packs. Outputs are compared as BF16 bit patterns;
  direct TX writes use changing descriptors and guards around both destinations.
- Timings: B/A/A/B brackets of warm and evicted GPU intervals. Explicit external
  CUDA events are captured around the native calls. Warm intervals repeat the
  calls 32 times; evicted intervals follow a 128 MiB flush outside the interval.
  Each bracket arm averages 16 replays. Inputs are nonzero during timing.

Use `python3 measurements/st_c1_input_pack_grid_20260915/summarize.py FILE.jsonl`
to regenerate the component table. A partial or failed run is refused.

## Evidence available

- Production native compile and load with CUDA hidden: **PASS**, `compile.json`.
  The build uses the same compiler-aware cache identity as the serving extension.
- Current CPU suite: **47 passed, 33 GPU-only skipped**, `cpu-tests.log` (80 total).
  The container uses `--workdir /repo`, with this checkout mounted read-only at
  `/repo` and `PYTHONPATH=/repo`. Earlier CPU invocations omitted `--workdir` and
  could import the image's `/opt/st/engine`; their results are superseded and
  are not qualification of this change. This suite covers dense callers,
  seven-row routing, KDA norms, forward paths, kernel boundaries, native cache
  and the new native prebuild integration.
- GPU numerical and component timing results: **queued, not yet measured**.
- TP4 onepass, consumer throughput, acceptance and quality: **not measured**.
  Reservation `c1pack-full-v4-a55c` (ticket `1789447070414497`) is paused until the
  component gate passes; it compares this candidate with `5871c559` using full
  validation (C=1 twice and C=4 once per boot). Two earlier preparation attempts
  were refused before taking GPUs because relevant main changes were missing;
  the candidate was rebased before the accepted reservation.

The block counts above are source facts. They do not establish a speedup.
Adoption must use the complete projection intervals as well as the pack interval;
a microbenchmark is not evidence of faster serving. Consumer claims require the
canonical TP4 onepass under D17.
