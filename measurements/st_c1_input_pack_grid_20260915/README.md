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

- Queue: `c1pack-grid-a55c`, ticket `178944615793684`, revision 2.
- Frozen source: `caff1db0`; native source SHA-256 is in `compile.json`.
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
- Related CPU suite: **25 passed, 10 GPU-only skipped**, `cpu-tests.log`.
- Dense, linear-family, kernel-glue and seven-row caller suite: **16 passed,
  23 GPU-only skipped**, `callers-tests.log`.
- GPU numerical and component timing results: **queued, not yet measured**.
- TP4 onepass, consumer throughput, acceptance and quality: **not measured**.

The block counts above are source facts. They do not establish a speedup.
Adoption must use the complete projection intervals as well as the pack interval;
a microbenchmark is not evidence of faster serving. Consumer claims require the
canonical TP4 onepass under D17.
