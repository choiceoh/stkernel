# C=1 input-pack launch geometry (2026-09-15)

## Decision: rejected

The launch-grid change passed bitwise numerical checks but did not give a
repeatable improvement in complete projection intervals. All input-pack
dispatch, kernel and probe API changes were removed. Production keeps its
original eight-warp layout. No TP4 boot was spent on this rejected candidate.

The complete sweep is in `gpu.jsonl`, with its queue log in `queue.log` and
the generated table in `components.md`. The focused query repeat is in
`query-repeat.jsonl` and `query-repeat.log`.

| Two warps vs eight: projection chain | Warm time change | Evicted time change |
| --- | ---: | ---: |
| KDA input | -0.69% | -0.02% |
| KDA output | +0.11% | -0.12% |
| MLA output | +0.19% | -0.41% |
| Query pair | -2.59% | +3.08% |
| MLP gate/up | -1.15% | -0.10% |
| MLP down | +4.40% | -1.75% |

Negative means less time. The isolated pack's changes were small (often tens
of nanoseconds), and gains did not consistently survive the consumer chain.
An apparent one-warp query win was retested across four independent graph
captures with alternating capture/allocation order and four B/A/A/B brackets.
Its aggregate chain time was **+2.61% warm / +0.08% evicted**; the single-layer
time was **+0.01% / +0.50%**. This confirmation also rejects a query-only default.

## Experiment

The C=1 dense input pack has eight independent warp rows per 128-column K block.
Its previous 256-thread CTA kept all eight rows together: K=1536/2048/3072/4096
launched 12/16/24/32 CTAs on GB10's 48 SMs. The candidate splits these rows across
CTAs with 1, 2 or 4 warps. Two warps per CTA launch 48/64/96/128 CTAs respectively.

Each warp retains the original load, amax reduction, FP32 scale and reciprocal,
FP8 conversion and output offsets. No inter-warp reduction, pack-layout change,
scratch growth or extra kernel launch is introduced. The PDL dependency remains.
Seven-row packing, wide-row packing and existing producer-owned packs retain
their paths. The prototype selected two rows per CTA only for eight-row inputs
of at most 4096 columns; the other sizes were measurement controls.

## Reproduction and gate

The historical canonical GPU probe was:

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
- Confirmation: `c1pack-query-a55c`, ticket `17894489101043755`, source `b5064362`.
  Four independent captures reverse the arms' capture/allocation order; each
  capture has four B/A/A/B brackets. Both query numerical groups passed.
- `prototype.tar.gz` contains the final experiment's source overlay, frozen
  from `b5064362`. Overlay it on `5871c559` in an isolated checkout to reproduce
  either sweep. The active engine no longer contains this prototype.

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
- GPU numerical checks: **18 groups passed** in the sweep; both confirmation
  groups passed. The largest allocated GPU footprint was 1,369,101,312 bytes.
- Component timing: **measured; no reliable projection-chain improvement**.
- TP4 onepass, consumer throughput, acceptance and quality: **not measured**.
  The paused full-validation reservation must not resume this rejected source.

These component measurements establish no serving speedup. The candidate was
retired before consumer validation, as required by the short numerical/timing
gate in D17.
