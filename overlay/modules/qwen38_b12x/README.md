# qwen38_b12x — the b12x lane on Qwen3.8-Flash-Next

Three files. Two are wrappers, one is a generated kernel override.

| file | kind | what |
|---|---|---|
| `qwen38_b12x_bounds.py` + `zz_qwen38_b12x_bounds.pth` | NEW | host-side capacity check around `launch_sm120_dynamic_moe`: prints `[b12x-bounds] state_E=.. rows a/b tiles c/d tasks e/f` and refuses a launch that would overrun its workspace. `DENEB_B12X_BOUNDS=0` disables. |
| `moe_dynamic_generic.py` | OVERRIDE of `blackwell_sm12x/_moe_dynamic/generic.py` | the stock SM120 dynamic MoE kernel plus the **unrouted-slot guard** — generated, not hand-edited: `tools/qwen38_b12x_guard_gen.py` |

## The guard, and why every Qwen3.8 b12x boot died before it

vLLM's routing contract lets a top-k slot be `-1`: a padded row, or — under
expert parallelism — an expert another rank owns. Its own backends honour it
(`fused_moe/utils.py`: "The kernel uses -1 to represent invalid topk_ids").
The stock b12x dynamic kernel never looks at the sign:

```
expert_id = topk_ids[hist_idx]
atomic_add_global_i32(row_counts + expert_id, 1)          # phase 1 histogram
row = atomic_add_global_i32(expert_write_rows + expert_id, 1)   # phase 2 route
```

`row_counts[-1]` is four bytes before the allocation. compute-sanitizer on the
exact arguments the profile run hands the kernel (2048 tokens × top-10, every
slot `-1`, dumped from the serving process and replayed):

```
Invalid __global__ atomic of size 4 bytes
  at ..._moe_dynamic generic MoEDynamicKernel ...
  Access to 0xf02a3c3d43fc is out of bounds
  and is 4 bytes before the nearest allocation at 0xf02a3c3d4400 of size 2048 bytes
```

2048 bytes = the `int32[512]` histogram. Whether that write faults or lands
in whatever the allocator placed before the histogram is an accident of
layout: replayed standalone the same call "succeeds"; in the serving process
it is the `cudaErrorIllegalAddress` that ended every boot. Same arguments
with valid ids: clean. It was never the shape — E=512, K=2560, N=640, top-10
runs on the stock kernel in all three b12x paths (dynamic/static/micro).

The guard is what the other backends do: a slot with `id < 0` contributes
nothing — no histogram count, no packed row, no task. Phase 0 already zeroes
the output, so a token whose every slot is unrouted stays zero, which is what
the EP all-reduce expects from a rank that owns none of its experts. That
makes it the expert-parallel path for this model too: no dummy expert, no
GEMM rows for the 3/4 of routes a TEP=4 rank does not own.

Measured (sanitizer matrix, `probes/qwen38_b12x_guard.py` is the same test):

| kernel | all `-1` | 75 % unrouted vs "valid id @ weight 0" | valid ids |
|---|---|---|---|
| stock | invalid atomics, IMA | invalid atomics | clean |
| guarded | clean, output 0 | clean, **bit-identical**, no-route token = 0 | clean |

## Regenerate / verify

```
S=<image>/usr/local/lib/python3.12/dist-packages/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/_moe_dynamic/generic.py
python3 tools/qwen38_b12x_guard_gen.py --stock $S --check overlay/modules/qwen38_b12x/moe_dynamic_generic.py
```

The generator pins the stock file's SHA-256 and anchors each edit on text that
must occur exactly once; a FlashInfer bump refuses to generate rather than
re-anchoring silently. The on-disk CuTe-DSL kernel cache hashes the kernel
source, so a guarded build never collides with a stock one.

## Probe (GPU, inside the image with the module mounted)

```
python3 probes/qwen38_b12x_guard.py        # refuses to run on an unguarded kernel
```
