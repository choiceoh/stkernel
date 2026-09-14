# Reduce the cost of direct MoE scale registers

Follow-up to merged #940 (`2ed1f047`), requested to increase useful work and
reduce overhead. Keep its 48 scale-publication barriers and 96 KiB of expanded
shared stores removed per H4096 expert/M16/128-intermediate work item.
Prepare packed addresses, shifts and the scale base once per pipeline stage,
then reuse them across that stage's K64 fragments. Remove the two unused
expanded-scale rings from shared allocation. Existing B backing supplies
layout-only views for creating the ordinary MMA fragments; this route never
reads or writes through those views. Compile-time checks verify the actual
Storage byte size and that the layout-only views fit their backing.

The existing direct-register default and scope are unchanged. Packed bytes,
quad exchange, MMA/rounding order, FP32 KDA, K=7, scatter and acceptance logic
are unchanged. Internal `sf6_registers=False` retains the ordinary control.
There are no additional runtime knobs or global allocations.

## Cost comparison

M8, same SM121 compiler/image, other axes fixed:

| Metric | Ordinary expanded scales | #940 | Follow-up |
|---|---:|---:|---:|
| Non-NOP static instructions | 3,474 | 3,724 | 3,619 |
| Extra instructions versus ordinary | — | +7.2% | +4.2% |
| Registers/thread | 121 | 103 | 119 |
| Shared storage/CTA | 97 KiB | 97 KiB | 89 KiB |
| Static BAR.SYNC instructions | 28 | 22 | 22 |
| Stack/local bytes | 0 / 0 | 0 / 0 | 0 / 0 |
| Base reads/lane/work item (source count) | — | 160 | 48 |

The follow-up removes 105 non-NOP instructions (-2.8% versus #940), 8 KiB of
shared allocation (-8.2%), and 112 repeated base reads per lane/work item
(-70%). The cost is 16 more registers than #940, still below the ordinary
control. Shared memory still limits the kernel to one CTA/SM; neither the
register nor shared-memory changes establish higher occupancy or step/s.
`compile.json` and `native-summary.json` preserve the final resources. #940's
source-bound record remains in `../st_moe_register_scales_20260914/compile.json`.

**Performance is unmeasured.** This reduces specified code/storage costs while
preserving #940's mechanism; it does not show a latency win or attainment of
24 step/s. That target still requires 50 → 41.67 ms/step at a 20 step/s anchor.
Whole-kernel timing, GPU numerical/replay checks, quality, acceptance and
consumer measurements remain required and were not submitted.

## Validation

- 27 CPU tests passed, including every scale base/code, all warp/quad rows,
  changing ring contents and canaries. The production FC1 executor also proves
  exactly one metadata preparation per gate/up stage, reuse across K blocks,
  and no use after release (`cpu-tests.log`).
- Seven full SM121 native variants compiled: M1/M7/M8, M8 ordinary control,
  stamped M8, and ordinary-path M16/M32. All stack/local bytes are zero.
- Four actual operand helpers compiled (ordinary/direct × FC1/FC2), and the
  actual copy mapping's 24,576 operand words matched the CPU encoder. The
  prepared GPU variant retains 64 replay/canary/operand checks per arm.
- Same-pack real-weight, repeated-graph and warm/evicted probes from #940 remain
  available. They have not run.

All remote work used the existing srv2 ST image
`sha256:f85de49afc0a41596cce3df2dab11af992a9aa5d21129c0f50ba719c30f68781`,
runc, CUDA hidden, no network, two CPUs/four GiB. No GPU context or queue,
engine/image build, model boot or restart was used.

```sh
python3 -m unittest -v tests.test_engine_moe_register_scales tests.test_engine_moe_compact_staging tests.test_engine_moe_fc1_reuse tests.test_engine_moe_sf6_staging tests.test_engine_moe_activation_store tests.test_engine_moe_scatter_config
python3 probes/engine_moe_sf6_compile.py --register-scales --sass --output /out/compile.json
python3 probes/engine_moe_register_scales.py --cpu --output /out/operands.json
```
