# C=1 MoE resident-wave experiment — default off

Base: `8c8b031b94bb80175cfccd49cfb6329d18cdecae`, also the immutable source of
the ongoing four-rank onepass. This experiment has not changed that boot.

The existing v4 kernel can choose 48/44/40/36/32 worker CTAs to reduce empty
slots in its final wave. For example, 16 active experts produce 64 work items:
32 workers can cover two full waves. Whether fewer concurrent workers repay the
extra route-counting atomics depends on the workload and requires GPU timing.

This patch exposes that existing schedule only through the private probe override
on 1..8-token `t,r,sf6` geometry, forwards it to the real kernel constructor, and
separates its memory/disk cache identity from the served handle. The old `e`/`k`
serving tokens remain rejected and no profile enables the option.

Validation prepared:

- Three CPU contract tests pass: serving exclusion, compile-order-independent cache
  separation and fail-fast shape rejection.
- `CUDA_VISIBLE_DEVICES= python3 -m probes.engine_moe_waves_compile --output /out/compile.json`
  compiles both native handles without a GPU and checks actual constructor forwarding
  and cache reuse. This complete CPU gate passed on `c13f2802` during the second
  consumer's preparation; `compile-c13f2802.json` retains both emitted handles
  and dispatcher/kernel hashes. It opened no GPU.
- The admitted `probes/engine_kernel_check.py --lanes moe_waves --ranks RANKS`
  resolves a named rank directory beside `facts.RANKS`, as the existing router
  probe does, and replays both graphs on identical real L3 TP4 packs, changing routing from 8 to 56
  active experts and back. FP32 atomic scatter retains the existing 0.001 relative
  repeat/graph tolerance; the report records measured pair and repeat error.
- Only after all numerical cases pass, warm and 64-MiB-evicted B/A/A/B kernel samples
  are recorded. They are component evidence, not an engine speed or quality verdict.

GPU numerics/timing and consumer adoption remain pending. Do not infer a speedup
or enable the experiment from CPU checks.
