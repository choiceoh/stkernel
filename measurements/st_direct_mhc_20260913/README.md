# GB10 TP4 direct MHC consumer

Status: implemented, default off; CPU arithmetic and SM121a compilation pass.
Native numerical, real TP4 ring replay, quality and onepass performance are
not yet qualified. `compile.json` is a compile-only record, not GPU proof.

The receiver passes four canonical rank pointers to the immediate MHC
consumer instead of materializing an all-reduced tensor. MHC sums in FP32
rank order 0,1,2,3 and rounds to BF16 before the unchanged post/pre equations.
The KDA state remains FP32. Auxiliary-feature FFNs and the final FFN keep a
materialized result; the full 45-layer model has 84 eligible boundaries.

An exchange owns its source and device descriptor until same-stream consumer
submission. Another device collective is refused while a packet is pending;
a consumer failure poisons the transport. Each graph captures the immediate
consumer before the next exchange. Ring slots cannot be overwritten by four
later exchanges without this rank first publishing its next exchange.
The 2+2 communication side-stream experiment is an incompatible plan.

This first implementation covers the receiver. GEMM writing directly to the
send ring is a separate producer extension: its output address must follow
the runtime ring sequence, with reservation before writes and publication
after every writing CTA. It is not claimed by this patch.

Validation on 2026-09-13:

- 29 CPU tests in the ST image passed: real four-rank LocalTP C=1/C=4
  output/auxiliary/cache-state equality, packet lifetime, defaults, boot paths,
  canonical cancellation sums and integer collectives.
- 23 fleet admission contract tests passed locally.
- Both production-flag SM121a native extensions compiled without a CUDA
  context (82.77 seconds, image
  `sha256:f7b81c6f085c056239c059e26b2101944bb117adf0d0a1df87cb22ae9938c162`).
- `probes/engine_direct_mhc_check.py` provides an admitted GPU gate: two
  coefficient layouts, rows 1/7/28/64, 96 changing-input/address graph
  replays against ordinary rank sum + existing MHC, exact comparison.

Before promotion, a same-build baseline/candidate bracket must include full
onepass C=1 twice and C=4 once per boot, 2K/32K/128K, output quality, actual
tok/s and TTFT, acceptance, length and per-step timing. No speedup is claimed.

## Requested sequence

1. Direct communication into MHC: receiver implemented here; producer extension pending.
2. Tile-ready prefill: next; begin with KDA input projection, preserving full
   recurrent token order and the existing FP8/BF16 transport boundary.
3. UMA NVMe staging: next; one mapped pinned allocation, lossless I/O.
4. Bounded GPU decode iterations: next; quantify available host savings and
   preserve stop, admission, prefix and context boundaries before integration.
