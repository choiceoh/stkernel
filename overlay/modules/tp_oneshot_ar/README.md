# tp_oneshot_ar

Host-register RDMA one-shot AllReduce for small decode tensors (27us against
NCCL's 67us on this fabric). Both files are new, so this half is portable;
the CUDACommunicator.all_reduce hook is per-image and lives in a
`*_oneshot_wiring` module.

Armed by `VLLM_DSV4_ONESHOT_AR` on both the DSV4 and GLM-5.3 lanes. Setup,
connect, and self-test failures fall back to NCCL only after an all-rank vote.
Once real OSAR traffic is committed, a local failure is fatal: silently moving
one rank to NCCL would split the collective and deadlock the other ranks.

One kernel launch per collective (`k_oneshot`, #89's 5->3 followed by 3->1):
ring guard and peer wait spin on globals inside the kernel, and the publish
step is taken by the last block via a never-reset monotonic completion counter
(`done_ctr % ARGRID`), which is what makes it cudagraph-replay-safe.

## SM121a launch geometry

The fixed grid is **48 blocks**, one per GB10 SM, rather than the inherited
256-block grid. GLM-5.3's DFlash `k=7` verify shapes are T=8/16/32 for
C=1/2/4, or 32,768/65,536/131,072 hidden elements; the old C=1 verify launch
used 128 data-owning blocks and 128 empty CTAs. Plain T=1 calls were worse:
only 16 blocks owned data and 240 were empty. Empty CTAs still crossed barriers
and incremented the completion counter before publish.
The shipped `MAXEL=131072` covers GLM's largest captured verify and DSV4
through DSpark C=5. DSV4 falls back to NCCL above that size. The crossover is
unmeasured, so `VLLM_DSV4_OSAR_MAXEL` is an opt-in build/eligibility override,
not a larger default. Every override uses a value-specific extension build
directory and is included in the boot fingerprint; all ranks must agree
because `MAXEL` participates in the remote receive-buffer stride. The fixed
48 blocks cover every admitted size through the existing grid-stride loops.

This is not a portable heuristic. `init()` requires compute capability 12.1
and exactly 48 SMs before allocating the RDMA state. A mismatch raises inside
the existing all-rank bootstrap vote, so every rank stays on NCCL. The fixed
48 increments per replay preserve the monotonic-counter invariant and CUDA
graph behavior. A block barrier between each thread's system fence and thread
0's completion atomic also guarantees that the last block cannot publish a
partially copied RDMA payload.

## L2 prefetch during the peer wait (2026-09-04, `VLLM_GLM53_AR_PREFETCH`)

The collective's wait is 38.7 of its 45.5 us (MEASUREMENTS 19차), ~100 times
per decode step, and during it DRAM is idle on every rank. `k_oneshot` now
takes `HintArgs` -- up to 8 (pointer, bytes) ranges of the weights the NEXT
kernel streams -- and warps 1..7 of every block walk them with
`prefetch.global.L2` (32 B sectors, interleaved across the grid so every
consumer block gets a uniform slice) while thread 0 polls the peer flags.
Owning blocks stop the moment the peers land; the budget (default 12 MB,
knob value = MB, 1..20; L2 is 24 MB) bounds the work either way. `n == 0` is
the old kernel byte for byte.

What to warm is LEARNED, not declared. The megakernel driver's launches
(`_gemm_call`, `_mhc_call`, `_kda_launch`) call `note_consumer(tensors)`;
the shim files the note under the ordinal of the most recent collective of
the current target forward; `begin_forward`/`end_forward` come from
`Glm5NextForConditionalGeneration.forward`, the class above the compiled
region (the drafter is a different class and never sees the table). A
forward that noted more consumers than the adopted table replaces it -- a
decode-shaped eager warmup does, a prefill forward (M > 32, stock GEMMs)
does not -- and the captured launches bake the adopted table's ranges into
their `HintArgs`. Every collective of the forward advances the ordinal,
whichever path serves it, so NCCL-served prefill collectives keep the keys
aligned. Needs MK-GEMM armed to have anything to learn.

`oneshot_ar_hint(x, ptrs, lens)` and `phase_counters()` are the probe-facing
bindings; `probes/oneshot_ar_disttest.py` times a 12 MB hint against none and
reads `t_wait` for both, which is where any DRAM contention with the NIC's
writes would show. Ceiling on the critical rank (its wait is the transfer,
~20 us = 4.6 MB at 230 GB/s): ~1.5-2.5 ms/step. Fleet bracket only.
