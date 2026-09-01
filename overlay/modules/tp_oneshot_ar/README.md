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
