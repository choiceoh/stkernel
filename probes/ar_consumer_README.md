# GLM53 communication / MHC / GEMM consumer preparation

`VLLM_GLM53_AR_CONSUMER_PDL=1` lets the next MHC load its immutable
projection weights while the current one-shot AllReduce waits for peers.
The existing GEMM PDL prologue can then prepare its weights along the same
stream. The profile default is **0**. This is an ordering experiment; no
serving speedup has been established yet.

The candidate requires `VLLM_GLM53_MK_PDL=1` and is bounded to at most
eight 4096-wide tokens, including the C=1 speculative verification bucket.
Larger inputs use the ordinary kernels. Compare both arms with the existing
AR hint prefetch disabled, as in the default profile: the candidate dispatch
takes precedence over those optional hints for these small collectives.

## Ordering contract

1. AR waits for its own producer before reading input or protocol state.
2. Every copy thread still performs the original system fence. The final
   CTA publishes the transmit sequence through the unchanged counter protocol.
3. Each AR CTA releases dependent launch before waiting for peer arrivals.
   All CTAs must reach their release points before CUDA can launch a consumer.
4. MHC reads only immutable `fn` weights before its dependency wait. The BF16
   consumer caches `[output, hidden, stream]` coefficients so four values use
   one aligned 64-bit load (24 vector loads instead of 96 scalar loads).
   Activation,
   workspace and counter accesses remain after the wait, including inactive
   CTAs that later obtain a tail ticket.
5. MHC arithmetic, BF16 bit patterns, projection reduction order and GEMM packs
   are unchanged. Scalar and vector layouts retain their own storage under one
   versioned cache entry; capture cannot allocate a missing pack. New MHC
   instantiations use their own occupancy queries. Distinct AllReduce entry
   points compile away the mode branch and preserve the ordinary kernel ABI.

CUDA may serialize these kernels. Correctness never depends on overlap or on
a successor making progress. Early release does not mean that AR output is
ready; the consumer wait remains mandatory. See NVIDIA's
[programmatic dependent launch contract](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html).

## Reproduction and evidence

`python3 probes/run_ar_consumer_cpu.py --out /absolute/fresh/path` compiles the
production CUDA and delayed producer in the exact serving image without GPU
access. It caps host memory and retains the generated objects for inspection.
Run the repository's CPU contracts before GPU admission.

On srv2, from a committed clean checkout based on current main, use the
canonical supervisor:

```bash
REPO="$PWD" bash /home/choiceoh/stkernel/bench/fleet.sh run --gpu \
  arconsumer0908v1 70 'AR/MHC consumer preparation: numerics then C1 B/A/B' -- \
  bash probes/run_ar_consumer_campaign.sh
```

The campaign verifies an idle service before stopping it. Its own GPU probe
containers stop on failure. It stops its loopback serving before releasing
the hold so public-port admission cannot mistake it for an unfinished boot.
The fleet supervisor owns approved-main recovery
or a validated transfer to the next boot job.

The probe first uses a deliberately delayed producer, then the real four-node
RDMA AllReduce. Peers receive a committed source archive in a fresh directory
and attest the source bytes; they do not need a pre-existing Git checkout.
Each run tests six token counts, two weight storage paths and
three changes behind fixed CUDA graph pointers: 36 cases, six outputs each,
with exact baseline/candidate comparisons. Independent CPU AR sums and FP64
MHC equations provide separate oracles. Both modes also run memcheck and
racecheck. C1 segment samples include warm and cold L2; they cannot establish
engine-step speed by themselves.

After the GPU gate the campaign uses the same deployed source for defaults,
candidate, defaults, retaining standard onepass quality, decode-window steps,
three fixed-length 2048-token requests, prefill contexts and SSE channels.
All four ranks must prove source hashes, flags and actual graph capture.
Raw receipts live under `/home/choiceoh/glm53-logs/ARCONSUMER-<session>/`.
