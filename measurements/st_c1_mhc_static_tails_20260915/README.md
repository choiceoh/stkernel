# C=1 MHC static tail ownership (2026-09-15)

## Candidate and current gate

The packed C=1 consumer has eight rows and 48 resident CTAs: three groups of
16 hidden-dimension chunks. Two groups process three rows apiece and the third
processes two. The candidate gives the third group's first eight CTAs one
tail each. Phase-one projection, per-token chunk arrivals, Sinkhorn, mixing,
normalization, rounding and the PDL dependency are unchanged.

Each launch removes 56 shared tail-ticket and 48 exit-ticket atomic updates.
This is a count from the source, not a measured latency claim. The 128 chunk
arrivals and each token's wait/reset remain. The native host gate requires
hidden size 4096, eight rows, lossless BF16 coefficients, an AR or direct-packet
consumer, and exactly 48 resident CTAs. All other shapes keep dynamic tails.

Source `f81800a0` on base `5871c559` keeps the production default dynamic.
Native `tail_mode=0/1/-1` selects dynamic/forced-static/shape-gated-auto for
same-build measurement. The default will change only after qualification.

## Evidence

- CPU native compile/load: **PASS**, `compile.json`. CUDA was hidden; both new
  native specializations were found, with 128 registers, 16 stack bytes,
  28,720 shared bytes and zero local spill allocation reported by cuobjdump.
- CPU tests: **37 passed, 8 GPU-only skipped** (`cpu-tests.log`, 45 total).
  The read-only task checkout was mounted at `/repo`, with `--workdir /repo`
  and `PYTHONPATH=/repo`; CUDA was hidden.
- GB10 exactness and component timing: queued as `c1mhc-tails-a55c`, ticket
  `17894505191390768`.
- TP4 transport and consumer performance: not measured yet.

## GPU gate

The probe loads all 90 real BF16-origin MHC coefficient tensors and measures
the 89 carried boundaries. It tests ordinary AR inputs and local four-rank
packet descriptors. Four magnitudes, forward/reverse replay and poisoned
outputs compare all four output fields bitwise against the original dynamic
kernel. Rank-fold and rounding canaries and descriptor rebinding exercise the
packet path. A mixed 1/7/8/16-row schedule checks 32 iterations of counter
rearming and automatic fallback; forced static tails reject unsupported rows.

Timings use external CUDA events captured around native calls, two independent
captures in opposite allocation order, four B/A/A/B brackets, and 16 replays
per bracket arm. Warm single-layer intervals repeat 32 calls; a distinct-layer
chain visits all 89 coefficient packs. Evicted intervals follow a 128 MiB
flush outside the timed interval. Local packet timings do not include network
transport and cannot establish serving throughput.

```sh
bash probes/run_engine_probe.sh probes/engine_kernel_check.py \
  --lanes mhc_c1_tails --samples 4 \
  --ranks /home/choiceoh/models/st-glm53-9391-up-gate-full/rank3of4.safetensors \
  --output /cache/c1-mhc-tails-a55c.jsonl
```

Runtime: `sha256:848e493f37af252865deea2fe6169916f6bac727343b5ab592cd74fcf3639544`,
Torch 2.13.0+cu132 / CUDA 13.2. Native source SHA-256:
`b3cd1b099ecf5e984cddcf712dbc751962c755dafea391f5623a6deebf85fdb4`.
GPU work runs only through the canonical fleet queue. Final adoption requires
the full TP4 onepass under `engine/CHARTER.md` D17.
