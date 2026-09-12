# Terminal mHC consumer and direct drafter feature packing

GLM's five drafter feature layers (4, 13, 23, 32, 41) each run
`mhc_post(...).float().mean(1).to(BF16)`. The forward pass then concatenates
their outputs. The final hidden state uses the same contraction before its
existing RMSNorm. The engine owns both the mHC carry and these output uses,
so a terminal consumer can produce the required mean directly into the
final feature tensor's column views. This is a model-specific opportunity
for ST; it does not imply that another engine cannot implement it.

`engine/kernels/mhc_contract.py` implements that boundary as one kernel per
feature. It loads the input and four residual channels once, computes the
four post-map results in the same FMA order as the served TileLang kernel,
rounds **each channel** to BF16, then computes the FP32 mean and writes BF16.
Folding the channel coefficients before their BF16 rounding is incorrect.

The supplied Torch 2.13 runtime's CUDA reduction uses serial accumulator
combination for this four-element, non-fastest-dimension mean. The probe
records that runtime header's hash. The GPU gate includes large cancelling
channels that distinguish serial from tree summation, as well as a canary
that fails if the four BF16 roundings move after the mean. These are exact
byte comparisons against the served TileLang and Torch operations.

The writer accepts an explicitly owned output column view with contiguous
columns and a wider row stride. Five calls can fill `[N,5*4096]` directly,
eliminating the final concat. Input aliases and overlapping output rows are
rejected; neighbour columns, padding and inputs must remain unchanged.
Changed-input graph replay must update every feature without reallocation.

## Scope and byte count

The intermediate BF16 post result, its FP32 conversion and FP32 mean account
for `(8 + 16 + 4) * N * H` bytes of storage written and then read. The fused
consumer avoids `56 * N * H` requested bytes per feature. Direct packing
also avoids reading and writing the concatenated feature data.

| Consumer | Local rows | Requested intermediate traffic removed |
| --- | ---: | ---: |
| One terminal contraction | 7 | 1.53125 MiB |
| Five features including concat | 7 | 8.203125 MiB |
| One terminal contraction | 1728 | 378 MiB |
| Five features including concat | 1728 | 2025 MiB |

1728 is a rank's token shard for the existing TP4 6912-token prefill chunk.
These are tensor byte counts, not measured DRAM traffic, memory savings or
latency. Cache residency and kernel occupancy can change the outcome.
The independent final contraction still feeds the existing RMSNorm.

The implementation remains unconnected to serving pending GPU validation.
An auxiliary feature read must not replace the residual carry needed by
the next layer. Serving integration should allocate the final feature tensor
once, fill columns in the existing `aux_layers` order, then preserve the
existing sequence-parallel gather. Probe callbacks and `finish=False` keep
their full carry semantics.

## Reproduction and remaining gate

```sh
# Same ST image, with no GPU device or CUDA context.
python3 probes/engine_mhc_contract_check.py --compile-only \
  --output /cache/mhc-contract-compile.json

# After the operator's priority window, through the canonical fleet queue.
bash bench/fleet.sh run --gpu --detach st-mhc-contract 5 \
  "Terminal mHC correctness and paired feature-assembly timing" -- \
  bash probes/run_engine_probe.sh probes/engine_mhc_contract_check.py
```

The GPU gate must pass all five tests without skips before timing. It covers
rows 1, 6, 7, 8, 65, 1728 and 6912; BF16 rounding and cancellation; strided
feature destinations; source/guard preservation; and changed-input replay.
Timing compares complete baseline contraction plus concat against direct
feature writing, using 20 alternating A/B and B/A samples per case in warm
and 64 MiB cache-evicted regimes. Model execution, communication and RMSNorm
are outside this measurement. Actual serving claims still require matched
onepass quality, output hashes, tok/s and TTFT with two runs per boot.

`compile.json` records both successful SM121 scalar-stride variants on
Torch 2.13.0+cu130 and Triton 3.7.1, without initializing CUDA. All seven
source hashes match the admitted tree. The ST-image package/audit check
passed 10 CPU tests and skipped the five GPU-only tests. Local package,
admission and step-tool checks passed 40 tests; after merging current main,
the audit/onepass/ledger check passed 39 tests. These suites overlap.

The code CI passed in run `34701264738`, including the engine suite and
onepass recording/consumer contracts. The previous failure in run
`34700561014` was the fleet audit hash during main's onepass-budget change;
merging main includes its reviewed hash repair from PR #764.

At 2026-09-12 15:10:56 UTC, KDA's operator-controlled reservation had resumed
as queue #1. This candidate was accepted as `stmhc-contract0913`, ticket
`17892258362318595`, after FP8 consumer ticket `17892258232315496`. Both
new probes use frozen source `25d2b730` in their own checkout and image
`st-engine:main-ff728f43`. KDA's original checkout remains at `19113939`.
No GPU result exists yet. Admission identities and reproduction commands
are recorded in `provenance.json`; queue positions are only that observation.
