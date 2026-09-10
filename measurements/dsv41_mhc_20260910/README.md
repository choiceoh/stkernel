# Shared MHC / DeepSeek V4.1 preparation — 2026-09-10

Implementation candidate, **not a serving or performance acceptance result**.
Base: `313e36c2ad915edbbd7823fdc4ec9b2a1af62e08`.

## Change

The existing common CUDA MHC implementation supports H4096/H5120 with HC4
and T1–32. H5120 uses twenty chunks instead of sixteen. The new V4.1 contract
shares the implementation but explicitly consumes the previous sublayer's
pre coefficients, returns current pre coefficients, and observes both BF16
boundaries before projection and RMS statistics. New paths skip weight loads
for token groups with no work.

The GLM 4096 legacy contract, serving admission, BF16-weight/AR gates, GEMM,
MLA, and model profiles are retained. Additional contracts use FP32 weights
only and are available through the probe API; no V4.1 serving hook is enabled.
MHC launches must remain serialized because device ticket counters are shared.

## Reference and runtime inventory

Read-only inventory of srv4 `/home/choiceoh/models/DeepSeek-V4.1-Flash`:

- HF revision: `fb2764a5cf321eaa5070ca8f9e892818f477c16d`.
- `config.json` SHA256:
  `8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879`.
- `inference/model.py` SHA256:
  `4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65`.
- `inference/kernel.py` SHA256:
  `1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455`.

`Block.hc_mixes/hc_pre/hc_post/forward`, `RMSNorm`, and
`hc_split_sinkhorn_kernel` define the arithmetic reference. The source has
hidden5120, four HC streams, twenty Sinkhorn iterations, norm epsilon `1e-20`,
HC epsilon `1e-6`, and `2 * sigmoid` post scaling. The current sublayer consumes
the previous sublayer's pre mix. The legacy fused contract differs at both
rounding boundaries and in coefficient ownership.

At inventory time, 46/48 weight shards existed, totaling 307,223,630,736 bytes,
versus 510,286,023,000 bytes in the index. Missing shards 47/48 contain Engram
weights. The downloader was still running; shard count is not completion
proof. No V4.1 profile or confirmed compatible serving image was present.
This is a timestamped inventory, not a statement about future availability.

## Validation

The reproducible checks are:

```sh
python3 -m unittest discover -s tests -p 'test_megakernel_mhc_geometry*.py'
python3 -m unittest discover -s tests -p 'test_megakernel*regressions.py'
python3 tests/test_logic.py --component core
```

Arithmetic tests require a CPU PyTorch installation; skips do not establish
their result. The CUDA source-extracted CPU checks exercise chunk coverage,
arrival publication, host argument admission and the changed rounding seams.
They do not execute device code.

`cpu_compile_runner.py` compiles the **whole** CUDA translation unit in a
fresh pinned container with no GPU devices, through normal `fleet --cpu`.
The inner `mk_mhc_geometry_bench.py --compile-only` checks both native exports
and that CUDA was never initialized. No model or serving package is loaded.
Compile success alone does not establish GPU numerical correctness.

The GPU probe contains H4096/H5120 × T1/5/6/8/10/12/16/17/24/32 for both
contracts, plus zero and near-zero V4.1 inputs. It checks all outputs against
separate PyTorch references, input immutability, A→B→A graph replay, exact
same-input replay and interference from independent GEMM work. It enforces
both pooled and worst-token relative error ≤1e-3; no weakened tolerance is
accepted. No concurrent MHC launch is claimed safe.

GPU execution remains pending: current fleet admission rejects standalone
component probes. No queue policy was weakened or bypassed. Whole-model
tok/s, TTFT, decode step/s, output quality and default promotion additionally
require the completed checkpoint and a verified compatible runtime.

Validation receipts and their exact tested source will be recorded below.
