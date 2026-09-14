# Packed decode replay inputs — CPU evidence

Base: `bcb2a09dec76d5bf6466a28f8496db1c9ee1e7ee` (PR #957 merged).

This change removes redundant metadata transfers and temporary storage in the
served decode paths. It makes no claim about measured step/s, token/s, prefill,
TTFT, or live acceptance. No GPU job was submitted or altered.

| Operation | Before | After |
| --- | --- | --- |
| Synchronous target metadata upload | 3 copies | 1 contiguous copy |
| Stochastic sampling policy upload | 4 copies | 1 contiguous copy |
| Burst reservation update | fresh pinned tensor + GPU temporary + H2D + D2D | reused pinned tensor + direct H2D |
| Greedy capture policy fields | 4 unused device buffers and their initialization | none |

Target token ids remain a separate copy. The asynchronous device-only target
fill keeps its existing four device inputs; the 3-to-1 reduction applies to the
synchronous host path. The sampling change affects stochastic replay; greedy
replay still uploads no policy and consumes no random draws. Burst reservation
staging adds one persistent `8 * max_seqs` byte host block in place of per-launch
host and device temporaries, with the existing one-live-burst retirement guard.

## Ownership and numerical contracts

Each live width uses the leading contiguous part of the host allocation;
there are no column-slice gaps at C=1, C=2 or C=3. Static device input addresses
stay fixed for every captured shape. Each target capacity keeps its own device
metadata. Host staging is reused after synchronous sampler readback, while the
asynchronous pipeline uses device-only inputs. Burst staging is not refilled
until the current burst has retired.

Top-k is an int32 view of packed 32-bit words, not a conversion to float. Other
sampling fields stay float32. Tests include top-k `2**24+1`, exact int64 request
ids above `2**33`, live-width churn, capacity switches, defaults after custom
policies, mixed greedy/stochastic rows, keyed uniforms and an excluded padded
vocabulary tail. Eager CPU sampling selects the same tokens from independent
field tensors. K=7, FP32 KDA and model/kernel arithmetic are unchanged.

## Validation

`cpu.log`: 86 tests considered, 80 passed and 6 CUDA-only tests skipped; no
failures. Tensor storage, aliasing, host fills, copy dispatch and CPU sampling
are real torch operations. Only graph execution and page locking are replaced
in the new CPU replay test. The existing burst oracle runs the production
launch/retire methods with CPU model and graph fixtures. Copy instrumentation
checks contiguous equal-size/equal-dtype sources and destinations, nonblocking
flags, copy counts and stable addresses. This is code/CPU evidence, not GPU
capture or throughput proof.

Executed on srv2 inside the existing image
`sha256:83080fb01fb9aab3efb34a364d045e9f50ce142885e8a1cc96dde1ca3aa0c052`,
with `--runtime=runc --network=none --cpus=2 --memory=4g --pids-limit=256`,
`CUDA_VISIBLE_DEVICES=` and `NVIDIA_VISIBLE_DEVICES=void`.
No image build, model boot or service restart was performed.

Command inside the image:

```sh
python3 -m unittest -v tests.test_engine_replay_metadata \
  tests.test_engine_graph_contracts tests.test_engine_burst_decode \
  tests.test_engine_pipeline tests.test_engine_sampling
```

`sources.sha256` identifies the exact tested source files. Existing CUDA
sampling/capture tests remain for an authorized GPU validation window.
