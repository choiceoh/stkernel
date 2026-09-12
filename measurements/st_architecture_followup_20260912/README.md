# Architecture follow-up while the GPU queue is occupied

## PR 760 tools used

The merged `step_sim`, `step_replay`, and `step_peek` tools were exercised.
These are host simulation, saved-record analysis, and read-only observation;
none establishes a serving improvement.

- Host simulation used draft width 6, token budget 6912, alignment 768, C=1,
  prompts 2K/32K/128K, and 256 generated tokens. On this Mac the Runner's
  decode median was 0.041 ms and prefill-chunk median 0.043 ms. This omits
  actual CUDA dispatch, communication, tokenization, and model execution.
  The NullModel's acceptance behavior is not the served model's acceptance.
  The original ring and `host-loop-replay.txt` preserve that narrower result.
- `onepass-replay.txt` regenerates 10.9625 median window step/s and 3.30945
  tokens/step from `st_onepass_20260912_0746/result.jsonl`. The replay reports
  no engine shape, revision, or session identity, so this file cannot be the
  matched baseline for the new experiments. Its printed cold/warm labels
  do not by themselves prove that prefix-cache state was controlled.
- The first real `step_peek` attempt ran on srv2 against its local port 8000.
  Connection was refused; zero samples were obtained. A later attempt against
  the canonical `10.10.10.2:8000` address also refused the connection while
  container `st-glm53` was running image `st-engine:main-f838f71a` on the host
  network. A running container is not proof that its HTTP server is ready.
  No traffic was sent
  to an inference endpoint and no fleet lease was acquired.
- Inspection found `_buckets` ignored its label filter. Prefill buckets
  could therefore enter the decode quantile calculation, despite the count
  and sum being filtered. This is fixed and covered by a mixed-kind,
  overlapping-boundary regression. All 14 step-tool tests pass.

## FP8 prefill consumer experiment

The existing sequence-parallel transport sends 2048-element FP8 blocks with
FP32 power-of-two scales. Its all-gather consumer writes the full BF16
activation. A native large-M dense projection then reads that activation to
quantize 128-column groups for DeepGEMM.

`prefill_collectives/consumer.py` combines these two kernels. It explicitly
rounds the unpacked value to BF16 in registers before recomputing each
128-column group's scale, preserving the existing numerical boundary.
Simply reusing a transport scale would not preserve that recipe.

For a 6912-by-4096 activation, removing the full BF16 write and following
read removes 108 MiB of requested intermediate traffic per rank per use.
This is a byte count, not a measured bandwidth or latency gain. Network
traffic, wire quantization, and BF16 rounding are unchanged.

The first potential serving consumer is `kda.in_proj`: after that projection
the KDA block needs only the input's shape, dtype, and device. DSA also reads
BF16 through its indexer, and MoE reads it through routing and experts; they
are not eligible for this replacement. An active calibration observer also
requires the BF16 input. Integration must explicitly check those consumers
and the bound FP8 projection instead of changing all transport outputs.

`compile.json` records a successful SM121 compilation on Torch 2.13.0+cu130
and Triton 3.7.1 without initializing CUDA. The local admission/tool gate ran
39 tests successfully; the focused package-import check also passes.

The current implementation is an unconnected kernel experiment. Its GPU
gate compares FP8 bytes and FP32 scales against ordinary unpack+quantize,
including padded packets, zero rows, differing 128-column scales, and graph
replay with changed received data. Timing includes the entire two-kernel
baseline versus the fused boundary for global row counts 4096 and 6912,
with paired A/B and B/A samples. Communication and GEMM are excluded.

```sh
# CPU compilation; no CUDA context is created.
python3 probes/engine_prefill_fp8_consumer_check.py --compile-only \
  --output /cache/prefill-fp8-compile.json

# GPU gate goes through the canonical fleet queue.
bash bench/fleet.sh run --gpu --detach st-prefill-consumer 5 \
  "FP8 packet consumer correctness and paired boundary timing" -- \
  bash probes/run_engine_probe.sh probes/engine_prefill_fp8_consumer_check.py
```

The KDA deferred-state reservation remains frozen in its original remote
checkout; these additional changes do not replace its queued source.
