# One launch for draft Q/K norms and rotary

Base: `f8fface71d153ea889416297c1ed3d880e67a641` (#934).

The fast drafter now sends its Q and K heads through one combined program grid.
Each program calls the existing `_norm_rope` body for one head. The only new
single-kernel parameter is a constant head offset, defaulting to zero; the
reduction, normalization roundings, weight multiply and rotary arithmetic body
are unchanged. `source.json` compares that body with the base AST.

The model-free lane is bound through `engine/base/lanes.py`, and both the single
and batched fast drafter paths use it by default. Q/K remain strided views of
their projection; the two outputs remain independent contiguous allocations.
There is no new persistent buffer, projection copy, collective or precision
setting. The reference drafter and observation-only K norm keep their calls.

## Source call counts

Five draft layers previously issued ten Q/K norm-RoPE calls per proposal; they
now issue five. This removes five launches per proposal at both C=1 and C=4.
It removes twenty if all four bounded decode iterations execute. These are
source counts, not measured latency or a prediction of consumer throughput.

The installed drafter config was read on srv2: five layers, 32 query heads and
8 KV heads before TP4, head dimension 128, epsilon `1e-5`, theta `10000`.
The GPU replay probe uses those values and K=7/T=8 row counts.

## Completed CPU evidence

- `cpu-tests.log`: 48 tests, 29 passed and 19 GPU-only skips, 7.626 seconds.
  The tests include the actual fast single/batched attention wrappers, unchanged
  packed V delivery, independent output ownership and common-lane boundaries.
- `interpreter.json`: four FP32 CPU cases match the independent single-head
  calls exactly, including packed input strides, separate Q/K head indices and
  unchanged input storage. This is address/arithmetic evidence, not BF16 CUDA
  evidence. The final interpreter run uses the model's epsilon `1e-5`.
- `compile.json`: both the TP4 and full-head BF16 kernels compile for SM121.
  They use 16 bytes of shared memory. This is compilation, not GPU execution.

All CPU checks used the ARM64 image
`sha256:09d9ba96a4c7e1113f91100b892a94c1ab859dae8e46db3e7b02dfa2564f93bc`,
runc, one CPU, 2 GiB, no network and no visible NVIDIA/CUDA device.

```sh
python3 -m unittest -v tests.test_engine_draft_qk tests.test_engine_norm_rope tests.test_engine_drafter tests.test_engine_draft_attention tests.test_engine_decode_buffers tests.test_engine_kernel_common
TRITON_INTERPRET=1 python3 probes/engine_draft_qk.py --mode interpreter --output interpreter.json
python3 probes/engine_draft_qk.py --mode compile --output compile.json
```

## Existing GPU reservation

`dsa_inputs` includes `draft_qk_pair`: same-build BF16 comparison with the
separate calls, changed/poisoned graph replay at 1/8/16/24/32 rows, context
changes through 32K/128K and rollback, and bounded B/A/A/B component timings.
It uses synthetic projection values and norm weights; it loads no extra model
weights. Replace `st-dsa-lengths0914` while preserving its five-minute estimate
and original queue age. No additional reservation or model boot is needed.

GPU numerical/replay results, component latency, consumer step/s, tok/s,
tokens/step and acceptance remain pending. Final consumer coverage remains
matched 32K/128K, C=1 twice and C=4 once.
