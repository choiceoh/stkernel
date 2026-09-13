# Token-major MLA prefill contractions

The two existing MLA einsums store their results in head-major order. Query
attention then calls `q_abs.contiguous()`, and output projection calls
`o.reshape(T, H*v_dim)`: both need a full tensor copy. This candidate computes
the same two contractions directly into fresh, contiguous token-major storage.
It applies throughout eligible long-prefill chunks, beyond the 2,051-token
dense prefix introduced by #887.

**Operator enablement:** the user explicitly requested immediate enablement
after reviewing the candidate. Production and experimental defaults are now
**`prefill_absorb_tiles=1`**. Experimental rollback is `STK_prefill_absorb_tiles=0`;
production rejects overrides. This changes selection, not GPU qualification.
The default-on evidence is retained under `default-on/`.
The operator-enabled `prefill_dense_prefix=1` remains on; indexer query sharding
remains off by default. No GPU queue submission, engine build/boot or GPU launch
was performed for this work.

## Implementation and boundaries

- Query: `[T,16,256] @ [16,256,512] -> [T,16,512]`.
- Output: `[T,16,512] @ [16,256,512].T -> [T,16,256]`.
- Read existing BF16 `kv_b` slices, including their head stride and output
  slice storage offset. There is no persistent weight repack.
- Preserve BF16 operands, FP32 accumulation and both BF16 output boundaries.
  The GPU GEMM reduction order changes and remains numerically unqualified.
- Route only a single, noncaptured, nonprobe step with 128–32,768 rows.
  The native wrapper additionally requires TP4 GLM geometry and eager CUDA.
  Short tails, decode, capture and multi-segment steps use the existing einsums.
- Runtime proof requires both query and output contractions on every DSA layer.
  Outputs own their storage; there is no shared output scratch or extra collective.

## CPU and compiler evidence

The existing ST image was used in a network-disabled runc container with no
GPU devices: CPU=2, memory/swap=4 GiB, pids=256, `NVIDIA_VISIBLE_DEVICES=void`,
`CUDA_VISIBLE_DEVICES=''`. The checkout was mounted read-only. Image:
`sha256:f85de49afc0a41596cce3df2dab11af992a9aa5d21129c0f50ba719c30f68781`.

At `29369ad4`, **49 CPU tests passed, zero skips**, 22.442 s (`cpu-tests.log`):

```sh
python3 -m unittest -v \
  tests.test_engine_prefill_absorb_tiles tests.test_prefill_dense_prefix \
  tests.test_prefill_indexer_shards tests.test_engine_execution_plans \
  tests.test_engine_native_execution tests.test_engine_knobs tests.test_engine_kda_ring_bench
```

The actual Triton body runs through checked CPU loads/stores, compared against
independent einsums. Cases cover native 16-head 256/512 geometry, both weight
slices, 131/132 rows, ragged reduction/output dimensions, output sentinels and
input/weight immutability. Relative output error is bounded below 0.0005 in these
CPU cases. The actual wrapper is checked for launch arguments, fresh contiguous
outputs, invalid layouts/dtypes and captured execution rejection. Its test file
is included in the normal `tools/check.py` CI selection.

TP4 model checks use the semantic reference lane, with dense prefix and indexer
sharding enabled on both arms. Hidden/auxiliary output, KDA/KV state and the next
seven-token decode match exactly for 131/132-row and tiled 259-row prefill. These
are model composition checks, not GPU numerical or language-quality proof.

The initial `fc2167b8` CPU run had one failure: the existing explicit knob-list
test needed the new option registered. This was fixed in `29369ad4` and is retained
in `cpu-tests-r1.log`. Merge `9193d7c2` incorporates main `6e08213f`, including
the draft precision changes. Focused post-merge validation ran 45 tests in 6.687 s:
**42 passed, 3 GPU-only checks skipped** (`cpu-main-merge.log`). The contraction
kernel, model routing, lane binding and execution plan are unchanged by that merge.

`compile_kernel.py` compiled the actual body for CUDA SM121 using the retained
checkpoint config in `../st_prefill_dense_prefix_20260913/model-config.json`.
At `fc2167b8`, all **8 variants compiled in 2.209 s**, with CUDA uninitialized.
`compile-r1.json` retains model/source hashes and resource reports; PTX/cubin
remain under `/home/choiceoh/glm53-logs/st-prefill-absorb-tiles-20260913/compile-r1`.

| Selected tile M/N/K, warps, stages | Registers | Stack/local bytes | Dynamic + static shared bytes |
|---|---:|---:|---:|
| Query 64/64/32, 4, 2 | 128 | 0 / 0 | 8,192 + 1,024 |
| Output 64/64/32, 4, 2 | 151 | 0 / 0 | 8,192 + 1,024 |

The other compiled geometries use up to 255 registers. They were not selected
without timing evidence. Compiler resources do not prove an improvement over
the existing cuBLAS/CUTLASS contractions.

## Removed explicit copy work

The existing head-major output layout and both materializations were reproduced
in Torch in the ST CPU image. `work_budget.py` derives dimensions/layer count from
the checkpoint config and counts eligible rows at the current chunk size 32,256.

| Input tokens | Copy reads + writes removed per rank across 11 DSA layers |
|---|---:|
| 2,000 | 1.007 GiB |
| 2,672, historical short prompt | 1.345 GiB |
| 32,000 | 16.113 GiB |
| 128,000 | **64.453 GiB** |

At a full chunk, the individual query and output copy allocations are 504 MiB
and 252 MiB. These occur at different points; their sum is not a measured peak
memory saving. The byte counts are explicit tensor-copy payload, **not measured
DRAM traffic, a TTFT reduction or a speed forecast**. New persistent bytes and
collectives are both zero. A final tail under 128 tokens remains on the old path.

## Upgraded Oracle #875

Oracle `e2bfbb9a` compared main `6e08213f` with candidate `9193d7c2`:

```sh
python3 /path/to/pr875/bench/storacle.py compare --tree /path/to/this/tree \
  --base 6e08213f --candidate 9193d7c2 \
  --config measurements/st_prefill_dense_prefix_20260913/model-config.json \
  --ctx 2000,32000,128000 --width 1 --set prefill_absorb_tiles=1 --json
```

`oracle-pr875.json` confirms the new execution switch, chunk=32,256 and zero
resident cache/state-layout deltas. Dense-prefix remains on and indexer sharding
off on both arms. **The prefill timing delta is null at every length** because
there is no matching kernel timing. The unchanged modeled subtotal is not a
prediction of equal performance. Drafter geometry remains the Oracle reference
assumption. The paired profile template binds the source/settings/model; blank
timings cannot serve as measurement evidence.

GPU numerical checks, real-weight quality/acceptance, fixed 1,024-token decode
and profiler-off, zero-prefix-reuse C=1 TTFT remain pending. The next consumer
targets are 2K >=3,300 tok/s and 128K >=4,000 tok/s; neither is claimed here.
