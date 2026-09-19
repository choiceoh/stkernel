# What the seed image carries for the sm_121a intake — one GB10, 2026-09-19

> 그대로 두는 기록 — 이 날 srv4 단일 GPU 레인에서 읽은 것과 그 원시 보고. 고치지 않는다.

One single-GPU lane ticket beside production (srv4, `st-glm53` serving), probe `probes/engine_sm121_inventory.py`
(`--lanes sm121_inventory`) from branch `sm121-intake` at `0c70936a`. It reads the seed image's installed flashinfer —
file names, a few words in candidate sources, imports and public signatures — and compiles or launches nothing.
Raw report: `sm121-inventory-0919a.json`.

| ticket | lane | what |
|---|---|---|
| `sm121-inv-0919a` | `--lanes sm121_inventory` | engine/SM121_INTAKE.md U0 |

## Toolchain in the container

GB10, capability (12, 1). torch 2.13.0+cu132 (arch list sm_80 … sm_120: no sm_121 SASS of its own; sm_120 cubins run on
12.1), triton 3.7.1, flashinfer-python 0.6.18.dev20260819 at `/usr/local/lib/python3.12/dist-packages/flashinfer`,
nvidia-cutlass-dsl 4.6.2, tilelang 0.1.12, nvcc `/usr/local/cuda-13.2/bin/nvcc`. `deep_gemm` is not an installed
distribution (the engine carries it, `engine/kernels/SOURCES.json`).

## Per intake item: what is already there

| item | in the image | imports |
|---|---|---|
| U3 MoE MXFP4 W4A8 | `fused_moe/cute_dsl/fused_moe_mxfp8_mxfp4.py` (`CuteDslMxfp8Mxfp4MoEWrapper`, MXFP8 activations × MXFP4 weights) and `b12x_moe.py` (`B12xMoEWrapper`, mentions mxfp4) | yes |
| U5 dense NVFP4 GEMM | `gemm/kernels/dense_blockscaled_gemm_sm120_b12x.py`, `gemm/gemm_mm_fp4_cute_dsl.py`, `gemm/gemm_bf16_fp4*.py`, `cute_dsl/blockscaled_gemm.py`; `gemm/__init__.py` names b12x (4) and mm_fp4 (3) | yes |
| U6 sparse MLA on SM120 | `mla/_sparse_mla_sm120.py` + `data/csrc/sparse_mla_sm120{,_prefill,_decode_dsv3_2,_decode_dsv4}.cu`; `cute_dsl/sparse/bsa_attn_sm120.py` | yes |
| U7 GQA paged decode | `decode.py` (`BatchDecodeWithPagedKVCacheWrapper`; `window_left` ×89, `sinks` ×28, `logits_soft_cap` ×82), `cute_dsl/attention/gqa_decode{,_paged}.py` | yes |
| U8 KV quantization | `decode.py`: `float8_e4m3fn` ×13, `nvfp4` ×34, `kv_cache_sf` ×31 | yes |
| U9 FlashKDA | `kda.py`, `kda_prefill.py`, `kda_decode.py` + `data/csrc/kda/flashkda_*.cu`, `cake_flashkda_*.cu` (79 files) | not imported by this lane |
| U13 GDN prefill | `gdn_prefill.py` (`chunk_gated_delta_rule`), `gdn_kernels/delta_rule_dsl/delta_rule_sm120.py` (`delta_rule_prefill_dsl`) | yes |

`engine/kernels/b12x` against the image's `fused_moe/cute_dsl/blackwell_sm12x`: 9 files the same bytes, 9 differ
(`__init__`, `_moe_dynamic/gated`, `_moe_dynamic/generic`, `moe_direct_micro_kernel`, `moe_dispatch`,
`moe_dynamic_kernel`, `moe_micro_kernel`, `moe_static_kernel`, `moe_w4a16_fp4_helpers`), 28 only in the engine, none
only in the image — the engine's b12x is a fork of the image's, not behind it.

## What came of it

Every item above binds a kernel the image already has instead of vendoring one (engine/SM121_INTAKE.md, the
"이미지" column); each item's own ticket compiles it on sm_121a and judges it against its engine/modules oracle. One
caution travels with U6: vllm#54929 reports that FlashInfer's SM120 sparse MLA livelocks under sustained load (GPUs at
100 % with no output) — the in-image kernel is a candidate to judge under load, not an answer.
