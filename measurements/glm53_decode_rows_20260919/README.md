# GLM-5.3's decode-row GEMMs against Qwen3.8's kernels — one GB10, 2026-09-19

> 그대로 두는 기록 — 이 날 srv4 단일 GPU 레인에서 잰 숫자와 그 원시 보고. 고치지 않는다.

Two single-GPU lane tickets beside production (srv4, `st-glm53` serving), probe `probes/engine_glm53_decode_rows.py`
from #1251 at main `1d268d1a`, synthetic weights of the served shape and format (deep_gemm's block-128 pack), arms
interleaved inside CUDA graphs so production's own steps land on every arm alike.

| ticket | lane | what |
|---|---|---|
| `glm53-head-0919a` | `--lanes glm53_head` | the rank's vocabulary head, 38,720 × 4,096 e4m3 (159 MB) |
| `glm53-gemv-0919a` | `--lanes glm53_gemv` | skinny_gemv against torch.mm at the step's BF16 GEMMs still on cuBLAS |

## Head, µs a call (median of 9 rounds, 8 calls a graph)

| rows | read only | verify: cuBLASLt (served) | verify: fp8_rows | **W8A16** | draft: cuBLASLt MX (served) | draft: fp8_rows | deep_gemm |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 706.2 | 891.0 | 736.4 | **733.1** | 853.0 | 747.2 | 927.0 |
| 8 | 703.5 | 905.9 | 739.6 | **731.3** | 859.4 | 740.7 | 925.9 |
| 14 | 693.8 | 929.0 | 742.6 | **734.0** | 850.4 | 738.7 | 937.2 |
| 16 | 699.6 | 911.0 | 738.9 | **742.9** | 908.3 | 743.2 | 935.6 |

Largest error against the BF16 product over its largest magnitude: 0.036–0.043 for every arm that quantizes the rows
(the served readers, fp8_rows, deep_gemm), **0.026–0.029 for W8A16**, which does not. Argmax agreement with the BF16
product over 7–16 random rows moved together across arms (1.0 / 0.75 / 0.93 / 1.0) and is too few rows to rank them;
fp8_rows' tile sweep at K = 4,096 put every tile within 737–757 µs, the served tile (32, 4, 4) among them.

GLM serves `MAX_SEQS = 2` and `SPEC_K = 7`, so a verify step reads the head at 8 or 16 rows and a draft pass at 7 or 14:
every served head call is in the lane's range. Against the served readers W8A16 is about 160–195 µs a call faster for
verify and 115–165 for draft — about 0.3 ms a step at C = 1 and 2 on a ~48 ms step, a component figure and not a fleet
measurement (D17).

## skinny_gemv against torch.mm, µs a call (weights rotated past the L2)

| shape | rows | torch.mm | best configuration | speedup |
|---|---:|---:|---:|---:|
| indexer wk+gate [256, 4096] | 8 | 13.70 | 11.88 `(32, 256, 4, 4, 3)` | 1.15 |
| | 16 | 13.85 | 12.25 | 1.13 |
| drafter context K/V [2560, 4096] | 8 | 96.49 | 92.98 `(64, 256, 1, 8, 3)` | 1.04 |
| | 16 | 97.27 | 93.93 | 1.04 |

Eleven DSA layers a step make the indexer pair about 19 µs a step at C = 1 — not wired: the split configuration needs a
`prepare` before capture in GLM's boot, a step for 0.04%.

## What came of it

The head's decode rows take W8A16 (`FP8Linear(decode_rows="w8a16")`, engine/kernels/dense), both the verify and the
draft call, by the operator's choice of 2026-09-19 (the verify head's logits change toward the BF16 product). The
cuBLASLt reader keeps batches past 16 rows and is the boot's cross-check (`qualify_decode_rows`).

Raw reports: `glm53-head-0919a.json`, `glm53-gemv-0919a.json` (the probe's `--output`, copied back by the lane).
