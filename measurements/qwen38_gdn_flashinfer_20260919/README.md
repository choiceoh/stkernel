# Qwen3.8's long prefill GDN on FlashInfer's SM120 kernel — one GB10, 2026-09-19

> 그대로 두는 기록 — 이 날 srv4 단일 GPU 레인에서 잰 숫자와 그 원시 보고. 고치지 않는다.

engine/SM121_INTAKE.md U13: `flashinfer.gdn_prefill.chunk_gated_delta_rule` (the seed image's flashinfer
0.6.18.dev20260819, SM120 CuTe-DSL delta rule) behind the served GDN chunk lane's contract
(`engine/kernels/gdn_prefill_sm120.chunk`), against the served `kda/chunk_decay.chunk_kda_with_decay`, at Qwen3.8's
per-rank cell (4 key / 12 value heads x 128) with a carried state.

| ticket | commit | lane | what |
|---|---|---|---|
| `sm121-gdndiag-0919d` | `sm121-candidates` `91cb92d3` | `sm121_gdn_diag` | the kernel as called first: NaN, twelve variants |
| `sm121-gdnnorm-0919f` | `sm121-results` `4cb480be` | `sm121_gdn` | q/k normalised first: correctness, kernel-only timings |
| `sm121-u13cells-0919g` | `sm121-u13-gdn` `f39bc495` | `qwen38_cells` | the boot's `lanes.qualify` and 93 GPU glue tests |
| `sm121-u13time-0919g` | `sm121-u13-gdn` `f39bc495` | `qwen38_gdn_flashinfer` | the whole lane against the served kernel, arms alternating in 9 rounds |

## Why it returned NaN, and what fixed it

Called as the served lane calls its own kernel (unnormalised q/k, `use_qk_l2norm_in_kernel=True`), every element of the
output was NaN in all twelve variants (value heads grouped 4/12 or widened to 12/12, the gate as alpha or as log decay,
the carried state, a zero state, the state's axes swapped). flashinfer#5255 (open, 2026-09-17): the native prefill path
ignores the flag, and unnormalised q/k grow the recurrence past fp32. With q and k L2-normalised before the call and the
flag False: no NaN, the output within 0.36–0.74% of the fp32 recurrence -- the served kernel's own 0.36–0.74%.

## The lane

`sm121-u13cells-0919g`: the boot's qualification (1,024 tokens, carried state, boundary states at 256 and 512 tokens)
o 0.003704, final state 0.006777, boundary states 0.005938 of the served kernel (band 0.015625); 93 GPU glue tests passed,
`GdnPrefillOnTheGpuTests` among them (4,096 tokens with boundaries every 768).

`sm121-u13time-0919g` (µs, median / minimum of 9, beside production):

| tokens | served chunk kernel | FlashInfer lane (normalise + copies + kernel) | speed (median) | o / state vs served |
|---:|---:|---:|---:|---|
| 1,024 | 935.2 / 910.4 | 683.4 / 667.1 | 1.37x | 0.0036 / 0.0056 |
| 2,048 | 1,087.8 / 1,063.7 | 755.0 / 746.4 | 1.44x | 0.0050 / 0.0053 |
| 4,096 | 1,857.2 / 1,844.5 | 913.7 / 890.2 | 2.03x | 0.0031 / 0.0055 |
| 8,192 | 3,591.8 / 3,579.3 | 1,438.1 / 1,420.6 | 2.50x | 0.0053 / 0.0038 |

Kernel only (`sm121-gdnnorm-0919f`, median / best): 128 tokens 322.0 / 86.0 against the served 121.8 / 106.4 -- slower,
hence `MIN_TOKENS` 1,024; 1,024 tokens 231.5 / 218.5 against 387.9 / 365.1; 8,192 tokens 993.7 / 871.9 against
4,245.0 / 4,001.2.

## What came of it

On by default for Qwen3.8's prefill segments of 1,024 tokens or more, by the operator's decision of 2026-09-19
("빠른건 기본에 켜"), qualified at boot, `--no-gdn-flashinfer` / `ST_GDN_FLASHINFER=0` the rollback
(engine/SERVING_DEFAULTS.md). **Fleet onepass not measured** (D17).
