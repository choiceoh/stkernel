# Qwen3.8 QSA prefill on the tile-union launch — one GB10, 2026-09-19

> 그대로 두는 기록 — 이 날 srv4 단일 GPU 레인에서 잰 숫자와 그 원시 보고. 고치지 않는다.

engine/SM121_INTAKE.md U12: vLLM PR 55430's tile-union prefill attention (`engine/kernels/qsa_tile_union.py`, Apache-2.0,
provenance in `engine/kernels/SOURCES.json`) against the served split-K launch (`qsa.qsa_sparse_paged_attention_blocks`)
at Qwen3.8's per-rank cell (6 query heads over one KV head of 256, pages of 768, the 2,048-token budget, ratio 4).

| ticket | branch @ commit | lane | what |
|---|---|---|---|
| `sm121-u12-0919c` | `sm121-u12-tile-union` @ `89d16d7e` | `--lanes qwen38_tile_union` | the gate, then timings (`probes/engine_qwen38_qsa_geometry.tile_union_arm`) |
| `sm121-u12cells-0919e` | `sm121-u12-tile-union` @ `8ef36903` | `--lanes qwen38_cells` | the boot's `lanes.qualify` with `qsa_tile_union.qualify`, and the GPU glue cases (91 tests, `TileUnionTests` among them) |

## Gate

Every case within the served sparse attention's band (two BF16 steps at the largest value, one in rms): largest error
over the largest value 0.0046–0.0066, rms 0.0010–0.0011, max |diff| 0.00024–0.00037. The boot's qualification
(1,024 rows, 12,000 positions deep): largest 0.005882, rms 0.000622. 91 GPU glue tests passed, none skipped.

## µs a launch (median / minimum of 7, beside production)

| rows | context | selection (Jaccard of neighbours) | split-K | tile-union | shared layout | union / split-K |
|---:|---:|---|---:|---:|---:|---:|
| 1,024 | 8,192 | prefill (0.975) | 4,808.7 / 2,941.3 | 1,753.8 / 1,743.8 | 1,622.8 / 1,619.0 | 0.36 / 0.59 |
| 1,024 | 28,000 | prefill (0.918) | 4,801.9 / 4,790.5 | 1,789.8 / 1,776.2 | 1,670.1 / 1,651.6 | 0.37 / 0.37 |
| 4,096 | 8,192 | prefill (0.965) | 19,067.0 / 17,624.0 | 14,119.4 / 12,706.8 | 13,900.2 / 12,531.7 | 0.74 / 0.72 |
| 4,096 | 28,000 | prefill (0.910) | 18,990.0 / 17,616.2 | 14,180.2 / 12,569.5 | 14,151.7 / 12,549.2 | 0.75 / 0.71 |
| 1,024 | 8,192 | independent (0.133) | 4,771.6 / 3,131.2 | 5,420.2 / 5,391.7 | 5,287.1 / 3,729.5 | 1.14 / 1.72 |
| 1,024 | 28,000 | independent (0.038) | 4,782.9 / 3,739.9 | 5,723.1 / 5,697.4 | 5,652.5 / 5,606.8 | 1.20 / 1.52 |
| 4,096 | 8,192 | independent (0.113) | 18,655.2 / 16,618.2 | 21,538.4 / 21,281.8 | 21,852.8 / 21,435.5 | 1.15 / 1.28 |
| 4,096 | 28,000 | independent (0.035) | 19,108.4 / 9,106.7 | 24,056.3 / 22,546.7 | 24,201.7 / 22,371.6 | 1.26 / 2.48 |

The split-K launch's minimum and median part by up to 2x in some rows (production's steps landing on it); the
tile-union launch's barely move. A prefill's neighbouring rows choose alike (vLLM measured Jaccard ~0.9 at 8k; the
"prefill" rows here are a segment-wide score plus a little of each row's own), and there the union is 1.7–2.7x faster at
1,024 rows and 1.33–1.41x at 4,096. Where neighbours choose apart it is slower.

## What came of it

On by default for Qwen3.8's prefill steps that `qsa_tile_union.admits` takes (at least 1,024 rows, 64 a segment on
average), by the operator's decision of 2026-09-19 ("빠른건 기본에 켜"), with the boot's qualification first and
`--no-tile-union` / `ST_QSA_TILE_UNION=0` the rollback (engine/SERVING_DEFAULTS.md). A step whose first rows the budget
covers (the chunk across the reach) keeps its covered + split-K pair. **Fleet onepass not measured** (D17).

Raw reports: `sm121-u12-0919c.jsonl`, `sm121-u12cells-0919e.jsonl`.
