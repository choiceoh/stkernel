# Qwen3.8 PLE 게이트 한 발사 — 4,096 토큰 청크의 PLE 57.6 → 34.7 ms (2026-09-19)

srv4 단일 GPU 레인, 프로덕션 옆, `--lanes qwen38_prefill`(랭크 파일 실가중치의 작은 net 넷 → 48 층 청크로 풀이). **한 랭크의 디바이스 시간이다 —
플릿 프리필 주장은 아직 없다(D17: 다음 창의 onepass).**

| 티켓 | 트리 | 내용 |
|---|---|---|
| `q38pfbase-0919a` | main `7dcc0f00` | 기준 |
| `q38pfmix-0919a` | `28cc6bbf` | 기준 + `gated_residual.mix_block`(이 PR 에 없음) + `ngram_gate`(이 PR) |

PLE 몫은 층 세트끼리의 차로 풀어서(`[1]` − fixed − GDN 층) 믹서 변경은 상쇄된다 — 두 티켓의 PLE 차는 이 PR 의 몫이다.

| | 컨텍스트 0 | 컨텍스트 4,096 |
|---|---:|---:|
| PLE (ms) | 57.62 → **34.69** | 58.89 → **35.99** |
| 그중 torch elementwise | 51.59 → 26.54 | 51.26 → 26.58 |
| `_gate` 한 발사 | — | 1.92 ms |

- 게이트·conv 정규화의 torch 연산 약 15 개(float 이항 11 발사 15.3 ms, pow 4.6, mean 2.3, bf16 복사 4.8 …)가 `_gate` 한 발사 1.9 ms 로.
  필요한 바이트(약 350 MB) 기준 바닥 1.3 ms.
- **남은 PLE torch 26.5 ms** 는 dilated causal conv 의 torch 형태(cat 4.1, float add 9.0, mul 6.6, silu 1.5)와 `gated + local` — 다음 레버.
- 청크 전체(같은 두 티켓)는 믹서 변경이 섞여서 이 PR 의 수치로 쓰지 않는다: main 960.2 ms/청크(4,032 tok/s 한 랭크).

정확성: 인터프리터에서 torch 형태와 바이트 동일(`tests/test_engine_ngram_gate`), 부팅 `lanes.qualify` 에 `ngram_gate`(2^-6 밴드).
