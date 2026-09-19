# Qwen3.8 — 부팅 qualify soak, #1225 뒤: GB10 에서 0 / 2000 (2026-09-19)

> 그대로 두는 기록 — srv4 단일 GPU 레인 티켓 `qwen38-qualify-soak-fixed-0919`(트리 `4a3ec736` = main `233de548` + #1225, 이미지 `st-engine:glm53`,
> 예산 8 GiB, 프로덕션 옆, 669.9 s). 뒤 실행은 이 파일을 고치지 않고 새 기록을 남긴다.

같은 프로브 같은 시드(`--lanes qwen38_qualify_soak:2000`)를 `norm_rope_partial` 이 주소마다 한 번만 저장하게 고친 트리에서:

| 트리 | 실패 | 통과한 호출의 결과 종류 | 기록 |
|---|---:|---:|---|
| `2665b3bf`(고치기 전) | **7 / 2000** — 전부 `norm_rope_4x128`, 헤드 2, 채널 32..63, 커널 쪽이 반복되지 않음 | 1 | [qwen38_qualify_soak_20260919](../qwen38_qualify_soak_20260919/README.md) |
| `4a3ec736`(#1225) | **0 / 2000** | 1 | [soak-4a3ec736.log](soak-4a3ec736.log) |

- 통과 값은 고치기 전과 소수점까지 같다: `norm_rope_4x128` (0.006897, 0.002325), `norm_rope_6x256` (0.006173, 0.001535) — 고침이 바이트를 바꾸지 않았다는 GPU 쪽 확인.
- 다섯 레인(gated residual, GDN, QSA norm-rope, skinny GEMV, FP8 rows) 모두 2000 회 한 가지 결과.
- 크기의 한계: 0 / 2000 은 실패율이 대략 0.15% 미만이라는 뜻(95%)이지 0 이라는 증명이 아니다. 원인(한 주소 두 번 저장)이 코드에서 사라졌고
  `tests/test_engine_qwen38_store_once.py` 가 그 모양의 재발을 막는다 — 이 soak 은 그 설명과 맞는 관측이다.
