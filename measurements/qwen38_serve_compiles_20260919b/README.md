# Qwen3.8 — 디코드 크기 워밍업 뒤 문 뒤에서 처음 쓰는 커널: 12 → 9 (2026-09-19)

> 그대로 두는 기록 — srv4 단일 GPU 레인 티켓 `qwen38-serve-compiles-headwarm-0919e`(브랜치 `decode-sized-warm` @ `68ed0840` = main `048270ef` +
> 헤드 패스 · 폭 1 · 8 · eager MoE 먼저, 이미지 `st-engine:glm53`, 예산 8 GiB, 프로덕션 옆). 뒤 실행은 이 파일을 고치지 않고 새 기록을 남긴다.

같은 census(`--lanes qwen38_serve_compiles`: 서빙 모델을 랭크 하나 · 층 1 과 7 · MTP · K=3 · 4 행으로 부팅 순서대로 세우고, 서빙 창의 요청 일곱 개를
러너로 흘리며 스텝마다 Triton · b12x 커널 캐시를 센다)를 main(`048270ef`, 기록 `qwen38_serve_compiles_20260919`)과 이 브랜치에서 돌렸다.

| | main `048270ef` | 이 브랜치 |
|---|---:|---:|
| 문 뒤에서 처음 쓴 커널 | 12 | **9** |
| `_qsa_covered_paged_gqa_kernel`(짧은 프리필, **첫 디코드 스텝 1.10 s**) | 2 | **0** |
| `_gates`(GDN, 짧은 프리필) | 1 | **0** |
| `_single_conv`(프롬프트 길이마다) | 5 | 5 — #1238 의 몫 |
| b12x dynamic MoE(타일 밴드) | 3 | 3 — 열림 |
| `_qsa_mqa_paged_group_kernel`(G 1, 3,223 토큰) | 1 | 1 — 열림 |

| 요청 | 프롬프트 | 디코드 스텝 | 가장 느린 디코드 스텝 s | 처음 쓴 커널 |
|---|---:|---:|---:|---:|
| 17x23 | 28 | 3 | 0.007 | 2 |
| capital | 30 | 20 | 0.007 | 1 |
| sky | 27 | 255 | 0.007 | 1 |
| transformer | 25 | 473 | 0.028 | 1 |
| long-summary | 3223 | 95 | 0.016 | 4 |
| transformer-again | 25 | 449 | 0.015 | 0 |
| sky-again | 27 | 255 | 0.015 | 0 |

- **첫 디코드 스텝의 컴파일이 사라졌다:** main 에서 첫 요청의 둘째 스텝이 1.10 s(덮인 QSA 의 작은 크기 특수화)였고, 이 브랜치에서는 모든 디코드 스텝이
  0.028 s 이하다.
- **부팅의 값:** 헤드 패스 20 개 합 8.35 s, 폭 1 · 8 은 0.431 · 1.47 s — 이 실행에서
  처음 컴파일된 몫이 대부분이다(main 의 census 가 먼저 돌아 레인의 디스크 캐시에 없던 커널들). 한 트리의 첫 부팅이 치르고, 그 뒤 부팅은 디스크에서 읽는다.
  이 net 의 헤드 패스는 MTP 헤드 한 층이라 48 층 부팅에서도 같은 크기다.
- 이 실행의 스텝 시간이 작은 것(첫 요청의 프리필 0.02 s 등)은 main 의 census 가 먼저 돌아 커널 대부분을 레인의 디스크 캐시에 남긴 때문이다 — 남은
  9 개도 여기서는 디스크 읽기였다. "처음 쓰인" 커널의 목록이 판정이고, 시간은 이 실행의 것이다(프로브 docstring).

남은 것:

| 커널 | 개수 |
|---|---:|
| `triton:_single_conv` | 5 |
| `b12x:_DYNAMIC_KERNEL_CACHE` | 3 |
| `triton:_qsa_mqa_paged_group_kernel` | 1 |

원시: [census-headwarm.json](census-headwarm.json), [로그](qwen38-serve-compiles-headwarm-0919e.log).
