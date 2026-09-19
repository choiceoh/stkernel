# Qwen3.8 — dynamic MoE 타일 밴드까지 데운 뒤 문 뒤에서 처음 쓰는 커널: K=1 8 · K=3 7, dynamic MoE 0 (2026-09-19)

> 그대로 두는 기록 — srv4 단일 GPU 레인 티켓 `qwen38-serve-compiles-k1-0919f` · `-k3-0919f`(브랜치 `eager-moe-dynamic-bands` @ `bfa81bc8`
> = main `b5613857` + dynamic 밴드 네 개의 eager 워밍업, 이미지 `st-engine:glm53`, 예산 8 GiB, 프로덕션 옆). 뒤 실행은 이 파일을 고치지 않고 새 기록을 남긴다.

같은 census(`--lanes qwen38_serve_compiles:K` — 서빙 모델을 랭크 하나 · 층 1 과 7 · MTP · 4 행으로 부팅 순서대로 세우고, 서빙 창의 요청 일곱 개를 러너로
흘리며 스텝마다 Triton · b12x 커널 캐시를 센다)를 서빙 기본값 K=1 과 운영자의 K=3 에서. main 의 이전 census 는 `qwen38_serve_compiles_20260919`
(12 개, `048270ef`) · `…20260919b`(9 개, 디코드 크기 워밍업 뒤).

| 문 뒤에서 처음 쓴 커널 | K=1 | K=3 |
|---|---:|---:|
| `triton:_mix_mean` | 1 | 1 |
| `triton:_qsa_merge_splitk_kernel` | 1 | 0 |
| `triton:_qsa_mqa_paged_group_kernel` | 1 | 1 |
| `triton:_single_conv` | 5 | 5 |
| **합** | **8** | **7** |

- **b12x dynamic MoE: 0** — main 의 census 에서 3 이었다(타일 밴드 16 · 32 · 64). eager 워밍업이 밴드마다 한 번씩(쌍 9 · 1,920 · 6,144 · 12,288):

| 쌍 | K=1 부팅 s | K=3 부팅 s |
|---:|---:|---:|
| 9 | 0.004 | 0.005 |
| 1920 | 0.006 | 0.009 |
| 6144 | 0.012 | 0.011 |
| 12288 | 5.927 | 0.02 |

  (K=1 이 먼저 돌아 12,288 쌍의 128 밴드를 처음 컴파일했다 — 한 트리의 첫 부팅이 치르는 값, 그 뒤는 디스크 읽기.)
- 남은 것: `_single_conv` 5 는 #1238(T 를 인자로). `_mix_mean` 1 — 짧은 프롬프트(28 행)가 고른 타일(256 폭, #1224 의 행 수 밴드)을 워밍업의 폭
  1 · 8 · 64 가 밟지 않았다. `_qsa_merge_splitk_kernel` 1(K=1 만) — 28 토큰 프리필의 split 프로필. `_qsa_mqa_paged_group_kernel` 1 — 3,223 토큰의 G 1.
  넷 모두 **유한한 집합**이다: 한 번 컴파일되면 디스크에 남는다. 프롬프트 길이마다 새로 생기던 것은 conv 뿐이었다.
- 모든 디코드 스텝이 K=1 0.01 s · K=3 0.017 s 이하. 다른 세션의 K=1 창에서 요청 중 보였던 `micro_m1/m3_…_t10_r80` 은 **재현되지 않았다**
  (census 는 C=1 이다 — 그 창은 C=4 요청도 돌렸다).

원시: [census-k1.json](census-k1.json), [census-k3.json](census-k3.json), 로그 [K=1](qwen38-serve-compiles-k1-0919f.log) · [K=3](qwen38-serve-compiles-k3-0919f.log).
