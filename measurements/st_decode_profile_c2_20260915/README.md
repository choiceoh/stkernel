# ST 디코드 스텝 커널 프로파일 — main `9c45086a`, C=1 대 C=2, K=7 (2026-09-15)

운영자 지시 "st커널 c=2 최적화 개선" 을 위한 분해다. 판정이 아니라 지도다. #950 뒤 GLM-5.3 은 요청을 둘까지 받으므로
프로덕션 디코드는 C=1(검증 8행)과 C=2(16행)뿐이다. C=2 커널 분해는 이 기록이 처음이다(#963 은 C=1 만 남았다).

> 2026-09-17: `9c45086a` 의 지도다. 그 뒤 main 이 바뀐 자리 셋 — mHC 의 C=2 FP32 커널(#972 뒤 BF16 팩), consumer 상한
> 32,768(#967 뒤 65,536), DSA 인덱서의 행별 루프(#971·#1010) — 는 `measurements/st_c2_levers_survey_20260917/README.md` §2 를 보라.

## 무엇을 어떻게 쟀나

- **트리.** main `9c45086a062278afa43096e134bdbe8e521e5645`(#963). 이미지는 `st-engine:bracket-9c45086a0622`, 운영 env, 서두 기본값이다.
- **플릿.** 운영자 "플릿 풀어도 돼" 로 `session/expert-requant0913` 리스를 `hold_lease.sh` 의 정지 파일로 풀었다. 그 뒤 큐 티켓
  `c2opt-profile-main963` 으로 `bench/st_bracket.sh hold` 를 올렸다. 창은 07:41:54 GO ~ 07:57:34 release 이고, 문은 07:49:25 에 열렸다.
- **부하와 도구.** `c2_profile.py` 가 `measurements/st_decode_profile_20260914/decode_profile.py` 의 부하와 창 함수를 그대로 쓴다.
  - 부하: 약 70 토큰 에세이 프롬프트, thinking 끔, max_tokens 6000, 끝나면 다시 넣는 긴 그리디 생성.
  - 창 순서: C=2 → C=1 → C=2. 한 프로세스의 셋째 프로파일은 비는 전례가 있어서 중요한 둘을 앞에 뒀다.
  - 창마다 `POST /v1/engine/profile {"steps": 32}` 로 32 버스트를 모든 랭크에서 잡고, 프로파일러 없는 8 초 창을 옆에 잰다.
- **스텝 환산.** 스텝 수는 `/metrics` 의 drafted/7/C 로 구한다.
- **요약.** `summarize_c1c2.py decode-profile-c1c2-9c45086a.json` 이 표를 만든다. 범주 정규식은 #963 의 `summarize.py` 와 같다.

## 창

| 창 | 버스트당 스텝 | 스텝 ms (프로파일러 / 없음) | 버스트 ms (프로파일러 / 없음) | 수락률 | 표 |
|---|---|---|---|---|---|
| C=2 #0 | 3.95 | 59.19 / 55.31 | 233.8 / 221.3 | 0.231 | 유효 |
| C=1 #0 | 3.97 | 53.11 / 44.32 | 210.8 / 179.4 | 0.198 | 유효 |
| C=2 #1 | 3.91 | 60.84 / 54.79 | 238.0 / 224.0 | 0.187 | 빈 표 (12 µs/버스트) |

- 프로파일러 없는 스텝은 C=1 44.3 ms, C=2 55.3 ms 로 **×1.25** 다.
- 프로파일러는 C=1 을 20%, C=2 를 7% 부풀렸다. 그러니 아래 표에서 믿을 것은 비율과 호출 수다.

## 스텝당 커널 시간 — C=1 대 C=2 (프로파일러 하)

커널 시간 합은 C=1 48.30 ms, C=2 59.67 ms(×1.24)다.

| 범주 | C=1 ms | C=2 ms | C2−C1 | × | C=1 호출 | C=2 호출 |
|---|---|---|---|---|---|---|
| MoE 전문가 | 22.03 | 27.62 | +5.59 | ×1.25 | 35 | 34 |
| TP 통신 | 3.94 | 6.46 | +2.52 | ×1.64 | 129 | 125 |
| mHC | 3.88 | 5.80 | +1.92 | ×1.49 | 81 | 79 |
| ST 글루 | 1.24 | 1.94 | +0.70 | ×1.57 | 181 | 217 |
| MLA / DSA | 1.41 | 2.01 | +0.60 | ×1.43 | 147 | 160 |
| KDA | 1.09 | 1.68 | +0.59 | ×1.54 | 86 | 83 |
| LM head · 드래프터 fc | 1.95 | 2.29 | +0.34 | ×1.18 | 3 | 2 |
| norm / elementwise / 복사 | 0.73 | 0.89 | +0.16 | ×1.22 | 219 | 224 |
| dense GEMM | 12.03 | 10.97 | **−1.06** | ×0.91 | 312 | 292 |

범주 안에서 본 것:

- **MoE** (+5.6 ms). `MoEStaticKernelV5` 층당 1회다. 8행이 층마다 고르는 전문가는 약 58, 16행은 약 104 로, 읽는 바이트가 는다.
  C2 전용 16행 타일은 codex 세션의 몫이다(#955 후보, #962).
- **TP 통신** (+2.5 ms).

  | 커널 | C=1 | C=2 |
  |---|---|---|
  | `k_publish_packets` | 1.56 ms (40.3회) | 2.94 ms (38.9회) |
  | `k_oneshot_moe_packets` | 1.30 ms (30.2회) | 2.34 ms (29.2회) |
  | 전체합 | PDL `k_oneshot_consumer` 0.91 ms (15.1회) | 일반 `k_oneshot` 0.98 ms (14.6회) |

  - C=2 전체합이 일반 경로로 가는 이유는 consumer 상한이 32,768 원소(8행)이기 때문이다.
  - 커널 시간으로 보면 consumer 와 일반 경로의 차이는 0.07 ms 뿐이다. consumer 의 이득은 생산자와 겹치는 벽시계인데, 이 표에는 보이지 않는다.
  - 증가분 대부분은 패킷 두 커널의 호출당 비용이다.
- **mHC** (+1.9 ms). C=1 은 BF16 압축 계수 `mk_mhc_packets_kernel<true>` 로 3.55 ms(70.5회, 호출당 50 µs)다. C=2 는 FP32
  `mk_mhc_packets_kernel<false>` 로 5.61 ms(68.1회, 호출당 82 µs)다. 압축 계수 경로는 `n <= 8` 에서만 쓰인다.
- **dense GEMM** (−1.1 ms). C=2 의 일반 경로가 C=1 의 특화 경로보다 싸다.

  | 행 | 커널 구성 |
  |---|---|
  | C=1 | `mk_gemm2_kernel<1,…>` 3.93 ms(105회), KDA in_proj `mk_gemm_input_cta_kernel<1,…,32,8>` 3.18 ms(28.5회), `mk_input_pack_kernel` 1.60 ms(80.6회), cutlass 1.06 ms, 순서 보존 CTA 두 종 1.29 ms, `mk_query_pair_kernel` 0.35 ms |
  | C=2 | `mk_gemm2_kernel<2,…>` 세 종 9.34 ms(188회), `mk_wide_input_pack_kernel` 0.58 ms(59.1회), cutlass 1.05 ms |

  C1 입력 pack 만 1.60 ms 다. 8행을 C2 식 넓은 pack 으로 돌리는 편이 나은지는 같은 빌드 비교로 확인할 후보다.
- **글루.**
  - `_commit_layers`(지연 KDA commit) 0.26 → 0.64 ms 다. 격자가 행 수에 비례한다.
  - `_activation` 호출이 2.5 → 36.5 회로 는다. C=2 의 공유 전문가는 겹침 없이 직렬로 돈다(`net.py` 의 `x.shape[0] <= spec_k+1` 게이트).
- **KDA.** `fused_recurrent_gated_delta_rule_fwd_kernel` 0.60 → 1.04 ms(27.5회), `_single_conv` 0.21 → 0.32 ms 다.
- **DSA.** `gatherTopK` 0.53 → 0.83 ms, `sm120_fp8_mqa_logits` 호출 9.2 → 17.8 회(행별 루프, `net.py` `_select_rows`), `_attend` 0.07 → 0.11 ms 다.

## 같은 트리·같은 날의 다른 기록

- **서빙 처리량.** 같은 main `9c45086a`, 2026-09-14 23:15 `expert-c1c2-0914-main963` 보류 부팅의 클라이언트 측정이다.
  원본은 srv2 `~/expert-capture/c1c2.log`, `c1c2-main963.jsonl` 이고, requant 리스를 쥔 세션이 쟀다.

  | 부하 | C=1 합산 | C=2 합산 | C=2 요청당 디코드 중앙 |
  |---|---|---|---|
  | 1024 토큰 고정, 2K 프롬프트 4개 | 66.65 / 65.04 tok/s | 79.28 / 87.95 tok/s | 52.0 / 49.4 tok/s (C=1 72.1 / 72.2) |
  | JSON 12문항 | 85.74 tok/s | 100.71 tok/s | 59.9 tok/s (C=1 89.7) |

- **행 수별 반복 비용.** 2026-09-13 onepass 네 판(옛 트리, MAX_SEQS=4)의 `latency.jsonl` `gpu_iteration` 을 행 수로 묶었다(`rows_cost.py`).
  2K 문맥에서 n=1 은 47.9~51.1 ms, n=2 는 67.7~77.0 ms(×1.41~1.53, forward ×1.44~1.56, 드래프터 ×1.10~1.12)였다.
  문맥이 2K 라 이번 70 토큰 창의 ×1.25 보다 크다. 그 사이 main 의 C1/C2 변경도 섞여 있다.
- **C=1 대 C=4 트레이스.** 같은 날 진단 트레이스를 두 부팅에서 비교했다(`trace_lanes.py`, 옛 트리).
  - KDA 재귀 커널이 ×5.2 로 행 수보다 빨리 는다.
  - 인덱서 logits 호출이 11 → 44 회, `_attend` 가 5 → 20 회로 는다.
  - dense GEMM 은 평탄했다.

## 한계

- 70 토큰 프롬프트와 thinking 끔 조건이다. 수락률은 0.19~0.23 으로 onepass(0.5 대)보다 낮고, DSA·MLA 가 볼 문맥이 작다.
  32K·128K 의 C=2 분해는 없다.
- C=2 둘째 창은 빈 표다(세 번째 프로파일). 그래서 C=2 표는 창 하나다.
- 프로파일러 하 커널 시간이다. 스트림이 겹치므로 합이 벽시계를 넘을 수 있다.

## 재현

1. 부팅: 플릿을 쥔 세션에서 `bench/st_bracket.sh hold <sha> 35`(여기서는 `fleet.sh run --gpu --detach` 티켓).
2. `python3 c2_profile.py --base <tree>/measurements/st_decode_profile_20260914/decode_profile.py --out <json> --label <label> --plan 2,1,2 --wait-minutes 60` (도어 `127.0.0.1:8001`).
3. `python3 summarize_c1c2.py <json>`.
