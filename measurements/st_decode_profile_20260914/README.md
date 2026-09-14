# ST 디코드 스텝 커널 프로파일 — main `de8bfff6`, C=1, K=7 (2026-09-14)

운영자 질문: "vLLM 때 디코딩 한계는 24 step 정도였다 — 지금 스텝은 어디서 시간을 쓰나." 이 기록은 판정이 아니라
분해다. 개선 후보를 고르기 위한 지도이고, 종단 step/s 판정은 원장의 onepass 몫이다.

## 무엇을 어떻게 쟀나

- 트리: main `de8bfff6092c759cb5bca1d47270e68f38f7e03c` (#956 머지 직후: CUDA 13.2.1 런타임, MLA E4M3 확장은 반 정밀도
  다리). Red Hat 랭크, 운영 env, srv2 플릿 브래킷 hold `expert-dprof-0914-main956b`, 서두 기본값.
- 부하(`decode_profile.py`): 긴 그리디 생성을 끊김 없이 유지한다. 약 70 토큰 에세이 프롬프트, thinking 끔,
  max_tokens 6000, 끝나면 같은 요청을 다시 넣는다. C=1 세 창, 이어서 C=4 두 창.
- 도구: 도어의 `POST /v1/engine/profile {"steps": 32}` 가 다음 32 디코드 스텝(버스트, 4 반복 묶음)을 모든 랭크에서
  torch profiler(CUDA 활동)로 잡는다. `GET /v1/engine/profile` 이 rank 0 의 커널 표(커널별 self device time, 호출 수)를
  준다. 각 창 앞뒤로 `/metrics` 의 `st:step_seconds{kind="decode"}` 와 drafted/accepted 카운터를 읽어 반복 수
  (drafted/7)와 벽시계를 구하고, 같은 길이의 프로파일러 없는 창을 바로 옆에 잰다.
- 요약: `summarize.py` (분류 정규식은 그 파일에). 원시 표: `decode-profile-de8bfff6.json`.

## 창

| 창 | 버스트 | 반복/버스트 | 버스트 ms (프로파일러 / 없음) | 반복 ms (프로파일러 / 없음) | 수락률 | 표 |
|---|---|---|---|---|---|---|
| C=1 #0 | 33 | 3.82 | 206.3 / 189.4 | 54.03 / 46.76 | 22.7% | 유효 |
| C=1 #1 | 33 | 3.94 | 211.8 / 187.9 | 53.77 / 47.83 | 20.7% | 유효 |
| C=1 #2 | 33 | 4.00 | 216.2 / 192.4 | 54.05 / 47.77 | 16.3% | 빈 표 (13.8 µs/버스트) |
| C=4 #0 | 39 | 7.74 | 291.6 / 289.5 | 37.65 / 35.50 | 19.2% | 빈 표 (28.8 µs/버스트) |
| C=4 #1 | 34 | 7.88 | 285.9 / 283.1 | 36.27 / 36.40 | 16.4% | 빈 표 (13.4 µs/버스트) |

- C=4 의 "반복" 은 행을 합친 drafted/7 이라 행-반복이다. 반복 ms 도 행-반복당이다.
- **한 프로세스에서 세 번째 프로파일부터 표가 비었다.** 원인은 확인하지 않았다. 그래서 C=4 분해는 없다.
- 프로파일러는 C=1 반복을 6~7 ms(13~16%) 늘린다. 반복당 커널 호출 약 1,359 회에 호출당 5 µs 쯤이면 맞는 크기다(추정).

## C=1 반복당 커널 시간 (유효 두 창 평균, 프로파일러 하)

커널 시간 합 54.89 ms, 호출 1,359 회/반복. 다른 스트림의 커널은 시간이 겹치므로 합이 벽시계(54.0 ms)를 넘을 수 있다.

| 항목 | ms/반복 | 비중 | 호출/반복 | 큰 커널 (ms) |
|---|---|---|---|---|
| MoE 전문가 | 25.13 | 45.8% | 40.2 | `MoEStaticKernelV5` (b12x CuTe, 층당 1회) 25.13 |
| dense GEMM | 13.65 | 24.9% | 355.9 | `mk_gemm2_kernel` 4.48, `mk_gemm_input_cta_kernel<1,…,32,8>` 3.58, `mk_input_pack_kernel` 1.79, cutlass bf16 GEMM 1.21 |
| TP 통신 | 4.43 | 8.1% | 147.3 | `k_publish_packets` 1.81, `k_oneshot_moe_packets` 1.43, `k_oneshot_consumer` 1.00 |
| mHC | 4.42 | 8.0% | 92.8 | `mk_mhc_packets_kernel` 4.04 (80.4회, 호출당 50 µs), `mhc_post_tilelang_kernel` 0.24 |
| LM head / 드래프터 fc (deep_gemm fp8×fp4, 이름은 형상에서 추정) | 2.21 | 4.0% | 2.9 | `sm120_fp8_fp4_gemm_1d1d` ×3: 4096 입력 7행·8행 0.86·0.86, 20480(=5×4096) 입력 0.50 |
| MLA / DSA | 1.57 | 2.9% | 167.5 | torch `gatherTopK` 0.60 (50.7회), `mk_mla_kernel` 0.33, torch `bitonicSortKVInPlace` 0.25 (41.2회) |
| 그 밖의 ST 글루 커널 | 1.41 | 2.6% | 205.7 | `_absorb` 0.49, `_commit_layers` 0.29, `_gate_partials` 0.11 |
| KDA | 1.25 | 2.3% | 97.6 | `fused_recurrent_gated_delta_rule_fwd` 0.68, `_kda_pair` 0.32, `_single_conv` 0.25 |
| norm / elementwise / 복사 | 0.83 | 1.5% | 249.0 | `vectorized_gather_kernel` 0.13, `splitKreduce_kernel` 0.12 |

전체 커널 목록은 `python3 summarize.py decode-profile-de8bfff6.json --all`.

## 같은 날, 같은 도구 계열의 다른 기록과 대조

- **onepass 09-13 23:48** (arm `32fb2892`, C=1 2K JSON 프롬프트, 2,023 반복, `measure-c1/latency.jsonl` 의 장치 단계 합):
  forward 46.6 ms, propose(드래프터) 3.05 ms, observe 0.66 ms, sample·commit·경계 합 약 0.1 ms. `gpu_iteration` 평균 50.5 ms
  (중앙 50.3, 최소 42.9). 이번 창의 약 47 ms 는 프롬프트가 짧아(약 70 토큰) DSA 가 볼 문맥이 작을 때의 값이다.
- **STEP_KERNEL_MAP 보충 분해 5·6** (vLLM 포크 시절, 같은 CUPTI 계열):
  - 09-04 무장 트레이스: 깨끗한 18 스텝 중앙 69.25 ms, MK MHC 89회 2.03 ms, `k_oneshot` AR 102회 5.28 ms, lm_head 둘 1.66 ms.
  - 09-05 프로덕션: 프로파일러 하 55.8 ms/스텝, 커널 1,166 개/스텝. MoE 전문가 42회 29.4 ms, mk_gemm2 147회 3.52 ms
    (KDA 레인이 in/o_proj 68회를 흡수), bf16 cutlass/cublas 103회 3.34 ms.
  - 그 트레이스들의 SPEC_K 는 이 기록에서 확인하지 않았다(아카이브의 09-07 onepass 는 SPEC_K=5, 디코드 중앙 21.4~21.9 step/s).
- **C=1 onepass JSON 12문항 클라이언트 step/s** (같은 날, 서두 기본값): `c58a8eb5` 18.51, `83140b14` 19.32, `b8bef130` 19.26
  (MEASUREMENTS 의 #956 기록).

## 읽을 때

- 프로파일러 하 커널 시간이다. 절대값은 부풀어 있고(벽시계 +15%), 쓸모 있는 것은 비율과 호출 수다.
- MoE 한 줄이 반복의 46% 다. 검증 8행이 층마다 고르는 서로 다른 전문가 수(top-8, 288 전문가, 기댓값
  288×(1−(280/288)^8) ≈ 58)가 읽는 바이트를 정한다.
- mHC `mk_mhc_packets_kernel` 은 호출당 50 µs 로, 보충 분해 5 의 MK MHC(호출당 23 µs)의 두 배다. 이 커널이 TP 패킷
  소비까지 포함하는지는 확인하지 않았다 — 포함한다면 두 수치는 같은 일을 재지 않는다.
- DSA 디코드 top-k 는 torch 선택(`gatherTopK` + `bitonicSortKVInPlace`, 92회 0.85 ms)이다. 프리필에는 네이티브
  `prefill_topk` 가 있다.
- dense GEMM 은 반복당 356 회 호출이고, 그중 입력 pack 이 92 회 1.79 ms 다.

## 같은 부팅의 메모리 장부 (rank 0)

작업 공간 상한 12 GiB, 프리필 최고치 9.89 GiB (`prefill/32256/1016320/prepared`), 준비 뒤 남는 작업 공간 1.99 GiB,
OOM 여유 최소 23.16 GiB. 아레나 55.45 GiB.

## 재현

1. 부팅: `bench/st_bracket.sh hold <sha> 45` (운영 env, 플릿 리스를 쥔 세션에서).
2. `python3 decode_profile.py --out decode-profile-<label>.json --label <label>` (도어 `127.0.0.1:8001`).
3. `python3 summarize.py decode-profile-<label>.json --all`.

프로파일러를 한 프로세스에서 세 번 이상 켜면 빈 표가 나올 수 있다. 창을 늘리려면 부팅을 나누는 편이 안전하다.
