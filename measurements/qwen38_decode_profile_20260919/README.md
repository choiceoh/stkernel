# Qwen3.8 디코드 스텝 커널 프로파일 — C=1 · K=3, GLM-5.3 09-14 기록과 같은 도구 (2026-09-19 17:12~17:16)

> 그대로 두는 기록 — 운영자 "프로파일 만들어서 기록 남겨봐". 뒤 실행은 이 파일을 고치지 않고 새 기록을 남긴다.
> 분해이지 판정이 아니다: 다음 레버를 고르기 위한 지도다(GLM 의 `measurements/st_decode_profile_20260914` 와 같은 자리).

## 무엇을 어떻게 쟀나

- **부팅:** 다른 세션의 운영자 창(`session/q38mtp-tune-0919`, MTP 헤드 파인튜닝)이 release 직전에 내어 준 10 분 슬롯이다. 프로덕션을 따로 내리지 않았다.
  트리 `~/st-worktrees/q38mtp-win` = `1aec0aac`(#1249 브랜치) + `mtp_tune.py` — **최신 main 이 아니다**: #1258(MTP 헤드 투영의 skinny GEMV),
  #1250, #1264 등이 없다. 명령([boot-cmd.txt](boot-cmd.txt)): `--spec-k 3 --draft-threshold off --no-tap-mtp-inputs --mtp-experts bf16
  --mtp-tuned .../st-qwen38-mtp-tuned3`(3차 파인튜닝 헤드) 와 런처 기본(one-shot, 4 행, shared overlap `one`, windowed-MTP 1,511). ready 64.1 s.
- **도구:** 도어의 `POST /v1/engine/profile {"steps": 32}` 가 다음 32 디코드 스텝을 모든 랭크에서 torch profiler(CUDA 활동)로 잡고, `GET` 이 랭크 0 의
  커널 표(커널별 device time, 호출 수; 상위 120)를 준다. 긴 greedy 에세이 요청을 끊김 없이 걸고, 각 프로파일 창 옆에 같은 부하의 프로파일러 없는
  8 초 창을 두어 `/metrics` 로 스텝 초와 drafted/accepted 를 읽었다. 드라이버 [qwen_profile.py](qwen_profile.py), 분류 [render_profile.py](render_profile.py),
  원시 [decode-profile-tuned3.json](decode-profile-tuned3.json), [profile.log](profile.log).

## 창

| 창 | 스텝 ms (프로파일러 / 없음) | 커널 합 ms/스텝 | 커널 몫(없음 기준) | 수용(드래프트 토큰당) | 스텝당 토큰(1 + 수용/스텝) |
|---|---:|---:|---:|---:|---:|
| C=1 #0 | 79.7 / 36.8 | 112.4 | — (무효) | 0.833 | 3.50 |
| **C=1 #1** | **39.0 / 36.9** | **32.0** | **87%** | 0.656 | **2.97** |
| C=4 #0 | 61.6 / 61.8 | 57.1 | 92% | 0.583 | 11.00(네 행 합) |

- **C=1 #0 은 무효:** 프로세스의 첫 프로파일에서 `k_oneshot_consumer` 가 스텝당 86.7 ms 로 부풀었다. 랭크들이 프로파일러를 서로 다른 순간에 켜면서
  피어를 기다린 몫으로 보인다. 둘째 창(#1)부터는 프로파일러가 스텝을 5.6% 만 늘렸다(39.0 / 36.9). 판정은 #1 과 C=4 로 한다.
- 스텝당 토큰은 카운터로 낸 값이다(`vllm:generation_tokens_total` 은 요청이 끝날 때 오르므로 이 창들에서 0). 에세이 greedy 는 드래프트가 쉬운
  텍스트다 — 속도 주장 없음(D17).

## C=1 스텝 한 개의 커널 (창 #1, 프로파일러 하, 랭크 0)

커널 합 32.05 ms, 발사 1,566 회/스텝. 스텝 36.9 ms(프로파일러 없음) 중 **커널 밖은 약 4.9 ms(13%)** — 호스트 · 발사 사이 공백.

| 묶음 | ms/스텝 | 몫 | 발사/스텝 |
|---|---:|---:|---:|
| 믹서 사이트 (skinny down+gates · up+mean · leave_norm 등) | 7.10 | 22% | 325 |
| TP 집합통신 (one-shot consumer · NCCL all-gather) | 6.85 | 21% | 114 |
| MoE 전문가 (b12x micro) | 5.95 | 19% | 48 |
| dense W4/FP8 (메가커널 GEMM) | 3.63 | 11% | 288 |
| 어휘 헤드 FP8 rows | 2.79 | 9% | 4 |
| MoE 라우팅 · 합 (router GEMV, softmax_topk, compact ids, swiglu, gated, sigmoid) | 1.92 | 6% | 337 |
| MTP 헤드 BF16 투영 (cuBLAS) | 1.35 | 4% | 22 |
| 그 밖 (torch elementwise · 복사 · 기타) | 1.04 | 3% | 275 |
| QSA | 0.79 | 2% | 81 |
| GDN (재귀 · conv · 게이트 norm) | 0.63 | 2% | 72 |

큰 커널:

| 커널 | ms/스텝 | 호출/스텝 |
|---|---:|---:|
| `k_oneshot_consumer(Ctrl*, __nv_bfloat16 const*, __nv_bfloat16*, int, int, HintAr` | 6.23 | 107 |
| `kernel_cutlass_kernel_enginekernelsb12xmoe_micro_kernelMoEMicroKernel_object_at_` | 5.95 | 48 |
| `_down_gates` | 3.55 | 106 |
| `_up_mean` | 3.25 | 106 |
| `_fp8_rows` | 2.79 | 4 |
| `void (anonymous namespace)::mk_gemm_input_kernel<false>((anonymous namespace)::M` | 2.35 | 96 |
| `void (anonymous namespace)::mk_gemm2_kernel<1, false, false, false, false>((anon` | 1.14 | 96 |
| `_skinny_gemv_kernel` | 0.97 | 48 |
| `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2` | 0.62 | 6 |
| `ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>)` | 0.55 | 4 |
| `std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, __nv_bflo` | 0.51 | 6 |
| `fused_recurrent_gated_delta_rule_fwd_kernel` | 0.50 | 36 |
| `_qsa_sparse_paged_gqa_splitk_kernel` | 0.41 | 15 |
| `_gate_up` | 0.35 | 3 |
| `_softmax_topk` | 0.31 | 51 |
| `_compact_topk_ids_kernel` | 0.29 | 48 |
| `_leave_norm` | 0.26 | 102 |
| `(anonymous namespace)::mk_input_pack_kernel((anonymous namespace)::MKInputPackCt` | 0.15 | 96 |

## GLM-5.3 (09-14, K=7) 과 나란히

| | GLM-5.3 C=1 K=7 (`de8bfff6`) | Qwen3.8 C=1 K=3 (이 기록) |
|---|---:|---:|
| 스텝(프로파일러 없음) | 약 47 ms | 36.9 ms |
| 커널 합 / 발사 | 54.9 ms / 1,359 | 32.0 ms / 1,566 |
| MoE 전문가 | 25.1 ms (46%) | 5.9 ms (19%) |
| dense GEMM | 13.7 ms (25%) | 3.6 ms (11%) |
| **TP 통신** | **4.4 ms (8%)** | **6.9 ms (21%)** |
| 잔차 믹서 (GLM mHC / Qwen 게이트 잔차) | 4.4 ms (8%) | 7.1 ms (22%) |
| LM 헤드 · 드래프터 | 2.2 ms (4%) | 헤드 2.8 ms (9%) + MTP BF16 투영 1.4 ms (4%) |
| 어텐션 계열 (MLA·DSA / QSA) | 1.6 ms (3%) | 0.8 ms (2.5%) |
| 선형 순환 (KDA / GDN) | 1.3 ms (2%) | 0.6 ms (2%) |

**읽기.**
- **커널 밖은 작다(13%).** 이 스텝은 대부분 커널이다. 앞서 부품 합으로 추정한 "3 분의 1 이 커널 밖" 은 틀렸다: 통신은 커널 안에서
  `k_oneshot_consumer` 가 피어를 기다리는 시간으로 잡힌다.
- **통신이 GLM 보다 무겁다: 6.9 ms, 21%**(GLM 8%). consumer 107 회 × 평균 58 µs — hidden 2560 의 작은 합이라 바이트가 아니라 지연이다.
  캐리 H4(leave 를 consumer 의 PDL 종속으로) · H5(leave 가 랭크 패킷을 직접 합산) · X1(작은 합에 compact 12-CTA consumer) · X2(TX 슬롯 직접 쓰기)가
  겨누는 자리가 스텝의 5 분의 1 이다.
- **믹서가 가장 크다: 7.1 ms, 22%**(GLM 의 mHC 8%). `_down_gates` 33.5 µs · `_up_mean` 30.7 µs 가 사이트마다 6.5 MB 씩 읽는다 — 약 200 GB/s, UMA 273 GB/s 의
  약 70%. 남은 레버는 바이트다(H6 의 FP8 은 디코드에서 +9.6% 로 기각됐다).
- MoE 는 GLM 의 4 분의 1(전문가 폭 640 · 네 행), dense W4 는 3.6 ms. 헤드 FP8 rows 는 호출당 0.70 ms × 4(검증 1 + 체인 3).
  MTP 헤드의 BF16 투영 1.4 ms(cuBLAS wmma · gemvx)는 이 트리에 없는 #1258(skinny GEMV)의 자리다.

## C=4 (창 #0, 네 행)

스텝 61.8 ms, 커널 57.1 ms(92%). MoE 가 커진다:

| 묶음 (render_profile 분류) | ms/스텝 | 몫 |
|---|---:|---:|
| MoE (b12x) | 22.98 | 40% |
| TP collective (one-shot/NCCL) | 9.75 | 17% |
| skinny GEMV (router, mixer down/up) | 9.04 | 16% |
| dense W4/FP8 (megakernel) | 4.26 | 7% |
| vocab head (FP8 rows / deep_gemm) | 2.83 | 5% |
| GDN / KDA / conv | 2.43 | 4% |
| other | 1.61 | 3% |
| QSA | 1.60 | 3% |

**안 한 것.** 최신 main 의 부팅(이 트리에 없는 변경들), 긴 컨텍스트(에세이는 수천 토큰 안), C=2, 다른 랭크의 표, 프로파일러 없는 커널 시간.
