# Qwen3.8 — 프리필 청크 하나의 내역, GB10 단일 레인: 4,096 토큰이 랭크 하나에 디바이스 약 1.0 s, 그 43% 가 믹서 사이트 (2026-09-19)

> 그대로 두는 기록 — srv4 단일 GPU 레인 티켓 `qwen38-prefill-census-0919e`(트리 main `048270ef`, 이미지 `st-engine:glm53`, 예산 8 GiB, 프로덕션 옆).
> 뒤 실행은 이 파일을 고치지 않고 새 기록을 남긴다.

`--lanes qwen38_prefill`(`probes/engine_qwen38_prefill.py`, #1223 · #1231): 서빙 모델의 `prefill` 그대로(타깃의 uncut forward + MTP 헤드의 관측)를
랭크 파일 실가중치의 작은 net 넷(층 [4,5,6,7] · [4,5] · [7] · [1])에서 4,096 토큰 청크 둘(컨텍스트 0 과 4,096)로 돌리고, 컴파일 · 벽시계 · CUDA
프로파일러 세 패스에서 fixed / GDN 층 / QSA 층 / PLE 넷을 풀어 48 층 청크로 더했다. 랭크 하나의 몫이고 집합통신은 없다(OneRankComm).

## 1. 디바이스 시간 — 두 컨텍스트에서 일관된다

48 층 청크: 컨텍스트 0 **1004.7 ms**, 컨텍스트 4,096 **1025.9 ms**(발사 6144.0 · 6386.0).
층마다: fixed 25.94 ms, GDN 층 17.23 ms, QSA 층 25.33 ms, PLE 54.44 ms.

| 패밀리 | 컨텍스트 0 ms | 컨텍스트 4,096 ms | 몫 | fixed | GDN 층당 | QSA 층당 | PLE |
|---|---:|---:|---:|---:|---:|---:|---:|
| hc gated residual | 226.3 | 227.5 | 23% | 1.6 | 4.663 | 4.73 | 0.043 |
| gemm (cublas/cutlass) | 214.4 | 221.0 | 21% | 9.065 | 4.332 | 3.847 | 3.263 |
| elementwise (torch) | 174.2 | 179.8 | 17% | 1.679 | 2.451 | 2.961 | 48.758 |
| moe b12x | 148.4 | 157.6 | 15% | 0.044 | 2.937 | 3.555 | 0.007 |
| qsa | 119.3 | 116.0 | 12% | 8.726 | 0.044 | 9.056 | 0.284 |
| other | 37.0 | 37.7 | 4% | 0.839 | 0.817 | 0.489 | 0.881 |
| gdn / kda | 35.2 | 35.1 | 4% | -0.025 | 0.97 | 0.025 | 0.018 |
| route / moe glue | 21.5 | 21.5 | 2% | 0.022 | 0.414 | 0.427 | 1.415 |
| dense w4/fp8 | 15.0 | 15.9 | 1% | 1.464 | 0.249 | 0.383 | -0.04 |
| conv | 6.8 | 6.7 | 1% | -0.006 | 0.187 | 0.006 | 0.006 |
| norm / rope | 6.5 | 7.0 | 1% | 2.543 | 0.166 | -0.156 | -0.196 |

- **믹서 사이트가 가장 크다:** 게이트 잔차 Triton 발사(`_leave_norm` · `_mix_mean` · `_norm_streams`) 226.3 ms 와
  cuBLAS GEMM 214.4 ms(대부분 믹서의 down · up 곱)를 합쳐 청크의 약 43%. `_leave_norm` 한 발사가
  4,096 행에 1.2 ms — 네 줄기 hidden(10,240 폭)을 읽고 쓰는 대역폭의 값에 가깝다: 줄일 길은 발사를 합쳐 줄기를 덜 오가는 것.
- **torch elementwise 174.2 ms:** PLE 주입 한 곳이 48.758 ms
  (해시 · 게이트 · conv 가 torch 형태 — `net._ple_feature` 의 docstring 이 적은 대로), 나머지는 층당 2.5 – 3 ms.
- MoE(b12x dynamic) 148.4 ms, 층당 약 3 ms 에 compact 합산(`index_add_`, 아래 표의 `indexFuncLargeIndex`) 약 0.8 ms.
- QSA 119.3 ms 는 QSA 층당 9 ms(sparse split-K 어텐션이 대부분). GDN 의 chunk 커널은 층당 1 ms 미만, dense W4/FP8 투영은 청크 전체 15 ms.

층 [4,5,6,7] · 컨텍스트 0 의 커널 상위(디바이스 ms, 층 넷 + MTP):

| 커널 | 발사 | ms |
|---|---:|---:|
| `_qsa_sparse_paged_gqa_splitk_kernel` | 2 | 17.173 |
| `kernel_cutlass_kernel_enginekernelsb12x_moe_dynamicgenericMoEDynamicKernel_object_at__tens` | 4 | 12.328 |
| `_leave_norm` | 10 | 12.085 |
| `_mix_mean` | 10 | 8.234 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_32x3_tn_align8` | 11 | 7.834 |
| `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_bf16_128x128_64x3_tn_align2>(cut` | 9 | 5.423 |
| `nvjet_sm121_tst_mma_128x208x64_2_32x104x64_tmaAB_alignCD4_bz_TNNN` | 2 | 4.398 |
| `void at::native::indexFuncLargeIndex<float, long, unsigned int, 2, 2, -2, true, at::native` | 4 | 3.295 |
| `void deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0u, 4224u, 2560u, 128u, 128u, 1u, 128u, 128u,` | 4 | 2.691 |
| `_norm_streams` | 4 | 2.378 |
| `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::Tenso` | 12 | 2.332 |
| `void cutlass::Kernel2<cutlass_75_tensorop_bf16_s1688gemm_bf16_128x128_tn_align1>(cutlass_7` | 4 | 2.127 |
| `_fp8_rows` | 2 | 1.409 |
| `void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, long, ` | 16 | 1.315 |

## 2. 벽시계와 호스트 몫 — 쓸 수 없다

| 층 세트 | 컨텍스트 | 벽시계 ms | 디바이스 ms | 호스트(벽 − 디바이스) ms | 발사 | 최대 GiB |
|---|---:|---:|---:|---:|---:|---:|
| 4,5,6,7 | 0 | 2448.12 | 102.97 | 2345.15 | 692 | 3.48 |
| 4,5,6,7 | 4096 | 110.92 | 103.72 | 7.2 | 712 | 3.48 |
| 4,5 | 0 | 73.9 | 60.41 | 13.49 | 453 | 2.88 |
| 4,5 | 4096 | 2742.54 | 59.76 | 2682.78 | 466 | 2.88 |
| 7 | 0 | 51.73 | 51.27 | 0.46 | 320 | 2.14 |
| 7 | 4096 | 51.72 | 51.44 | 0.28 | 319 | 2.14 |
| 1 | 0 | 107.54 | 97.62 | 9.92 | 424 | 2.71 |
| 1 | 4096 | 110.67 | 97.57 | 13.1 | 421 | 2.71 |

두 칸(층 [4,5,6,7] 의 컨텍스트 0, 층 [4,5] 의 컨텍스트 4,096)의 벽시계 패스에 2.3 – 2.7 s 가 한 번 끼었다. 나머지 칸의 호스트 몫은 0.3 – 13 ms 다.
이 레인은 프로덕션 GLM 과 같은 GPU 옆에서 돈다. 벽시계는 한 번 재는 값이라 그 경합이 그대로 들어가고, 넷을 푼 48 층 값은 음수까지 나왔다(원시
JSON 의 `chunk_48`). 디바이스 시간은 프로파일러가 커널마다 잰 것이라 두 컨텍스트에서 1 % 안으로 맞는다. **이 기록의 판정은 디바이스 표뿐이다.**
벽시계를 쓰려면 패스를 되풀이해 중앙값을 내야 한다(프로브의 다음 손질).

속도 주장 없음(D17): 한 랭크의 계산이고 플릿의 프롬프트 시간이 아니다. 원시: [prefill-048270ef.json](prefill-048270ef.json),
[로그](qwen38-prefill-census-0919e.log).
