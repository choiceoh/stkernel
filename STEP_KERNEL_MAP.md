# 스텝 커널 지도 — 우리가 안 만든 것들

C=1 디코드 스텝의 커널은 **1,582 개**, 그중 이 리포에서 컴파일되는 것이 **402 개
(25.4%)** 다(2026-09-04 무장 트레이스). 나흘 전 첫 인구조사(08-31 스톡)에서는
1,886 개 중 **144 개(7.6%)** 였다. 나머지 1,200 개가 어디서 나오고, 무엇이 그걸
정하고, 손댈 수 있는지를 적는다. 이 문서는 "우리 커널만 최적화 대상"이라는 잘못된
전제를 깨기 위한 것이다 — 이 레포의 오버레이 체계 자체가 벤더 코드를 접수해 고치는
물건이고, `b12x_moe.py`·`gpu_model_runner.py`·`glm5next_model.py` 는 이미 그렇게
쓰고 있다. 메가커널(EXP-6)은 그 접수를 한 단계 더 밀어 **구간 전체를 우리 커널로**
바꾼 것이고, 지도의 4 분의 1 이 그래서 우리 것이 됐다.

**이 문서가 서 있는 세 트레이스** (전부 rank 0, `census.py` 로 같은 방식):

| 트레이스 | 부팅 | 커널/스텝 | 우리 것 | 어디에 |
|---|---|---|---|---|
| 08-31 16:51 | 스톡(노브 전부 off) | 1,886 | 144 (7.6%) | 아래 본문 · 보충 분해 1 |
| 09-01 15:07 | 스톡 | 1,837 | 144 (7.8%) | 보충 분해 2 · 3 |
| **09-04 18:42** | **무장 (MK MHC+GEMM)** | **1,582** | **402 (25.4%)** | **보충 분해 5** |

09-04 트레이스는 28차 §3 이 프로파일한 그 부팅(18:25 cand, 깨끗한 18 스텝, 69.25 ms)
이다. **오늘의 프로덕션 기본값은 그보다 한 발 앞서 있다** — MK-MLA · 드래프터 W4
전량 · 서빙 PDL 이 그 뒤에 기본이 됐다(앞의 둘은 28차 §8, PDL 은 27차 프로브 · PR #290
으로 20:57). 그 셋이 지도의 어느 칸을 움직이는지는
보충 분해 5 의 마지막 표에 **미실측으로** 적어 둔다. "무장 ≠ 서빙"(22차)은 문서에도
적용된다 — 트레이스가 없는 칸은 트레이스가 있는 칸과 같은 표에 숫자로 앉히지 않는다.

**정정(2026-09-04)**: 첫 판의 "우리 소유 186 개(9.9%)" 는 틀렸다. `census.py` 의
osar 그룹 정규식 `k_reduce` 가 앵커 없이 deep_gemm 의 `sm120_split_k_reduce_impl`
42 발/스텝을 우리 AR 로 셌다(우리 osar 모듈의 `__global__` 은 `k_oneshot` 하나뿐이고
102 발/스텝이다 — 아래 구조표와도 어긋났었다). 정규식에 `^` 앵커를 넣어 고쳤고,
그 42 발은 원래 자리인 GEMM 그룹으로 갔다. 원장의 같은 부팅 표("커널 개수의 실측
가격")는 처음부터 **148(AR 104 + 게이트 44)** 로 적고 있었다 — 어긋난 쪽은 지도였다.

> ⚠ 트레이스의 **절대 시간은 트레이스 사이에서 비교하지 말 것**. CUPTI 가 GPU 바쁜
> 시간을 부풀린다(앞선 세션: 136 ms vs 실측 스텝 47.6 ms; 09-04 무장 트레이스는
> 69.25 ms 대 서빙 61~63 ms 로 ~10%). **개수가 정본**이고, 시간은 같은 트레이스
> 안에서의 상대 비교로만 쓴다.

## 이 모델이 무엇인가 — 커널 개수가 전부 여기서 나온다

`glm5_next_text`, 45 층, hidden 4096. **하이브리드 어텐션**이다:

| | |
|---|---|
| `layer_types` | `linear_attention` **34** + `deepseek_sparse_attention` **11** |
| `full_attn_layers` | `[3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43]` — 4 층마다 하나 |
| `kda_layers` | 나머지 34 (Kimi Delta Attention, `short_conv_kernel_size: 4`) |
| MoE | `first_k_dense_replace: 3` → 층 0~2 는 dense MLP, **3~44 가 MoE** (42 층) |
| 전문가 | `n_routed_experts: 288`, `num_experts_per_tok: 8`, `n_shared_experts: 1` |
| MHC | `mhc: True` — 전 층 |

**커널 개수가 이 구조와 정확히 맞는다.** 이게 인구조사가 신뢰할 만하다는 증거다:

| 커널 | /스텝 | = |
|---|---|---|
| `fused_recurrent_gated_delta_rule_fwd_kernel` | 34 | KDA 층 34 × 1 |
| `_causal_conv1d_update_kernel` | 34 | KDA 의 short conv |
| `layer_norm_gated_fwd_kernel` | 34 | KDA 의 gated norm |
| `mhc_pre_big_fuse_with_norm` / `mhc_fused` | 90 / 89 | 45 층 × 각 1 |
| MoE static kernel · topk | 42 · 42 | MoE 층 42 |
| `_deneb_gate_partial_kernel` (우리) | 42 | MoE 층 42 |
| `k_oneshot` (우리 AR) | **102** | ≈ 45 층 × 2 + 12 |
| MLA 어텐션 | 33 | 풀어텐션 11 층 × 3 |

`k_oneshot` 102 는 부수로 **콜렉티브/스텝을 실측 확정**한다. 앞서 로그 창을
느슨하게 맞춰 추정한 73 은 틀렸다.

## 그룹별 지도

`census.py` 의 그룹 규칙으로 자른 커널 **개수**다(시간 아님). 왼쪽은 08-31 스톡
부팅, 오른쪽은 09-04 무장 부팅 — 사이에 메가커널 MHC·GEMM 세그먼트가 들어왔다.

| 그룹 | 스톡 08-31 | 무장 09-04 | 출처 | 지금 상태 |
|---|---|---|---|---|
| elementwise 글루 | **483** | **446** | torch/inductor 생성 | 안 움직였다 — MK 는 글루를 흡수하지 않는다. `custom_ops` A/B(EXP-2) 미실행 |
| **메가커널 세그먼트 (우리)** | — | **263** | `glm53_megakernel` | mk_gemm 185 + mk_mhc 89 (아래 ⑥) |
| cutlass/cublas GEMM | 439 | 229 | cuBLASLt + deepgemm | 밀집 fp8 197 발이 MK 레인으로. 남은 건 드래프터 bf16 + 저랭크 b-절반(②) |
| 정규화/양자화 | 301 | 129 | vLLM 커스텀 op + W8A8 대가 | `per_token_group_quant` 179 → 2 — MK GEMM 이 커널 안에서 양자화한다(③) |
| mhc (TileLang) 2 종 | 185 | 14 | 이미지의 glm5next 모델 코드 | **접수 완료** — mk_mhc 가 89 발로 대체(④) |
| MoE b12x | 99 | 110 | flashinfer | 손 안 댐. 개수 차는 08-31→09-01 사이의 경로 변경(09-01 스톡도 110) |
| **osar AR (우리)** | 102 | 98 | `tp_oneshot_ar` | 이미 5→1. `k_oneshot` 하나뿐이다 |
| 기타 | 82 | 86 | 혼합 | |
| KDA/FLA 청크 | 68 | 77 | fla 라이브러리 | `MK_SEG_KDA=0` — 유일하게 남은 미무장 세그먼트 |
| 복사/산포/수집 | 47 | 50 | torch | |
| **게이트 (우리)** | 42 | 40 | `moe_gate_sm121` | #96 로 반감, 레버 아님(보충 분해 3) |
| 어텐션 (MLA) | 33 | 33 | flashinfer/trtllm | 오늘 기본값은 `MK_SEG_MLA=1` — 이 트레이스엔 없다 |
| 샘플러/스펙 | 5 | 5 | vLLM | |
| NCCL | (복사 그룹) | 4 | 로짓 AllGather | |
| **합계** | **1,886** | **1,582** | | −304 (−16%) |

- 스톡 열의 "기타 150"(첫 판)은 그 뒤 KDA 행이 생기면서 82 + 68 로 갈렸다 — 같은
  트레이스, 같은 도구, 그룹 규칙만 자란 것이다(보충 분해 1 이 그 150 을 분해했다).
- 그룹 규칙은 정규식 휴리스틱이다. 깨끗한 스텝만 잘라 시간까지 붙인 표는 보충 분해 5
  에 있고, 그쪽이 커널 **시간**의 정본이다.

### ① elementwise 483 개 (25.6%, 스톡 08-31) — 가장 큰 덩어리, 원인은 설정

45 층에 483 개면 **층당 10.7 개**다. inductor 는 이런 걸 주변 커널에 접어
넣는 게 일인데(`combo_kernels: True` 도 켜져 있다) 483 개가 살아남았다.

원인은 `custom_ops: ['all', '+quant_fp8', '+quant_fp8']` 로 보인다. **커스텀 op
하나하나가 inductor 의 융합 장벽**이라, 그 사이의 residual add·scale·mask 가
독립 커널로 남는다. 즉 `custom_ops` 는 *"수제 커널이 빠르다"* 와 *"융합이 커널을
없앤다"* 의 교환인데 **이 스택에서 A/B 된 적이 없다.**

- 시험: `custom_ops` 를 `['none']` 또는 선택 목록으로 놓고 부팅 1 회. **코드 변경 0.**
- 상한: 483 × 5.4 us = 2.6 ms = 스텝의 3.9%

**무장 09-04**: 483 → **446**. MK 는 글루를 흡수하지 않는다(세그먼트는 GEMM·MHC
구간만 접수한다). 그리고 **상한 2.6 ms 는 낙관이었다** — 무장 트레이스의 깨끗한
스텝에서 elementwise 481 발의 시간 합은 **1.00 ms(스텝의 1.4%)** 다. 5.4 us/커널은
#89 의 역산 상한이고 그래프 안 글루의 실제 점유는 ~2 us(보충 분해 2)다. EXP-2
(`custom_ops` A/B)는 여전히 미실행이고, 이제 그 기대값은 1 ms 아래다.

### ② cutlass bf16 GEMM 145 개 (스톡 08-31) — 어디가 아직 bf16 인가

`cutlass_80_wmma_tensorop_bf16` 107 + 38 = **145/스텝**. W8A8(#92/#94)이 180 개를
fp8 로 옮겼는데도 남았다. `glm53_fp8_dense` 의 `_INCLUDE` 와 대조하면:

```python
r"\.self_attn\.(in_proj_qkvbfg_a|fused_qkv_a_proj|q_proj|k_proj|v_proj|o_proj|out_proj)$"
r"\.mlp\.shared_experts\.(gate_up_proj|down_proj)$"
r"\.mlp\.(gate_up_proj|down_proj)$"
```

잡히는 것: KDA 의 q/k/v/b/f_a/g_a 는 로더가 `in_proj_qkvbfg_a` 로 **병합**하므로
(`glm5next_model.py:843-848`) 첫 패턴에 걸린다. shared expert 의 gate/up 도
`gate_up_proj` 로 병합돼 걸린다.

**안 잡히는 것** (≈145 와 맞는다):

| | 개수 | 무엇 |
|---|---|---|
| `f_b_proj`, `g_b_proj` | 34 × 2 = 68 | KDA 저랭크 게이트의 **b 절반** |
| `q_b_proj`, `kv_b_proj` | 11 × 2 = 22 | 풀어텐션 층의 **b 절반** |
| `indexer.wq_b`, `indexer.wk` | ~24 | 희소 어텐션 인덱서 |

전부 **저랭크 사영의 뒤쪽 절반**이다. 바이트는 작아서(랭크 차원) 대역폭 이득은
크지 않지만 **커널 개수로는 7.7%** 다. `_INCLUDE` 에 `_b_proj` 계열을 더하는 것은
한 줄이고, 그 층들이 `ignore` 에 있는 이유(정밀도 민감)를 감안하면 **게이트를
붙여 재야 한다.**

**무장 09-04**: cutlass/cublas 그룹 전체가 439 → **229**. 밀집 fp8 GEMM 197 발이
mk_gemm 레인으로 갔고, 남은 201 발(깨끗한 스텝)은 **드래프터 34 발(꼬리, 2.88 ms)**
+ 저랭크 b-절반 ~112 발 + `splitKreduce` 23 + 인덱서 `gemmSN` 11 이다. 즉 이 절의
표적(b_proj 계열, EXP-4)은 **그대로 남아 있고**, 그 위에 드래프터가 얹혔다 —
오늘의 기본값은 드래프터 30 발을 MK W4 로 옮겼다(28차, 스텝 −1.1 ms). 인덱서
`gemmSN` 11 발(EXP-9 표적)도 아직 stock 이다.

### ③ 정규화/양자화 301 개 (스톡 08-31) — W8A8 의 대가가 보인다

`per_token_group_quant_8bit_packed_register_kernel` 이 **137 + 42 = 179/스텝**.
dense GEMM 마다 활성화를 fp8 로 양자화하는 비용이다. #94 가 −6.9 ms 를 벌면서
약 1 ms 를 여기에 지불한다(5.4 us/커널 기준). 순이득이지만 **공짜가 아니다.**

`pass_config` 의 `fuse_norm_quant`·`fuse_act_quant` 는 이미 `True` 로 확정된다.
137 개가 남았다는 것은 그 융합이 닿지 않는 자리가 있다는 뜻이다.

**무장 09-04**: 301 → **129**. `per_token_group_quant_8bit_packed_register_kernel`
이 **179 → 2 발**이다 — MK GEMM 이 커널 안에서 활성화를 양자화하므로 W8A8 의
"공짜가 아닌 대가" 항목이 사라졌다. 남은 2 발은 두 lm_head(타깃·드래프트) 앞의 것.

### ④ mhc 185 개 (9.8%, 스톡 08-31) — 우리 AR 보다 많다

`mhc_pre_big_fuse_with_norm_tilelang_kernel` 90 + `mhc_fused_tilelang_kernel` 89.
**TileLang 커널**이고 45 층마다 두 번 돈다. GLM-5.3-next 고유의 Multi-Head
Compression(`mhc_sinkhorn_iterations` 등)이다.

이미지의 모델 코드에서 나오므로 **오버레이로 접수해 고칠 수 있다**(dsv4 레인엔
이미 `dsv4_mhc_tilelang` 모듈이 있다). 둘을 하나로 접으면 −89/스텝(1.5%).

**무장 09-04**: 185 → **14**(mk_mhc 89 발이 대체, stock 잔여 7). 이 절이 예상한
"둘을 하나로 접으면 −89/스텝" 이 그대로 일어났다 — pre 와 post 가 사이트당 한 발로
접혔다. 시간은 2.54 ms → **2.03 + 0.04 ms**(발당 13.7 → 22.8 us, 발사 수가 절반).

### ⑤ MoE b12x 99 개 (스톡 08-31) — 제약이 문서화돼 있다

`flashinfer_b12x_moe.py`(오버레이)가 지원 조합을 코드로 못박는다:

```python
return (weight_key, activation_key) in (
    (kNvfp4Static, kNvfp4Dynamic),   # W4A4
    (kNvfp4Static, None),            # W4A16 -- 커널 안에서 BF16->FP4
)
```

**가중치는 NVFP4 고정.** W4A8 은 통과하지 못한다. 그리고:

```python
def _supports_parallel_config(cfg): return not cfg.use_ep
```

**b12x 는 expert parallelism 을 지원하지 않는다** — 288 전문가가 전 랭크에
복제된다. 스텝의 62% 가 전문가 읽기인데 그게 **샤딩 없는 전량 읽기**라는 뜻이고,
EP 를 쓸 수 있다면 그 항이 줄어들 여지가 있다. 이 축을 여는 것은 커널 교체이며
지금까지 논한 어떤 변경보다 크다.

**무장 09-04**: 99 → **110**, 손 안 댔다. 그리고 27차가 이 축의 다음 칸을 닫았다 —
**MK_SEG_MOE 는 열 이유가 없다**: b12x static 커널(U=40)의 실효 대역폭 197 GB/s 는
같은 형상에서 우리 MK 레인의 196 GB/s 와 같다(둘 다 발사 안 구조에 묶여 있고, 읽기
전용 참조 245 와의 20% 격차는 공통). 재개 조건은 persistent 타일 스트림
마이크로커널이 240 GB/s 에 닿는 것. EP(EXP-1)는 여전히 **미실측 부팅 게이트**다 —
구현은 끝났고 브래킷만 남았다(RUNBOOK EXP-1).

### ⑥ 메가커널 세그먼트 263 개 (우리) — 접수의 결과

| | 발/스텝 | ms/스텝 | 발당 | 무엇을 대체했나 |
|---|---|---|---|---|
| `mk_gemm_kernel` | 185 | 14.13 | 76 us | deep_gemm 밀집 fp8×fp4 197 발 + `per_token_group_quant` 177 발 |
| `mk_mhc_kernel` | 89 | 2.03 | 23 us | TileLang `mhc_pre`+`mhc_fused` 179 발 |

**커널 수는 −304, 시간은 제자리다.** 09-01 스톡 트레이스의 deep_gemm 밀집 GEMM 은
197 발 14.06 ms 였고, 무장 트레이스의 mk_gemm 은 185 발 **14.13 ms** 다(같은 도구,
다른 부팅 — 브래킷 아님). MHC(−0.5 ms)와 MoE 글루(2.71 → 1.20 ms, 양자화가 GEMM
안으로 들어가면서)에서만 시간이 줄었다. 이 문서가 처음부터 경계한 것 — "커널 하나
5.4 us" 는 상한이고 개수 축의 이득은 그보다 훨씬 작다 — 커널 −304 발 규모에서
다시 확인된 셈이다.

세그먼트 넷 중 **KDA 만 미무장**이다(`MK_SEG_KDA=0`; conv 상태 dtype 계약 때문에
`MAMBA_CACHE_DTYPE` 노브가 09-04 22:24 에 붙었다). MLA 는 09-04 19:55 부터 기본값
이지만 이 트레이스에는 없다.

## 이미 닫힌 축 (재조사 금지)

| 축 | 판정 |
|---|---|
| `fuse_act_padding` · `fuse_mla_dual_rms_norm` · `fuse_rope_kvcache` · `fuse_qk_norm_rope_kvcache` | **ROCm 전용 하드게이트.** CUDA 에서 영구 불가 |
| `fuse_gemm_comms` · `enable_sp` | O2 기본값이 `IS_DENSE`. MoE 라 정책상 off (플랫폼 게이트는 없음) |
| `fuse_allreduce_rms` | 확정값 `True` 인데 **런타임 미지원** — `FI_ALLREDUCE_FUSION_MAX_SIZE_MB` 에 sm_121(=121) 키가 없다. 강제해도 FlashInfer 콜렉티브 자체가 이 플릿에서 선택되지 않는다 |
| `enable_qk_norm_rope_fusion` | GLM-5.3-Flash 는 **NoPE** — rope 가 없다 |

**교훈**: `pass_config` 의 확정값이 `True` 라도 **런타임 지원 게이트가 한 겹 더
있다.** 융합 판정의 전제는 확정값이 아니라 **경고 부재**여야 한다.

## 커널 개수 축의 실측 천장

| | |
|---|---|
| 첫 판의 실측 가격 | **5.4 us/커널** (#89 의 −209 커널 → −1.13 ms) — **상한** |
| 그 가격으로 1,886 개 전부 | 10.2 ms = 스텝의 15.4% (낙관치) |
| 실제로 없앤 것 (MK 세트) | **−304 개**, 그중 시간이 준 것은 MHC −0.5 · MoE 글루 −1.5 ms |
| 무장 스텝의 글루 481 발 총점유 | **1.00 ms (1.4%)** — 발당 ~2 us |

**이 축에 처음으로 "소유권 없이" 도전하는 시도**: `overlay/modules/glm53_megakernel`
(2026-09-01) — 콜렉티브 사이 구간을 persistent 48블록 런치 하나로 흡수. 소유하지
않은 커널을 하나씩 줄이는 대신 구간 전체를 원시 CUDA 세그먼트로 재작성한다.
수치·계약 게이트와 브래킷 계획은 RUNBOOK EXP-6.

| 세그먼트 | 예측(2026-09-01) | **실측(09-04 무장 트레이스)** |
|---|---|---|
| MHC | 179 → 45 | **179 → 89** + stock 잔여 7 |
| quant + 밀집 GEMM | ~360 → ~180 | **376 → 187** |
| KDA 블록 | ~510 → 34 | 미무장(`MK_SEG_KDA=0`) |
| 합계 천장 | 약 4.9 ms | 시간이 준 자리는 MHC −0.5 · 양자화 −1.5 ms 뿐 |

⚠ **그런데 이 근거는 강해지지 않고 약해지고 있다.** #89 이후 #90·#93·#96 이
324 개를 더 없앴는데 W8A8 몫을 뺀 스텝은 변하지 않았다(예상 65.8 vs 실측 66.1).
5.4 us 는 **상한이고 실제는 더 작을** 공산이 크다. 위 표의 "없애면" 값은 낙관치로
읽어야 한다.

**그리고 메가커널이 그 경고를 규모로 확인했다(2026-09-04)**: 세트가 커널을 304 개
지웠는데 밀집 GEMM 구간의 시간은 197 발 14.06 ms → 185 발 14.13 ms 로 제자리였다
(⑥). 개수 축에서 실제로 시간이 나온 자리는 **커널이 사라진 곳이 아니라 일이 합쳐진
곳**이다 — 양자화가 GEMM 안으로 들어가고(−1.5 ms), MHC 의 두 발이 한 발이 된 것
(−0.5 ms). 다음 후보를 개수로 정렬하지 말 것.

비교를 위해 — 같은 날 측정된 다른 축:

| 축 | 실측 |
|---|---|
| W8A8 dense 도입 (#92·#94) | **−6.9 ms (+10.4%)** |
| 커널 축 PR 6 개 합계 | −1.1 ms (+1.5%) |
| 스텝의 전문가 읽기 몫 | **41 ms (62%)** — 대역폭 91%, 짤 것 없음 |

## 보충 분해 (2026-09-01 — 기존 트레이스 재분석, 무부팅)

### 기타 150 개의 정체

| 내용 | /스텝 | 비고 |
|---|---|---|
| KDA recurrent (`fused_recurrent_gated_delta_rule`) | 34 | SSM 상태 읽·쓰기(~68MB/스텝)가 지배 — 대역폭 하한 |
| KDA short conv (`_causal_conv1d_update`) | 34 | 상기 recurrent에 붙는 전처리 |
| kpool 인덱서 갱신 (`_kpool_decode_update*` 등) | 22 | 11 풀어텐션 층 × 2 |
| 인덕터 글루 오분류 (`triton_poi_fused*`) | ~27 | elementwise 483 의 일부 (그룹 패턴 밖 이름) |
| `kernel_mha` | 5 | DFlash2 드래프터 5레이어 어텐션으로 판단 (드래프터 축 D≈0 닫힘) |
| 스펙 메타데이터 prep 1-오프 | ~16 | 각 2-5 us |

전부 0.1~0.6% class — 새 >CV 레버 없음.

### bf16 GEMM 145 개의 grid 귀속

트레이스 grid 차원으로 소유자 특정:

| grid | 개수 | 귀속 |
|---|---|---|
| (8,32,1) | 33 | **q_b+kv_b+wq_b (11x3) — bproj fp8 암의 표적과 정확히 일치** (bf16 중 최대 시간 4163us†) |
| (8,16,1) | 68 | f_b+g_b (34x2) — min(shape)=128 가드 제외 판정 재확인 |
| (8,48,1) | 5 | 드래프터 QKV 투영 (드래프터 축 닫힘) |
| (8,2,8) 등 splitK | 44 | wk_weights_proj·기타 소형 |

bproj 암(#131)이 이 리포에서 가장 큰 게이트 자산임이 데이터로 확정.

### KDA 꼬리 — 융합 후보가 소스 접촉으로 소멸

`o_norm` 꼬리(`core_attn_out.copy_(self.o_norm(core_attn_out, g2))`)를 융합
후보로 스코핑했다가 소스에서 반전: `layer_norm_gated_fwd`는
`y = x if out_dtype is None` — **norm 이 이미 in-place** (커널이 x에 직접 기록),
self `copy_`는 ATen 쇼트서킷으로 커널 미발생. conv→recurrent 융합만 남지만
상태 읽기가 지배라 0.1~0.2% — 채택 바 아래로 확정. 교훈: 인구조사의 그룹 합계는
**소스를 읽기 전까지는 개선 여지가 아니라 의문점**이다.

## 보충 분해 2 (2026-09-02 — 9월 1일 트레이스 재분석, 커널 밖)

앞의 지도는 GPU 커널만 셌다. 같은 트레이스(229 스텝)를 **스트림 합집합**으로
보면 스텝의 12%가 GPU 유휴이고, 그 위치가 하나다.

| 구간 (스텝 72.1 ms, 프로파일러 하) | 유휴 | 무엇 |
|---|---|---|
| 1.6 ms 지점 | 0.6 ms | DtoH 회수 뒤 드래프터 그래프 발사 대기 |
| 6.6~12.3 ms | **5.7 ms** | eager 입력 준비: aten 호출 1,028개, memcpy 45개, 1~3 us 커널 ~100개, 커널 사이 50~430 us 공백 (호스트가 발사를 못 따라감) |
| 12.5~14.0 ms | **1.43 ms** | `cudaGraphLaunch` — 1,640 노드 타깃 그래프 제출 호스트 소요 (50스텝 중앙, p10 1.34 p90 1.59); 노드당 ~0.87 us |
| 스텝 끝 | 0.8 ms | 샘플러 → 다음 스텝 전환 |
| 합 | 8.9 ms | |

프로파일러가 호스트를 부풀리므로 절대값은 과장이다. 프로파일러 없는 앵커는
원장의 "디코드 중 GPU 93%" (같은 정의) → 실제 유휴 4~5 ms/스텝. 원장의
"호스트 병목" 기각은 32 ms 잔차의 설명으로서 맞았고, 5 ms 짜리 항목으로는
살아 있다. dflash 는 async scheduling 을 잃지 않는다(2026-09-03 정정: `dflash` ∈ `EagleModelTypes`, 이미 켜져 있음)는 것과 준비
구간 동안 GPU 가 완전히 비는 트레이스가 일치한다.

준비 구간의 코드: `v1/worker/gpu/input_batch.py`·`block_table.py`(Triton 5개),
`model_states/mamba_hybrid.py`, KV 그룹 7개(MLA+indexer 1, kpool tail 1,
mamba 4, 드래프터 1)의 빌더 — GDN 빌더 4개가 각각 `to`·`sub`·`arange`·`index`×2·
`copy_`×5 를 내고(트레이스의 4회 반복 패턴), indexer 빌더가 `floor_divide`×3·
`diff`·fill·`_prepare_uniform_decode`·deep_gemm 스케줄을 낸다. 전부 이미지 파일이고
오버레이돼 있지 않다. 접수 방식은 `overlay/modules/glm53_runtime`(EXP-7):
파일을 덮지 않고 러너 메서드를 패치하며 preimage 를 고정한다.

**그래프 안 글루 605개의 정체**(같은 트레이스, 위치 귀속):

| 위치 | 개/스텝 | 무엇 | 처리 |
|---|---|---|---|
| 풀어텐션 11층 인덱서 | 275 | 꼬리 풀 강제 8개, seq_len 유도, fill, 확장·변환 주변 | 우리 `sparse_attn_indexer_kpool.py` — `VLLM_GLM53_KPOOL_FUSED_TOPK`(기본 off, 미측정)에 접기 |
| KDA 34층 | 136 | conv 뒤 `reshape` 가 split 뷰를 q/k/v 세 번 복사 | MK-KDA 가 흡수 |
| MoE 42층 | 42 | shared expert 덧셈 | osar copy-in 에 접기 |
| 나머지 | ~150 | 드래프터 inductor 글루, 꼬리 native RMSNorm, eager memcpy | 작음 |

커널당 점유는 CUPTI 로 길이 중앙 2.0 us + 그래프 안 간격 0.2 us — 원장의
5.4 us 상한과 부합하고, 605개 전부가 2.7 ms(부풀린 값)다.

**읽다가 나온 것**: (1) 인덱서의 fp32 head-gate `torch.mm(hidden.float(), _wp_fp32)`
가 cuBLAS gemmSN 2블록 커널로 층당 86 us, 11층 0.95 ms/스텝(CUPTI) — 우리
stock `attention.py` `Indexer.forward` 의 것(`FUSED_K_GATE=0` 이라 우리 fastpath 사본은 안 돈다),
split-K 로 수 us 감 → `glm53_indexer_gate_splitk` (EXP-9, opt-in, N=32 콜드 88 → 12.7 us,
~1.2%/스텝; 첫 판의 N=16 수치는 리뷰로 정정). (2) 드래프터 fc 투영
814 us(eager bf16, 5층 hidden cat, 168 MB 읽기, `ReplicatedLinear` 라 fp8 dense
패턴 밖) 포함 드래프터 커널 합 ~3 ms(CUPTI) — 원장 D≈0 과 긴장, 직접 측정 전
판단 보류. (3) `KpoolTailMetadataBuilder` 의 원형 tail 슬롯 매핑은 러너가
`positions` 를 안 넘겨 이 이미지에서 잠들어 있다(generic 매핑 사용) —
`glm53_tail_slot_persistent` 의 고정 버퍼도 그래서 무효; C>=2 수치 축.

## 보충 분해 3 (2026-09-02 밤 — 9월 1일 트레이스, 스텝 구성과 오프라인 레버 소진 판정)

같은 트레이스(rank 0, 223 스텝, 중앙 71.0 ms)를 **범주별 커널 시간**으로 자른 것
(`tools/trace_step_composition.py`). 스트림이 둘(210 = 본 그래프, 17 = 공유 전문가·헤드·
드래프터, 239 = 16개)이라 유휴는 **스트림 합집합** 기준이다: 합집합 busy 61.8 ms, 유휴
9.24 ms, 범주 합(65.0 ms)과 합집합의 차 3.2~3.6 ms 가 두 스트림의 동시 실행분. (첫 판은
단일 스트림 가정으로 유휴를 5.65 ms 로 과소 계산했다 — 리뷰로 정정; 보충 분해 2 의 합집합
8.9 ms/72.1 ms 와 이제 일치한다.)

| 범주 | ms/스텝 (중앙) | 개/스텝 | 비중 | 레버 |
|---|---|---|---|---|
| MoE 전문가 커널 (flashinfer cute-DSL) | 30.18 | 42 | 42.5% | 대역폭 바닥 (원장 273 GB/s 재확인) — 없음 |
| deep_gemm fp8×fp4 밀집 GEMM | 14.06 | 197 | 19.8% | EXP-6 MK-GEMM W4 (stock 대비 -20~30%/발사) |
| **GPU 유휴 (스트림 합집합 기준)** | 9.24 | — | 13.0% | **EXP-7** (async 는 이미 켜져 있고 이 구간을 못 가린다) |
| `k_oneshot` AR | 5.76 | 102 | 8.1% | p10 30.6 / p50 43.5 / p90 80.4 us — p10 이 고유 지연이면 ~2.5 ms 가 랭크 간 대기. EXP-7/8 이 호스트 편차를 줄여 같이 준다 |
| bf16/cuBLAS GEMM | 5.17 | 165 | 7.3% | EXP-4 bproj(33개), EXP-9 gemmSN(11개 × 88 us; N=32 콜드도 88 us, split-K 12.7 → **~1.2%**), 헤드 794 us |
| MoE 글루 (quant·act·topk·게이트) | 2.71 | 331 | 3.8% | 아래 게이트 항목 참조 — 없음 |
| MHC (TileLang 2종) | 2.54 | 185 | 3.6% | MHC 스윕(EXP-3) |
| KDA (recurrent·conv·norm) | 1.27 | 102 | 1.8% | EXP-6 MK-KDA |
| elementwise 글루 | 0.95 | 466 | 1.3% | 보충 분해 2 표 — 각 <0.5% |
| MLA 디코드 | 0.66 | 22 | 0.9% | — |
| nccl AllGather (헤드 로짓) | 0.52 | 3 | 0.7% | 452 us 1회 — TP 어휘 샤드 로짓 수집 |
| 인덱서 (logits·top-k·확장) | 0.39 | 44 | 0.6% | 층당 top-k 24 us + 소형 10개 ≈ 0.4% — 모듈 가치 없음 |
| 기타/드래프터·샘플러 분류분 | 0.79 | 176 | 1.1% | |

**스텝의 시간 구조(rank 0)**: [0~5.7 ms 준비: 0.84 ms 공백 ×4 = GDN 빌더 4개의 호스트
지연, 1.09 ms 첫 공백] → [5.7 ms 에서 1.98 ms 공백 = 그래프 제출] → [~7.7~62.8 ms 타깃
그래프] → [62.8~65.2 ms 헤드: fp8 GEMM 830 us + AllGather 452 us + bf16 GEMM 794 us]
→ [0.41 ms 공백] → [65.6~70.5 ms 드래프터: 소형 커널 3.5 ms + fc GEMM 809 us(M=7,
deep_gemm fp8×fp4 — 42~84 MB 를 809 us 면 52~104 GB/s, split-K 없는 K=20480 직렬
스케줄)] → 다음 스텝의 준비.

**D≈0 의 조건**: 드래프터 ~4.3 ms 는 다음 스텝의 호스트 준비(5.7 ms 유휴) 뒤에
숨는다 — 호스트가 병목인 동안만 공짜다. EXP-7(준비 0.15 ms)·EXP-8(스케줄 겹침)이
붙으면 은신처가 사라지고 드래프터가 임계경로에 올라온다(천장 ~6%: 소형 3.5 + fc
0.6). #104 판정을 지금 재론하는 것이 아니라, **EXP-7/8 이 붙은 뒤 D 를 다시 재야
한다**는 조건부 기록이다.

**게이트 커널은 레버가 아니다**: `_deneb_gate_partial_kernel` 은 그래프 안 중앙 37.5 us
지만 GLM 형상(E=288, K=4096, 2.36 MB bf16)에서 단독 콜드 13.6 us(stock F.linear 18.3).
BN∈{8,16,32}×BK∈{256,512,1024}×warps×stages 스윕(가중치 46개 순환, 그래프 리플레이)
에서 비트 동일 변형 60여 개가 전부 13.4~14.5 us — 더 빠른 타일이 없다. 그래프 안
초과분은 stream 17 의 공유 전문가 quant+GEMM 과 겹친 시간(42회 합 1.53 ms) 이라
임계경로가 아니다.

**판정**: 오프라인(무부팅)으로 닫을 수 있는 >1% 레버가 없다. 남은 것은 부팅 게이트 —
EXP-7(유휴 13% 가 표적) > EXP-6 MK(W4 GEMM 20~30%/발사 × 14 ms) > EXP-9(~1.2%) >
EXP-4(0.9%). EXP-8 은 기각(이미 켜져 있었다, 2026-09-03).

## 보충 분해 4 (2026-09-04 — 09-03 무장 트레이스, 스텝의 꼬리)

21차는 `execute_context_0(0)_generation_1(8)` 주석 안만 봤다. 그 주석은 **타깃 forward**
이고(스트림 17 과 203 에 하나씩 — 6개 주석 = 3 스텝; 21차 표의 ms/step 은 그래서 절반
값이고 비중은 유효), 러너의 준비 커널(`_gather_block_tables_kernel`)에서 자르면 스텝은
이렇다(rank 0, 두 깨끗한 스텝):

| 구간 | ms | 안에 있는 것 |
|---|---|---|
| 타깃 forward | 58.7 / 62.4 | 21차 표 그대로 (MoE 31, mk_gemm 13.5 = 173발 중 126발이 공유 전문가 스트림 203, AR 6.7 …) |
| **꼬리** | **7.5 / 8.5** | 타깃 헤드 fp8 836 us + AllGather 409~567 + 샘플러 ~0.1 → 드래프터: **fc bf16 792 us**(cutlass wmma, 168 MB, 212 GB/s), 5층 × ~660 us(GEMM 51+66+43+52+236+118 = ~567 us bf16, 200~235 GB/s 의 DRAM 하한 + AR 2회 ~80 + 글루), 드래프트 헤드 fp8 812 us(W8A16, 158 MB, ~195 GB/s), 셀렉터/topk ~0.15 |

꼬리 동안 GPU 는 95% 바쁘다(union busy 7.1/7.46, 8.06/8.46) 이고 다음 스텝의 준비
커널이 꼬리 끝에서 시작한다 — 임계경로다. 드래프터의 bf16 GEMM 3.2~3.6 ms/스텝이
디코드에 남은 가장 큰 커널 쪽 레버이고, 오프라인 프로브(RUNBOOK EXP-10)로 MK W4 가
1.25 ms 까지 내린다(−3.0%/스텝). 드래프터는 이 부팅에서 TP=4 로 돌았다(층마다
o_proj/down_proj 뒤 `k_oneshot`, qkv 12.6 MB 스트림) — 프로필의 `DRAFT_TP=1` 은 서빙에
닿지 않았고, 대역폭으로는 TP=4 가 맞는 쪽이다. 재현: 스크래치의 `tail_region2.py`
패턴 — 스텝을 준비 커널에서 자르고, 스텝 안에 완전히 들어오는 주석만 forward 로 본다.

**그 뒤(2026-09-04)**: 드래프터 W4 는 28차 §4 브래킷을 통과해 19:38 부터 기본값이고
(step/s 15.95 → 16.235, −1.1 ms), 꼬리의 bf16 3.2~3.6 ms 중 그만큼이 MK 레인으로
갔다. 남은 꼬리 레버는 RUNBOOK EXP-16(드래프터 메가커널, 정정 상한 −0.8~1.0 ms)
하나다. 무장 트레이스로 다시 잰 꼬리는 보충 분해 5.

## 보충 분해 5 (2026-09-04 — 09-04 무장 트레이스, 무부팅 재분석)

28차 §3 이 프로파일한 그 부팅(18:25 cand — MK MHC+GEMM 무장, MK-MLA off, 드래프터는
fc 만 W4, 서빙 PDL 은 그 브랜치에 아직 없었다)의 트레이스를
`tools/trace_step_composition.py` 로 다시 잘랐다. 깨끗한 **18 스텝, 중앙 69.25 ms**,
유휴 8.18 ms, 두 스트림 동시 실행분 6.63 ms.

| 범주 | ms/스텝 | 발/스텝 | 비중 | 09-01 스톡(보충 분해 3) |
|---|---|---|---|---|
| MoE 전문가 (flashinfer cute-DSL) | 32.52 | 42 | 47.0% | 30.18 / 42 |
| **MK GEMM (우리)** | 14.13 | 185 | 20.4% | deep_gemm 14.06 / 197 |
| **GPU 유휴 (스트림 합집합)** | 8.18 | — | 11.8% | 9.24 |
| bf16/cuBLAS GEMM + 헤드 게이트 | 6.39 | 201 | 9.2% | 5.17 / 165 |
| `k_oneshot` AR | 5.28 | 102 | 7.6% | 5.76 / 102 |
| **MK MHC (우리)** | 2.03 | 89 | 2.9% | MHC 2.54 / 185 |
| lm_head 둘 (deep_gemm fp8) | 1.66 | 2 | 2.4% | (bf16 GEMM 안에) |
| KDA (recurrent·conv·norm) | 1.33 | 102 | 1.9% | 1.27 / 102 |
| MoE 글루 | 1.20 | 136 | 1.7% | 2.71 / 331 |
| elementwise 글루 | 1.00 | 481 | 1.4% | 0.95 / 466 |
| 로짓 AllGather | 0.63 | 3 | 0.9% | 0.52 / 3 |
| MLA 디코드 | 0.41 | 22 | 0.6% | 0.66 / 22 |
| 인덱서 | 0.39 | 44 | 0.6% | 0.39 / 44 |
| 기타 | 0.41 | 110 | 0.6% | — |
| 드래프터/샘플러 분류분 | 0.17 | 11 | 0.2% | — |

⚠ 오른쪽 열은 **다른 부팅**(09-01 스톡)이다 — 브래킷이 아니라 같은 도구로 잰 두 장의
사진이다. 종단 step/s 판정은 원장의 몫이고 이 표가 대신하지 않는다. 그래도 읽히는 것:
**MoE·AR·KDA·글루·인덱서는 그대로**이고, 움직인 세 칸이 MK GEMM(양자화를 삼킴),
MHC(발사 절반), MoE 글루(−1.5 ms)다.

**스텝의 두 토막** — 준비 커널(`_gather_block_tables_kernel`)에서 자르고, 그 안에서
마지막 MoE 전문가 커널로 다시 자르면:

| 구간 | ms | 무엇 |
|---|---|---|
| 타깃 forward | 61.88 | MoE 32.5 + MK GEMM 대부분 + AR + MHC + KDA |
| **꼬리** | **7.37** | GPU 는 이 동안 93% 바쁘다(union busy 6.84) — 임계경로 |

| 꼬리 항목 | ms | 발 |
|---|---|---|
| 드래프터 5 층의 bf16 cutlass GEMM | **2.88** | 34 |
| lm_head 둘 (타깃 832 us + 드래프트 824 us, fp8 W8A16) | 1.66 | 2 |
| 로짓 AllGather | 0.63 | 3 |
| 드래프터 AR (`k_oneshot`) | 0.54 | 12 |
| 드래프터 fc — **이미 MK W4 레인** | 0.50 | 6 |
| 드래프터 어텐션 `kernel_mha` | 0.11 | 5 |
| 글루·샘플러·캐시 기록 | ~0.42 | ~120 |

보충 분해 4(09-03 트레이스)의 꼬리 7.5~8.5 ms 와 같은 그림이고, 그때 792 us 였던
fc 가 여기서는 mk_gemm 6 발 500 us 다 — 28차 §3 의 "fc 만 서빙" 팔이 이 트레이스다.

### 오늘의 기본값은 이 트레이스보다 한 발 앞서 있다 (미실측)

| 노브 | 기본값이 된 시각 | 지도에서 움직일 칸 | 원장 근거 |
|---|---|---|---|
| 드래프터 W4 **전량** (`VLLM_DFLASH2_FP8_DENSE=1` + 컴파일 캐시 키 수정) | 09-04 19:38 (cf2e16c) | 꼬리의 bf16 cutlass 30 발 → mk_gemm | 28차 §4: step/s 15.95 → **16.235** (−1.1 ms) |
| `VLLM_GLM53_MK_MLA=1` | 09-04 19:55 (414b1bc) | MLA 디코드 22 발 → `mk_mla_kernel` | 28차 §5: 디코드 +1.0%, 프리필 +15~18% |
| `VLLM_GLM53_MK_PDL=1` | PR #290 (09-04) | mk_gemm 발당 −7.6% | 27차: 58.0 → 53.6 us, 스텝 −0.76 ms. **우리 커널의 발사 속성**이고, 이미지가 GB10 에서 봉인한 TileLang mhc 의 PDL 과는 다른 축이다(원장 09-04) |
| 마스터 `VLLM_GLM53_MEGAKERNEL=1` | 09-04 20:35 (3db84dc) | (이 트레이스가 이미 무장) | 28차 §8 |

다음 캡처에서 **먼저 확인할 세 줄**: (1) `mk_mla_kernel` 11 발/스텝이 있는가(없으면
MLA 는 또 공허 무장이다), (2) cutlass bf16 201 → ~171 인가, (3) mk_gemm 185 → ~215
인가. 세 개가 다 맞아야 "오늘의 기본값이 서빙되고 있다" 는 말이 트레이스로 증명된다.

### 지도를 다시 그리며 고친 도구 셋

1. **분류기가 코드보다 늦어 있었다** — `tools/trace_common.py::category()` 가
   메가커널을 몰라서 mk_gemm+mk_mhc 274 발 14.5 ms 를 `other` 로 흘렸다(그래서 첫
   실행의 1 위 범주가 "기타" 였다). MK GEMM/MHC/MLA/KDA·prep_fused 범주를 넣고,
   이 리포에서 컴파일되는 커널을 판정하는 `owner()` 를 같은 파일에 뒀다.
2. **`census.py` 의 osar 정규식이 남의 커널을 세고 있었다** — 앵커 없는 `k_reduce`
   가 deep_gemm 의 `sm120_split_k_reduce_impl` 42 발/스텝을 우리 AR 로 셌다. `^`
   앵커로 고쳤고, 이 문서의 "우리 소유 186 개(9.9%)" 는 **144 개(7.6%)** 로 정정.
   문서 안의 구조표(`k_oneshot` 102)와 그룹표(144)가 어긋나 있던 것이 신호였다.
3. **`census.py` 가 부팅 중인 노드에서 위험했다** — `json.load` 로 트레이스를 통째로
   올려 40 MB 캡처에 수 GB 를 썼다. 로드 중인 노드의 MemAvailable 은 한 자릿수 GiB
   까지 떨어지고(26차) 그 상태의 큰 파이썬 프로세스는 earlyoom 감이다. 이벤트 하나씩
   흘려보내는 파서로 바꿨다 — **최대 RSS 17 MB**, 08-31 트레이스 출력은 이전 판과
   **비트 동일**, 실행 시간은 오히려 짧다.

## 보충 분해 6 (2026-09-05 — 프로덕션 부팅의 첫 노드 인구조사, 34차)

체인22 PRODSET 부팅(v2 레인 + prep-fused + MK-MLA + 드래프터 W4 + 서빙 PDL; 그 부팅은 KDA·
LOCALQ·AR 프리페치까지 켠 팔이었다)에서 `bench/profile-step.py decode:600` 으로 rank 0·3 을
캡처했다(245 스텝, 프로파일러 아래 55.8 ms/스텝). 새 도구 `tools/trace_step_nodes.py` 는
스트리밍이라 부팅 중 노드에서도 안전하고(RSS 수십 MB), 커널 수가 최빈값인 "깨끗한 스텝"
239개의 중앙값을 낸다.

| | 09-04 무장 | **09-05 프로덕션** |
|---|---|---|
| 커널/스텝 | 1,582 | **1,166** (−416) |
| 유휴(스트림 합집합) | 8.18 ms | **1.20 ms** (prep-fused) |
| 밀집 GEMM | mk_gemm 185 발 14.13 | **mk_gemm2 147 발 3.52** (KDA 레인이 in/o_proj 68 발을 삼킴, v2 발당 24 µs) |
| bf16 cutlass/cublas | 201 발 6.39 | **103 발 3.34** (드래프터 30 발이 W4 로; 남은 22 × 67 µs = MLA 층의 q_b·wq_b) |
| mk_mla | 0 | **11** (보충 5 의 "다음 캡처 3줄" 중 첫째 확인) |
| MoE 전문가 | 42 발 32.5 | 42 발 29.4 |
| elementwise 글루 | 481 발 1.00 | 318 발 0.68 |

보충 분해 5 가 미실측으로 남긴 세 줄: (1) `mk_mla_kernel` 11 발/스텝 **있다**, (2) cutlass bf16
201 → **103**, (3) mk_gemm 185 → v2 레인 **147**(KDA 레인 on 이라 −68) — 셋 다 오늘의 기본값이
서빙되는 것으로 확인됐다.

**프로파일러 자신의 두 가지 왜곡**(전 범주 이벤트 창 `--gap` 으로 확인): (a) 스텝 시작의
타깃 그래프 `cudaGraphLaunch` 가 rank 0 에서 2.8 ms(안에 CUPTI "Activity Buffer Request" 1.6 ms)
— 스텝 시작 갭 1.7 ms 는 프로파일러 값이고 rank 3 은 0.4 ms; 31차의 "그래프당 7~10 µs" 가
프로파일러 없는 값이다. (b) 호스트 파이썬도 부풀어 아래 드래프터 갭의 절대값은 상한이다.

**드래프터 앞 갭**(스텝의 유일한 GPU 유휴 덩어리): `precompute_and_store_context_kv` 의 마지막
`reshape_and_cache` 뒤 `Memcpy DtoH (Device -> Pageable)` 3.6 µs + `cudaStreamSynchronize` →
파이썬 ~290 µs(rank 0)/~460 µs(rank 3) → 드래프터 그래프 `cudaGraphLaunch`(192/259 µs) → 첫
커널. rank 0 의 드래프터 첫 `k_oneshot` 이 245 µs = rank 3 의 더 긴 갭을 기다리는 시간. 정체는
`DFlashSpeculator.propose` 의 `_build_draft_attn_metadata`(FlashInfer 빌더가 비인과 어텐션이라
`seq_lens.cpu()` 를 읽고 FULL 재생이 안 읽는 dict 를 만든다) — `glm53_drafter_prep` 이 처방.

**MLA 층 한 개의 노드 열(rank 0, 중앙 스텝)**: mk_mhc 19 → qkv_a(mk_gemm2) 24 → q/k norm 2 →
**q_b 67 → wq_b 66**(cutlass bf16 128x1, 12.6 MB 씩) → wk_weights_proj 10+2 → 복사 3.5 → **head gate
`gemmSN` 89** → layer_norm 1.8 → fwht_quant 5.4 → mul 1.2 → gate_score 8+2 → fill 1.1 → 복사 1.7 →
kpool_decode_update 9.3 → mqa_logits 4.6 → [seq_len·tail 강제 10 발 ≈ 18] → topk 2.5+23.3 →
expand 1.2 → 복사 1.9 → concat_and_cache 3.3 → **q·W_UK 26.5** → fill·fill·convert·복사 7 →
mk_mla 41 → **W_UV 22** → o_proj(mk_gemm2) 51 → k_oneshot. 층당 ~500 µs 가 한 스트림에 직렬이고
그중 bf16 GEMM 4발 200 µs(EXP-4 의 자리), head-gate 89(EXP-9), 글루 12발 ~30(EXP-24 융합).

**MoE 층의 두 스트림**: aux 스트림에 gate_partial 20 → topk 3.8 → MoE 655~1,000 → add 1.8 →
k_oneshot 이 직렬(임계경로), 메인 스트림에는 공유 전문가 쌍(v2 25 + act 1.9 + v2 8)뿐이라
1.2~1.4 ms 씩 논다. gate_partial 은 공유 gate_up 과 겹쳐 20~23 µs(단독 콜드 13.6).

**꼬리(드래프터)**: 층당 norm 2.5 → kernel_projection(mk_gemm2) 15.5 → conv 글루 2.4 → qkv 13.3 →
q/k norm 1.6 → rotary 2.4 → cache 3.2 → kernel_mha 31 → o_proj 31 → k_oneshot 64 → post norm 6.8
→ kernel_projection 13 → 글루 2 → gate_up 41 → act 2.2 → down 50 → k_oneshot 51 ≈ 335 µs × 5.
드래프터 GEMM 은 전부 v2 레인(EXP-10 이 서빙된다). 앞의 fc 5 청크(mk_gemm2 60~75 + 청크 사이
글루 ~10) 390 µs 가 EXP-15 의 자리, `precompute_and_store_context_kv` 의 bf16 GEMM 101 µs(K/V
문맥 사영, 레인 밖 "1 of 31")가 남는다.

재현: `python3 tools/trace_step_nodes.py <trace.gz> --dump step.txt --gap 52400,53200`.

## 재현

```bash
# 1) 캡처 — 부팅된 서버에서 (프로파일 캡처는 엔진을 죽일 수 있다: 레그 마지막에)
curl -X POST localhost:8000/start_profile && sleep 12
curl -X POST localhost:8000/stop_profile

# 2) 인구조사 — 개수·그룹·소유권 (스트리밍 파서, RSS ~17 MB)
python3 census.py ~/vllm-prof/dp0_pp0_tp0_dcp0_ep0_rank0.*.pt.trace.json.gz

# 3) 시간 구성 — 깨끗한 스텝 중앙값, 유휴는 스트림 합집합
python3 tools/trace_step_composition.py <trace.gz>            # 한 장
python3 tools/trace_step_composition.py <base.gz> --diff <cand.gz>

# 4) 꼬리 — 마지막 MoE 전문가 커널 뒤(헤드·샘플러·드래프터)
python3 tools/trace_step_tail.py <trace.gz>
```

- `census.py` 는 스텝 분모를 `_get_num_sampled_and_rejected` 로 **세고**(추정 금지),
  그룹별 커널/스텝 · **소유권(우리 리포 vs 이미지)** · 상위 25 커널을 낸다.
- 소유 판정 목록은 `tools/trace_common.py` 의 `OURS` 하나뿐이다 — 새 커널을 만들면
  거기에 심볼을 넣어야 지도가 그걸 우리 것으로 센다.
- **플릿 규율**: 이 분석들은 rank 노드의 CPU 를 쓴다. 디코드 레그가 도는 동안에는
  돌리지 말고(21:08 사고: 단일 코어 파이썬이 step/s 15.2 → 9.1), 부팅 창에서
  `nice -n 19 taskset -c 19` 로. 큰 트레이스라도 메모리는 이제 문제가 아니다(위 3번).
- 이 문서의 트레이스: srv2 `~/vllm-prof/` 의 rank 0 캡처 —
  `…1788162670231631279`(08-31 스톡) · `…1788242845451080109`(09-01 스톡) ·
  `…1788514956111126568`(09-04 무장). rank 3 사본은 srv4 의 같은 경로에 있다.

