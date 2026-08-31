# 스텝 커널 지도 — 우리가 안 만든 것들

C=1 디코드 스텝의 커널 1,886 개 중 우리 소유는 **186 개(9.9%)** 뿐이다. 나머지
1,700 개가 어디서 나오고, 무엇이 그걸 정하고, 손댈 수 있는지를 적는다. 이 문서는
"우리 커널만 최적화 대상"이라는 잘못된 전제를 깨기 위한 것이다 — 이 레포의
오버레이 체계 자체가 벤더 코드를 접수해 고치는 물건이고, `b12x_moe.py`·
`gpu_model_runner.py`·`glm5next_model.py` 는 이미 그렇게 쓰고 있다.

측정: 2026-08-31, 프로파일 캡처(`start_profile`), 트레이스 328,164 이벤트 ·
고유 126 종. **스텝 분모는 추정하지 않고 셌다** — 스텝당 1 회 도는
`_get_num_sampled_and_rejected_kernel` × 174. 분석기는 `census.py`(아래).

> ⚠ 트레이스의 **절대 시간은 쓰지 말 것**. CUPTI 가 GPU 바쁜 시간을 부풀린다
> (앞선 세션: 136 ms vs 실측 스텝 47.6 ms). **개수만 정본**이다.

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

| 그룹 | /스텝 | 비중 | 출처 | 손댈 수 있나 |
|---|---|---|---|---|
| elementwise 글루 | **483** | 25.6% | torch/inductor 생성 | **설정** — `custom_ops` 가 융합 장벽 |
| cutlass/cublas GEMM | 397 | 21.0% | cuBLASLt + deepgemm | 일부 — bf16 145 개는 양자화 표적 |
| 정규화/양자화 | 301 | 16.0% | vLLM 커스텀 op + W8A8 대가 | 일부 — `fuse_*_quant` 는 이미 True |
| mhc (TileLang) 2 종 | 185 | 9.8% | 이미지의 glm5next 모델 코드 | **예** — 오버레이로 접수 가능 |
| 기타 | 150 | 8.0% | 혼합 | |
| **osar AR (우리)** | 144 | 7.6% | `tp_oneshot_ar` | 이미 5→1 |
| MoE b12x | 99 | 5.2% | flashinfer | **예** — #101 로 이미 접수 |
| 복사/산포/수집 | 47 | 2.5% | torch | |
| **게이트 (우리)** | 42 | 2.2% | `moe_gate_sm121` | #96 로 반감 |
| 어텐션 (MLA) | 33 | 1.7% | flashinfer/trtllm | |
| 샘플러/스펙 | 5 | 0.3% | vLLM | |

### ① elementwise 483 개 (25.6%) — 가장 큰 덩어리, 원인은 설정

45 층에 483 개면 **층당 10.7 개**다. inductor 는 이런 걸 주변 커널에 접어
넣는 게 일인데(`combo_kernels: True` 도 켜져 있다) 483 개가 살아남았다.

원인은 `custom_ops: ['all', '+quant_fp8', '+quant_fp8']` 로 보인다. **커스텀 op
하나하나가 inductor 의 융합 장벽**이라, 그 사이의 residual add·scale·mask 가
독립 커널로 남는다. 즉 `custom_ops` 는 *"수제 커널이 빠르다"* 와 *"융합이 커널을
없앤다"* 의 교환인데 **이 스택에서 A/B 된 적이 없다.**

- 시험: `custom_ops` 를 `['none']` 또는 선택 목록으로 놓고 부팅 1 회. **코드 변경 0.**
- 상한: 483 × 5.4 us = 2.6 ms = 스텝의 3.9%

### ② cutlass bf16 GEMM 145 개 — 어디가 아직 bf16 인가

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

### ③ 정규화/양자화 301 개 — W8A8 의 대가가 보인다

`per_token_group_quant_8bit_packed_register_kernel` 이 **137 + 42 = 179/스텝**.
dense GEMM 마다 활성화를 fp8 로 양자화하는 비용이다. #94 가 −6.9 ms 를 벌면서
약 1 ms 를 여기에 지불한다(5.4 us/커널 기준). 순이득이지만 **공짜가 아니다.**

`pass_config` 의 `fuse_norm_quant`·`fuse_act_quant` 는 이미 `True` 로 확정된다.
137 개가 남았다는 것은 그 융합이 닿지 않는 자리가 있다는 뜻이다.

### ④ mhc 185 개 (9.8%) — 우리 AR 보다 많다

`mhc_pre_big_fuse_with_norm_tilelang_kernel` 90 + `mhc_fused_tilelang_kernel` 89.
**TileLang 커널**이고 45 층마다 두 번 돈다. GLM-5.3-next 고유의 Multi-Head
Compression(`mhc_sinkhorn_iterations` 등)이다.

이미지의 모델 코드에서 나오므로 **오버레이로 접수해 고칠 수 있다**(dsv4 레인엔
이미 `dsv4_mhc_tilelang` 모듈이 있다). 둘을 하나로 접으면 −89/스텝(1.5%).

### ⑤ MoE b12x 99 개 — 제약이 문서화돼 있다

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
| 실측 가격 | **5.4 us/커널** (#89 의 −209 커널 → −1.13 ms) |
| 전부 없애면 | 1,886 × 5.4 us = **10.2 ms = 스텝의 15.4%** |
| 우리 소유만 | 186 개 = 1.0 ms = **1.5%** |

⚠ **그런데 이 근거는 강해지지 않고 약해지고 있다.** #89 이후 #90·#93·#96 이
324 개를 더 없앴는데 W8A8 몫을 뺀 스텝은 변하지 않았다(예상 65.8 vs 실측 66.1).
5.4 us 는 **상한이고 실제는 더 작을** 공산이 크다. 위 표의 "없애면" 값은 낙관치로
읽어야 한다.

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

## 재현

```bash
# 부팅된 서버에서 (프로파일 캡처는 엔진을 죽일 수 있다 -- 마지막에)
curl -X POST localhost:8000/start_profile
sleep 12
curl -X POST localhost:8000/stop_profile
python3 census.py /home/choiceoh/vllm-prof/*.pt.trace.json.gz
```

`census.py` 는 스텝 분모를 `_get_num_sampled_and_rejected` 로 **세고**, 그룹별
커널/스텝과 상위 25 개 개별 커널을 낸다. 앞선 인구조사가 스텝 수를 **추정**해서
틀렸으므로 그 부분을 고정했다.
