# sm_121a 업스트림 도입 — vLLM·SGLang 의 GB10 작업 중 이 엔진에 들일 것

> 살아 있는 참조 — **고정된 후보 목록과 항목마다의 상태. 항목이 닫히면(머지·기각) 이 표의 상태 칸을 고친다.** 여기가 틀리면 그건 버그다.

출발점은 2026-09-19 의 조사다. vLLM·SGLang 에서 8~9 월에 SM12x(sm120/sm121, GB10, DGX Spark)를 다룬 PR
275 개를 훑고, 이 엔진의 입구(`tools/onboard.py`, `engine/kernels/cells.py`)와 코드에 대 보았다. 질문은
"다른 새 모델이 오면 이 엔진이 얼마나 서빙할 준비가 됐나"였고(D5: 하드웨어는 특정, 모델은 한정하지 않는다),
그래서 이 목록의 첫 기준은 **입구가 '거절'·'변환'을 내는 자리를 줄이는 것**이다.

## 운영자 결정 (2026-09-19)

| # | 결정 |
|---|---|
| I1 | 분류한 항목 **전부 도입**("전부 도입"). 이미 있는 것과 안 가져오는 것(아래 끝 절)은 빼고 |
| I2 | 이 목록은 여기서 고정한다. 항목마다 머지 또는 기각과 기록으로 닫는다. 도중에 나온 새 아이디어는 다음 목록으로 |

## 규칙 — 이 목록이 헌장과 만나는 자리

- **D3 폴백 금지.** 업스트림의 "없으면 None 을 돌려 다음 백엔드로"(예: SGLang `try_sm120_fp8_linear`)는 그대로
  가져오지 않는다. 새 커널은 **입구가 형상으로 고르는 레인**이 되고, 자격(오라클 대비)을 못 맞추면 부팅이 죽는다.
  "메가커널 인스턴스가 없는 MLA 형상은 Triton 레인"도 런타임 분기가 아니라 `cells` 판정이다.
- **D8 커널 언어.** 가져오는 코드는 원래 언어 그대로(CuTe-DSL·Triton·.cu). 통일을 위한 재작성은 안 한다.
- **D17.** 속도가 근거인 항목(B 절)은 단일 레인 수치로 **커널 증거**까지 가고, 최종 채택은 플릿 onepass 다.
  플릿 창은 운영자의 명시 지시로만 연다.
- **의존성.** 런타임 이미지는 `--network none` 으로 굳은 시드다(`engine/runtime/dependencies.json`:
  flashinfer 0.6.18.dev20260819, cutlass-dsl 4.6.2, triton 3.7.1, CUDA 13.2.1). 이미지에 이미 있는 것은
  **바인딩**, 없는 것은 **벤더링**(`engine/kernels/SOURCES.json`·`THIRD_PARTY_NOTICES.md` 에 출처·라이선스)이다.
  어느 쪽인지는 U0 이 GB10 이미지 안에서 정한다.

## 항목

판정: `cpu` = WSL 두 레인(`tools/check.py --list`, 커널은 `TRITON_INTERPRET=1`), `gpu` = 단일 GB10 레인(오라클 대비),
`fleet` = 플릿 onepass(D17). 상태: `열림` → `PR #N` → `머지`/`기각`.

### 0. 준비

| ID | 무엇 | 출처 | 종류 | 판정 | 상태 |
|---|---|---|---|---|---|
| U0 | 이미지 인벤토리: 아래 항목의 후보 구현이 시드 이미지 안에 있는지(import·시그니처) | — | measure | gpu | 닫음: `sm121-inv-0919a`, [기록](../measurements/sm121_inventory_20260919/README.md) — U3·U5·U6·U7·U8·U9·U13 이 이미지 안에 있다 |

### A. 모델 범위 — 입구의 '거절'·'변환'을 줄인다

| ID | 무엇 | 출처 | 종류 | 판정 | 상태 |
|---|---|---|---|---|---|
| U1 | 입구가 `hf_quant_config.json` 을 읽는다: quant_algo(NVFP4·W4A16_NVFP4·FP8·MXFP4·MIXED_PRECISION 층별), group_size, kv_cache_quant_algo, exclude_modules. 설정 읽기의 나머지 빈틈도(바깥 `mtp_config`, `dense_intermediate_size`, `local_layer_ids`/`sliding_window_size`, 바깥 `model_type`) | vllm#56050, #56535 | door | cpu | PR (이 PR) |
| U2 | 모양(`kernel_shape.Attention`)이 윈도·상대 편향·k/v conv 를 말하고, `cells` 가 그런 어텐션을 평범한 GQA 로 통과시키지 않는다(D3) | 조사(Inkling) | door | cpu | PR (이 PR) |
| U3 | MoE MXFP4 **W4A8** — b12x 의 MXFP4 전문가를 FP8 활성으로(cells 레시피 (c)) | sglang#34878 | kernel | gpu | 열림 |
| U4 | MoE 활성 정밀도를 체크포인트가 정한다 — W4A16 체크포인트가 W4A4 로 돌지 않게(U1 이 읽은 값으로 셀 선택) | vllm#56535 | fix | cpu+gpu | PR (이 PR): 입구가 `nvfp4-a16` 으로 읽고 셀이 이름으로 거절 — A16 셀 자체는 다음 목록 |
| U5 | dense NVFP4 GEMM 레인 — 어텐션 투영·공유 전문가까지 NVFP4 인 체크포인트를 변환 없이 | sglang#38685, #38170, vllm#54614 | kernel | gpu | 열림 |
| U6 | sparse-MLA Triton 레인 — 메가커널 인스턴스가 없는 MLA/DSA 형상(DSv3.2·GLM-5.2 류)을 입구가 이 레인으로 판정 | vllm#54929 (대안 vllm#54976 B12X) | kernel | gpu | 열림 |
| U7 | GQA 레인 — paged KV + 윈도 + sink(가장 넓은 가족: gpt-oss·Gemma·Mistral·Qwen3 류) | vllm#50022, #55078(재료) | kernel | gpu | 열림 |
| U8 | KV 캐시 양자화(FP8, NVFP4) — 긴 컨텍스트와 큰 모델을 128 GB 에 | vllm#46329, #55976, #56550, #55557, sglang#37798 | kernel | gpu+품질 | 열림 |

### B. 이미 서빙하는 모델의 속도 (커널 증거까지; 채택은 플릿)

| ID | 무엇 | 출처 | 종류 | 판정 | 상태 |
|---|---|---|---|---|---|
| U9 | GLM KDA chunked prefill 을 FlashKDA 로(업스트림 GB300 1.7~3.8×; sm_121a 빌드부터) | vllm#55737 | kernel | gpu→fleet | **기각(하드웨어)**: FlashInfer 의 FlashKDA 는 `_FLASH_KDA_SUPPORTED_COMPUTE_CAPABILITIES = {(10, 0), (10, 3)}`(kda_prefill.py:40) — SM100 의 tcgen05/TMEM 커널이라 sm_121a 에서는 부를 수 없다. 업스트림의 1.7~3.8× 는 GB300 수치. 이식은 재설계(D8)라 다음 목록으로 |
| U10 | skinny FP8 GEMM(M=1 GEMV, M≥2 CUTLASS)을 `dense/fp8_rows`·W8A16 과 GB10 에서 대조 | sglang#38082 | measure | gpu | **기각(측정된 상한)**: 이 엔진이 FP8 로 읽는 디코드 행은 헤드뿐이고(나머지 dense 는 W4A8), 헤드의 W8A16 은 이미 가중치를 한 번 읽는 바닥의 96% 다 — `glm53-head-0919a` 8 행 731.3 vs read-only 703.5 µs, 16 행 742.9 vs 699.6 µs([기록](../measurements/glm53_decode_rows_20260919/README.md)). 어느 skinny GEMM 도 4~6% 넘게 줄일 수 없다 |
| U11 | FP8 prefill GEMM 의 L2 절벽(가중치 > 24 MiB, M ≥ 8k) — 우리 cuBLASLt 에도 있나, 있으면 래스터 스위즐 | vllm#55180 | measure | gpu | 열림 |
| U12 | Qwen3.8 QSA prefill 타일 합집합(연속 행이 고른 블록의 합집합을 한 번씩) | vllm#55430 | kernel | gpu→fleet | 열림 |
| U13 | Qwen3.8 GDN prefill 을 FlashInfer 로, GDN gate 투영 | vllm#55715, #57318 | kernel | gpu→fleet | 열림 |
| U14 | Qwen3.8 PLE 표를 NVFP4 로 묶어 상주(파일 읽기와 메모리의 교환) | vllm#56273 | kernel | gpu+품질 | 열림 |

### C. GB10 정확성 — 모든 모델, 조용히 틀리는 종류

| ID | 무엇 | 출처 | 종류 | 판정 | 상태 |
|---|---|---|---|---|---|
| U15 | 로드된 id 로 주소를 만드는 gather 의 clamp·mask 감사, 단일 레인 compute-sanitizer 한 번 | vllm#49049 | audit | cpu+gpu | 감사 끝(아래) — 서빙 경로 버그 없음, 방어 빈틈 1, 죽은 커널 2. sanitizer 실행 남음 |
| U16 | 긴 컨텍스트(≥120k) 디코드 정확성 검사 — SM121 에서만 토큰 0 을 내던 종류 | sglang#36845 | test | gpu | 열림 |
| U17 | PDL: wait 앞의 읽기가 부팅 상수뿐인가 | sglang#38290 | audit | cpu | 닫음: 감사(아래) — 버그 없음 |
| U18 | 캡처 뒤 패딩·null 슬롯의 비유한 값(0×NaN)이 실제 행을 오염시키나 | vllm#57158 | audit | cpu | 닫음: 감사(아래) — 버그 없음 |
| U19 | 드래프터 상태가 TP 랭크마다 어긋나는 자리 | sglang#33614 | audit | cpu | 감사 끝(아래) — GLM 안전, Qwen3.8 은 랭크 간 대조가 없다 |
| U20 | DeepGEMM 스케일: FP32 스케일(2 의 거듭제곱 아님)을 받으면 부팅에서 거절 | sglang#39482, vllm#57512, #54600 | fix | cpu | 브랜치 `sm121-u20-scale-guard`(PR 전: GLM 부팅이 지나는 `FP8Linear` 바인드를 바꾼다) — 우리 스케일은 이미 UE8M0(`packing.fp8_block_scales`, `fp8.py:14`) |

### D. 통합 메모리·플랫폼

| ID | 무엇 | 출처 | 종류 | 판정 | 상태 |
|---|---|---|---|---|---|
| U21 | 통합 메모리 회계: NVML 이 장치 메모리를 못 읽을 때, 프로세스 자신의 사용량으로 KV 를 잰다 | vllm#57378, #49760, #55828 | audit | cpu | **닫음(이미 있음)**: 엔진은 NVML 을 쓰지 않는다. KV 는 부팅 전에 선언한 예산(`engine/base/budget.py` "GATE: KV … declared before load")이고, 아레나(`engine/base/arena.py`)는 `/proc/meminfo` 의 MemAvailable 과 earlyoom 하한으로 받으며, `box.check_box` 가 장치 총량 == MemTotal(통합 메모리)을 단정한다 — vllm#57378·#49760·#55828 의 세 문제가 설 자리가 없다 |
| U22 | 가중치 스트리밍(O_DIRECT, 읽기 전용 매핑)을 `mapped_staging` 과 부팅 시간으로 대조 | sglang#37680, #38441 | measure | gpu | **닫음(이미 있음)**: `engine/base/loader.py` 가 같은 설계다 — 연속 바이트 구간을 O_DIRECT 로(페이지 캐시가 아레나와 같은 풀이라), 핀 버퍼 둘로 읽기와 업로드를 겹치고, 텐서는 장치 블록 하나의 뷰. 디스크 5.1~8.3 GB/s, 2026-09-16 부팅의 load 47.8 GB / 13.2 s([기록](../measurements/st_boot_20260916/README.md)) |

## U0 이 정한 것 — 이미지에 있는 후보

`sm121-inv-0919a`(시드 이미지 flashinfer 0.6.18.dev20260819, [기록](../measurements/sm121_inventory_20260919/README.md)):

| 항목 | 이미지 안의 후보 | 그러므로 |
|---|---|---|
| U3 | `fused_moe/cute_dsl/fused_moe_mxfp8_mxfp4.py`(MXFP8 활성 × MXFP4 가중치), `b12x_moe.py` | 바인딩 |
| U5 | `gemm/kernels/dense_blockscaled_gemm_sm120_b12x.py`, `gemm/gemm_mm_fp4_cute_dsl.py` | 바인딩 |
| U6 | `mla/_sparse_mla_sm120.py` + `sparse_mla_sm120*.cu`(dsv3_2·dsv4 decode, prefill) | 바인딩 후보 — vllm#54929 는 이것이 부하에서 livelock 한다고 한다. 부하 판정이 먼저, Triton 레인은 그 대안 |
| U7 | `decode.py`(`window_left`·`sinks`·`logits_soft_cap`), `cute_dsl/attention/gqa_decode_paged.py` | 바인딩 |
| U8 | `decode.py` 의 FP8·NVFP4 KV(`kv_cache_sf`) | 바인딩 |
| U9 | `kda_prefill.py` + `csrc/kda/flashkda_*.cu` | 바인딩(JIT, nvcc 는 이미지에 있다) |
| U13 | `gdn_prefill.chunk_gated_delta_rule`, `delta_rule_dsl/delta_rule_sm120.py` | 바인딩 |

엔진의 `engine/kernels/b12x` 는 이미지의 `blackwell_sm12x` 의 포크다(같은 바이트 9, 다름 9, 엔진에만 28, 이미지에만 0).

## 감사 기록 (2026-09-19, 코드 읽기)

**U15 — 로드된 id 로 만든 주소.** 서빙 경로에서 마스크·clamp 없이 주소가 되는 id 는 없다. `qsa.py` 는 요청·페이지를
clamp 하고 모든 로드를 `physical_page >= 0 & < num_pages` 로 가린다. `decode_topk.cu:176-177`, `prefill_topk.cu:264` 는
길이를 clamp 한다. 메가커널은 빈 레인이 자기 행의 유효 슬롯을 다시 읽고 `ok ? … : -INF` 로 버린다.
- **방어 빈틈 하나:** 예약되지 않은 블록표 항목(-1)을 0 으로 clamp 한 사본으로 **쓴다** —
  `engine/profiles/glm53/decode_graphs.py:105`(`clamp_min_(0)`) → `engine/kernels/mla/decode_inputs.py:21-24`의 저장,
  Qwen 은 `engine/kernels/step_addresses.py:29,43,46`. 블록 0 은 널 블록이 아니라 살아 있는 요청의 것이다. 지키는 것은
  호스트의 예약 검사(`engine/base/slot_caches.py` `prepare`, `qwen38/decode_graphs.py` `publish`)뿐이고, 그 검사가
  틀리면 GB10 은 fault 없이 **남의 KV 를 덮는다.** 읽기 쪽은 선택이 버리므로 안전하다.
- **죽은 커널 둘:** `causal_conv.py` 의 `_causal_conv1d_update_kernel`(PAD_SLOT_ID -1 을 주소로 읽는다),
  `kpool.py` 의 쓰기 커널들(마스크가 없으면 loc -1 로 저장) — 엔진 어디서도 부르지 않는다.
- `draft_attention.py:75` 의 값 로드는 창(`near`) 밖 컨텍스트 칸도 읽고 확률 0 을 곱한다(0 × 유한). 링이 부팅·캡처·입장
  때 0 으로 채워지고 실제 출력만 쓰이므로 안전하다 — 불변식으로 지켜지는 자리다.

**U17 — PDL wait 앞의 읽기.** PDL 은 `dense/kernels.cu`, `mla/glm53_megakernel.cu`, `mhc/tilelang_kernels.py`,
`oneshot/dsv4_oneshot_ar.cu`, `causal_conv.py`(두 호출 모두 `launch_pdl=False`)에만 있다. wait 앞에서 읽는 것은
부팅 때 한 번 채운 가중치(`wq4`·`ws4`)와 mHC 계수(`a.fn`)뿐이다. 라우터 커널에는 PDL 이 없다. 남은 것: `dense/kernels.cu:42`
의 "No PDL is emitted" 는 낡은 주석이다(PDL 은 3337-3370 에서 켜져 있다) — 네이티브 빌드 해시를 바꾸므로 그 파일을 다음에
고치는 PR 에서 같이 고친다.

**U18 — 캡처 뒤 패딩·널 슬롯.** 모든 캡처 경로가 끝나고 캐시를 리셋한다(GLM `decode_graphs.py:393-395, 583-584`,
`burst_decode.py:228-231`, Qwen `decode_graphs.py:126-127, 253-254`, `warmup.py:144`). 리셋은 페이지와 상태를 0 으로,
블록표를 -1 로 채운다. GLM 은 행 수마다 캡처해 패딩 행이 없고, Qwen 의 패딩 행은 마지막 토큰의 복제(유한)다.

**U19 — TP 랭크 간 드래프터 상태.** GLM: 드래프트 토큰은 랭크 0 방송(`agree_walk`), 샘플 판정도 방송(`agree_verdict`),
greedy 는 int64 MAX all-reduce, 합은 랭크 순서 고정, 호스트가 결과를 랭크 간 대조(`_agree_outcome`). Qwen3.8: 픽과 확률은
집합 통신으로 같고 샘플러는 결정적이지만, **랭크 간 대조(tripwire)가 없다** — GLM 은 torch cumsum 이 랭크를 갈라놓은 적이
있다(`draft_agreement.py:37-45`). 빈틈이지 확인된 버그는 아니다.

## 이미 있는 것 — 가져오지 않는다

- topk 동점 결정성(vllm#56749, #55122): `decode_topk.cu:22`, `prefill_topk.cu:46` 이 낮은 인덱스를 고른다.
- 파일로 읽는 PLE(sglang#37068, #39126): `engine/profiles/qwen38/ple_table.py`(`MappedTable`).
- 정렬 안 맞는 N(vllm#48588): `engine/kernels/dense.PaddedDenseLinear`.
- NoPE MLA(vllm#53969, #55778): GLM-5.3 이 회전 없는 MLA 다.
- PDL 라우터 bias(sglang#38290 의 원래 버그): 우리 라우터 커널(`moe_route`, `router_fused`, `router_fp32`)은 PDL 을 쓰지 않는다 — U17 은 PDL 을 쓰는 나머지 커널을 본다.

## 가져오지 않는 것

PCIe IPC all-reduce(sglang#34528 — 우리는 RoCE one-shot), diffusion, AMD, SM90/SM100 전용 커널, 쿡북, 파서.
