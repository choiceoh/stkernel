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
| U0 | 이미지 인벤토리: 아래 항목의 후보 구현이 시드 이미지 안에 있는지(import·시그니처·sm_121a 컴파일) | — | measure | gpu | 열림 |

### A. 모델 범위 — 입구의 '거절'·'변환'을 줄인다

| ID | 무엇 | 출처 | 종류 | 판정 | 상태 |
|---|---|---|---|---|---|
| U1 | 입구가 `hf_quant_config.json` 을 읽는다: quant_algo(NVFP4·W4A16_NVFP4·FP8·MXFP4·MIXED_PRECISION 층별), group_size, kv_cache_quant_algo, exclude_modules. 설정 읽기의 나머지 빈틈도(바깥 `mtp_config`, `dense_intermediate_size`, `local_layer_ids`/`sliding_window_size`, 바깥 `model_type`) | vllm#56050, #56535 | door | cpu | 열림 |
| U2 | 모양(`kernel_shape.Attention`)이 윈도·상대 편향·k/v conv 를 말하고, `cells` 가 그런 어텐션을 평범한 GQA 로 통과시키지 않는다(D3) | 조사(Inkling) | door | cpu | 열림 |
| U3 | MoE MXFP4 **W4A8** — b12x 의 MXFP4 전문가를 FP8 활성으로(cells 레시피 (c)) | sglang#34878 | kernel | gpu | 열림 |
| U4 | MoE 활성 정밀도를 체크포인트가 정한다 — W4A16 체크포인트가 W4A4 로 돌지 않게(U1 이 읽은 값으로 셀 선택) | vllm#56535 | fix | cpu+gpu | 열림 |
| U5 | dense NVFP4 GEMM 레인 — 어텐션 투영·공유 전문가까지 NVFP4 인 체크포인트를 변환 없이 | sglang#38685, #38170, vllm#54614 | kernel | gpu | 열림 |
| U6 | sparse-MLA Triton 레인 — 메가커널 인스턴스가 없는 MLA/DSA 형상(DSv3.2·GLM-5.2 류)을 입구가 이 레인으로 판정 | vllm#54929 (대안 vllm#54976 B12X) | kernel | gpu | 열림 |
| U7 | GQA 레인 — paged KV + 윈도 + sink(가장 넓은 가족: gpt-oss·Gemma·Mistral·Qwen3 류) | vllm#50022, #55078(재료) | kernel | gpu | 열림 |
| U8 | KV 캐시 양자화(FP8, NVFP4) — 긴 컨텍스트와 큰 모델을 128 GB 에 | vllm#46329, #55976, #56550, #55557, sglang#37798 | kernel | gpu+품질 | 열림 |

### B. 이미 서빙하는 모델의 속도 (커널 증거까지; 채택은 플릿)

| ID | 무엇 | 출처 | 종류 | 판정 | 상태 |
|---|---|---|---|---|---|
| U9 | GLM KDA chunked prefill 을 FlashKDA 로(업스트림 GB300 1.7~3.8×; sm_121a 빌드부터) | vllm#55737 | kernel | gpu→fleet | 열림 |
| U10 | skinny FP8 GEMM(M=1 GEMV, M≥2 CUTLASS)을 `dense/fp8_rows`·W8A16 과 GB10 에서 대조 | sglang#38082 | measure | gpu | 열림 |
| U11 | FP8 prefill GEMM 의 L2 절벽(가중치 > 24 MiB, M ≥ 8k) — 우리 cuBLASLt 에도 있나, 있으면 래스터 스위즐 | vllm#55180 | measure | gpu | 열림 |
| U12 | Qwen3.8 QSA prefill 타일 합집합(연속 행이 고른 블록의 합집합을 한 번씩) | vllm#55430 | kernel | gpu→fleet | 열림 |
| U13 | Qwen3.8 GDN prefill 을 FlashInfer 로, GDN gate 투영 | vllm#55715, #57318 | kernel | gpu→fleet | 열림 |
| U14 | Qwen3.8 PLE 표를 NVFP4 로 묶어 상주(파일 읽기와 메모리의 교환) | vllm#56273 | kernel | gpu+품질 | 열림 |

### C. GB10 정확성 — 모든 모델, 조용히 틀리는 종류

| ID | 무엇 | 출처 | 종류 | 판정 | 상태 |
|---|---|---|---|---|---|
| U15 | 로드된 id 로 주소를 만드는 gather 의 clamp·mask 감사, 단일 레인 compute-sanitizer 한 번 | vllm#49049 | audit | cpu+gpu | 열림 |
| U16 | 긴 컨텍스트(≥120k) 디코드 정확성 검사 — SM121 에서만 토큰 0 을 내던 종류 | sglang#36845 | test | gpu | 열림 |
| U17 | PDL: wait 앞의 읽기가 부팅 상수뿐인가 | sglang#38290 | audit | cpu | 열림 |
| U18 | 캡처 뒤 패딩·null 슬롯의 비유한 값(0×NaN)이 실제 행을 오염시키나 | vllm#57158 | audit | cpu | 열림 |
| U19 | 드래프터 상태가 TP 랭크마다 어긋나는 자리 | sglang#33614 | audit | cpu | 열림 |
| U20 | DeepGEMM 스케일: FP32 스케일(2 의 거듭제곱 아님)을 받으면 부팅에서 거절 | sglang#39482, vllm#57512, #54600 | fix | cpu | 열림 |

### D. 통합 메모리·플랫폼

| ID | 무엇 | 출처 | 종류 | 판정 | 상태 |
|---|---|---|---|---|---|
| U21 | 통합 메모리 회계: NVML 이 장치 메모리를 못 읽을 때, 프로세스 자신의 사용량으로 KV 를 잰다 | vllm#57378, #49760, #55828 | audit | cpu | 열림 |
| U22 | 가중치 스트리밍(O_DIRECT, 읽기 전용 매핑)을 `mapped_staging` 과 부팅 시간으로 대조 | sglang#37680, #38441 | measure | gpu | 열림 |

## 이미 있는 것 — 가져오지 않는다

- topk 동점 결정성(vllm#56749, #55122): `decode_topk.cu:22`, `prefill_topk.cu:46` 이 낮은 인덱스를 고른다.
- 파일로 읽는 PLE(sglang#37068, #39126): `engine/profiles/qwen38/ple_table.py`(`MappedTable`).
- 정렬 안 맞는 N(vllm#48588): `engine/kernels/dense.PaddedDenseLinear`.
- NoPE MLA(vllm#53969, #55778): GLM-5.3 이 회전 없는 MLA 다.
- PDL 라우터 bias(sglang#38290 의 원래 버그): 우리 라우터 커널(`moe_route`, `router_fused`, `router_fp32`)은 PDL 을 쓰지 않는다 — U17 은 PDL 을 쓰는 나머지 커널을 본다.

## 가져오지 않는 것

PCIe IPC all-reduce(sglang#34528 — 우리는 RoCE one-shot), diffusion, AMD, SM90/SM100 전용 커널, 쿡북, 파서.
