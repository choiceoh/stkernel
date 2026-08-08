# 실측 원장 (measurement ledger)

이 스택에서 내린 모든 성능 판정과 그 수치. **여기 없는 주장은 미실측이다.**
방법론·함정은 [README의 bench 표](README.md#bench--측정-도구와-함정) 참조.
기준 환경: 4× GB10(SM121) · TP=4 · CRS812 · `production-hybrid-1.6` 이미지.

## 기준선 (슬리밍+인시던트픽스, 노브 전부 OFF — 현행 프로덕션)

| 지표 | 값 |
|---|---|
| 프리필 (단일, 60K, 캐시무효) | **2,503–2,528 tok/s** |
| 디코드 (bench-dec, 768tok 생성) | C=1 57.4 · C=2 95.6 · C=4 141.6 tok/s |
| 짧은 디코드 (bench-tp4) | 73–110 진동 (dspark 수용률; 평균 ~95) |
| 장문 디코드 (IndexCache freq=4) | 32K +13.6% · 128K +16.0% vs freq=0 |
| GPU KV cache | 3,190,072 tokens (28.6 GiB) |
| 품질 | 리트리벌 9/9 (2K/32K/128K × 사실 3) |

## 커널 귀속 — 프리필의 완전 분해 (2026-08-09)

torch profiler 32K 프리필 캡처, rank0 (`--profiler-config` CLI로 라우트 개방,
트레이스 오프라인 파싱). **GPU busy 99.6% — 런치갭/CPU 병목설 사망.**

| 버킷 | 비중 | 비고 |
|---|---|---|
| MoE (b12x W4A16) | 24.2% | probe 계측(27.6%)과 정합 |
| **comms (NCCL AllReduce ×846)** | **23.4%** | 단일 최대 커널 3,116ms. 회당 3.7ms ≈ 4랭크 링 이론치 = **대역폭 바닥** |
| GEMM (deep_gemm 투영들) | 18.6% | tf32 hc_prenorm 487ms 포함(mHC 계열) |
| attn/indexer (sparse MLA) | 12.5% | |
| mHC 계열 합산 | **~16%** | mhc_post 1,034 + mhc_pre 620 + hc_prenorm 487 — 버킷 3곳에 분산 |
| norm/rope/quant 나머지 | ~5% | |
| memops | 0.9% | |

## 판정 원장

### 채택 (프로덕션)

| 항목 | 수치 근거 | 일자 |
|---|---|---|
| 업스트림 5건 (#51209/#51042/#51252/#49059/#51202) | 장문 디코드 +13~16%, 정합성/크래시 가드, 프리필 무회귀 | 08-08 |
| `VLLM_DSV4_INDEXER_SP=1` | +5.4%, bit-exact | 08-08 |
| 슬리밍(PR#1–4) + threshold 인시던트 픽스 | 무회귀 검증 (9/9, 프리필/디코드 동급) | 08-09 |
| IDXFREQ=4 | freq=2 대비 짧은디코드 우위, 장문 동급 | 08-08 |
| SPEC_TOKENS=5 | 3은 −7% (5 위쪽 미탐) | 08-08 |

### 기각 — 측정으로 (재론 금지, 수치가 근거)

| 항목 | 수치 | 일자 |
|---|---|---|
| `TRIMIDX=1` (skip-topk KV 미할당) | 프리필 −3.4% (2,431), KV +0.3% (주장 +10%) | 08-09 |
| `C4AREUSE=1` / `SPFAST=1` | 완전 중립 (2,508–2,528 / 2,518–2,521) | 08-09 |
| `MAX_NUM_BATCHED=8192` | 중립 (2,507–2,511) → **잔차는 토큰 비례 비용** 확정 | 08-09 |
| `NCCL_PROTO=Simple` | 중립 (2,489–2,510) → comms는 프로토콜 아닌 대역폭 바닥 | 08-09 |
| #47474 포팅 (token_to_req 캐시) | 완전 중립 (포크에 빠른 경로 기존재) | 08-08 |
| #51430 포팅 (eager 축소) | 프로덕션 경로에서 데드코드 + BRKCG=1에선 CUDA illegal access | 08-08 |
| `VLLM_USE_BREAKABLE_CUDAGRAPH=1` | 장문 디코드 ~5배 저하 + 엔진 사망 | 08-08 |
| `B12X_INDEXER_STREAM/KV_STREAM=1` | 디코드 45–62로 폭락 (기준 85–102) | 08-08 |
| SP(#46789류) / EP / DP / DCP | b12x가 EP 거부 · DCP 프리필 반토막 · DP 디코드 −47% | 08-08 |
| SPEC_TOKENS=3 | 디코드 평균 83.5 vs 90 | 08-08 |

### 비가용 — 구조적으로 (환경이 근거)

| 항목 | 이유 | 일자 |
|---|---|---|
| `VLLM_USE_B12X_MHC=1` (mHC cute 교체) | **3회 에스컬레이션 전부 컴파일 호스트 OOM** (기본 → GPU_MEM 0.45 → 0.5+inductor 1스레드; 22–25분 지점). 가중치 로드 상태의 inductor 재컴파일이 128GB 통일 메모리에 안 들어감 | 08-09 |
| #49236 / #51323(HiSparse) 직접 포팅 | 신규 `.cu` — 단, **컨테이너에 nvcc 13.2+cmake+ninja 존재 확인** → JIT 확장 빌드 경로는 열려 있음 (미착수) | 08-08 |
| #51318 / #51395 / #48993 / #50911 / #49302 | 대상 코드경로가 이 빌드에 부재/비활성 | 08-08 |

## 남은 검증된 레버 (우선순위순)

1. **GPU 클럭 2200MHz 영속**: 프리필 **+8.7% 실측**, GPU 56→67°C. srv4 하드
   파워오프 이력(열 추정)과의 트레이드 — 운영자 결정 대기.
2. **컨테이너 내 nvcc 커널 빌드**: #49236(TTFT +3.9% 주장)류. 툴체인 확인 완료,
   `torch.utils.cpp_extension.load()` + 오버레이 바인딩 심 필요 (~중간 작업).
3. 그 외는 하드웨어(패브릭 대역폭)·모델(mHC 구조) 영역 — 소프트웨어 밖.

## 업스트림 비교체크 (2026-08-09, PR·이슈 ~25건 vs 이 스택)

로컬 사본(sai/moe/oproj probe·p474 히스토리)·이미지 파일·부팅 로그·포렌식 대조.

### 판정

| 건 | 판정 | 증거 |
|---|---|---|
| #49897 인덱서 프리필 NaN→IMA(SM12x) | 포크 자체 해결 — 무사 | 문제 커널 쌍이 이미지에 가드 없이 실존 + `VLLM_USE_B12X_SPARSE_INDEXER=False` 기본 확인 = 프로덕션이 그 deepgemm 경로로 상시 통과 중 |
| #51467/#50924 구조화출력+dspark 엔진사망 | **무사 — 런타임 시험 통과** | 유휴 상태 `response_format: json_object` 1발 정상 응답 + 직후 api/chat 건강 (08-09) |
| #49547 그래프 강등 | 변종 확인 — 신규 레버 | `fuse_attn_quant` 비호환 경고로 FULL 강제 → 인덱서 UNIFORM_BATCH 제약으로 **FULL_DECODE_ONLY** (프리필 piecewise 0; 디코드 FULL 1..256 유지). GPU busy 99.6%라 실손실은 작을 것 |
| #50773 fuse_norm/act_quant GB10 오염 보고 | 워치 | 요청 안 한 두 패스가 resolved에서 자동 True 확인 — 품질 9/9라 포크판 정상 추정 |
| #48031 엔진 ready 600s | 채택 (보험) | 이미지 `envs.py:27` 기본 600s + supervisor 이력 'boot grace exceeded' → 런처에 `VLLM_ENGINE_READY_TIMEOUT_S=3600` |
| #49921 라우터 GEMM 게이트 | **기각 — 자체감사로 정정** | 게이트(`gate_linear.py:53 family(100)`)는 실존하나, 이 포크의 생성자는 `force_fp32_compute` 미전달(nvidia/model.py:607) → 현행 폴백이 이미 **bf16 F.linear + [N,256] fp32 캐스트**. (4096,256)은 Tier-1/2 부적격이라 게이트를 열어도 실효는 캐스트 1커널 융합뿐(기대 ~0). 포팅 시도 철회 |
| #49027 TP 웻지(상한 근접) | 이론 잔존 | 포렌식 전건 grep — NCCL 카운트 불일치 시그니처 무발견. 430K 근접 세션 시 위험, 증상 시그니처는 포렌식 로그로 식별 가능 |
| #51489/#50774/#48210/#50365/#51106 | 무관/기커버 | C128A 빌더 zero-가드(p474 L286)·tile_sched 소비 무·bmm_fp8 미검출·atomic 미검출·#49059 가드 확인 |
| #51340/#51009/#50011/#46307 | 워치/기해결 | 워밍업 행 무증상 · 수용률은 bench-dec 열 감시 · sleep 미사용 · UMA는 drop_caches 대응 |

### 구성 해석 드리프트 (요청 vs resolved, 부팅 로그 실증)

- 요청 `FULL_AND_PIECEWISE` → 실효 **FULL_DECODE_ONLY** (`fuse_attn_quant`→splitting_ops=[] → FULL → 인덱서 제약)
- 요청 `fuse_gemm_comms:true` → resolved **False** (조용히 드랍 — 런처 플래그는 장식)
- 미요청 `fuse_norm_quant/fuse_act_quant` → resolved **True** (자동 활성)

### 신규 A/B 큐 (재기동 단위, 하나씩)

1. `fuse_attn_quant=false` — 진짜 FULL_AND_PIECEWISE 회복 vs attn-quant 융합 상실
2. `VLLM_USE_B12X_SPARSE_INDEXER=1` — 미실측 노브 (SM120 전용 요건 충족; 켜면 우리 indexer.py의 b12x 스케줄 분기가 라이브)

## 인시던트 로그

| 일자 | 증상 | 근본 원인 | 수정 |
|---|---|---|---|
| 08-09 | 워밍업 `c128a_prefill_topk_indices=None` 크래시 | 슬리밍이 `_spec_mult` 하드코딩 — 실환경은 `VLLM_DSPARK_IMPL=upstream`이라 비오버레이 `sparse_mla.py`(threshold 11)와 오버레이(6)가 같은 배치를 다르게 분할 | env-게이트 복원 (51a1a52) + raise에 `[diag]` 자가진단 부착 |
| 08-09 | `/start_profile` 404 ×3 | 라우트는 `--profiler-config` **CLI 인자**로만 attach — env(`VLLM_TORCH_PROFILER_DIR`/`VLLM_SERVER_DEV_MODE`)는 무관 | 런처 serve 인자에 배선 (454f49f) |
| 08-09 | 프로파일 캡처 후 엔진 웨지 | `stop_profile`의 트레이스 덤프가 120s+ | 캡처는 전용 부팅에서만; 트레이스는 서버가 마저 쓰므로 **오프라인 파싱** (`load_events`/`bucket_of` 재사용) |

## 방법론 교훈 (체인 운용)

- 컨테이너 env는 런처 `-e` 명시분만 전달 — 신규 변수는 **sed로 런처 변형에 굽고**, 성공 경로에 `printenv` 의도 검증을 넣을 것 (NCCL 헛시험 사례).
- 폴백 분기("실패 시 변형")는 1차가 *의도와 다르게* 성공하면 안 탄다.
- 대기 함수에 teardown을 넣지 말 것 (자기 스택 삭제 사고).
- `pkill -f`는 자기 ssh 명령줄과 매칭되면 자살 — 브래킷 패턴(`[b]ench`) 사용.
- 디코드 비교는 `bench-dec.py`(768tok 생성)로만 — 짧은 벤치는 10% 효과도 못 가른다.
- 구성 A/B에서 한쪽만 추세로 보이면 설명 전에 **콜드 재기동 재현**부터.
