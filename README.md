# stkernel

**TP=4 DGX Spark GB10 MoE 서빙 스택** — 4× NVIDIA DGX Spark (GB10, sm_121a, 48 SM),
CRS812 스위치드 패브릭, vLLM eldritch/b12x 포크 이미지.

베이스 이미지는 서드파티라 소스 트리가 없다. 이 리포는 그 이미지를 **재빌드 없이**
개선하기 위한 스택이다 — 파이썬 파일을 `site-packages` 위에 read-only 바인드
마운트하는 오버레이 방식이며, 모든 수정부에 `# deneb fork:` 마커가 있다.

```
-v <overlay>/nvidia_model.py:/opt/venv/.../vllm/models/deepseek_v4/nvidia/model.py:ro
```

## 무엇을 물으면 어디를 여나

| 질문 | 문서 |
|---|---|
| 지금 무엇이 켜져 서빙되나 | [`profiles/README.md`](profiles/README.md) — 프로필별 모듈·기본 노브 표 |
| 이 수치가 실측인가 | [`MEASUREMENTS.md`](MEASUREMENTS.md) — **여기 없으면 미실측**. 맨 앞에 판정 규율 8줄과 찾아보기 |
| 디코드 스텝이 무슨 커널로 이루어지나 | [`STEP_KERNEL_MAP.md`](STEP_KERNEL_MAP.md) — 개수·소유권·시간 구성·꼬리 |
| 다음에 무엇을 부팅하나 | [`RUNBOOK_KERNEL_CAMPAIGN2.md`](RUNBOOK_KERNEL_CAMPAIGN2.md) — EXP 상태 표(부팅 필요 여부 포함) |
| 이 모듈은 무엇을 접수하나 | `overlay/modules/<name>/README.md` + 같은 폴더의 `manifest.tsv` |
| 무엇으로 재나, 어디에 함정이 있나 | 이 문서의 `bench/` · `probes/` · `tools/` 절 |

## 모듈과 프로필

오버레이는 **모듈 단위**다 (`overlay/modules/<name>/`, 각자 `manifest.tsv` 보유).
어떤 모듈을 싣는지는 **프로필**이 정한다 (`profiles/<model>.env`의 `MODULES=`).
모델이 하나였을 때는 단일 매니페스트로 충분했지만, 같은 플릿에서 GLM-5.3-Flash와
Qwen3.8-Flash-Next를 올려보면서 **모델이 아니라 하드웨어에 속한 발견**이 쌓였고
그걸 둘 곳이 없었다.

범위가 모델을 넘는 모듈:

| 모듈 | 범위 |
|---|---|
| `tp_oneshot_ar` | 4노드 TP one-shot AllReduce — 모델·아키텍처 무관 |
| `moe_gate_sm121` | 융합 MoE 라우터 게이트 — **GB10의 모든 MoE**. 이미지의 `GateLinear`가 가속 티어를 SM90/SM100 계열로만 판정해 sm_121은 항상 bf16 Tier-4로 떨어진다 (dsv4 실측 C=1 +3.5%) |
| `spec_fp8_head` | W8A16 fp8 보캡 헤드 — 드래프터 일반 |
| `mla_indexer` · `mla_sparse_swa` | MLA 인덱서·슬라이딩윈도우 백엔드 — DeepSeek-MLA 계열 |

`launchers/compose-overlays.sh <profile>`가 모듈을 합성해 배포기·런처 프리플라이트·
4노드 SHA-256 검증이 기대하는 **평평한 디렉터리 + 단일 매니페스트**를 만든다.
노드가 보는 계약은 그대로다.

## b12x 우선 정책

b12x는 GB10에서 FP4 텐서코어에 닿는 유일한 MoE 경로다 (marlin·bf16 폴백은 먼저
디퀀트한다). 그래서 **백엔드는 항상 명시**한다 — `--moe-backend b12x`를 주면 후보가
하나로 좁혀져 불가할 때 `NotImplementedError`로 **크게 실패**하고, `auto`로 두면
marlin으로 **조용히 떨어진다**.

가능 여부는 부팅 전에 판정할 수 있다:

```bash
tools/b12x-preflight.py --scan ~/models 4
```

커널 진입점은 여전히 `num_local != num_experts` 를 거부한다 (flashinfer #3383).
오버레이가 EP를 **로컬 전용 MoE처럼** 보이게 한다 — 기본 direct 경로는
로컬 weight row만 유지하고, 전역 top-k 를 리맵한 뒤 원격 슬롯을 커널 전에
제거한다. 기본 디코드는 8 row씩 `top_k=1` micro 호출로 나눠 C=1/2/4를
8/16/32회에 처리한다. `VLLM_B12X_EP_STOCK_TOPK_MICRO=1` 실험은 원격
슬롯을 같은 토큰의 zero-weight 로컬 ID로 바꾸고 원래 `top_k=8` 모양을
최대 5토큰/40 routed-row로 나눠 이를 2/4/7회로 줄인다. 실GPU 수치와
CUDA graph replay가 아직 미검증이라 기본 0이다. 프리필(`tokens * top_k >
640`)은 원격 슬롯을 빼서 GEMM하지 않는다. 각 fixed workspace는 강하게
보유해 prefill cache 확장이 CUDA graph 주소를 무효화하지 못하게 한다.
`VLLM_B12X_EP_ZERO_WEIGHT_MICRO=1` 실험은 E=72를 유지한 채 안정적인
8/16/32-token shape만 top-k=8의 1/2/4 micro 호출로 줄인다. zero-weight
sentinel은 row append 전에 버리지만 GPU 수치/E2E 이득은 아직 미검증이라 기본 0이다.
`VLLM_B12X_EP_NO_DUMMY=0`은 zero-weight dummy와 wrapper 경로를 복원하는
롤백이다. vLLM 의 EP all-reduce 가 랭크를 합친다. `ENABLE_EP=1` 이
`--enable-expert-parallel` 이다. EP 를 끄면
랭크당 gate+up 행이 128의 배수여야 하고, 켜면 **전체** intermediate 가 그
정렬을 만족하면 된다 (Qwen3.8 의 640 은 TP=4 에서 깨지고 EP=4 에서 산다).
패딩하면 부팅은 되는데 출력이 깨진다. 이미지 자신도 같은 말을 한다:
`mxfp4_round_up_hidden_size_and_intermediate_size`가 **B12X에만 크기를 그대로
돌려주고** MARLIN·DEEPGEMM·TRTLLM은 올림한다.

## 구성

| 디렉터리 | 내용 |
|---|---|
| `overlay/modules/<name>/` | 오버레이 **모듈** 25개 — 각자 소스 · `manifest.tsv` · `README.md`(25/25) · 의존 선언 `requires`. glm53 전용 25개는 34차(2026-09-05)에 다섯 묶음(`glm53_model`·`glm53_kernels`·`glm53_drafter`·`glm53_moe`·`glm53_runtime`)으로 접혔다 — 행·계약·노브 불변 |
| `profiles/<model>.env` | 어떤 모듈을 싣고 어떤 노브로 뜨는지 (`MODULES=` + 서빙 env). dsv4 18 · glm53 8 · qwen38 1 모듈 |
| `build/<profile>/` | `compose-overlays.sh` 가 렌더한 평평한 디렉터리 + 합성 매니페스트 — 배포기·런처가 보는 것 (생성물, 손으로 고치지 않는다) |
| `launchers/` | 프로덕션 런처 + 슈퍼바이저 + systemd 유닛 + manifest 기반 4노드 배포·SHA-256 검증 + 런타임 경계 감사 + A/B 하네스(`ab-glm53.sh`) |
| `bench/` | 검증·측정 도구 (아래 표) |
| `probes/` | 계측 빌드 (CUDA 이벤트 단계 분해, `DENEB_ATTN_PROF=1`) — 원본 + 계측 diff만: `attn_prof`/`fi_prof`는 오버레이 현행본 기준(`diff overlay/… probes/…`로 검증 가능), `moe_prof`/`oproj_prof`/`sai_probe`는 이미지 원본 기준 |
| `tools/` · `census.py` | 트레이스 분석 — 커널 인구조사·스텝 시간 구성·꼬리 절단 (아래 `tools/` 절) |
| `tests/` | GPU/vllm 없이 도는 순수 로직 검증 (`python3 tests/test_logic.py`, 44.6K checks) — 청커 예산·skip-topk 규칙·SP 샤드·DSpark 범위/preimage 계약·manifest 불변식·도구 계약 |
| `MEASUREMENTS.md` | **실측 원장** — 모든 판정과 수치 (여기 없는 주장은 미실측) |
| `STEP_KERNEL_MAP.md` | 디코드 스텝의 커널 지도 — 무엇이 몇 발 돌고 누가 소유하며 어디가 레버인지 |
| `RUNBOOK_KERNEL_CAMPAIGN2.md` | 부팅이 필요한 실험(EXP-1~19)의 절차·게이트·중단 기준 |

## 매니페스트 합성 · 런타임 경계 감사

**매니페스트가 파일 목록·컨테이너 마운트 목적지·베이스 preimage의 유일한 원본**이다.
모듈마다 `overlay/modules/<name>/manifest.tsv` 를 갖고, `compose-overlays.sh <profile>`
가 프로필의 `MODULES=` 를 합쳐 `build/<profile>/manifest.tsv` 하나로 렌더한다(dsv4 23행 ·
glm53 40행). 배포기·런처·검증이 보는 것은 그 합성본이다 — 루트에 `overlay/manifest.tsv`
는 더 이상 없다. 세 번째 열은 교체 대상의 production-hybrid-1.6 SHA-256 또는 새 파일의
`absent` 계약이다.
배포기는 manifest와 그 안의 모든 파일을 4노드에 복사하고 SHA-256을 대조하며,
런처도 같은 manifest를 읽어 바인드 마운트와 기동 전 스큐 검사를 만든다.
고정된 image ID를 확인한 뒤 **마운트되지 않은 이미지 원본**의 모든 preimage를
대조하므로, 태그·API·소스가 어긋나면 기존 컨테이너를 내리기 전에 중단한다.
새 오버레이는 shell 스크립트를 고치지 않고 **모듈 하나(디렉터리 + manifest 행) 추가 +
프로필의 `MODULES=` 한 단어**로 들어간다. 두 모듈이 같은 소스 파일명이나 같은 컨테이너
경로를 주장하면 합성이 중단된다 — 그게 모듈이 정말 독립인지 검사하는 자리다.

고정한 베이스 이미지의 registry layer에서 sampler/hash-MoE/expert-map 실제
소스를 복원해 감사한 결과는 **PASS=0, FAIL=7, UNKNOWN=0**이다. 즉 요청한
상·하한 보호가 prod1.6에는 없다. 이미지·레이어·파일 SHA-256과 각 실패 항목은
[`RUNTIME_GUARD_AUDIT.md`](RUNTIME_GUARD_AUDIT.md)에 고정했다. 실행 컨테이너의
보이는 소스도 다음 명령으로 재검사할 수 있으며, 소스가 생략된 파일은 안전으로
간주하지 않고 UNKNOWN을 반환한다.

```bash
python3 launchers/audit-runtime-guards.py --container hy4
```

## overlay/ — 프로덕션 채택 5건

| PR | 파일 | 내용 | 실측 |
|---|---|---|---|
| [#51209](https://github.com/vllm-project/vllm/pull/51209) | `attention.py` | IndexCache — C4A 레이어가 이전 top-k 재사용 (`index_topk_freq=4`) | 장문 디코드 32K **+13.6%** / 128K **+16.0%** |
| [#51042](https://github.com/vllm-project/vllm/pull/51042) | `flashinfer_sparse.py` + `sparse_swa_dsv4.py` | DSpark 비인과 decode SWA 버퍼 폭을 메타데이터로 전달 (window_size 오판독 수정) | 정합성 |
| [#51252](https://github.com/vllm-project/vllm/pull/51252) | `indexer.py` | 프리필 청커 예산을 `compress_ratio`로 나눠 소비 버퍼와 단위 일치 | 정합성 · 프리필 무손실 |
| [#49059](https://github.com/vllm-project/vllm/pull/49059) | `flashinfer_sparse.py` | 빈 프리필 청크(0-element reshape) 크래시 가드 | 예방 |
| [#51202](https://github.com/vllm-project/vllm/pull/51202) | `flashinfer_sparse.py` | prefill/decode 게이트를 요청수 대신 토큰수로 | 예방 |

검증: 프리필 2,437–2,533 tok/s(무회귀) · 장문 리트리벌 9/9(2K/32K/128K) · traceback 0.

## DSpark 단일스트림 가속 실험: FP8 draft head + top-k Markov

두 최적화는 기본값이 모두 **OFF**다. registry layer에서 추출한
`production-hybrid-1.6`의 `dspark_v2.py`, `speculator_v2.py`,
`utils_v2.py`가 공개 복원본과 논리 diff 0임을 확인했고, 그 실제 raw SHA-256을
manifest에 고정했다. 런처는 원본 hash와 `VLLM_DSPARK_IMPL=upstream` v2 경로를
확인한 뒤 최소 변경된 세 파일을 read-only 마운트한다. 노브가 0이면 새 분기는
도달하지 않고 기존 DeepGEMM FP8/밀집 Markov 경로를 그대로 쓴다.

| 런처 노브 | 런타임 env | 변형 | 보호 조건 |
|---|---|---|---|
| `FP8HEAD=1` | `VLLM_DSPARK_FP8_DRAFT_HEAD=1` | target과 alias된 draft LM head의 row-wise E4M3 복사 + `_scaled_mm` | SM89+, float8/API, 2-D weight, hidden 차원 일치; 기존 DeepGEMM FP8과 동시 사용 금지 |
| `MARKOV_TOPK=512` | `VLLM_DSPARK_DRAFT_TOPK=512` | base-logit 상위 K개에만 Markov W2 행을 gather해 `baddbmm_` | `1 <= K <= vocab(129280)`, full replicated W2 `[V,R]`, target/draft vocab 동일, TP all-gather |

top-k는 잘린 proposal `q` 전체를 기존 Gumbel/rejection 버퍼에 그대로 기록한다.
따라서 target 검증 분포는 바꾸지 않지만 draft proposal과 수용률은 바뀔 수 있어
반드시 수용률과 품질을 같이 측정한다. 로컬 TP4 실측 전에는 성능 수치를 확정하지
않는다. 이 이미지에는 기존 DeepGEMM draft-head FP8이 기본 ON이므로
`FP8HEAD=1`은 BF16 대비가 아니라 **row-wise `_scaled_mm` 대 기존 FP8** A/B다.
vLLM [#47584](https://github.com/vllm-project/vllm/pull/47584)의 외부 GB10
단일 디코드 +3–5%를 이 스택의 예상 이득으로 그대로 쓰지 않는다. top-k 설계
근거는 [#49969](https://github.com/vllm-project/vllm/pull/49969)이지만, 후자는
Qwen3/다른 동시성 결과이므로 역시 이 스택 수치로 인용하지 않는다.

한 번에 한 축씩 cold restart하여 아래 2×2를 `bench-dec.py` 3회 이상과
`check-quality.py` 9/9로 비교한다. 마지막 결합 arm은 두 단독 arm이 모두
통과한 뒤에만 실행한다.

```bash
# baseline / FP8 only / top-k only / combined
FP8HEAD=0 MARKOV_TOPK=0   bash launchers/start-hy4-tp4.sh
FP8HEAD=1 MARKOV_TOPK=0   bash launchers/start-hy4-tp4.sh
FP8HEAD=0 MARKOV_TOPK=512 bash launchers/start-hy4-tp4.sh
FP8HEAD=1 MARKOV_TOPK=512 bash launchers/start-hy4-tp4.sh

python3 bench/bench-dec.py
python3 bench/check-quality.py
docker exec hy4 grep -E "rowwise FP8|top-k Markov" /tmp/hy4.log
```

롤백은 두 노브를 0으로 되돌린 뒤 재기동하면 된다. preimage mismatch로 중단하면
강제 우회하지 말고 `docker run --rm --entrypoint sha256sum "$IMAGE" ...`와
`docker cp`로 실제 소스를 추출해 manifest pin과 diff를 다시 리뷰한다.

## DSpark 수용률 실험 (2026-08-11): 2-pass 정제 + Markov 사이드로드 — **당일 실측 종결 (기각)**

**결과 (08-11 브래킷, MEASUREMENTS 기각 표)**: `REFINE=1`은 수용률 ×0.55
붕괴·C=1 −33%로 **기각** (마스크 학습 backbone의 OOD — 재제안 금지), ngram
하이브리드는 `ngram-ceiling.py` 상한 +3.7%로 **부팅 없이 기각**. 남은 수용률
경로는 드래프터 재훈련/신규 이미지뿐. 아래는 구현·절차 기록이다.

원장 08-10 판정으로 수용률 **노브 축**은 소진됐다(temp·SPEC_TOKENS·method·
scale — 재제안 금지). 남은 축은 **제안 분포 q 자체를 바꾸는 구조 레버**이며,
이 포크의 seeded-Gumbel 커플링 검증(수용 = 동일 (seed,pos) 노이즈에서
`argmax(log q + g) == argmax(log p + g)`; 방출 토큰은 항상 verifier 쪽
argmax)이 **q가 무엇이든 타깃 분포를 불변**으로 만들므로, 이 축은 구조적으로
품질 무손실이고 유일한 지표는 bench-dec 수용률이다. 배경: DSpark 논문
[arXiv:2607.05147](https://arxiv.org/abs/2607.05147) — 병렬 드래프터의
"rapid acceptance decay"(우리 실측 per-pos 78.5→21.5%, 감쇠 ~0.72)가 명시된
병목이고, 블록 내 조건화는 1차 Markov 헤드뿐이다.

| 런처 노브 | 런타임 env | 내용 | 비용/리스크 |
|---|---|---|---|
| `REFINE=1` | `VLLM_DSPARK_REFINE_PASS=1` | **2-pass 자기정제**: pass-1 드래프트 토큰을 노이즈 슬롯에 되먹여 backbone+순차 샘플링 재실행 (동일 Gumbel 키 = 커플링 유지 필수). Jacobi 1회 반복 | 드래프트 스텝당 backbone 1패스 추가(3층, 스텝 −5~7% 추정). backbone이 노이즈 입력으로 학습돼 **OOD 리스크** — 수용률이 안 오를 수 있음. 이론상한 net +3~5% = 채택 하드룰 경계선 |
| `MARKOV_SIDELOAD=<path>` | `VLLM_DSPARK_MARKOV_SIDELOAD` | 서빙 도메인 코퍼스로 재적합한 Markov W1/W2 교체 (`tools/markov_refit.py` 산출물, `/home/choiceoh/models` 하위 필수 — 4노드 preflight 존재 검사) | SCALE 평탄성(원장)이 낮은 상한을 시사 — 기대 낮음. 전량 교체 아닌 블렌드(기본 0.5) + 랭크 재인수분해 |

검증 절차 (한 축씩 cold restart, 브래킷 기준→후보→기준):

```bash
REFINE=1 bash launchers/start-hy4-tp4.sh
python3 bench/bench-dec.py          # 수용률 열이 유일 지표 (acc=50 회귀 정규화)
python3 bench/check-quality.py      # 9/9 위생 게이트
docker exec hy4 grep -E "refinement enabled|sideloaded" /tmp/hy4.log
```

**ngram/copy 하이브리드는 코드 전에 상한 실측** (하드룰): `bench/ngram-ceiling.py`가
실트래픽 형태 텍스트(jsonl prompt/completion)로 prompt-lookup 수용률 상한을
시뮬레이션한다 — 커플링 덕에 "실현 텍스트와의 일치 = 정확한 수용 시뮬"이
성립. 상한이 한자리 후반 %를 못 넘기면 엔진 구현(FULL 그래프와 싸움) 착수
금지. 토크나이저는 노드에서 (`--mode chars`는 배관 스모크 전용).

confidence head(체크포인트 `mtp.*.confidence_head.*`, 현 로더는 드롭)는 논문의
동시성 스케줄러용이라 C=1 무익 판정 유지 — 신규 이미지 배선 대기.
`launchers/watch-image-tags.sh` + `dsv4-image-watch.timer`가 Docker Hub
태그·digest 변경(신규 태그 + 기존 태그 re-push 모두)을 srv2에서 일일 감시
— **08-11 설치·가동** (09:19 KST, 상태 `~/.local/state/stkernel/`, 변경 시
`~/hybrid-stack/NEW-IMAGE-TAGS.txt`에 기록).

## TP4/GB10 전용 슬리밍 (2026-08-09)

오버레이는 업스트림 범용 코드의 포팅이라 이 배포에서 **구조적으로 도달 불가능한**
경로를 담고 있었다. 아래를 제거했고, 전제가 어긋나면 조용히 오동작하는 대신
기동 시점에 죽도록 fail-fast assert를 심었다.

| 제거 항목 | 근거 |
|---|---|
| SM100(B200급) 어텐션 클래스 + plain-row bf16/per-tensor-fp8 KV 경로 | GB10=SM121은 항상 `DeepseekV4FlashInferSM120Attention` + `fp8_ds_mla`(uint8). 클래스명은 import 안전 스텁으로 보존 |
| breakable-cudagraph forward 분기 (`VLLM_USE_BREAKABLE_CUDAGRAPH`) | 확정 스위치 0 고정 — 켜면 장문 디코드 5배 저하 + 엔진 사망 |
| b12x WO projection 경로 + 커스텀 op 2종 | 런처가 env를 노출하지 않음. `setup_b12x_wo_projection()`은 외부 호출 대비 no-op 스텁으로 보존 |
| DCP/PCP(context parallel) 분기·커널, PP(IndexCache 랭크 계산) | 런처에 CP/PP 플래그 없음. 메타데이터의 `dcp_*` 필드는 소비자(`sparse_attn_indexer`) 계약상 1/0/1 고정으로 유지 |
| FlashMLA 타일 스케줄러 (`build_tile_scheduler`) | SM120은 b12x 경로라 스케줄러 미사용, `_flashmla_C`도 sm_121a 미빌드 — 기존에도 항상 스킵. `tile_sched_*` 필드는 None 고정으로 유지 |
| FP4 인덱서 캐시, `index_topk_pattern`, compress_ratio==1(V3.2/GLM) 인덱서 빌더 경로 | FP4는 SM10x 전용, pattern은 미노출(freq만 사용), V4-Flash 인덱서 캐시는 전부 C4A(ratio 4) |
| ROCm/XPU 분기, 업스트림 DSpark parallel-drafting 2배 threshold | CUDA 4노드 고정, fork DSpark impl 사용 |
| `experimental/` 5파일 (#51430, #47474 포팅) | p430은 breakable CG 전용(프로덕션 fatal), p474는 성능 중립 미채택 — 복원은 git history(`e95f82f`) |
| 런처 `SP`/`EPFLAG` 노브 | b12x MoE TP 전용 → enable_sp(#46789)·EP 계열 구조적 불가 |

최적화: `VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD`·`VLLM_DSV4_INDEXER_SP`·
`B12X_PAGED_INDEX_SUPERTILE_K`·`VLLM_USE_B12X_SPARSE_INDEXER`·
`VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` env 조회를 핫패스(매 스텝 metadata build
포함)에서 init/import 시점 1회로 호이스팅, `_indexer_sp_owned_ranges`의
per-call import/TP 조회 캐싱, 인덱서 디코드 빌드의 `seq_lens.max().item()`
동기화 폴백 제거(ratio>1 상시라 호스트 상한으로 대체).

**유지한 가변 경로** (런처 노브라 상수가 아님): `IDXSP`(indexer SP on/off),
`IDXFREQ`, `V2RUNNER`(batch_topology 유무), `SPEC_TOKENS`(flatten vs native
디코드 — next_n∈{1,2}는 native deepgemm), C128A 레이어 분기, B12X vs DeepGEMM
인덱서 스케줄 분기.

### 2차 최적화 — 실측 판정: C4A 재사용 **중립·미채택** (MEASUREMENTS.md)

- **C4A top-k 글로벌라이즈 재사용**: IndexCache(#51209)로 S 레이어는 F 레이어의
  top-k 버퍼를 그대로 읽으므로 글로벌라이즈(로컬 top-k → 페이지 슬롯 id) 결과도
  동일하다. 공유 메타데이터의 `flashinfer_sparse_index_cache`에 스텝당 1회만
  계산하고 S 레이어는 재사용 (`c4a_decode_global`/`c4a_prefill_global`).
  F 레이어가 매 스텝 무조건 재계산·갱신하고 첫 C4A 레이어는 항상 F라서 stale
  불가. freq=4 기준 스텝당 C4A 레이어 ~3/4의 글로벌라이즈 커널 제거
  (C128A는 F/S 구조가 없어 갱신 보장이 안 되므로 레이어별 유지).
- 인덱서 스케줄 빌더(deepgemm `has_deep_gemm()` 프로브·b12x import)를 빌더
  init 1회로 호이스팅 — 매 디코드 스텝 반복 제거.
- **런처 preflight 버그 수정**: 기존 검사는 `overlay/attention.py`를 봤지만
  실제 마운트는 `overlay-b12x/` — 잘못된 디렉터리였고 1파일뿐이었다. 이제
  단일 manifest와 그 안의 모든 파일을 마운트 디렉터리에서 SHA-256으로 헤드와
  대조해 **노드 간 오버레이·목록 스큐를 기동 전에 차단**한다.
- supervisor `BOOT_GRACE` 1500→3600s: 오버레이 소스 변경 후 재컴파일이
  25분을 넘기면 부트 중 재기동 루프에 빠지던 것 방지 (API가 뜨면 즉시
  통과하므로 정상 부팅엔 영향 없음).
- attn_prof의 decode/prefill 분류 임계값을 하드코딩 64에서
  `MAX_NUM_SEQS×(1+SPEC_TOKENS)`(프로덕션 96)로 — 동시 10요청 초과 디코드
  배치가 prefill로 오분류되던 것 수정.

### 3차 최적화 — 실측 판정: KV 트림 **기각** (프리필 −3.4%, KV +0.3%; MEASUREMENTS.md)

- **skip-topk 레이어의 인덱서 KV 캐시 미할당**: IndexCache(#51209)로 top-k를
  재사용하는 S 레이어는 자기 인덱서를 절대 실행하지 않으므로 그 k-cache
  (압축토큰당 132B + 같은 페이지에 패킹되는 compressor state)는 한 번도
  쓰이거나 읽히지 않는다. 해당 `DeepseekV4IndexerCache.get_kv_cache_spec`이
  None을 반환해 페이지 할당 자체를 제거 — SWA-only 어텐션 레이어가 이미 쓰는
  검증된 메커니즘과 동일. freq=4 기준 C4A 인덱서 캐시의 ~3/4이 사라져 KV 풀
  바이트 기준 대략 10%대 용량 회수(레이어 구성에 따라) → 같은 GPU_MEM에서
  컨텍스트/동시성 여유 증가. 인덱서 **모듈·가중치는 유지**(체크포인트 로딩
  불변), `IDXFREQ=1`이면 전 레이어가 F가 되어 이전과 동일하게 전량 할당.
  **검증**: 기동 로그의 "GPU KV cache size" 증가 확인 + `check-quality.py` 9/9
  + `bench-dec.py` 무회귀.
- supervisor `wait_for_fleet`가 **로컬 docker 데몬도 폴링**: 유저 유닛의
  `After=docker.service`는 시스템 유닛에 대해 무효라 부팅 직후 첫 launch가
  로컬 docker 미기동으로 헛돌 수 있었다 (자가 복구는 됐지만 사이클 낭비).
- `check-quality.py` 문서 수정: 사실을 심는 실제 깊이는 25/50/75%.

### 4차 최적화 — 실측 판정: SPFAST **중립·미채택** (MEASUREMENTS.md)

- **indexer-SP 단일 구간 zero-copy 고속경로**: `_indexer_sp_owned_ranges`가
  인접 구간을 병합해 반환하고(소유 행 집합 불변 — `tests/test_logic.py`로
  검증), 결과가 단일 연속 구간이면 — 순수 프리필 단일 청크는 모든 랭크에서,
  혼합 배치는 rank 0에서 해당 — arange/cat·index_select×3·index_copy×2-3을
  전부 dim-0 슬라이스 뷰(zero-copy)와 연속 copy 1회로 대체. 계산되는 행과
  배치 위치는 동일.
- sparse_swa 빌드의 `slot_mapping >= 0` **임시 bool 텐서+copy 제거**:
  `torch.ge(slot_mapping, 0, out=is_valid_token)` 제자리 쓰기 — 매 스텝
  할당 1 + 커널 1 절감.
- `profile-step.py` 확장: 커널 구간 **합집합 기반 busy/idle 분해**(launch
  glue를 스트림 중첩과 구분해 직접 측정) + **스텝당 ms 환산**(probe 트리의
  429/268/57/40과 같은 단위로 비교).
- `tests/test_logic.py` 신설: 청커 예산 수학(#51252)·IndexCache F/S 규칙
  (첫 C4A는 항상 F — 캐시 재사용·spec 제거의 전제)·SP 샤드 커버리지
  (랭크 합집합=전체, 대형 청크 무중복)를 vllm/GPU 없이 AST 추출로 실행 검증.
  `deploy-overlays.sh`가 배포 전 자동 실행.
- `launchers/deploy-overlays.sh` 신설: 단일 manifest를 기준으로 리포 → 4노드
  `-b12x` 디렉터리 배포 + SHA-256 검증 (preflight 스큐 검사의 쓰기 쪽 반쪽).

### 5차 수제 미세 최적화 (µs급 잡비용 소거 — 합계는 스텝의 1% 미만)

- indexer 빌드: compressed seq_lens(`seq_lens // 4`) 계산을 프리필 브랜치로
  이동 — 소비처가 프리필 청커뿐인데 **디코드 전용 스텝마다** 커널+할당을
  내던 것 제거.
- sparse_swa: 프리필 gather_lens 커널 출력을 매 빌드 `torch.empty` 대신
  **영속 버퍼**로 (기존 index/lens 버퍼와 같은 스트림-순서 규율).
- attention: `aux_streams[:3]`/`[:2]`/이벤트 리스트 슬라이싱을 per-call에서
  init 1회로 — 레이어당 eager 호출마다 만들던 리스트 객체 4개 제거.
- flashinfer: `_as_sparse_cache` dim-fix 뷰를 **identity 가드 캐싱** — KV
  텐서는 기동 시 1회 바인딩되므로 레이어당 호출 2-3회의 view 생성이 전부
  재사용으로 바뀜 (`is` 가드라 리바인딩에도 안전).

### 6차 수제 최적화 — CPU 텐서 스칼라 산술 소거

- **프리필 청크 루프의 오프셋 산술을 파이썬 int로**: `_forward_prefill`이
  청크마다 `query_start_loc_cpu[i]` 인덱싱→뺄셈→슬라이스 바운드 변환을
  CPU 텐서 dispatch로 하고 있었다 (레이어 61 × 청크 × 연산 다수 ≈ **ms급/
  스텝** — 5차 항목들보다 한 자릿수 큼). SWA 빌더가 스텝당 1회
  `tolist()`로 만든 `query_start_loc_py`(파이썬 int 리스트)를 메타데이터에
  실어, 레이어 쪽 루프는 순수 int 산술 + int 슬라이싱만 남김. `getattr`
  폴백으로 파일 단위 롤백(이미지 빌더 + 우리 flashinfer) 호환 유지.
- per-call 텐서 별칭 생성 소거(identity 가드): norm weight `.data`(레이어당
  호출마다 2회), SWA 캐시 `view(N,-1)`(fused insert마다), `split` 크기
  리스트, `swa_cache_layer.prefix` 체인 2곳.

### 8차 (2026-08-11) — 판정 종결된 킬스위치 3종 코드 제거 (동작 불변, 순 −176줄)

인시던트 바이섹트가 무혐의를 확정한 뒤 개별 실측까지 끝난 세 노브
(원장 08-09: `TRIMIDX` 기각 — 프리필 −3.4%·KV +0.3%, `C4AREUSE`/`SPFAST`
완전 중립)는 재론 금지 판정이라 다시 켤 일이 없다. 게이트와 그 뒤의
도달-불가 경로를 오버레이·probes·런처에서 제거해 기본값 경로만 남겼다:

- `attention.py`: `DeepseekV4IndexerCache.spec_enabled` 체인 제거(전 레이어
  spec 상시 할당 — 이전과 동일 동작)와 그로써 고아가 된
  `DeepseekV4Indexer(skip_topk=)` 파라미터 제거(인덱서 실행 게이트는
  레이어 쪽 `self.skip_topk`가 그대로 담당). indexer-SP 경로의 단일구간
  슬라이스 팔 제거 — `index_select`/`index_copy_` 단일 경로. 인접 구간
  병합은 gather 인덱스 단축 효과가 있어 유지(tests 불변식 그대로).
- `flashinfer_sparse.py`: C4A 글로벌라이즈 재사용 분기 제거.
  `flashinfer_sparse_index_cache` 메타데이터 필드는 이미지 SM100 경로가
  자기 키로 쓰므로 보존(우리 쪽 읽기/쓰기만 소멸).
- 런처 `-e DENEB_*` 3종 제거, probes(attn_prof/fi_prof) 동일 반영.

제거 기준은 **"측정 종결·미채택 + 우리가 심은 코드"**다. 같은 기각이라도
`FP8HEAD`는 원장이 "실험용 잔존(기본 0)"으로 명시해 유지했고, 이미지 유래
디버그 경로(`VLLM_DSPARK_REFERENCE_HC` 등)는 env로 도달 가능하므로 남긴다.

### 7차 — 코드 자체 낭비 제거 (동작 불변, 순 −110줄)

- `attention_impl`의 `wq_b_kv_insert` 클로저 중복 정의 2회 + 인라인 반복
  1회 → 단일 정의 (세 오버랩 토폴로지 공유).
- SP 경로 단일/다중 구간 병렬 중복 ~30줄 → `_take`/`_emplace` 인수분해
  (`SPFAST` 킬스위치 게이트 보존).
- `get_kv_cache_shape`: SM10x 분기 제거 후 부모 호출만 남은 순수
  pass-through 오버라이드 삭제 · 단일 호출자 경유 메서드
  `_forward_sparse_impl` 인라인 · `swa_k_cache` 데드 파라미터 체인 제거.
- `DeepseekV4Indexer` 죽은 저장 4개, 이중 `if prefix`, `sm_count` 경유
  변수 정리. indexer/sparse_swa 모듈 docstring의 "threshold doubling
  제거됨" 문구를 인시던트 수정(env 게이트 복원)에 맞게 정정.

## 인시던트 2026-08-09 — 워밍업 실패, 킬스위치 바이섹트

증상: 워밍업 4분 내 실패, C128A 레이어의 `c128a_prefill_topk_indices`가
None (원본 오버레이와 구조 동일한 조회 — None을 **만드는 쪽**이 바뀐 것).
`c128a_*`는 이미지의 FlashMLA 빌더가 KV 그룹 단위로 채우므로, **그룹 구성을
바꾸는 skip-topk 인덱서 KV 스펙 제거(3차)가 1순위 용의**, C4A 글로벌라이즈
재사용(2차)이 2순위다.

동작을 바꾸는 최적화 3건을 **기본 OFF 킬스위치**로 전환 — git 바이섹트
대신 env 플립 + 재기동(4분 실패 창)으로 판정한다:

| 런처 노브 | env | 내용 |
|---|---|---|
| `TRIMIDX=1` | `DENEB_TRIM_SKIP_INDEXER_KV` | skip-topk 인덱서 KV 스펙 제거 (KV ~10%대 회수) |
| `C4AREUSE=1` | `DENEB_C4A_GLOBALIZE_REUSE` | C4A top-k 글로벌라이즈 재사용 |
| `SPFAST=1` | `DENEB_SP_SINGLE_SPAN` | indexer-SP 단일 구간 zero-copy |

절차: ① 전부 OFF로 기동 → 워밍업+`check-quality.py` 통과 확인(안 되면 1차
슬리밍 자체가 원인 → `e95f82f` 오버레이로 롤백). ② 노브를 **하나씩** 켜고
재기동 — 실패를 재현하는 노브가 범인. ③ 통과한 노브는 켠 채 유지, 범인은
OFF 고정 후 근본 원인 분석. `torch.ge(out=)` 마이크로옵은 무혐의 입증
전까지 원본 2-step으로 되돌림.

**종결 (`51a1a52`)**: 커밋 바이섹트 결과 실패는 **1차 슬리밍(f734e8e)부터**
— 위 용의자 랭킹은 반증됐고 킬스위치 3건은 전부 무혐의. 근본 원인은
슬리밍이 하드코딩한 `_spec_mult=1`: serve.sh가 `VLLM_DSPARK_IMPL=upstream`을
설정하므로 **미오버레이** 이미지 빌더(sparse_mla.py)는 threshold 1+2N(=11),
슬림 빌더들은 1+N(=6) — 7토큰 워밍업 행을 이미지=decode / 오버레이=prefill로
엇갈리게 분류 → C128A prefill 메타데이터 None. env 조건 멀티플라이어 복원으로
수정 (부팅 200s · 리트리벌 9/9 · 프리필 2,503–2,523 무회귀). **교훈: 오버레이
하지 않는 이미지 코드와 락스텝인 값은 절대 하드코딩으로 접지 말 것 — 런처가
안 쓰는 env라도 이미지 내부 스크립트가 쓸 수 있다.** 킬스위치 3종은 무혐의
확정이므로 하나씩 켜서 실측할 가치가 있다 (KV +10%대의 `TRIMIDX` 우선).
→ **후속 종결**: 3종 전부 개별 실측 완료(기각/중립/중립 — 판정 원장 08-09),
8차(2026-08-11)에서 게이트·코드 자체를 제거했다.

## 외부 실측 반영 — NVIDIA 포럼 #378890 (2× Spark · 동일 모델·유사 포크)

같은 DeepSeek-V4-Flash-0731을 2× GB10 TP=2로 돌린 팀의 교정된 실측에서
반영한 것:

- **supervisor `CHAT_TIMEOUT` 60→300s**: 초장문 인제스트가 신규 요청을
  프리필 종료까지 블록(262K에서 145–159s 실측; 우리 430K는 ~170s+ 추정).
  60s 프로브면 supervisor가 **정상 인제스트 중에** 3연속 오탐→강제 재기동.
- **런처 [1.5/5] 페이지 캐시 드롭 + `free` 로깅**: GB10 UMA에서 기동 메모리
  체크가 회수 가능 페이지 캐시를 불가용 취급 → KV 풀이 부팅마다 출렁임
  (동일 구성 4.22→5.34GiB 실측). 결정성 확보는 TRIMIDX KV 측정의 전제.
- **bench-dec 수용률 병기**: 답글 제안(프리필 배치 크기 ↔ MTP 수용률) 반영.
- 대기 항목: **`MAX_NUM_BATCHED` 4096→2048 A/B** — 그들 실측 8192→2048에서
  KV 풀 +63–67%(활성화 예약 회수), 프리필 중 디코드 점유 2.9×, 프리필
  −7.7%. 이미 런처 노브라 재기동만으로 시험 가능; TRIMIDX와 같은 축(KV)
  이므로 한 번에 하나씩.
- 클라이언트 지침: 프롬프트는 append-only + 변경 필드는 하단에 — 상단 변경
  시 턴당 비용 **39.6×**(DSpark 실측), 프리픽스 캐시 재사용 97–99% 유지.
- 우리 bench 도구는 그들의 계기 결함 3종(SSE 이벤트 카운팅·프리필 창
  오염·콜드 참조)에 해당 없음 확인 — 전부 `usage` 토큰 카운트 기반.

이 이하 남은 개선은 코드가 아니라 측정이다: `profile-step.py`로 미귀속
~49%를 쪼갠 결과가 다음 작업(통신 튜닝 / mHC probe)을 정한다.

주의: 오버레이 소스가 바뀌었으므로 다음 기동에서 torch.compile/AOT 해시가
갈려 **1회 장시간 재컴파일 워밍업**이 발생한다 (커널 캐시 마운트는 그대로).
롤백은 기존과 동일 — 해당 `-v ...:ro` 마운트 제거 또는 git으로 이전 오버레이
복원.

## 배포 레이아웃

스택마다 경로가 다르고, `dsv4` 는 노드마다 또 다르다(런처
`start-hy4-tp4.sh:220` 의 `overlay_dir()` 가 정본).

| 스택 | 노드 | 오버레이 경로 |
|---|---|---|
| `dsv4` (hy4) | srv2(head) · srv3 | `~/hybrid-stack/overlay-b12x/` |
| `dsv4` (hy4) | srv1 · srv4 | `~/hybrid-stack-port/overlay-b12x/` |
| `glm53` | 4노드 동일 | `~/overlays/glm53/` (런처 `OVERLAY_DIR` 기본값) |

- 기동: `launchers/start-hy4-tp4.sh` · `launchers/start-glm53-nvfp4-tp4.sh`
  (둘 다 srv2에서; 워커 3대는 ssh로 원격 기동)
- 상시 운영(dsv4): `dsv4-tp4.service`(user unit) → `dsv4-tp4-supervisor.sh`
  — 부팅 시 자동 기동, 실생성 헬스프로브 3연속 실패 시 포렌식 덤프 후 재기동.
  **수동 A/B 전에는 supervisor 를 멈춘다**(안 그러면 부팅 중에 개입한다)
- A/B: `launchers/ab-glm53.sh` — base 팔이 `VLLM_GLM53_MEGAKERNEL=0` 을 **명시**한다
  (프로필 기본값이 메가커널 세트라, 명시하지 않으면 base 가 조용히 cand 가 된다)
- 배포: `launchers/deploy-overlays.sh <profile>` 가 합성본을 4노드에 복사하고
  SHA-256 을 대조한다. **재배포는 서빙 코드 교체이므로 재기동의 일부가 아니라
  브래킷 대상**이다 — 재기동 기본은 이미 배포된 상태 그대로다
- **롤백**: 런처의 해당 `-v ...:ro` 마운트 한 줄 제거 → 재기동. 오버레이가 없으면 이미지 원본 그대로.

## bench/ — 측정 도구와 함정

| 도구 | 용도 | 함정 |
|---|---|---|
| `bench-tp4.py` | 프리필+짧은 디코드 (매회 고유 프롬프트 = 프리픽스캐시 무효) | 디코드는 dspark 수용률 탓 73–110 진동 — 구성 비교엔 프리필 사용 |
| `bench-dec.py` | **디코드 비교 전용** — 768토큰 생성으로 수용률을 평균화, 샘플별 dspark 수용률(`/metrics` 델타) 병기 | 짧은 벤치로는 10% 효과도 판별 불가 · 수용률 열로 이상치가 운(수용률)인지 실효과인지 판별 |
| `bench-ctx.py` | 장문 TTFT/디코드 분리 (스트리밍) | 비스트리밍이면 프리필이 디코드에 섞여 오측 |
| `check-quality.py` | 2K/32K/128K에 사실 3개를 25/50/75% 깊이로 심고 리트리벌 | 인덱스 stride 버그가 산문 열화가 아닌 **검색 실패**로 드러남 |
| `bench-conc.py` | 동시성 스윕 C=1/2/4 | 한 스윕 내 C값들은 상관 표본 — 단독 스윕으로 판정 금지 |
| `profile-step.py` | **미귀속 잔차 귀속** — torch profiler로 NCCL·융합 커널 포함 전 커널 버킷팅 + top-N 테이블(mHC 커널 식별용). srv2에서 실행 | `VLLM_TORCH_PROFILER_DIR` 반영 재기동 1회 필요 · 캡처 중 오버헤드로 절대값 부풀음(비율로 판단) · 멀티스트림 중첩 시 합계>벽시계 · 서빙 트래픽·supervisor 프로브가 섞일 수 있어 한가할 때 실행 |
| `korean-corruption.py` | **채택 게이트** — 한국어 출력 손상을 세어 "간헐적"을 비율로 만든다 (0/16) | n=16 의 잡음이 1/16 급이다(28차: base 팔에서 1/16 이 나왔고 판정에 쓰지 않았다) |
| `check-quality.py` · `needle-256k.py` | **채택 게이트** — 2K/32K/128K 리트리벌 9/9 · 같은 패턴의 256K needle | 인덱스 stride 버그는 산문 열화가 아니라 검색 실패로 드러난다 |
| `bracket.py` | base→cand→base 브래킷의 기록·판정 도구 (판정 규칙 자체는 원장) | 도구가 판정을 대신하지 않는다 — 게이트 통과 여부는 사람이 원장에 적는다 |
| `streamgap.py` | 동시 수용 매끄러움 — 디코드 스트리밍 중에 34K 프리필을 끼얹는다 | 서빙 품질 축이라 step/s 와 다른 신호다 |
| `pp-ctx-benchy.py` · `tg-llama-benchy.py` | llama-benchy 호환 축(pp2048 프리필 · ctx_tg@dN · tg128) — 외부 수치와 견줄 때 | 우리 브래킷 채널(C=1 step/s)과 정의가 다르다. 비교용이지 판정용이 아니다 |
| `bench_common.py` | 위 도구들의 공용 하네스(엔드포인트·프롬프트·수용률 델타 읽기) | |

## probes/ — 계측 빌드 · 오프라인 프로브

`DENEB_ATTN_PROF=1` + 해당 파일을 마운트하면 단계별 CUDA 이벤트 타이밍이 로그로 나온다
(`[deneb-prof]`/`[deneb-prep]`/`[deneb-fi]`). 캡처 가드(`is_current_stream_capturing`) 필수 —
없으면 `cudaErrorStreamCaptureInvalidated`로 기동 실패.

프리필 분해는 **완결됐다** — probe 스팬(스텝 예산) + torch profiler 커널 귀속
(`bench/profile-step.py`, 32K 캡처)의 최종 수치와 전체 판정 원장은
**[MEASUREMENTS.md](MEASUREMENTS.md)**. 요약: GPU busy 99.6%,
MoE 24.2 · comms 23.4(대역폭 바닥) · GEMM 18.6 · attn 12.5 · mHC ~16%.
과거 "미귀속 ~49%"는 comms+mHC+GEMM 일부로 전량 귀속됐고, Amdahl f≈0.57의
정체는 comms(TP 무관)+mHC(랭크 복제)였다.

### 오프라인 프로브 — 부팅을 아끼는 자리

캠페인 항목(EXP-\*)의 대부분은 **부팅 전에 프로브가 먼저 답한다**. 전부 서빙
컨테이너가 아니라 **새 컨테이너**에서 돈다 — 서빙 컨테이너의 docker-exec CUDA 는
TP 콜렉티브를 세운 전력이 있다. 래퍼가 합성 오버레이를 실제 이미지 경로에 바인드해
"리포의 소스 그대로" 를 보장한다:

```bash
bash probes/run_mk_probe.sh probes/<probe>.py       # 아무 프로브나, 합성 소스로
bash probes/run_megakernel_bench.sh                 # 메가커널 수치+타이밍 일괄
bash probes/run_mhc_glm53_bench.sh                  # MHC 발사설정 스윕
bash probes/run_prep_fused_check.sh                 # prep-fused 수치·발사 수
```

| 축 | 프로브 | 무엇을 답하나 |
|---|---|---|
| 메가커널 | `megakernel_glm53_bench.py` · `mk_pdl_graph_check.py` · `osar_build_check.py` | 세그먼트 수치·발사당 µs · 그래프 캡처 아래 PDL 이 실제로 걸리는가 · 확장이 프리페치 바인딩을 갖고 빌드되는가 |
| GEMM/양자화 | `gemm_fuse_bench.py` · `fp8_scale_granularity.py` · `nvfp4_dense_accuracy.py` · `mla_q_precision_check.py` | 작은 M 융합이 대역폭을 되찾나 · 128블록 스케일 계약값 · nvfp4/e4m3 강등의 오차 대가 |
| MoE | `moe_decode_stream_probe.py` · `moe_gate_tile_sweep.py` | **서빙되는** b12x 커널이 디코드 형상에서 내는 GB/s(EXP-14 를 닫은 프로브) · 라우터 게이트 타일 |
| MLA/인덱서 | `mk_mla_bench.py` · `mk_mla_prefill_check.py` · `indexer_gate_check.py` · `kda_conv_state_map.py` | MK-MLA 대 FlashInfer(수치·GPU 시간·호스트 계획) · head-gate split-K · KDA conv 상태 슬롯 출처 |
| 드래프터 | `drafter_fc_check.py` · `drafter_dense_path_check.py` · `head_w4_check.py` · `accept_profile.py` | 꼬리가 무엇을 지불하나(팔별) · 밀집 경로가 실제로 서빙되나 · W4 보캡 헤드의 로짓 영향 · 위치별 수용률 |
| 프리필 | `prefill_ladder.py` · `kda_prefill_bench.py` · `glm53_prefill_profile.sh` | 길이 사다리로 후보 분리 · KDA 청크 발사설정 스윕 · 32K 한 번 캡처 후 census |
| 통신 | `oneshot_ar_disttest.py` · `nccl_fingerprint.py` · `uma_datapath.cu` | 4랭크 정합 · 부팅 로그의 NCCL 지문 · RDMA 등록 메모리를 GPU 가 읽나 |
| 하드웨어 상한 | `gb10_mma_rates.cu` · `gb10_gather_roof.cu` | sm_121a 텐서코어가 포맷별로 실제 발행하는 양 · LPDDR gather 천장 |
| 트레이스 보조 | `trace_precise.py` · `gap_concurrent.py` | 정밀 구간 · **가장 큰 공백 동안 다른 스트림에서 무엇이 도나**(그래프 안 커널이 임계경로인지 가르는 질문) |
| 재현 진단 | `mhc_replay.py`(+`run_mhc_replay.sh`) · `refine_ab.py` | 그래프 리플레이에서만 나는 MHC 불안정을 좁힌다 · REFINE_PASS A/B 하네스(판정은 기각) |

**규율**: 프로브 숫자는 원장에 그대로 들어가지 않는다 — 오프라인 값은 상한이거나
형상이 다르고(27차: 상한이 프로브 하나에 절반으로 줄었다), 서빙 판정은 브래킷이다.
프로브가 하는 일은 **부팅할 가치가 있는지**를 먼저 가르는 것이다.

## tools/ · census.py — 트레이스 분석

프로파일 캡처(`start_profile`)로 뜬 torch 트레이스에서 **커널이 몇 발 돌고 어디에
시간이 가는지**를 뽑는다. 무엇을 믿을지의 규칙은 하나다 — **개수는 정본, 시간은 같은
트레이스 안에서만 상대 비교**(CUPTI 가 GPU 바쁜 시간을 부풀린다).

| 도구 | 용도 | 함정 |
|---|---|---|
| `census.py` | 커널/스텝 · 그룹별 분포 · **소유권(우리 리포 vs 이미지)** · 상위 25 커널 | 스텝 분모를 세지 않고 추정하면 틀린다(그래서 `_get_num_sampled_and_rejected` 로 센다). 그룹 정규식은 계약이 아니다 — 앵커 없는 패턴이 남의 커널을 우리 칸에 넣은 전례(`k_reduce` ← deep_gemm `split_k_reduce`) |
| `census.py --after REGEX [--depth N]` | **인접성** — 지목한 커널 직후 같은 스트림에서 무엇이 도는지 | "AR 뒤에 이 체인이 붙는다" 류 전제를 소스 추정 대신 트레이스로 확정한다. 다른 스트림의 커널은 '직후'가 아니다 |
| `tools/trace_step_composition.py` | 깨끗한 스텝의 범주별 ms/발 중앙값, **유휴는 스트림 합집합** | 단일 스트림 합산은 유휴를 과소 계산한다(9.24 vs 5.65 ms 오차 전력). `--diff` 는 두 트레이스를 나란히 놓지만 **개수 채널이 정본**이고 시간은 부팅이 다르면 브래킷이 아니다 |
| `tools/trace_step_tail.py` | 스텝을 forward / 꼬리(마지막 MoE 전문가 커널 뒤 = 헤드·샘플러·드래프터)로 자르고 꼬리를 분해 | 꼬리는 GPU 93% 바쁜 임계경로다 — "드래프터는 공짜" 는 호스트가 병목이던 시절의 값 |
| `tools/trace_common.py` | 위 도구들의 공용 로더·스텝 절단·범주·**소유 판정** | 새 커널을 만들면 `OURS` 에 심볼을 넣어야 지도가 우리 것으로 센다. 분류기가 코드보다 늦으면 1위 범주가 "기타"가 된다 |

```bash
python3 census.py ~/vllm-prof/dp0_pp0_tp0_dcp0_ep0_rank0.*.pt.trace.json.gz
python3 tools/trace_step_composition.py <trace.gz> [--diff <cand.gz>]
python3 tools/trace_step_tail.py <trace.gz>
```

**플릿 규율**: 이 분석들은 rank 노드의 CPU 를 쓴다. 디코드 레그가 도는 동안 돌리지
말고(단일 코어 파이썬이 step/s 15.2 → 9.1 로 떨어뜨린 전력), 부팅 창에서
`nice -n 19 taskset -c 19` 로. 메모리는 문제가 아니다 — 파서가 이벤트를 흘려보내므로
40 MB 캡처에도 RSS 17 MB 다. 결과 지도는 [`STEP_KERNEL_MAP.md`](STEP_KERNEL_MAP.md).

## 미채택 포팅 (파일은 제거, 근거 보존 — 복원: git history `e95f82f`)

| 파일(구 `experimental/`) | PR | 미채택 사유 |
|---|---|---|
| `attention_p430.py` | #51430 | eager 구간 축소는 **breakable CG 전용** — 프로덕션은 `VLLM_USE_BREAKABLE_CUDAGRAPH=0`이라 죽은 코드. 켜면 이 포팅은 CUDA illegal access(전제인 #49236 `eager_scratch.py`가 이미지에 없음), 켜는 것 자체도 장문 디코드 5배 저하+엔진 사망 |
| `*_p474.py` | #47474 | `token_to_req_indices` 3중 계산 캐시 — 포팅·검증 완료했으나 **성능 중립**(우리 포크에 이미 빠른 경로 2개 존재), 최광역 파일(`v1/attention/backend.py`)이라 미채택 |

## 확정 스위치 (재시험 불필요)

- `VLLM_DSV4_INDEXER_SP=1` — 유일하게 켬 (+5.4%, bit-exact)
- `VLLM_DSV4_GATE_FUSED=1` (런처 `GATEFUSE`, 기본 ON) — 융합 small-M 라우터
  게이트: GB10은 GateLinear 가속 티어 장치판정 전탈락 → Tier-4 3커널 체인
  (1.71ms/step)이었던 것을 트리톤 단일 커널(fp32 직출력)로. 디코드 C=1
  **+3.5%**(브래킷), 프리필 무접촉, 롤백 `GATEFUSE=0` (MEASUREMENTS.md 08-10)
- `VLLM_USE_BREAKABLE_CUDAGRAPH=0` **고정** — 1이면 장문 디코드 ~5배 저하 + 엔진 사망. 해당 forward 분기는 오버레이에서 제거됨
- `B12X_INDEXER_STREAM`/`B12X_KV_STREAM` off — 켜면 디코드 반토막
- b12x MoE는 **TP 전용** — EP/DP 불가 → SP(#46789)·EP 계열 전부 구조적 불가. 런처의 `SP`/`EPFLAG` 노브도 제거됨

## 라이선스

`overlay/`·`probes/`의 vLLM 파생 파일은 원본 SPDX 헤더 그대로
**Apache-2.0**이며, 리포 전체가 같은 라이선스를 따른다 (`LICENSE`).
