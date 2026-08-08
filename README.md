# stkernel

DeepSeek-V4-Flash-0731 · **TP=4** 프로덕션 오버레이 스택
(4× NVIDIA DGX Spark GB10 · CRS812 스위치드 패브릭 · vLLM eldritch/b12x 포크 이미지)

베이스 이미지 `aidendle94/sparkrun-vllm-ds4-gb10:production-hybrid-1.6`(2026-06-26 계열)은
서드파티라 소스 트리가 없다. 이 리포는 그 이미지를 **재빌드 없이** 개선하기 위한
전체 스택이다 — 파이썬 파일을 `site-packages` 위에 read-only 바인드 마운트하는
오버레이 방식이며, 모든 수정부에 `# deneb fork: port of upstream PR #NNNNN` 마커가 있다.

```
-v <overlay>/attention.py:/opt/venv/.../vllm/models/deepseek_v4/attention.py:ro
```

## 구성

| 디렉터리 | 내용 |
|---|---|
| `overlay/` | **프로덕션 오버레이 4파일** (업스트림 PR 5건 포팅 + TP4/GB10 전용 슬리밍) |
| `launchers/` | 프로덕션 런처 + 슈퍼바이저 + systemd 유닛 + `deploy-overlays.sh`(4노드 배포+md5 검증) |
| `bench/` | 검증·측정 도구 |
| `probes/` | 계측 빌드 (CUDA 이벤트 단계 분해, `DENEB_ATTN_PROF=1`) — 오버레이와 동일 코드 + 계측 |
| `tests/` | GPU/vllm 없이 도는 순수 로직 검증 (`python3 tests/test_logic.py`) — 청커 예산·skip-topk 규칙·SP 샤드 커버리지 |

## overlay/ — 프로덕션 채택 5건

| PR | 파일 | 내용 | 실측 |
|---|---|---|---|
| [#51209](https://github.com/vllm-project/vllm/pull/51209) | `attention.py` | IndexCache — C4A 레이어가 이전 top-k 재사용 (`index_topk_freq=4`) | 장문 디코드 32K **+13.6%** / 128K **+16.0%** |
| [#51042](https://github.com/vllm-project/vllm/pull/51042) | `flashinfer_sparse.py` + `sparse_swa_dsv4.py` | DSpark 비인과 decode SWA 버퍼 폭을 메타데이터로 전달 (window_size 오판독 수정) | 정합성 |
| [#51252](https://github.com/vllm-project/vllm/pull/51252) | `indexer.py` | 프리필 청커 예산을 `compress_ratio`로 나눠 소비 버퍼와 단위 일치 | 정합성 · 프리필 무손실 |
| [#49059](https://github.com/vllm-project/vllm/pull/49059) | `flashinfer_sparse.py` | 빈 프리필 청크(0-element reshape) 크래시 가드 | 예방 |
| [#51202](https://github.com/vllm-project/vllm/pull/51202) | `flashinfer_sparse.py` | prefill/decode 게이트를 요청수 대신 토큰수로 | 예방 |

검증: 프리필 2,437–2,533 tok/s(무회귀) · 장문 리트리벌 9/9(2K/32K/128K) · traceback 0.

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

### 2차 최적화 (미실측 — 다음 배포에서 bench-dec/check-quality로 검증)

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
  4파일 전부를 마운트 디렉터리에서 md5로 헤드와 대조해 **노드 간 오버레이
  스큐를 기동 전에 차단**한다.
- supervisor `BOOT_GRACE` 1500→3600s: 오버레이 소스 변경 후 재컴파일이
  25분을 넘기면 부트 중 재기동 루프에 빠지던 것 방지 (API가 뜨면 즉시
  통과하므로 정상 부팅엔 영향 없음).
- attn_prof의 decode/prefill 분류 임계값을 하드코딩 64에서
  `MAX_NUM_SEQS×(1+SPEC_TOKENS)`(프로덕션 96)로 — 동시 10요청 초과 디코드
  배치가 prefill로 오분류되던 것 수정.

### 3차 최적화 (미실측 — 기동 로그·리트리벌로 검증)

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

### 4차 최적화 (미실측 — bench-tp4/bench-ctx 프리필로 검증)

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
- `launchers/deploy-overlays.sh` 신설: 리포 → 4노드 `-b12x` 디렉터리 배포 +
  md5 검증 (preflight 스큐 검사의 쓰기 쪽 반쪽).

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

| 노드 | 오버레이 경로 |
|---|---|
| srv2(head)·srv3 | `~/hybrid-stack/overlay-b12x/` |
| srv1·srv4 | `~/hybrid-stack-port/overlay-b12x/` |

- 기동: `launchers/start-hy4-tp4.sh` (srv2에서; 워커 3대는 ssh로 원격 기동)
- 상시 운영: `dsv4-tp4.service`(user unit) → `dsv4-tp4-supervisor.sh`
  — 부팅 시 자동 기동, 실생성 헬스프로브 3연속 실패 시 포렌식 덤프 후 재기동
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

## probes/ — 계측 빌드

`DENEB_ATTN_PROF=1` + 해당 파일을 마운트하면 단계별 CUDA 이벤트 타이밍이 로그로 나온다
(`[deneb-prof]`/`[deneb-prep]`/`[deneb-fi]`). 캡처 가드(`is_current_stream_capturing`) 필수 —
없으면 `cudaErrorStreamCaptureInvalidated`로 기동 실패.

실측 트리 — 프리필 스텝 예산 최종 (4,096토큰, 벽시계 ~1,552ms @ 2,640 tok/s):

| 구간 | ms/스텝 | 비중 |
|---|---|---|
| MoE (라우터+experts+shared) | 429 | 27.6% |
| attention_impl (prep+FlashInfer 커널) | 268 | 17.3% |
| 입력 GEMM (wq_a/wkv/인덱서 헤드) | 57 | 3.7% |
| o_proj einsum | ~40 | 2.6% |
| **미귀속 잔차** (mHC + TP collectives + launch glue) | ~758 | **~49%** |

미귀속 잔차는 Amdahl f≈0.57 비스케일 비율과 정합 — **다음 표적**. 파이썬
스팬으로는 못 쪼갠다 (`fuse_gemm_comms`/`fuse_allreduce_rms`가 통신을 컴파일된
커스텀 op 안으로 융합) → `bench/profile-step.py`로 커널 레벨 귀속.

## 미채택 포팅 (파일은 제거, 근거 보존 — 복원: git history `e95f82f`)

| 파일(구 `experimental/`) | PR | 미채택 사유 |
|---|---|---|
| `attention_p430.py` | #51430 | eager 구간 축소는 **breakable CG 전용** — 프로덕션은 `VLLM_USE_BREAKABLE_CUDAGRAPH=0`이라 죽은 코드. 켜면 이 포팅은 CUDA illegal access(전제인 #49236 `eager_scratch.py`가 이미지에 없음), 켜는 것 자체도 장문 디코드 5배 저하+엔진 사망 |
| `*_p474.py` | #47474 | `token_to_req_indices` 3중 계산 캐시 — 포팅·검증 완료했으나 **성능 중립**(우리 포크에 이미 빠른 경로 2개 존재), 최광역 파일(`v1/attention/backend.py`)이라 미채택 |

## 확정 스위치 (재시험 불필요)

- `VLLM_DSV4_INDEXER_SP=1` — 유일하게 켬 (+5.4%, bit-exact)
- `VLLM_USE_BREAKABLE_CUDAGRAPH=0` **고정** — 1이면 장문 디코드 ~5배 저하 + 엔진 사망. 해당 forward 분기는 오버레이에서 제거됨
- `B12X_INDEXER_STREAM`/`B12X_KV_STREAM` off — 켜면 디코드 반토막
- b12x MoE는 **TP 전용** — EP/DP 불가 → SP(#46789)·EP 계열 전부 구조적 불가. 런처의 `SP`/`EPFLAG` 노브도 제거됨

## 라이선스

`overlay/`·`probes/`의 vLLM 파생 파일은 원본 SPDX 헤더 그대로
**Apache-2.0**이며, 리포 전체가 같은 라이선스를 따른다 (`LICENSE`).
