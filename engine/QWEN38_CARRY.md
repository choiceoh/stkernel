# Qwen3.8 이식 캠페인 — GLM 최적화를 Qwen3.8 에서 '그대로'

> 살아 있는 참조 — **고정된 후보 목록과 항목마다의 상태. 항목이 닫히면(머지·기각) 이 표의 상태 칸을 고친다.** 여기가 틀리면 그건 버그다.

출발점은 [`MODEL_DEPENDENCE_20260917.md`](MODEL_DEPENDENCE_20260917.md) 다. 그 조사에서 GLM 시절 최적화 203건 중
Qwen3.8 에 **그대로** 닿는 것은 36.5% 였고, 나머지는 재측정·어댑터·수식 차이로 막혀 있었다.
이 캠페인은 그 나머지를 할 수 있는 만큼 '그대로' 로 옮긴다.

## 운영자 결정 (2026-09-17, grill-me)

| # | 결정 |
|---|---|
| Q1 | 가능한 것부터: Qwen3.8 중심, DSv4.1 은 단일 GPU 로 되는 것만 |
| Q2 | CPU + 단일 GPU 레인. 4노드 플릿은 쓰지 않는다 |
| Q3 | 어댑터(glue) 셀도 GPU 판정과 실측 기록이 붙으면 `admitted` |
| Q4 | MoE 셀은 커널 기록(오라클 2% + 타일 스윕)으로 `admitted`. 엔진 속도(D17)는 따로 미실측으로 적는다 |
| Q5 | 수식이 달라 같은 코드를 못 쓰는 GLM 최적화는 아이디어를 Qwen3.8 자체 커널(QSA·GQA·게이트 잔차·MTP)로 옮긴다 |
| Q6 | 수치가 바뀌는 커널 작업도 포함한다. 단일 GPU 에서 `engine/modules` 오라클로 판정한다 |
| Q7 | GLM 과 공유하는 커널(b12x·KDA 링·conv·dense·one-shot)도 개선한다 |
| Q8 | 공유 커널을 바꿔도 GLM 은 단일 GPU 판정(실가중치 레인 검사 포함)으로 기본 켬. **헌장 D17 규칙 2 의 예외**이며, 해당 PR·원장 항목마다 "운영자 결정 2026-09-17: 플릿 미실측" 을 적는다 |
| Q9 | 이 목록을 시작할 때 고정한다. 항목마다 머지 또는 기각과 기록으로 닫는다. 도중에 나온 새 아이디어는 다음 목록으로 넘긴다 |

## 세는 법

- **크기:** Qwen3.8 C=1 디코드 한 스텝(타깃 검증 그래프 + MTP 드래프트 그래프)에서 줄어드는 발사·복사·바이트 수다.
  `SPEC_K=1` 이라 두 그래프 모두 2 토큰이다. 층은 GDN 36, QSA 12, MoE 48 이고, MTP 헤드가 QSA 1·MoE 1 을 더한다.
  **소스 계수 추정이지 실측이 아니다**(GLM 접기 PR 들과 같은 방식).
- **종류:**
  - `fold`: 바이트 동일 접기·융합
  - `kernel`: 수치·타일·정밀도·알고리즘이 바뀜
  - `native`: 어댑터를 네이티브 경로로 대체
  - `measure`: GPU 에서 고를 디스패치 선택
  - `fix`: 결함 수정
- **판정:**
  - `cpu`: stk-test 에서 `TRITON_INTERPRET=1` 로 참조와 바이트 일치
  - `gpu`: 단일 GB10 에서 오라클 대비
  - `glm`: GLM 커널이 바뀌어 GLM 실가중치 단일 GPU 검사도 필요
- **상태:** `열림` → `PR #N` → `머지` 또는 `기각(기록)`.

## 0. 선행 — 결함과 하네스

| ID | 내용 | 대상 | 종류 | 판정 | 비용 | 상태 |
|---|---|---|---|---|---|---|
| P1 | C=1 캡처 스텝의 `rows_req` 가 stride-0 뷰라 QSA 커널이 저장소 너머를 읽고 어텐션이 스텝을 거부 | `profiles/qwen38/net.py:step_meta`, `kernels/qsa.py` | fix | cpu | 시간 | 머지 #1084 |
| P2 | Qwen3.8 서빙 커널의 CPU 인터프리터 하네스. QSA ops·게이트 잔차·GDN·캡처 `step_meta` 를 `engine/modules` 오라클에 대조. 지금은 테스트가 0 건이라 `cpu` 판정의 전제 | `tests/` | fix | cpu | 일 | 머지 #1099 |
| P3 | 서빙 프리필이 768 토큰 블록마다 타깃 forward 를 따로 돈다. `served_step` 이 `marks` 를 버려서 청크당 최대 42 forward | `profiles/qwen38/adapter.py`, `base/composed.py` | fix | gpu | 일 | 열림 |
| P4 | 부팅이 `prepare_dense` 에 `consume_weights` 를 주지 않아 BF16 원본이 랭크당 약 1.9 GB 상주. 프리샤드가 패딩 크기를 예약해야 함 | `profiles/qwen38/fleet.py`, `preshard.py` | fix | cpu | 일 | 열림 |

## 1. 셀 판정 — 단일 GPU 레인 기록으로 `cells.py` 에 `admitted`

| ID | 내용 | 대상 | 종류 | 판정 | 비용 | 상태 |
|---|---|---|---|---|---|---|
| C1 | 셀 판정 프로브 모드(`engine_kernel_check --lanes qwen38_cells`). 단일 레인 티켓은 main 에 있는 프로브만 돌린다 | `probes/engine_qwen38_cells.py` | fix | cpu | 시간 | 머지 #1089, 티켓 `qwen38-cells-0917` 대기 |
| C2 | dense 패딩 어댑터 GPU 판정 + W4A8/FP8 전환 행 수 실측: Qwen3.8 hidden 2560 · 중간 160, DSv4.1 576 | `kernels/dense`, `cells.py` | measure | gpu | 시간 | 프로브 머지 #1096, 티켓 `qwen38-dense-0917` 대기 |
| C3 | KDA decay 어댑터(ring·chunk·recurrent) GPU 판정 + BV 8/16/32 스윕(4/12×128×128, T=2, 1–4 행, 정확 롤백 게이트) | `kernels/kda`, `cells.KDA_MEASURED_CELLS` | measure | gpu | 시간 | 프로브 머지 #1098, 티켓은 K1 뒤(서빙 진입점을 잰다) |
| C4 | MoE EP 셀(로컬 128/512, I640, top-10, silu): 오라클 2% + micro 타일·MAC 사다리 + 프리필 `tile_m` 핀 | `kernels/b12x/moe_dispatch.py`, `cells.py` | measure | gpu | 일 | 프로브 이 PR |
| C5 | DSv4.1 mHC V41 이음매(`MHCV41`) GPU 판정 | `kernels/dense/mhc.py`, `cells.py` | measure | gpu | 시간 | 열림 |
| C6 | Qwen3.8 자체 레인 GPU `qualify`(게이트 잔차·QSA·GDN). 수치 변경 작업의 기준점 | `profiles/qwen38/lanes.py` | measure | gpu | 시간 | 열림 |
| C7 | `cells.py`: 측정된 어댑터 셀을 `admitted` 로(Q3). `summarize.py` 는 서빙 커널이 PR 이 최적화한 커널과 같을 때만 '그대로' 로 셈. 재집계 | `kernels/cells.py`, `measurements/st_model_dependence_20260917/summarize.py` | fix | cpu | 시간 | 장치는 머지 #1095(측정 튜플 넷은 기록이 붙을 때 채움) |

플릿이 필요해 이번 목록에서 뺀 것: one-shot·프리필 통신의 hidden 2560 실측(4랭크), Qwen3.8 부팅 onepass(D17 속도 기록).

## 2. QSA — 인덱서와 어텐션

| ID | 내용 | GLM 출처 | 대상 | 종류 | 크기 | 판정 | 비용 | 상태 |
|---|---|---|---|---|---|---|---|---|
| Q1 | 죽은 글루 복사 제거: 읽히지 않는 `positions…expand(N,1,3).contiguous()`, `ik.contiguous()`, `first[:,0].contiguous()` 등 | #547 #926 #933 | `net.py:_qsa`, `qsa.py:norm_rope_partial` | fold | −39 발사 | cpu | 시간 | 머지 #1090 |
| Q2 | 캡처 `step_meta` 를 Triton 한 발사로(타깃·드래프트 두 번) | #543 #546 #819 #821 | `net.py:step_meta`, `kernels/step_addresses.py` | fold | 약 −80 발사 | cpu | 시간 | 머지 #1094 |
| Q3 | QSA 입력 융합: q/k norm+rope 와 K/V 저장을 (행, 헤드)당 한 발사로, 압축→norm→rope→인덱스 키 쓰기를 한 발사로, 링 쓰기를 K/V 저장에 합침 | #914 #936 #547 #582 #921 | `qsa.py`, `net.py:_qsa` | fold | 층당 11→1, 약 −130 발사 | cpu | 일 | 열림 |
| Q4 | 출력 게이트를 어텐션 최종 저장 안에서 적용 | #919 #899 | `qsa.py:qsa_sparse_paged_attention` | fold | −65 발사, 프리필 청크당 −21.6 GiB | gpu | 시간 | 열림 |
| Q5 | 어텐션이 블록 id 를 타일 루프 안에서 위치로 확장(확장 발사 제거) | #819 #887 | `qsa.py` 선택·어텐션 | fold | −13 발사, 프리필 청크당 −8 GiB | cpu | 시간 | 이 PR |
| Q6 | 확장이 블록 id 를 오름차순 정렬: 어떤 선택기가 돌았든 합산 순서가 같게(지금은 디코드와 프리필이 다르게 반올림할 수 있음) | #546 #693 | `qsa.py:_expand_qsa_indices_kernel` | kernel | 0 | gpu | 일 | 열림 |
| Q7 | 디코드 블록 선택을 `st_dsa_select` 로(GLM `decode_topk.cu` 그대로, 열 수 컷오프) | #1010 #926 | `qsa.py:select_blocks` | native | 약 −130 발사 | gpu·glm | 일 | 열림 |
| Q8 | 점수 커널이 요청의 키 타일을 최대 4 쿼리 행에 한 번만 읽기 | #971 #1010 | `qsa.py:_qsa_mqa_paged_kernel` | kernel | 128K 에서 스텝당 −107 MB | gpu | 일 | 열림 |
| Q9 | QSA 커널 GB10 발사 기하 스윕: norm_rope·압축·저장·확장·점수, 어텐션 split 프로필(지금은 상류 GB300 프로필로 디코드마다 64 split) | #554 #556 #641 #658 #737 | `qsa.py` | measure | split 1 이면 −13 발사 | gpu | 일 | 열림 |
| Q10 | 덮인 앞부분(≤2,050 위치)의 dense causal 프리필 커널 | #887 #889 | `qsa.py` 새 커널 | kernel | 2K 이하 프롬프트 K/V 읽기 약 −50% | gpu | 일 | 열림 |
| Q11 | 프리필 인덱스 쿼리 행을 랭크별로 나눠 점수(`QueryShard`) | #881 | `net.py:_qsa` | fold | 긴 프롬프트 점수 행 −75% | cpu(LocalTP) | 일 | 열림 |
| Q12 | QSA 어텐션을 메가커널로(`mla/glue.gqa`, FP8 KV) | 메가커널 계열 | `lanes.py`, `caches.py` | measure | KV 바이트 ½ | gpu | 일 | 열림 |
| Q13 | K/V 를 한 영역의 레코드로 두는 캐시 배치 | #641 | `caches.py` | measure | 0 발사 | gpu | 시간 | 열림 |

## 3. 게이트 잔차 — mHC 계열 아이디어

| ID | 내용 | GLM 출처 | 대상 | 종류 | 크기 | 판정 | 비용 | 상태 |
|---|---|---|---|---|---|---|---|---|
| H1 | down+inject GEMM 을 `leave_norm` 안으로 접기(디코드 ≤16 행) | MK_SEG_MHC | `kernels/gated_residual.py` | kernel | 사이트당 −1, −100 발사 | gpu | 일 | 열림 |
| H2 | gates + up GEMM + `mix_mean` 한 발사 | MK_SEG_MHC | `gated_residual.py:mix` | kernel | 사이트당 −2, −200 발사 | gpu | 일 | 열림 |
| H3 | hidden 폭 커널을 512 폭 5 타일로 재배치 | #634 | `gated_residual.py` | measure | 0 발사 | cpu·gpu | 시간 | 열림 |
| H4 | `leave` 를 one-shot consumer 의 PDL 종속으로 | MK AR consumer, #689 | `gated_residual.py`, `lanes.py` | measure | 약 0.4 ms/스텝 추정 | gpu | 일 | 열림 |
| H5 | `leave` 가 TP4 랭크 패킷을 직접 합산(생산자 TX 슬롯과 함께) | #812 #826 | `gated_residual.py`, `kernels/oneshot` | fold | 스텝당 약 1 MB | cpu | 일 | 열림 |
| H6 | `--hc-fp8` 레인 실측(믹서 가중치 읽기가 스텝당 1.32 GB) | — | `net.py:_prepare_hc_fp8` | measure | 바이트 −40%, 발사 +300 | gpu | 시간 | 열림 |
| H7 | 작은 접기: 임베딩 `repeat`, PLE 층의 분리된 leave 와 out-of-place 덧셈 | — | `net.py` | fold | −3 발사 | cpu | 시간 | 기각: 임베딩 `repeat` 는 all-reduce 뒤라 접을 자리가 없고, PLE 층 둘은 게이트 잔차 커널에 변형을 하나 더 들여야 해서 스텝당 2 발사의 값이 없다 |

## 4. MTP 드래프터

| ID | 내용 | GLM 출처 | 대상 | 종류 | 크기 | 판정 | 비용 | 상태 |
|---|---|---|---|---|---|---|---|---|
| D1 | MTP 체인 K>1 을 드래프트 재생 안에서(SPEC_K 2–4 스윕). 지금 K=1 은 스텝당 최대 2 토큰이고, CPU 참조 레인 K=3 은 라운드당 3.67 토큰. k≠1 가드만 풀면 `streams=None` 에서 죽는다 | #869 #627 | `decode_graphs.py:DraftGraphs`, `adapter.py:ServedMTP`, `facts.py` | native | 스텝당 토큰 | gpu | 일 | 열림 |
| D2 | 검증 pick 을 전 vocab gather 없이: argmax 키 + max all-reduce, 캡처 샘플링 | #563 #929 | `decode_graphs.py:TargetGraphs`, `adapter.py` | fold | −1 all_gather(993 KB), −1 H2D | gpu | 일 | 열림 |
| D3 | 드래프트를 GPU 에 남김: 스텝당 host sync 제거, streams 인계를 복사 한 번으로 | #605 #627 | `adapter.py`, `decode_graphs.py` | fold | −1 sync, −2 복사 | gpu | 일 | 열림 |
| D4 | MTP 프롬프트 관측을 한 forward 로 묶기(P3 이 먼저 닫히면 불필요) | #627 #722 | `adapter.py:ServedMTP.observe` | fold | 청크당 최대 −41 헤드 forward | gpu | 일 | 열림 |
| D5 | argmax partials/finish 를 한 warp 로 | #1004 | `kernels/common/vocab_candidates.py` | measure | 0 발사 | cpu·glm | 시간 | 열림 |
| D6 | MTP dense 투영을 FP8 로 디코드하는 수용률 팔 | #862 #863 #871 | `net.py:prepare_dense` | measure | 수용률 | gpu | 시간 | 열림 |

## 5. 공유 커널 — Qwen3.8 셀이 닿게

| ID | 내용 | GLM 출처 | 대상 | 종류 | 크기 | 판정 | 비용 | 상태 |
|---|---|---|---|---|---|---|---|---|
| K1 | GDN 게이트를 링 커널 안에서 계산(in_proj 조각을 stride 로 읽음) | #569 #571 | `kernels/kda/ring.py`, `fused_recurrent.py`, `net.py:_gdn_rows` | native | −36 발사 | cpu·glm | 시간 | 열림 |
| K2 | `gated_norm` 이 z 를 stride 로 읽음 | #569 | `kernels/gdn.py` | fold | −36 발사·복사 | cpu | 시간 | 머지 #1091 |
| K3 | chunk 파이프라인이 헤드별 decay 를 네이티브로(widen·repeat_interleave 제거) | #615 #811 | `kernels/kda/chunk_decay.py`, `kda.py` | native | 프리필 청크당 약 −20 GiB 쓰기 | cpu·glm | 일 | 열림 |
| K4 | strided q/k l2norm 을 4 헤드에서도 admit | #811 | `kda.py:_glm53_qk_l2norm_strided` | fold | 프리필 층당 −3 발사 | cpu·glm | 시간 | 열림 |
| K5 | GDN norm 이 out_proj 의 W4 입력 팩을 씀(S2 뒤) | #968 #978 | `gdn.py`, dense | kernel | −36 발사 | gpu·glm | 일 | 열림 |
| M1 | MoE 출력 finalizer 한 발사: BF16(routed + shared·gate), sigmoid 는 torch 에 둠(b12x FP32 평면 직접 소비는 다음 목록) | #904 #906 | `kernels/moe_output.py:gated_sum`, `lanes.py`, `net.py:_moe` | fold | −196 발사 | cpu·gpu | 시간 | 머지 #1092 |
| M2 | 라우팅(softmax top-10, 재정규화, BF16 반올림, EP 리맵) 한 발사 | #789 #810 #779 | `lanes.py:route_softmax_topk` | kernel | 약 −700 발사 | gpu | 일 | 열림 |
| M3 | micro 레인의 EP 추가 경로(direct FP32 FC2 scatter, shared FC1 A, M16 타일)를 E128/H2560/I640/silu 로 | #955 #920 #974 | `b12x/moe_dispatch.py`, `moe_micro_kernel.py` | kernel | FC1 입력 로드 ½ | gpu·glm | 일 | 열림 |
| M4 | EP-local dynamic 프리필 커널(층마다의 host sync `nonzero` 제거) | #895 #811 | `b12x/moe_dynamic_ep_local.py`, `lanes.py:moe` | kernel | 프리필 층당 −6 발사, −2.3 GiB | gpu | 일 | 열림 |
| M5 | shared expert 를 overlap 스트림에서 | #789 | `dense/shared_mlp.py`, `net.py:_moe` | measure | 층당 약 5 발사 겹침 | gpu | 시간 | 열림 |
| S1 | swiglu 가 0 패딩된 sh_down 입력을 직접 씀 | #569 #973 | `kernels/common/swiglu.py`, `PaddedDenseLinear` | fold | −98 발사 | cpu·gpu | 시간 | 머지 #1097 |
| S2 | W4 입력 재사용을 m 2–8, 4120/4224×2560, 2560×1536 에서 admit + hidden 2560 ksr 스윕 | #1050 #888 #946 #939 #969 | `kernels/dense/kernels.cu` | measure | GEMM 98 개의 입력 양자화 대폭 감소 | gpu·glm | 시간 | 열림 |
| S3 | 부팅 자기 보정(GPTQ W4/FP8·헤드) 배선. 지금 Qwen3.8 팩은 전부 round-to-nearest | #650 #659 #661 #673 #779 | `profiles/qwen38/fleet.py` | kernel | 품질 | gpu | 일 | 열림 |
| X1 | 작은 합에 compact 12-CTA one-shot consumer | #967 #944 #957 | `kernels/oneshot` | measure | 합 101 개가 48→12 CTA | gpu·glm | 시간 | 열림 |
| X2 | 생산자·MoE finalizer 가 TX 슬롯에 직접 씀(hidden 을 형상에서) | #826 #904 #906 | `kernels/oneshot`, dense | kernel | 발사 중립, 스텝당 −98 복사 | gpu·glm | 일 | 열림 |

## 진행 순서

1. **선행:** P1, P2. `cpu` 판정의 전제다.
2. **셀 판정:** C1 을 머지 → C2–C6 단일 레인 티켓 → C7.
3. **CPU 로 닫히는 접기:** Q1, Q2, M1, K2, S1, K1, H7, Q5, Q3, K4.
4. **GPU 로 판정하는 커널:** M2, H2, H1, Q4, Q6→Q7, D1, D2, D3, P3, Q8, K3, M3, M4, K5.
5. **스윕과 메모리:** Q9, H3, S2, X1, H6, M5, D5, D6, Q13, P4.
6. **크고 위험한 것:** Q10, Q11, Q12, H4, H5, X2, S3, D4.

## 옮기지 않는 GLM 최적화

| 계열 | 항목 | 이유 |
|---|---|---|
| mHC | #1009, #972, #689 TileLang, #634 TMA 자체, #860 | CTA 티켓·FP32 계수·TileLang 패스가 Triton 게이트 잔차에 없음. 믹서 가중치는 이미 BF16 이고, MTP 는 스트림 전체를 읽음 |
| 인덱서 | #907, #961, #821 헤드 게이트, #556 Hadamard/FP8 | 점수가 이미 `visible` 에서 멈춤. 트리 검증 없음. QSA 에 학습된 헤드 가중치와 FP8 인덱서 없음 |
| 인덱서 | #582 #815 #971 #921 #926 | 이미 있음: vLLM 커널이 행 전부를 한 발사로 처리 |
| 어텐션 | #956 #952, #641·#554 SM121 명령, #658·#737 prefill32 CUDA, #698 | FP8 잠재 확장이 없음(BF16 KV). PTX 명령과 메가커널 코드는 Triton 에 대응이 없음. #698 은 GLM 에서도 기각 |
| 드래프터 | #1003, #982, #894·#900, #777, #735 #736 #729 #725, #951 #947, #879, #756, #871 | DFlash2 전용·비전 없음·FC 가 이미 BF16. 구조가 이미 있음(argmax 키, 융합 norm, KV 헤드당 읽기, 캐시된 역주파수). 트리 없음. GLM 에서도 기각 |
| 공유 | b12x 정적 v4/v5 계열, SP FP8 패킷 프리필 계열, deferred KDA 커밋, FP16 상태, CTA 셀 계열, GLM 라우터 | EP 셀이 정적 레인에 닿지 않음(아이디어는 M3 로). 프리필이 일반 TP all-reduce(M4 로). T=R=2 에서 같은 바이트에 발사만 늘어남. GLM 에서도 되돌림. GLM 폭 컴파일(S2 로). 라우터 수식이 다름(M2 로) |
