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

## 운영자 결정 (2026-09-19, 통신·지연 — H4·H5·X1·X2, grill-me)

운영자의 틀: "커널 밖 1/3 을 줄일 후보". 조사에서 확인한 것 — 이 넷은 합 하나에 붙은 GPU 쪽 부대비용(발사 간격, reduce 단계,
빈 CTA, 복사)만 깎고, one-shot 의 RDMA 바닥(GLM 게이지 0행 외삽 합당 16–20 µs)과 낙오 편차는 그대로다. GLM 쪽 근거도 항목마다
다르다: H4 의 원형(#473)만 C=1 −0.54 ms 실측이 있고, H5·X2 의 원형(#812·#826·#904·#906)은 속도 주장 없이 기본 켬, X1 은 ST 엔진이
컴파일한 적이 없다.

| # | 결정 |
|---|---|
| 1 | 넷을 carry 규칙대로 PR 하나씩 닫는다(단일 GPU 판정, Q8 예외로 기본 켬). 세 PR(H4·H5·X2)이 머지되면 세션이 `fleet.sh st-hold` 로 플릿 창을 잡는다(quiet gate 로만 양보): 같은 빌드 OFF→ON→OFF, C=1·C=4 같은 요청 묶음, ON 과 첫 OFF 에서 4랭크 진단 트레이스(`decompose_trace.py`)로 합의 바닥·낙오 편차·커널 사이 빈 시간을 분해. ON 이 잡음 밖으로 느리면 기본을 끄고 다음 창에서 항목별로 가른다 |
| 2 | X1 은 측정 없이 기각한다(전제 오류, 아래 표) |
| 3 | X2 는 MoE 쪽(패킷 커널의 복사 단계 안에서 Qwen 의 게이트 합)만 만든다. dense 쪽(W4 GEMM 이 TX 슬롯에 직접)은 먼저 GLM 형상의 기존 직접 생산자로 GPU 쪽 시간을 재고, 합당 1 µs 이상 이길 때만 N=2560 으로 옮긴다 |
| 4 | H4 는 PDL leave 에 더해, 합의 대기 동안 그 사이트 믹서의 down projection 을 L2 로 미리 읽는다(#473 의 "대기 동안 다음 MHC 의 불변 가중치 준비"를 Qwen 의 쪼개진 믹서로). 단일 GPU 부품 프로브의 세 팔(off·pdl·prefetch)로 판정하고, 프리페치는 사이트 시간이 줄 때만 기본으로 둔다 |

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
| P3 | 서빙 프리필이 768 토큰 블록마다 타깃 forward 를 따로 돈다. `served_step` 이 `marks` 를 버려서 청크당 최대 42 forward | `profiles/qwen38/adapter.py`, `base/composed.py` | fix | gpu | 일 | 이 PR: 컴포지션이 `takes_mark` 로 통과 중에 떠낼 경계를 밝히고(`net.Step.marks` → `caches.mark_gdn`·`mark_ple`), `ComposedModel.prefill` 은 거절된 경계에서만 자른다. 32,256 토큰 청크가 타깃 1 forward + MTP 관측 1 회(전에는 각 42). 블록 격자 밖에서 시작한 스텝은 첫 경계에서 한 번만 자르고 나머지 경계를 다 챙긴다. CPU: 배관과 reference 레인의 GDN 상태·탭 대조(`tests/test_engine_qwen38_prefill_marks.py`). 머지 #1183. **GPU 판정 — 남겼던 둘 다 기록이 답했다:** (1) uncut 32,256 토큰 forward 의 워크스페이스 최대 **8.29 / 12 GiB**(플릿 부팅 창 2026-09-18 부팅 2 랭크 0 의 `prefill/32256/0/memory`, `measurements/qwen38_boot_window_20260918` §2), (2) `chunk_kda_with_decay(states_at=)` 의 Qwen3.8 셀은 GB10 단일 레인에서 `KdaDecayKernelTests`(mark 상태를 재귀식 오라클에)·`NativeChunkTests`(mark 상태까지 바이트 동일)가 통과(`qwen38_qsa_folds_20260918`, `qwen38_lane_20260919`). 프리픽스 재사용의 플릿 onepass 는 미실측(D17) |
| P4 | 부팅이 `prepare_dense` 에 `consume_weights` 를 주지 않아 BF16 원본이 랭크당 약 1.9 GB 상주. 프리샤드가 패딩 크기를 예약해야 함 | `profiles/qwen38/fleet.py`, `preshard.py` | fix | cpu | 일 | 머지 #1109(sh_down 원본 랭크당 약 40 MB 는 프리샤드 예약 전까지 남김) |

## 1. 셀 판정 — 단일 GPU 레인 기록으로 `cells.py` 에 `admitted`

| ID | 내용 | 대상 | 종류 | 판정 | 비용 | 상태 |
|---|---|---|---|---|---|---|
| C1 | 셀 판정 프로브 모드(`engine_kernel_check --lanes qwen38_cells`). 단일 레인 티켓은 main 에 있는 프로브만 돌린다 | `probes/engine_qwen38_cells.py` | fix | cpu | 시간 | 머지 #1089; GPU 케이스 7건 중 픽스처 오류 3건은 #1107 에서 링 8칸으로 수정(`measurements/qwen38_lane_20260917`) |
| C2 | dense 패딩 어댑터 GPU 판정 + W4A8/FP8 전환 행 수 실측: Qwen3.8 hidden 2560 · 중간 160, DSv4.1 576 | `kernels/dense`, `cells.py` | measure | gpu | 시간 | 프로브 머지 #1096; 첫 실행 FP8 320×2560 4행 오차 0.22 발견·멈춤, #1107 에서 broken_arms 로 계속, 재실행 예정 |
| C3 | KDA decay 어댑터(ring·chunk·recurrent) GPU 판정 + BV 8/16/32 스윕(4/12×128×128, T=2, 1–4 행, 정확 롤백 게이트) | `kernels/kda`, `cells.KDA_DECAY_MEASURED_CELLS` | measure | gpu | 시간 | 프로브 머지 #1098, 티켓 `qwen38-kda-0917` 대기(8c25c625, K1 의 GDN 진입점) |
| C4 | MoE EP 셀(로컬 128/512, I640, top-10, silu): 오라클 2% + micro 타일·MAC 사다리 + 프리필 `tile_m` 핀 | `kernels/b12x/moe_dispatch.py`, `cells.py` | measure | gpu | 일 | 프로브 머지 #1100; 첫 실행 디코드 통과, 정적 프리필 반복 불일치 발견·멈춤, #1107 에서 진단 추가, 재실행 예정. **플릿에서 확인(2026-09-18 첫 부팅):** 즉시 프리필이 routed pairs < 640 이라 정적 커널을 (행, 전문가) 조합마다 JIT(약 4 s, 랭크 0 이 12 분에 75 개). **PR #1180**: expert-local 즉시 프리필은 dynamic 커널로(`select_sm120_moe_backend`); 창 2 의 checks-only 프로브가 16/64/128/1024/4096 토큰을 오라클 0.5~0.9% 로 통과(디코드 micro 0.5~0.6%, 재생 바이트 동일). 타일·MAC 스윕(`prefill_sweep`/`decode_sweep`)은 레인 티켓 `qwen38-moe-0917` 의 몫; **정적 디코드 10·12 토큰 illegal access → `lanes.static_pad` 16행 패딩, 14~32 통과 (2026-09-18 srv4 창, PR #1192)** |
| C5 | DSv4.1 mHC V41 이음매(`MHCV41`) GPU 판정 | `kernels/dense/mhc.py`, `cells.py` | measure | gpu | 시간 | 열림 |
| C6 | Qwen3.8 자체 레인 GPU `qualify`(게이트 잔차·QSA·GDN). 수치 변경 작업의 기준점 | `profiles/qwen38/lanes.py` | measure | gpu | 시간 | 머지 #1107(GPU qualify 통과, measurements/qwen38_lane_20260917) |
| C7 | `cells.py`: 측정된 어댑터 셀을 `admitted` 로(Q3). `summarize.py` 는 서빙 커널이 PR 이 최적화한 커널과 같을 때만 '그대로' 로 셈. 재집계 | `kernels/cells.py`, `measurements/st_model_dependence_20260917/summarize.py` | fix | cpu | 시간 | 장치는 머지 #1095(측정 튜플 넷은 기록이 붙을 때 채움) |

플릿이 필요해 이번 목록에서 뺀 것: one-shot·프리필 통신의 hidden 2560 실측(4랭크), Qwen3.8 부팅 onepass(D17 속도 기록).

## 2. QSA — 인덱서와 어텐션

| ID | 내용 | GLM 출처 | 대상 | 종류 | 크기 | 판정 | 비용 | 상태 |
|---|---|---|---|---|---|---|---|---|
| Q1 | 죽은 글루 복사 제거: 읽히지 않는 `positions…expand(N,1,3).contiguous()`, `ik.contiguous()`, `first[:,0].contiguous()` 등 | #547 #926 #933 | `net.py:_qsa`, `qsa.py:norm_rope_partial` | fold | −39 발사 | cpu | 시간 | 머지 #1090 |
| Q2 | 캡처 `step_meta` 를 Triton 한 발사로(타깃·드래프트 두 번) | #543 #546 #819 #821 | `net.py:step_meta`, `kernels/step_addresses.py` | fold | 약 −80 발사 | cpu | 시간 | 머지 #1094 |
| Q3 | QSA 입력 융합: q/k norm+rope 와 K/V 저장을 (행, 헤드)당 한 발사로, 압축→norm→rope→인덱스 키 쓰기를 한 발사로, 링 쓰기를 K/V 저장에 합침 | #914 #936 #547 #582 #921 | `qsa.py`, `net.py:_qsa` | fold | 층당 9→2, −91 발사(링을 쓰기 전에 읽어야 해서 두 발사) | cpu | 일 | 머지 #1106 |
| Q4 | 출력 게이트를 어텐션 최종 저장 안에서 적용 | #919 #899 | `qsa.py:qsa_sparse_paged_attention` | fold | −65 발사, 프리필 청크당 −21.6 GiB | gpu | 시간 | 머지 #1108 |
| Q5 | 어텐션이 블록 id 를 타일 루프 안에서 위치로 확장(확장 발사 제거) | #819 #887 | `qsa.py` 선택·어텐션 | fold | −13 발사, 프리필 청크당 −8 GiB | cpu | 시간 | 머지 #1103 |
| Q6 | 확장이 블록 id 를 오름차순 정렬: 어떤 선택기가 돌았든 합산 순서가 같게(지금은 디코드와 프리필이 다르게 반올림할 수 있음) | #546 #693 | `qsa.py:_expand_qsa_indices_kernel` | kernel | 0 | gpu | 일 | 이 PR |
| Q7 | 디코드 블록 선택을 `st_dsa_select` 로(GLM `decode_topk.cu` 그대로, 열 수 컷오프) | #1010 #926 | `qsa.py:select_blocks` | native | 약 −130 발사 | gpu·glm | 일 | 이 PR(Triton 형태; `st_dsa_select` 의 CUDA 가 아님): `kernels/qsa_select.select` — 캡처 스텝의 블록 선택이 행마다 한 프로그램, 한 발사. k 번째 값은 정렬이 아니라 float 비트를 순서 보존 int32 키로 바꿔 32 번의 세기로 찾고, 동률은 낮은 블록부터(`prefill_topk`·`st_dsa_select` 와 같은 규칙 — 디코드와 프리필이 같은 점수에서 같은 집합을 고른다; torch.topk 는 동률 순서 미정), id 는 prefix sum 으로 오름차순(Q6 이 어차피 정렬하는 순서), 나머지는 -1. `select_blocks` 의 torch 형태(arange·비교·not·masked_fill·topk·비교·캐스트·where·복사 — QSA 층당 12 발사 이상, 스텝 13 층)를 대체. 32,768 열(131K 토큰)까지만 받고 그 위 버킷은 torch 형태 그대로(한 프로그램의 스레드 한계). 판정: CPU 인터프리터 + RTX 5050(sm_120) — 규칙(안정 정렬의 앞 k 개) 대조, 동률 없는 점수에서 torch.topk 집합과 동일. 같은 카드의 그래프 재생 sanity(2 행, k 512): 1K 열 30→8 µs, 4K 36→13, 16K 65→20, 32K 53→38, 65K 는 59→231 이라 제외. 버킷당 컴파일 0.2–0.5 s. `tests/test_engine_qwen38_qsa_select.py`. **GB10 판정(2026-09-19, srv4 단일 GPU 레인, 티켓 `qwen38-cells-0919b`): 통과** — `SelectTests` 6 건 포함 GPU 케이스 57 건 전부(`measurements/qwen38_lane_20260919`). `WIDEST` 의 교차점은 Q9 의 레인 실측. 머지 #1202 |
| Q8 | 점수 커널이 요청의 키 타일을 최대 4 쿼리 행에 한 번만 읽기 | #971 #1010 | `qsa.py:_qsa_mqa_paged_kernel` | kernel | 128K 에서 스텝당 −107 MB | gpu | 일 | 머지 #1193: `_qsa_mqa_paged_group_kernel` — 한 요청의 연속 행(최대 4)을 한 프로그램이 맡아 키 타일을 한 번만 읽는다. 행별 내적은 행 커널과 같은 모양이라 logits 가 **바이트 동일**(표의 `kernel` 이 아니라 사실상 fold). 캡처 스텝은 행당 토큰 수의 4 이하 최대 약수(K=1 → 2, K=3 → 4), 한 세그먼트 호스트 스텝은 4, 세그먼트 여럿은 1. 판정: WSL `TRITON_INTERPRET=1` 95/95, RTX 5050(sm_120) 실 CUDA 34/34, **GB10 단일 레인 통과**(`measurements/qwen38_qsa_folds_20260918`). 크기(소스 계수): 검증 스텝 점수의 인덱스 키 읽기 K=1 에서 1/2(128K 에서 −107 MB), K=3 에서 1/4, 프리필 청크 1/4 |
| Q9 | QSA 커널 GB10 발사 기하 스윕: norm_rope·압축·저장·확장·점수, 어텐션 split 프로필(지금은 상류 GB300 프로필로 디코드마다 64 split) | #554 #556 #641 #658 #737 | `qsa.py` | measure | split 1 이면 −13 발사 | gpu | 일 | PR #1221(도구 #1208): GB10 단일 레인 4 회. **split 프로필을 GB10 의 표로** — 16 폭 타일·4 warps, split 은 프로그램 수로 ≤2 → 64, ≤8 → 16, ≤256 → 4, 그 위 → 1(merge 없음): eager sparse 4,096 행 −26%, covered 2,048 행 −52%, 캡처 N=4·8·16 은 cold −9·−3·−3%, C=1(N=1·2)은 64 split 그대로. **프리필 점수** 128×32×4 로 −6~−16%(점수 바이트 513 팔 전부 동일). 그대로: `qsa_select.WIDEST`·warps 규칙(GB10 에서도 맞음), 입력 발사 4 warps, 디코드 점수. 게이트: 1,926 팔 오라클 밴드 안, covered = sparse 바이트(프로필 공유 유지). norm_rope·압축·저장·확장은 Q1–Q5 가 접어 더는 발사되지 않아 팔이 없다. 발사 하나의 시간이며 엔진 속도는 미실측. `measurements/qwen38_qsa_geometry_20260919` |
| Q10 | 덮인 앞부분(≤2,050 위치)의 dense causal 프리필 커널 | #887 #889 | `qsa.py` 새 커널 | kernel | 2K 이하 프롬프트 K/V 읽기 약 −50% | gpu | 일 | PR #1196: `_qsa_covered_paged_gqa_kernel` — 예산이 덮는 호스트 스텝(`Qwen38Net._covers`, 2,051 위치)은 블록 id 없이 dense causal 발사 하나로 어텐드한다. 타일의 위치는 열 번호 그대로라 512 id 로드·정렬·확장이 없고, 한 요청의 연속 행 최대 4 개가 K/V 타일을 한 번 읽어 나누며, 묶음에서 가장 먼 행에서 멈춘다(예산의 33 타일 대신 ceil(프롬프트/64)). 행별 softmax 는 sparse 커널의 연산을 sparse 발사의 타일·split 그대로(`_split_profile` 공유) 밟아 출력이 **바이트 동일**(표의 `kernel` 이 아니라 사실상 fold; K/V 읽기는 표의 −50% 가 아니라 1/4). 판정: sparse 발사 대비 `torch.equal` + raw bits — WSL `TRITON_INTERPRET=1` 106/106, RTX 5050(sm_120) 실 CUDA 모델 실폭 43/43; **GB10 단일 레인 통과**(`measurements/qwen38_qsa_folds_20260918`). 캡처 스텝은 대상이 아니다(컨텍스트가 디바이스 값) |
| Q11 | 프리필 인덱스 쿼리 행을 랭크별로 나눠 점수(`QueryShard`) | #881 | `net.py:_qsa` | fold | 긴 프롬프트 점수 행 −75% | cpu(LocalTP) | 일 | 머지 #1185(덮인 절반: 가장 긴 세그먼트가 2,051 위치 안에서 끝나는 호스트 스텝은 점수·top-k 없이 선택) · 머지 #1188(랭크 절반: 한 세그먼트 eager 프리필의 인덱스 쿼리를 랭크가 1/4 씩 점수하고 id 를 층당 한 번 all-gather; 분할은 양쪽의 모든 점수 호출이 radix 선택기를 타는 스텝에서만 — `prefill_topk.admits_calls` · `qsa.shards_select_alike` · `lanes.qsa_select_alike`). **운영자 결정 2026-09-18: 기본 켬, 플릿 미실측**(PR #1190) — 롤백은 `fleet.py --no-query-shards`, 런처 `ST_QUERY_SHARDS=0`. cpu 판정: WSL `TRITON_INTERPRET=1` 에서 서빙 커널 케이스 포함 23/23, RTX 5050(sm_120) 실 CUDA 23/23(radix 빌드는 GB10 전용이라 거기서는 64 행 이하 스텝). **GB10 단일 레인 통과**(`measurements/qwen38_qsa_folds_20260918`): 양쪽이 네이티브 radix select 를 타는 넓은 스텝에서 분할한 선택은 모든 행에서 분할하지 않은 선택과 같은 집합이다(radix select 가 행 안에 남기는 순서는 발사마다 다르다 — 어텐션은 정렬해서 읽는다). 남은 것: 플릿 onepass(D17), 분할 없는 부분 덮임 |
| Q12 | QSA 어텐션을 메가커널로(`mla/glue.gqa`, FP8 KV) | 메가커널 계열 | `lanes.py`, `caches.py` | measure | KV 바이트 ½ | gpu | 일 | 열림 |
| Q13 | K/V 를 한 영역의 레코드로 두는 캐시 배치 | #641 | `caches.py` | measure | 0 발사 | gpu | 시간 | 기각 — PR #1221: 같은 K/V 값을 오늘의 블록 배치와 레코드 배치로 두고 sparse 어텐션을 GB10 에서 비교 — 같은 발사·같은 바이트, 캡처 N=2·8·32 에서 ±1%, eager 4,096 행 +1%. `caches.layout` 은 그대로. `measurements/qwen38_qsa_geometry_20260919` §6 |

## 3. 게이트 잔차 — mHC 계열 아이디어

| ID | 내용 | GLM 출처 | 대상 | 종류 | 크기 | 판정 | 비용 | 상태 |
|---|---|---|---|---|---|---|---|---|
| H1 | down+inject GEMM 을 `leave_norm` 안으로 접기(디코드 ≤16 행) | MK_SEG_MHC | `kernels/gated_residual.py` | kernel | 사이트당 −1, −100 발사 | gpu | 일 | 기각 제안 — H2 와 같은 구조(down+inject GEMM 15.0 µs 도 대역폭 한계치, 접으면 Triton matvec 이 행마다 다시 읽음). H2 의 실측과 프로브 참조 GB10 주: 이 행의 근거(Triton GEMV 가 cuBLAS 에 진다)는 RTX 5050 의 것 — GB10 에서는 skinny GEMV 가 down+inject 를 cuBLAS 보다 1.11 배 빨리 읽는다(q38gemv-0919c). 남은 장벽은 GEMV 가 아니라 `leave_norm` 의 행 전체 norm 이 곱 앞에 있어야 한다는 격자 의존이다. H2 는 게이트를 down 발사에 태워 닫혔다(아래) |
| H2 | gates + up GEMM + `mix_mean` 한 발사 | MK_SEG_MHC | `gated_residual.py:mix` | kernel | 사이트당 −2, −200 발사 | gpu | 일 | 머지 #1207: `gated_residual.mix_rows` — 디코드 1..16 행에서 게이트를 down(+inject) 곱의 저장에, `mix_mean` 을 up 곱의 저장에 접어 사이트 5 → 3 발사(−2, H2 의 크기; 게이트는 up 대신 down 발사에 탄다). 두 곱은 `kernels/common/skinny_gemv`(행 16 패딩 tensor-core dot, split-K 는 마지막 도착 프로그램이 split 순서로 합산). 같은 곱의 네 발사 산술과 **바이트 동일**, qualify 통과(max 0.0063). **GB10 실측** (q38site-0919a, 16 사이트 한 그래프·가중치 회전): 믹서 cuBLAS 66.5~70.5 µs(+복사 1) → 60.3~62.7 µs(1·4·8·16 행). #1197·#1201 의 RTX 5050 판정(Triton 접기 기각)은 GB10 에서 반대다. 발사 접기만의 몫은 사이트당 2~3 µs(#1215 의 headroom, 1~8 행) — 나머지는 skinny GEMV 가 cuBLAS 보다 빨리 읽은 몫. 라우터도 같은 GEMV(4~16 행 1.96~2.53 배). 스텝 A/B 약 −1.3 ms(C=1, K=3, 단일 GPU; measurements/qwen38_decode_gemv_20260919) |
| H3 | hidden 폭 커널을 512 폭 5 타일로 재배치 | #634 | `gated_residual.py` | measure | 0 발사 | cpu·gpu | 시간 | PR #1233: GB10 레인(`qwen38-mix-tiles-0919c`, M5 창 안)에서 `_mix_mean` 의 타일 축(#1224, 바이트 동일)을 스윕 — 캡처 4·17·32 행에서 256 폭 × 4 warps 가 한 블록(4,096×8) 대비 −51·−41·−30%(4.63→2.28, 4.79→2.83, 4.93→3.44 µs/사이트), 64 행부터는 대역폭 한계라 모든 기하가 ±4%. 규칙 `_mix_tile(hid, rows)`: 32 행까지 256×4, 그 위는 한 블록 그대로. `leave_norm`·`norm_streams` 는 채널 방향 축약(norm)이라 타일링하면 반올림 순서가 바뀌어 대상 밖. 엔진 속도는 미실측(32 행 캡처 스텝당 약 −0.14 ms 추정). `measurements/qwen38_mix_tiles_20260919` |
| H4 | `leave` 를 one-shot consumer 의 PDL 종속으로 | MK AR consumer, #689 | `gated_residual.py`, `lanes.py` | measure | 약 0.4 ms/스텝 추정 | gpu | 일 | 열림 |
| H5 | `leave` 가 TP4 랭크 패킷을 직접 합산(생산자 TX 슬롯과 함께) | #812 #826 | `gated_residual.py`, `kernels/oneshot` | fold | 스텝당 약 1 MB | cpu | 일 | 열림 |
| H6 | `--hc-fp8` 레인 실측(믹서 가중치 읽기가 스텝당 1.32 GB) | — | `net.py:_prepare_hc_fp8` | measure | 바이트 −40%, 발사 +300 | gpu | 시간 | 기각(디코드) — 2026-09-19 두 번째 운영자 창(main `4148c35f`, 런처 기본 K=1·one-shot·`one`): `ST_HC_FP8=1` 이 기본 대비 C=1 디코드 28.35 → 31.08 ms/스텝(**+9.6%**, 다섯 요청 모두 +5.8~+15.3%; 30 ms 이하 스텝 비율 80–93% → 11–67%), C=4 45.25 → 47.29(+4.5%, 두 바퀴 −1.5·+10.1% — 부팅 간 잡음 안). 수용(토큰/스텝)은 같은 분포. 부팅 ready 61.2 → 98.0 s(warm prefill +16 s, capture decode +21 s), prepare dense +0.75 GiB. 한 쌍·순서 one→fp8. 믹서 바이트 절반(산수로 약 −2.4 ms)을 발사 증가와 H2 접기의 이탈(FP8 투영이 두 BF16 곱을 대신하니 접기가 비킨다)이 넘는다. 품질은 판정 안 함(느리므로 불요). 프리필(4,096 행, 믹서 사이트가 청크 디바이스 시간의 43% — f8 census #1245)은 미실측 — 연다면 따로. `measurements/qwen38_s2h6_window_20260919` §2 |
| H7 | 작은 접기: 임베딩 `repeat`, PLE 층의 분리된 leave 와 out-of-place 덧셈 | — | `net.py` | fold | −3 발사 | cpu | 시간 | 기각: 임베딩 `repeat` 는 all-reduce 뒤라 접을 자리가 없고, PLE 층 둘은 게이트 잔차 커널에 변형을 하나 더 들여야 해서 스텝당 2 발사의 값이 없다 |

## 4. MTP 드래프터

| ID | 내용 | GLM 출처 | 대상 | 종류 | 크기 | 판정 | 비용 | 상태 |
|---|---|---|---|---|---|---|---|---|
| D1 | MTP 체인 K>1 을 드래프트 재생 안에서(`fleet --spec-k K`, `decode_graphs.draft_chain`). 두 행 C=1 실측 K=3: 450 토큰 2.43 토큰/스텝·44.3 ms/스텝·46.2 tok/s(K=1 1.69·34.7·39.5). 4 행은 (3 행 × 4) 12 토큰의 정적 MoE 커널이 캡처에서 죽어 격리 전(C4 에 디코드 12/16 검사) | #869 #627 | `decode_graphs.py:DraftGraphs`, `adapter.py:ServedMTP`, `fleet.py` | native | 스텝당 토큰 | gpu | 일 | PR #1182 (2 행 실측; 4 행 열림) |
| D2 | 검증 pick 을 전 vocab gather 없이: argmax 키 + max all-reduce, 캡처 샘플링 | #563 #929 | `decode_graphs.py:TargetGraphs`, `adapter.py` | fold | −1 all_gather(993 KB), −1 H2D | gpu | 일 | 열림 |
| D3 | 드래프트를 GPU 에 남김: 스텝당 host sync 제거, streams 인계를 복사 한 번으로 | #605 #627 | `adapter.py`, `decode_graphs.py` | fold | −1 sync, −2 복사 | gpu | 일 | 이 PR(호스트 쪽 절반): 검증 스텝의 PLE 스테이징이 디바이스를 되읽지 않는다 — `TargetGraphs.run` 은 호스트가 방금 올린 ids 를 `ids.tolist()` 로, `net.stage_ple` 은 각 행의 직전 ngram_size−1 토큰을 id 링에서 `.cpu()` 로(작은 발사 6–8 개와 함께) 읽고 있었다. 링이 그 위치에 든 것은 거기에 먹인 토큰, 곧 행의 이력이라 `adapter.ServedModel` 이 둘 다 쥐고 있다: `forward(host=(ids, carried))` → `TargetGraphs.run(known=)` → `stage_ple(carried=)`. 호스트 사본이 없는 호출자(기본 `ComposedModel.decode`)는 전처럼 디바이스에서 읽는다. RTX 5050 + x86 호스트에서 스텝당 스테이징 370 → 72 µs(그동안 GPU 는 타깃 재생 앞에서 놀았다; GB10 의 Grace 코어에서는 미실측). `tests/test_engine_qwen38_decode_host.py`. **표의 원래 형태(드래프트를 GPU 에 남김)는 막혔다:** PLE 표를 SSD 로 내린 운영자 결정 (2026-09-18) 뒤로 호스트가 검증 스텝의 ids 를 알아야 표 행을 읽는다 — 드래프트 재생의 `.tolist()` 는 남는다. streams 인계의 복사 줄이기는 열림 |
| D4 | MTP 프롬프트 관측을 한 forward 로 묶기(P3 이 먼저 닫히면 불필요) | #627 #722 | `adapter.py:ServedMTP.observe` | fold | 청크당 최대 −41 헤드 forward | gpu | 일 | P3 의 PR 로 닫힘: 청크가 한 조각이라 `ServedMTP.observe` 도 청크당 한 번이다 |
| D5 | argmax partials/finish 를 한 warp 로 | #1004 | `kernels/common/vocab_candidates.py` | measure | 0 발사 | cpu·glm | 시간 | 기각 — PR #1213: GB10 실측(레인 티켓 `vocab-argmax-warps-0919`)에서 두 발사 모두 4 warps(지금 기본값)가 가장 빠르거나 동률, 1 warp 는 +7~+75%(Qwen3.8 샤드 2 행 3.34→5.51 µs), 2 warps 는 ±. #1004 의 후보 선택은 MAX 축약 16 연발이라 warp 간 교환이 지배했고, greedy 는 1,024 열의 MAX 한 번이라 처방이 반대로 나온다. 패킷은 어느 warps 에서도 바이트 동일(정수 키). 남긴 것: `ARGMAX_WARPS = 4` 선언, 프로브 팔 `argmax_run`(레인 `vocab_argmax`), 커널 경로의 첫 직접 테스트 `tests/test_engine_vocab_argmax_kernel.py`. `measurements/vocab_argmax_warps_20260919` |
| D6 | MTP dense 투영을 FP8 로 디코드하는 수용률 팔 | #862 #863 #871 | `net.py:prepare_dense` | measure | 수용률 | gpu | 시간 | 열림 |

## 5. 공유 커널 — Qwen3.8 셀이 닿게

| ID | 내용 | GLM 출처 | 대상 | 종류 | 크기 | 판정 | 비용 | 상태 |
|---|---|---|---|---|---|---|---|---|
| K1 | GDN 게이트를 링 커널 안에서 계산(in_proj 조각을 stride 로 읽음) | #569 #571 | `kernels/kda/ring.py`, `fused_recurrent.py`, `net.py:_gdn_rows` | native | −36 발사 | cpu·gpu | 시간 | 머지 #1101 |
| K2 | `gated_norm` 이 z 를 stride 로 읽음 | #569 | `kernels/gdn.py` | fold | −36 발사·복사 | cpu | 시간 | 머지 #1091 |
| K3 | chunk 파이프라인이 헤드별 decay 를 네이티브로(widen·repeat_interleave 제거) | #615 #811 | `kernels/kda/chunk_decay.py`, `kda.py` | native | 프리필 청크당 약 −20 GiB 쓰기 | cpu·glm | 일 | 이 PR: 파이프라인의 다섯 커널(K K^T 둘·`recompute_w_u`·`chunk_gla_fwd_kernel_o`·상태 재귀)이 `QG`(q·k 는 H/QG 헤드, 값 헤드 i 는 키 헤드 i // QG)와 `G_HEAD`(헤드별 게이트 [T,H] 를 행마다 값 하나로 읽어 키 채널로 브로드캐스트)를 받고, `chunk_kda_with_decay` 는 넓힌 decay(층당 189 MiB fp32)와 반복한 q·k(2 × 94.5 MiB)를 만들지 않는다 — 32,256 토큰 청크의 GDN 36 층에 약 13 GiB 쓰기(표의 20 GiB 는 과대 추정). **stride 0 블록 포인터로 읽으면 안 된다:** Triton 의 coalesce 패스가 포인터 연속성으로 텐서의 스레드 레이아웃을 골라서 K 축 합의 리덕션 트리가 바뀌고 A·Aqk 의 9% 가 FP32 마지막 비트에서 달라졌다(1-D 로드 + 브로드캐스트로 해결). 결과는 넓힌 형태와 **바이트 동일**(출력·최종 상태·mark 상태; `widen=True` 가 비교용으로 예전 형태를 만든다), `QG` 1·`G_HEAD` 거짓이면 모든 오프셋이 예전 그대로(KDA 경로). 판정: CPU 인터프리터 + RTX 5050(sm_120, triton 3.6) 바이트 동일, 같은 카드의 sanity 로 T=8192 피크 281→185 MiB·4.3→2.7 ms(행마다 exp2 한 번). `tests/test_engine_gdn_chunk_native.py`, `SOURCES.json` 핀 갱신. **GB10 판정(2026-09-18, srv4 단일 GPU 레인, 티켓 `qwen38-qsa-folds3-0918`, 다른 세션의 실행): 통과** — `measurements/qwen38_qsa_folds_20260918/cells-9d784bc7.log`, GPU 케이스 51 건 전부(`NativeChunkTests`: 네이티브 형태가 넓힌 형태와 sm_121a 에서도 바이트 동일, KDA 형태는 재귀식 대조 통과; 같은 실행에서 `KdaDecayKernelTests` 도 통과). 머지 #1194. **GLM 실가중치 단일 GPU 검사는 미판정**(운영자 결정 2026-09-17: 플릿 미실측) |
| K4 | strided q/k l2norm 을 4 헤드에서도 admit | #811 | `kda.py:_glm53_qk_l2norm_strided` | fold | 프리필 층당 −3 발사 | cpu·gpu | 시간 | 머지 #1105 |
| K5 | GDN norm 이 out_proj 의 W4 입력 팩을 씀(S2 뒤) | #968 #978 | `gdn.py`, dense | kernel | −36 발사 | gpu·glm | 일 | 열림 — S2 기각으로 전제가 바뀜: o_proj 2560×1536 의 입력 재사용은 따로 도는 pack 발사 때문에 GB10 에서 +4.5~+13.6% 였다. K5 는 그 pack 을 norm 이 쓰게 해 발사를 없애는 것이므로, o_proj admit 을 스스로 다시 넣고 `mk_gemm_input_kernel` 단독 대 ordinary 로 판정해야 한다 |
| M1 | MoE 출력 finalizer 한 발사: BF16(routed + shared·gate), sigmoid 는 torch 에 둠(b12x FP32 평면 직접 소비는 다음 목록) | #904 #906 | `kernels/moe_output.py:gated_sum`, `lanes.py`, `net.py:_moe` | fold | −196 발사 | cpu·gpu | 시간 | 머지 #1092 |
| M2 | 라우팅(softmax top-10, 재정규화, BF16 반올림, EP 리맵) 한 발사 | #789 #810 #779 | `lanes.py:route_softmax_topk` | kernel | 약 −700 발사 | gpu | 일 | 이 PR: `kernels/moe_route.softmax_topk` — 캡처 스텝의 라우터와 EP 리맵(`local_routes`, 발사 형상의 sentinel 포함)이 점수 행 하나에 한 발사(`Lanes.route_local`). 즉시(compact) 스텝과 reference 레인은 그대로 조합. 동률은 가장 낮은 expert id 부터(torch.topk 는 순서 미정). **shared 게이트의 sigmoid 는 접지 않음**: Triton 의 exp 가 torch 의 것이 아니라 2^20 값 중 32% 가 torch.sigmoid 와 다르고(최대 7.9e-7), 게이트는 FP32 로 소비돼 반올림이 차이를 흡수하지 않는다 — M1 이 sigmoid 를 torch 에 둔 것과 같은 이유(−98 발사 포기). 실측은 RTX 5050(sm_120, triton 3.6·torch 2.11): 98,304 행에서 expert 집합 100% 동일, k 안의 동률 순서는 14% 행에서 다름, 가중치 99.9996% 바이트 동일·나머지 BF16 한 단계. CPU 인터프리터 판정(`tests/test_engine_qwen38_moe_route.py`), GPU 케이스는 `probes/engine_qwen38_cells` 목록. **GB10 판정(2026-09-18, srv4 단일 GPU 레인, 티켓 `qwen38-qsa-folds3-0918`, 다른 세션의 실행): 통과** — `measurements/qwen38_qsa_folds_20260918/cells-9d784bc7.log`, GPU 케이스 51 건 전부(`SoftmaxTopkTests` 8 · `LayerTests` 2: 서빙 가중치가 `exact.to(bf16)` 와 바이트 동일, 동률 없는 점수에서 torch.topk 와 같은 expert·순서). 머지 #1189; #1203 이 `route_local` 경로의 정적 커널 패딩을 더함 |
| M3 | micro 레인의 EP 추가 경로(direct FP32 FC2 scatter, shared FC1 A, M16 타일)를 E128/H2560/I640/silu 로 | #955 #920 #974 | `b12x/moe_dispatch.py`, `moe_micro_kernel.py` | kernel | FC1 입력 로드 ½ | gpu·glm | 일 | 열림 — 이식 검토(2026-09-19): 옮길 변형이 GLM 의 EP 셀에 레지스터 수준으로 묶여 있다 — `moe_dispatch._ep_micro_direct_scatter` 는 `(num_topk, max_rows, skip id) == (8, 64, 72)`·`swigluoai_uninterleave`·타일 (16|32, 128) 만 받고, `moe_micro_kernel._validate_ep_direct_scatter_layout` 는 스레드당 scatter 쌍 16 개의 좌표를 컴파일 시점에 단언한다. Qwen3.8 셀(top-10·로컬 128·I640·silu)은 어느 것도 맞지 않아, 이식이 아니라 top-10/silu 변형을 새로 짓는 일이다(cutlass 가 있는 GB10 에서만 컴파일) |
| M4 | EP-local dynamic 프리필 커널(층마다의 host sync `nonzero` 제거) | #895 #811 | `b12x/moe_dynamic_ep_local.py`, `lanes.py:moe` | kernel | 프리필 층당 −6 발사, −2.3 GiB | gpu | 일 | 열림 — 이식 검토(2026-09-19): `moe_dynamic_ep_local.py`(1,159 줄)는 stock gated 커널(I ≤ 512)의 서브클래스로 E72·I2048·warp 당 ≤8 로컬 라우트 캐시에 묶여 있고 스스로 "experimental; no GPU correctness claim" 이다. Qwen3.8 은 I640 이라 generic 구현을 타서 gated 변형부터 없다. 층마다의 `nonzero` host sync(`lanes.py` 의 eager `moe`)를 없앤다는 목적은 유효 — 재설계 항목 |
| M5 | shared expert 를 overlap 스트림에서 | #789 | `dense/shared_mlp.py`, `net.py:_moe` | measure | 층당 약 5 발사 겹침 | gpu | 시간 | PR #1233: 플릿 A/B(운영자 "플릿 잡고 해", main 048270ef, 런처 기본 K=1·one-shot, 프로덕션 다운 572 s) — `--shared-overlap all` 이 C=1 디코드 스텝 29.01→27.55 ms(−5.0%, 다섯 요청 모두 0.93–0.975; 30–40 ms 스텝 비율 17–41% → 2–10%), C=4(8 행) 43.35→45.93 ms(+6.0%). 한 쌍·순서 off→all 이라 순서 효과가 섞임(마지막 반복 두 요청만 −3.5·−2.5%). 포크는 한 요청의 행(`one`)에서만 이득 쪽. **운영자 결정 2026-09-19("켜"): 기본 `one`** — PR #1234(`fleet --shared-overlap` 기본 one, 롤백 `ST_SHARED_OVERLAP=off`, 부팅 로그에 `shared expert:` 한 줄); 순서를 바꾼 확인 쌍(one→off)은 다음 창. 2026-09-19 두 번째 창의 쌍은 **무효** — `one` 부팅이 레인 프로브 옆에서 거부돼 순서가 off→one 으로 바뀌었고, off 의 요청 내내 다른 세션의 레인 티켓(`q38win-0919a`)이 srv4 GPU 에서 돌았다(off 가 C=1 +12.6%·C=4 +23% 느리게 나온 까닭). 그 창의 깨끗한 `one` 값: C=1 28.35 ms/스텝, C=4 45.25. 확인 쌍은 여전히 다음 창. `measurements/qwen38_shared_overlap_20260919`, `measurements/qwen38_s2h6_window_20260919` |
| S1 | swiglu 가 0 패딩된 sh_down 입력을 직접 씀 | #569 #973 | `kernels/common/swiglu.py`, `PaddedDenseLinear` | fold | −98 발사 | cpu·gpu | 시간 | 머지 #1097 |
| S2 | W4 입력 재사용을 m 2–8, 4120/4224×2560, 2560×1536 에서 admit + hidden 2560 ksr 스윕 | #1050 #888 #946 #939 #969 | `kernels/dense/kernels.cu` | measure | GEMM 98 개의 입력 양자화 대폭 감소 | gpu·glm | 시간 | 기각 — GB10 레인(`qwen38-input-reuse-0919b`, 2026-09-19 두 번째 운영자 창 안, main `4148c35f`)에서 #1241 의 admit 을 쟀다: `InputReuseTests` 3 개 통과, 12 셀(형상 3 × 행 2·4·6·8) 전부 ordinary 와 **바이트 동일**(eager·재생). 시간(콜드 중앙값, 가중치 48 MiB 회전): GDN in_proj 4120×2560 −0.2~−1.8%, QSA in_proj 4224×2560 +0.8~+1.9% — 둘 다 ±2% 안(37 µs 발사) — o_proj 2560×1536 **+4.5~+13.6%**(14.8 → 16.8 µs; 따로 도는 pack 발사가 ksr 3 의 짧은 GEMM 에 얹힌다). K=4096 인 GLM 에서는 타일마다의 입력 양자화가 비싸 이기지만 Qwen3.8 의 K 2560·1536 에서는 pack 값을 못 번다. admit 을 되돌림(PR 이 PR #1241 을 revert — `kernels.cu` 는 #1241 앞과 같다). hidden 2560 ksr 스윕은 미실측으로 남김(ksr 을 바꾸면 합산 순서가 바뀌어 바이트 판정이 아니라 품질 판정이다). `measurements/qwen38_s2h6_window_20260919` §3 |
| S3 | 부팅 자기 보정(GPTQ W4/FP8·헤드) 배선. 지금 Qwen3.8 팩은 전부 round-to-nearest | #650 #659 #661 #673 #779 | `profiles/qwen38/fleet.py` | kernel | 품질 | gpu | 일 | 열림 |
| X1 | 작은 합에 compact 12-CTA one-shot consumer | #967 #944 #957 | `kernels/oneshot` | measure | 합 101 개가 48→12 CTA | gpu·glm | 시간 | 기각(측정 없이, 운영자 결정 2026-09-19) — 전제 오류: ST 엔진은 compact 12-CTA 를 한 번도 컴파일하지 않았다(`build()` 가 `OSAR_COMPACT_CTA` 를 정의하지 않는다; vLLM overlay 의 선택 옵션이었고 기본 0 — #967 의 기록). #967 의 이득(16행 합 −9~−10%)은 48-CTA consumer 에서 데이터 없는 CTA 가 표만 내고 대기 없이 빠지는 몫이고, Qwen3.8 의 합은 이미 그 경로다(4행 × 2560 = 1,280 벡터, 48 CTA 중 5 개가 데이터). compact 가 더 줄이는 것은 그 빈 CTA 36 개의 발사뿐(추정 합당 1 µs 미만)인데, `OSAR_COMPACT_CTA=1` 은 빌드 전체의 발행 판정을 바꾸고(모든 호출이 wrap-safe 표) GLM C=1 의 합(8행 × 4096 = 32,768, compact 상한)도 compact 로 보내 GLM 판정까지 부른다. 플릿 분해(결정 1)에서 발행이 CTA 스케줄링에 밀리는 게 보이면 다음 목록에서 다시 연다 |
| X2 | 생산자·MoE finalizer 가 TX 슬롯에 직접 씀(hidden 을 형상에서) | #826 #904 #906 | `kernels/oneshot`, dense | kernel | 발사 중립, 스텝당 −98 복사 | gpu·glm | 일 | 열림 |

## 진행 순서

1. **선행:** P1, P2. `cpu` 판정의 전제다.
2. **셀 판정:** C1 을 머지 → C2–C6 단일 레인 티켓 → C7.
3. **CPU 로 닫히는 접기:** Q1, Q2, M1, K2, S1, K1, H7, Q5, Q3, K4.
4. **GPU 로 판정하는 커널:** M2, H2, H1, Q4, Q6→Q7, D1, D2, D3, P3, Q8, K3, M3, M4, K5.
5. **스윕과 메모리:** Q9, H3, S2, X1, H6, M5, D5, D6, Q13, P4.
6. **크고 위험한 것:** Q10, Q11, Q12, H4, H5, X2, S3, D4.

## 운영자 지시 (2026-09-18) — 목록 밖의 결정

| 지시 | 내용 | 상태 |
|---|---|---|
| "그냥 mtp로 해" | 드래프터는 체크포인트의 MTP 헤드(`SPEC_K=1`)로 간다. PixelML 의 Flash-Next DFlash 드래프터(srv2 `~/models/qwen38-flash-next-dflash-pixelml`; 자체 측정에서 수학만 네이티브 MTP 를 이기고 채팅은 느림)는 붙이지 않는다 | 결정 |
| "e뭐시기 그건 ssd로 내리고" | PLE 표(랭크당 11.92 GiB)를 아레나에서 빼 `ple-r{r}of4.weight` 로 랭크 파일 옆에 두고 행을 번호로 읽는다(`profiles/qwen38/ple_table.py`; 레이아웃 `st-qwen38-tep4-modelopt-v3`) | PR |
| "이미지는 파트로 사전 샤딩해서" | NVIDIA 허브 체크포인트(fc694b54)를 `preshard.py` 로 네 랭크 파일 + 네 표 파일로 자른다(MTP 전문가 FP8→NVFP4, 전문가 그룹은 랭크별, memmap 읽기) | 산출물 완성(srv4, 2026-09-18 15:38, 699 s, 3 GiB 캡): `~/models/st-qwen38-tep4`, 랭크 0 표 파일 sha = 09-11 파일과 동일; 랭크 r 을 GLM 과 같은 노드로 배포 |
| "올려봐" | 세션 창(quiet gate 양보)에서 `launchers/start-st-qwen38.sh` 로 첫 플릿 부팅 | 창 1: 네 랭크 부팅·문 열림(107 s, 캐시 뒤 40 s), 토큰 0 — 즉시 프리필의 정적 MoE 커널이 (행, 전문가) 조합마다 JIT(12 분에 75 개), one-shot 은 그 사이 STALL. |
| "두 문제 해결해 / 폴백경로 말고 자체 경로 / 부팅도 한번 해보지그래" | 프리필은 b12x dynamic 커널(행 수 무관 아티팩트)로 보내는 규칙 하나; one-shot 은 기본 그대로 | 창 2(PR #1180): C4 프로브 오라클 안(프리필 0.5~0.9%, 디코드 0.5~0.6%), 플릿 53.5 s ready, STALL 0, 한글·영어 답 정상, 450 토큰 39.4 tok/s, 디코드 35.3 ms/스텝·1.74 토큰/스텝(수용 72%). `measurements/qwen38_fleet_boot_20260918` |
| "k=3 정도 하지" / "k=3가 안되면 k=2까지만해" | `fleet --spec-k K`(launcher `ST_SPEC_K`): 드래프트 재생 안에서 헤드 K 번(`decode_graphs.draft_chain`), 검증 스텝 K+1 토큰; 고정 링 검사 K ≤ 4 | 창 3(PR #1182): 4 행 부팅은 12 토큰 정적 MoE 커널(`static_m12`)의 첫 런치에서 illegal access 로 캡처 실패, 2 행 부팅 OK. 두 행 C=1 짝: 450 토큰 K=3 2.43 토큰/스텝·44.3 ms/스텝·디코드 54.9·프리필 포함 46.2 tok/s vs K=1 1.69·34.7·48.8·39.5 (+12.5% / +17%). K=2 는 3 행에서 9 토큰으로 같은 정적 경로라 폴백이 아니다 |
| "8행으로 가자" | 8행 부팅이 밟는 정적 MoE 디코드 모양(10~32 토큰)을 C4 프로브로 판정, 죽는 모양은 16행 패딩(`lanes.static_pad`), 레인 승인·`fleet_prepare` 수정 | srv4 세션 창(PR #1192): 정적 10·12 illegal access(네 행 K=3 부팅의 그것), 14~32 통과; 패딩 뒤 10·12·14 통과. 8행 K=3 플릿 부팅·C=1~8 측정은 다음 창 |

## 옮기지 않는 GLM 최적화

| 계열 | 항목 | 이유 |
|---|---|---|
| mHC | #1009, #972, #689 TileLang, #634 TMA 자체, #860 | CTA 티켓·FP32 계수·TileLang 패스가 Triton 게이트 잔차에 없음. 믹서 가중치는 이미 BF16 이고, MTP 는 스트림 전체를 읽음 |
| 인덱서 | #907, #961, #821 헤드 게이트, #556 Hadamard/FP8 | 점수가 이미 `visible` 에서 멈춤. 트리 검증 없음. QSA 에 학습된 헤드 가중치와 FP8 인덱서 없음 |
| 인덱서 | #582 #815 #971 #921 #926 | 이미 있음: vLLM 커널이 행 전부를 한 발사로 처리 |
| 어텐션 | #956 #952, #641·#554 SM121 명령, #658·#737 prefill32 CUDA, #698 | FP8 잠재 확장이 없음(BF16 KV). PTX 명령과 메가커널 코드는 Triton 에 대응이 없음. #698 은 GLM 에서도 기각 |
| 드래프터 | #1003, #982, #894·#900, #777, #735 #736 #729 #725, #951 #947, #879, #756, #871 | DFlash2 전용·비전 없음·FC 가 이미 BF16. 구조가 이미 있음(argmax 키, 융합 norm, KV 헤드당 읽기, 캐시된 역주파수). 트리 없음. GLM 에서도 기각 |
| 공유 | b12x 정적 v4/v5 계열, SP FP8 패킷 프리필 계열, deferred KDA 커밋, FP16 상태, CTA 셀 계열, GLM 라우터 | EP 셀이 정적 레인에 닿지 않음(아이디어는 M3 로). 프리필이 일반 TP all-reduce(M4 로). T=R=2 에서 같은 바이트에 발사만 늘어남. GLM 에서도 되돌림. GLM 폭 컴파일(S2 로). 라우터 수식이 다름(M2 로) |
