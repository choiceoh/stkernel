# C=2 DSA 인덱서 선택 접기 — 두 행의 logits·지평 마스크·top-k 를 한 런치씩 (2026-09-15, PR #971)

운영자 지시 "st커널 c=2 최적화 개선" 캠페인의 한 조각이다. 판정 범위는 **캡처된 선택 컴포넌트**다. 엔진 tok/s·step/s,
수용률·답 품질은 재지 않았다.

## 무엇이 문제였나

`Glm53Net._select_rows` 는 캡처된 디코드 스텝의 행 전부를 선택한다. 후보 gather·지평 마스크·finalize 는 이미 접혀 있었지만,
DSA 층마다 행마다 세 런치가 남아 있었다.

- DeepGEMM `sm120_fp8_mqa_logits`
- `_mask_horizon`
- torch top-k

그래서 C=2 는 C=1 의 두 배를 띄웠다. #965 의 C=2 프로파일에서 logits 호출이 9.2 → 17.8 회였고, 128K 버킷에서는 top-k 가
다중 블록(`mbtopk`)이라 호출 하나가 커널 21 개다.

## 바꾼 것 (기본값 켜짐, 노브 없음)

1. **`engine/profiles/glm53/net.py` `_select_rows(..., joined=True)`.** 두 행 이상이면 한 스텝에 한 번씩 돈다.
   - 행들의 후보를 이어 붙인다(`keys_all.view(rows*n_cand, d)`).
   - 질의마다 제 행의 창 `[r*n_cand, r*n_cand + ke)` 을 DeepGEMM **압축 logits**(`max_seqlen_k=n_cand`)로 한 런치에 점수 매긴다.
     출력의 행 m 열 c 가 곧 그 행의 후보 c 다(행별 런치와 같은 열).
   - `[rows*t, n_cand]` 블록에 `_mask_horizon` 한 번, `torch.topk` 한 번.
   - `joined=False` 는 프로브의 같은 빌드 대조군이다.
2. **게이트** — 아래를 모두 만족할 때만 합친다. 나머지는 예전 행별 코드 그대로다.
   - `1 < rows`: C=1 은 코드가 바뀌지 않는다.
   - `rows*t <= JOINED_SLICES(20)`: torch 는 슬라이스 수와 크기로 단일/다중 블록 선택을 고른다
     (`should_use_multiblock`). 20 슬라이스까지는 크기 기준 하나(20,000 열)라서, 16 슬라이스 블록이 8 슬라이스 행과
     같은 알고리즘을 탄다.
   - `t % 4 == 0 and n_cand % 4 == 0`: DeepGEMM 은 질의를 4 개씩 블록으로 묶는다(32 헤드). 각 블록의 KV 범위는
     블록 첫 창 시작을 4 로 내림한 곳에서 시작한다. 이 조건이면 블록 하나가 한 행만 담고, 제 행의 창 시작점(행별 런치의 0 과 같은
     상대 위치)에서 시작한다. 1 토큰 스텝(프로덕션 폭이 아니다)은 블록이 행을 넘어 모든 창을 점수 매기게 되므로 행별로 남긴다.
3. **`DecodeRows.lengths` 에 `width` 추가.** 서빙(`engine/kernels/indexer.py` 새 Triton `_row_windows`)과 참조
   (`engine/modules/sparse_indexer.py`) 둘 다다.
   - 한 런치가 `seq_lens`·`ke` 에 더해 창 `(i*width, i*width + ke)` 를 낸다.
   - `GraphCaches.row_lengths` 가 순전파당 한 번 캐시하고, 11 DSA 층이 공유한다(키에 width 포함).
   - width 없는 호출은 예전 `_row_lengths` 커널 그대로다.
4. **레인 계약.** `Lanes.indexer_logits` 에 `width` 를 더했다.
   - 서빙: `max_seqlen_k` 로 넘긴다(`engine/kernels/deep_gemm.py` 가 전달).
   - 참조: 질의별 창을 gather 해 einsum 한다.
5. **프로브** `probes/engine_decode_select_rows.py`, 레인 `engine_kernel_check.py --lanes select_rows`.
   - 대조군·후보, 그리고 옵션 두 개를 11 층 캡처 그래프로 비교한다.
   - 서빙 레인, 합성 페이지 풀 레코드를 쓴다.
6. **테스트**
   - `tests/test_engine_decode_rows.py`: 합친 선택 = 루프 선택(행 2·폭 4/8), 무엇이 루프에 남는지, 경계 동점, 짧은 문맥, 11 층 길이 공유.
   - `tests/test_engine_decode_lengths.py`: width 캐시·불일치 거부.
   - `tests/test_engine_indexer_rows.py`: `_row_windows` 대 참조, 인터프리터.
   - `tests/test_engine_kernels.py`: 래퍼 kwargs.

## 수치 게이트 — GPU 무허용오차 (srv4 GB10, 단일 GPU 레인, 프로덕션 옆, 모델 부팅 없음)

두 티켓 모두 PASS 다. 슬롯·카운트를 원소 단위로 비교했고, 재생 순서는 정/역 두 번이다.

| 티켓 | 소스 | 결과 |
|---|---|---|
| `st-c2idx-sel0915` | `9b41064c` (첫 구현, 게이트 전) | 모든 팔 불일치 0 |
| `st-c2idx-final0915` | `6982cdf4` (최종 게이트, 이 PR 의 코드) | 모든 팔 불일치 0 |

- 버킷: C=2·t=8 은 4096 / 8192 / 16384 / 32768 / 65536 / 131072 / 202752(n_cand 1024~50688). C=1·t=8 은 4096 / 32768 / 131072.
  C=2·t=1 은 4096 / 32768. 시간 구간에서 3 버킷을 한 번 더 돌렸다.
- 버킷마다 다섯 단계를 돈다(문맥은 [행0, 행1]).
  1. 무작위 키, 블록표 재결합, 문맥 [용량 끝, 절반+3]
  2. 동점 키(레코드 1/4 이 질의 0 을 겨눈 같은 키·스케일 1, 1/8 이 다른 같은 키), [끝−5, 37]
  3. 영 키 절반과 대부분 음수 게이트(0.0 동점), 재결합, [3, 끝]
  4. 동점 키, 재결합, 문맥이 뒤로 감, [용량/3, 끝−1]
  5. 무작위, [1, 700] (top-k 폭 512 보다 짧은 행)
- 층 0 에서 k 번째 점수가 top-k 경계에 걸친 질의 수를 단계별로 셌다. 예: 32K 에서 [0, 4, 8, 11, 0] / 16. 동점 깨기가 실제로 판정됐다.
- 네 팔 모두 모든 버킷·단계·순서에서 불일치 원소 0 이다.
  - `wide` 는 65536 버킷에서 wide top-k 가 다중 블록(32768 열), 대조군이 단일 블록(16384 열)으로 알고리즘이 달랐는데도 같았다.

## 시간 — 캡처 11 층 재생당 ms, B/A/A/B 두 번 (B·A 각 4 표본; warm 48 재생, evicted 24 재생 앞 128 MiB 쓰기)

최종 티켓 `st-c2idx-final0915`(`6982cdf4`)의 값이다. 첫 티켓의 같은 칸은 괄호다.

| 문맥 (버킷) | 캐시 | 대조군 평균 / 최소 | **joined** 평균 / 최소 | 비율 | 절약 |
|---|---|---|---|---|---|
| 2K ×2 (4096) | warm | 0.749 / 0.749 | **0.404 / 0.404** | 0.539 (0.540) | −0.345 ms |
| 2K ×2 (4096) | evicted | 0.809 / 0.808 | **0.451 / 0.450** | 0.557 (0.553) | −0.358 ms |
| 32K ×2 (32768) | warm | 1.762 / 1.760 | **1.016 / 1.014** | 0.577 (0.579) | −0.746 ms |
| 32K ×2 (32768) | evicted | 1.853 / 1.853 | **1.092 / 1.092** | 0.589 (0.589) | −0.761 ms |
| 128K ×2 (131072) | warm | 3.715 / 3.692 | **2.447 / 2.440** | 0.659 (0.671) | −1.269 ms |
| 128K ×2 (131072) | evicted | 3.864 / 3.729 | **2.487 / 2.481** | 0.644 (0.670) | −1.377 ms |

- **모든 문맥 버킷에서 이긴다.** 128K 에서 비율이 줄어드는 것은 logits 계산(질의당 ke 키)이 접기로 줄지 않기 때문이다.
  줄어드는 몫은 런치와 top-k 의 행 병렬이다.
- 128K evicted 대조군의 마지막 B 표본 하나가 4.250 ms 였다(프로덕션 옆 간섭). 최소값 비율은 0.665 다.
- **C=1 은 코드가 같다.** 대조군 대 joined 는 0.998~1.000(2K 0.431/0.430, 32K 0.927/0.927, 128K 1.765/1.765 warm)이다.
  이것이 이 측정의 잡음 바닥이다.

### 옵션 평가 (같은 티켓, 같은 대조군 브래킷)

| 후보 | 런치/재생 (2K·32K / 128K) | 2K warm | 32K warm | 128K warm | 판정 |
|---|---|---|---|---|---|
| 대조군: 행별 logits·마스크·top-k | 113 / 553 | 1 | 1 | 1 | — |
| **(c) joined**: 압축 logits 1 + 마스크 1 + top-k 1 | **58 / 278** | **0.539** | **0.577** | **0.659** | **채택** |
| (b) `rows_cat`: 행별 logits 2 + cat + 마스크 1 + top-k 1 | 80 / 300 | 0.683 | 0.731 | 0.618~0.851* | 기각: joined 보다 느림 |
| (a) `wide`: 비압축 logits(clean) 1 + wide top-k + id 뺄셈 | 80 / 300 | 0.716 | 0.859 | 0.898 | 기각: top-k 가 질의당 2 배 열 |

\* 128K `rows_cat` warm 브래킷의 B 표본이 흔들렸다(평균 5.539, 최소 3.420). evicted 는 0.851 이고, 첫 티켓은 0.831 이다.

- **(a) wide** 는 "긴 문맥에서 질까" 가 걱정이었다. 대조군보다는 빠르지만 문맥이 길수록 joined 와의 차이가 벌어진다.
  마스크 하나 대신 `smxx_clean_logits`·id 뺄셈(int32→int64 복사 포함)이 들어가, joined 보다 층당 두 런치가 많다.
- **(b)** 는 DeepGEMM 이 미리 잡은 출력에 쓰지 못한다(`out=` 가 없다). 그래서 cat 복사가 들고, logits 두 런치가 남는다.
- **(c)** 는 DeepGEMM 에 이미 있는 배치 레이아웃이다. `fp8_fp4_mqa_logits(..., max_seqlen_k)` 의 압축 logits 는
  `kIsCompressedLogits` 템플릿으로 저장 위치만 `ks` 만큼 당긴다(`sm120_fp8_mqa_logits.cuh`).

### 런치 인구조사 (torch profiler, 캡처 재생 4 회 평균, 11 층)

- **2K·32K: 113 → 58 (층당 −5).**
  - `sm120_fp8_mqa_logits<…false…>` 22 → `<…true…>` 11
  - `_mask_horizon` 22 → 11
  - `sbtopk::gatherTopK` 22 → 11
  - `direct_copy` 22 → 0: `topk(out=)` 가 행 조각에 쓰던 복사
  - `_row_lengths` 1 → `_row_windows` 1
- **128K: 553 → 278 (층당 −25).** `mbtopk` 한 호출이 커널 21 개다(digit count·cumsum·within-k 넷씩, scan_by_key·memset 둘씩,
  kth count·fill·gather 하나씩). 층당 top-k 호출이 둘에서 하나가 되니 −21 이고, logits·마스크 −1 씩, 복사 −2 를 더해 −25 다.

### 1 토큰 스텝 (프로덕션 폭 아님)

- 첫 티켓(게이트 전)에서 C=2·t=1·32K 를 합치면 정확했고 0.675 배였다. `rows_cat` 도 0.735 배였다.
- 최종 게이트는 t=1 을 행별로 남긴다. 최종 티켓의 joined 대 대조군은 0.981 / 1.001 이다.
- 이유: t % 4 ≠ 0 이면 DeepGEMM 질의 블록이 행을 넘는다. 그러면 블록의 질의마다 모든 창을 점수 매기므로 일이 행 수에 비례해 는다.
  C=2 에서만 쟀기 때문에 넓히지 않았다.

## 크기 가늠 (컴포넌트 수치 → 스텝)

- 11 DSA 층 선택이 C=2 순전파당 −0.35 ms(2K), −0.75 ms(32K), −1.27 ms(128K) 다.
- #965 의 짧은 문맥 C=2 스텝(프로파일러 없이 55.3 ms)에 대면 2K 몫은 약 0.6% 다.
- 32K·128K 의 C=2 스텝 시간은 이 기록에 없다. 그 비율은 주장하지 않는다.
- 합성 풀 레코드, 프로덕션 옆 단일 GPU 측정이다. 소비자 tok/s·step/s 판정은 onepass 몫이다(D17). 미측정.

## 재현

이미지는 `st-engine:bracket-9c45086a0622`
(`sha256:b45454b5fdc138bfaafa6470cf9cb49bbcd74783ceffb343b0a7b1d0cf87fcd5`, torch 2.13.0+cu132, CUDA 13.2)다.

CPU(srv4, CUDA 숨김) — `cpu-tests.log`: 14 모듈 145 테스트 통과, GPU 스킵 11.
`interpreter-tests.log`: `_row_windows`·`_mask_horizon` 이 `TRITON_INTERPRET=1` 에서 참조와 바이트 동일.

```sh
docker run --rm -e CUDA_VISIBLE_DEVICES= -e CUTE_DSL_ARCH=sm_121a -e PYTHONPATH=/repo -v "$PWD":/repo:ro -w /repo \
  --entrypoint python3 st-engine:bracket-9c45086a0622 -m unittest -v tests.test_engine_decode_rows \
  tests.test_engine_decode_lengths tests.test_engine_kernels tests.test_engine_graph_contracts
docker run --rm -e CUDA_VISIBLE_DEVICES= -e TRITON_INTERPRET=1 -e PYTHONPATH=/repo -v "$PWD":/repo:ro -w /repo \
  --entrypoint python3 st-engine:bracket-9c45086a0622 -m unittest -v tests.test_engine_indexer_rows
```

GPU(srv2 동결 체크아웃에서 큐의 단일 GPU 레인):

```sh
git -C ~/stkernel worktree add --detach ~/st-worktrees/c2idx-6982cdf4 6982cdf4
cd ~/st-worktrees/c2idx-6982cdf4
ST_IMAGE=sha256:b45454b5fdc138bfaafa6470cf9cb49bbcd74783ceffb343b0a7b1d0cf87fcd5 ST_PROBE_TREE=st-c2idx-6982cdf4 \
  bash bench/fleet.sh run --gpu --detach st-c2idx-final0915 15 '...' -- \
  bash probes/run_engine_probe.sh probes/engine_kernel_check.py --lanes select_rows --output /cache/st-c2idx-6982cdf4.json
python3 summarize.py gpu-6982cdf4.jsonl
```

## 파일

| 파일 | 내용 |
|---|---|
| `gpu-6982cdf4.jsonl`, `gpu-6982cdf4.log` | 최종 게이트 티켓 원본 사건과 큐 로그. identity 의 source_sha256: net.py `eb6b66c8…`, lanes.py `862a2e38…` |
| `gpu-9b41064c.jsonl`, `gpu-9b41064c.log` | 첫 티켓(게이트 전 구현): net.py `db213b05…`, lanes.py `4c262c75…` |
| `summarize.py` | 수치·시간 표·런치 인구조사 요약 |
| `cpu-tests.log`, `interpreter-tests.log` | CPU 게이트 |

## 못 한 것 · 한계

- 네 노드 onepass(수용률·tok/s·step/s)는 돌리지 않았다. 운영자 상시 지시("결과 오면 검증하지말고 바로 기본값 pr 머지해")에 따라
  컴포넌트 게이트 뒤 병합한다.
- 입력은 합성 FP8 풀 레코드와 질의다. 실제 가중치나 실제 인덱서 분포는 아니다. 정확성은 커널 산술의 행 독립성으로 판정했고,
  동점 단계로 top-k 동점 깨기를 확인했다.
- 1 토큰 스텝과 C≥3 은 합치지 않는다(게이트). C=4 가 돌아오면 32 슬라이스는 torch 의 다른 크기 규칙(10,000 열)을 타므로 따로 판정해야 한다.
