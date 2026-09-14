# C=2 의 16행 전체합을 one-shot PDL consumer 로 — 2026-09-15

운영자 지시("st커널 c=2 최적화 개선" → "전부 개선해")의 one-shot 몫이다.
- C=2 스텝은 16행을 검증한다. 타깃·드래프터의 전체합은 전부 16 × 4096 = 65,536 BF16 원소다.
- 지금까지 consumer 상한은 8행(32,768 원소)이었다. 그래서 C=2 의 전체합은 전부 일반 `k_oneshot` 으로 갔다.
- 이 변경은 그 합을 C=1 과 같은 `k_oneshot_consumer` 로 보낸다. **커널 소스는 한 바이트도 바꾸지 않았다.**

**판정 없음.** TP4 부팅·onepass 는 캠페인 리드의 몫이다. 이 기록은 정확성 증거와 크기 추정만 담는다.

## 왜 8행이었나 — 이 빌드에서는 커널 한계가 아니었다

과제의 전제는 "12 CTA 가 4 티켓씩, 두 벡터 stash 가 32,768 원소를 묶는다" 였다. 그 형태는 ST 엔진이 컴파일하지 않는다.

| 형태 | 어디서 컴파일되나 | 격자 | 발행 티켓 | stash | stash 가 덮는 원소 |
|---|---|---|---|---|---|
| **ST consumer** `k_oneshot_consumer` = `k_oneshot_impl<true>` | `engine/kernels/oneshot/__init__.py` `build()`: `OSAR_COMPACT_CTA` 를 정의하지 않는다(= 0) | 48 CTA × 256 | CTA 당 1, `done_ctr % 48 == 47` | `mine[VECITER]`, MAXEL 262,144 에서 VECITER = 3 | 3 × 48 × 256 × 8 = 294,912 ≥ MAXEL |
| compact consumer | vLLM overlay 의 선택 옵션 `VLLM_GLM53_AR_COMPACT_CTA=1` 뿐(기본 0) | 12 CTA × 256 | CTA 당 4, wrap-safe | `mine[OSAR_COMPACT_VECITER]` = 2 | 2 × 12 × 256 × 8 = 49,152 (자격 상한 32,768) |

- compact 형태라면 16행은 8,192 벡터 ÷ 3,072 레인 = 스레드당 3 trip 이라 두 벡터 stash 를 넘는다. 리드의 설명은 그 형태에서만 맞다.
- ST 의 8행은 커널이 아니라 측정에서 왔다.
  - vLLM 시절 consumer 도입(a6119955, 09-08)이 "C=1 검증 버킷을 포함한 8 토큰 이하" 만 A/B 했다(`probes/ar_consumer_README.md`).
  - ST 이식(9051b7c9)이 `8*4096` 를 그대로 옮겼다.
  - #820 이 그 값을 `cells.py` 로 올리면서 "the 12-CTA PDL consumer's element bound" 라고 적었다. 그 설명이 틀렸다.
- 그래서 설계는 **커널 변경 없음, 배정 상한만 16행** 이다. 일반 경로(17행 이상)는 그대로다.

## 바꾼 것

- `engine/kernels/cells.py`: `ONESHOT_CONSUMER_MAX_ELEMENTS` 8 × 4096 → **16 × 4096**. 주석은 실제 컴파일 형태로 고쳤다.
- `engine/kernels/oneshot/__init__.py`:
  - 부팅 자체 시험(합 == NCCL)에 16행을 더했다. 두 커널 모두 돈다. consumer 자격은 `rows <= 8` 대신 서빙 배정과 같은 `numel <= CONSUMER_MAX_ELEMENTS` 로 판단한다.
  - rank 순서 소거 시험에 16행을 더하고, 캡처 재생(배율 0, 2⁻⁸, 2⁸)을 7행과 16행에서 한다.
  - 지연 게이지에 `sum_16rows` 셀을 더했다. 부팅 게이지 `oneshot_sum_16rows_us` 와 `st:lane_info` 에 실린다.
  - 부팅 비용 증가: one-shot 집합통신이 자체 시험에서 12회(NCCL 대조 합 16행 2회, 소거·캡처·재생 두 커널 × 5회), 게이지에서 211회(3 워밍업 + 16 캡처 + 12 × 16 재생) 늘고, NCCL all-reduce 가 1회 는다.
- `dsv4_oneshot_ar.cu`·`dsv4_oneshot_transport.h`·`SOURCE.json` 은 main 과 바이트가 같다. 네이티브 캐시 키가 같으니 후보 부팅이 재컴파일하지 않고, 확장은 프로덕션과 같은 바이너리다.
- 검증 도구:
  - `tests/test_engine_oneshot_consumer.py` (CPU): 엔진 소스에서 격자·stash·소유·티켓을 뽑아 g++ 로 대조한다. `reduce()` 배정도 고정한다.
  - `tests/test_engine_oneshot_consumer_cuda.py` (단일 GPU): consumer 대 일반 커널의 바이트 대조.
  - `probes/oneshot_producer_oracle.cu`: 위 GPU 시험이 쓰는 `oneshot_ar`·`oneshot_ar_consumer` 바인딩과 PDL 이웃 커널 `staged_copy`.
  - `probes/engine_kernel_check.py --lanes oneshot_consumer`: 단일 GPU 레인 진입점. 위 시험과 기존 `test_engine_moe_output_transport`(8/16/24/32행 패킷)를 돌린다.
  - `tests/test_engine_oneshot_latency.py`·`test_engine_kernel_shape.py`: 새 셀과 새 상한으로 갱신.

## 정확성

### 1. 커널: 스택·소유·티켓 (CPU, g++, 엔진 소스에서 추출)

`test_stash_and_ownership_cover_every_consumer_size_through_maxel` 은 `dsv4_oneshot_ar.cu` 의 `ARGRID`·`ARTHREADS`·`VECITER` 정의와 `osar_block_owns` 를 그대로 뽑는다. MAXEL 과 consumer 상한은 `cells.py` 값이다.
- 모든 (블록, 스레드)에 대해 커널의 벡터 루프와 스칼라 꼬리를 같은 경계·보폭으로 돈다.
  - 범위: 163,840 원소(상한 + 한 보폭)까지 8원소마다, MAXEL 까지 격자 경계 ±9 와 홀수 꼬리, 모든 행 배수.
  - 판정 1: stash trip ≤ VECITER. MAXEL 에서 정확히 3 이다.
  - 판정 2: consumer 소유 == 실제로 데이터를 만지는 블록(+ 블록 0). 일반 소유 ⊇ 만지는 블록.
- 16행에서 consumer 는 32 CTA 가 소유한다. 일반 커널은 48 CTA 전부다. 8행은 각각 16 과 48 이다.
- 티켓: 일반·consumer(CTA당 1, 모듈러), 패킷(CTA당 1, wrap-safe), MAX·gather(1 CTA 가 48)를 섞은 20,000 발행. 매번 발행자는 하나이고 마지막 티켓이며, 진입 때마다 `done_ctr == 48 · seq` 다.
- 변이 시험(`mutation.log`): 네 가지 변이를 넣었고 넷 다 잡혔다.
  - 16B 레인을 32B 로 센 소유, 블록 0 무조건 소유 제거, stash 한 trip 부족, 일반 소유 축소.
  - 변이하지 않은 소스는 통과한다.

### 2. 조기 출발하는 후속 커널 감사 (main 9c45086a, C=2 경로)

consumer 는 발행 뒤 `griddepcontrol.launch_dependents` 를 부른다. 그래서 다음 PDL 커널은 피어 대기와 reduce 가 끝나기 전에 출발할 수 있다. 그 커널이 dependency wait 전에 합의 출력을 읽으면 틀린다(09-12 NVFP4 사고와 같은 계열). C=2 에서 16행 전체합 바로 뒤에 오는 커널은 다음과 같다.

| 합 | 생산자 | 후속 | PDL 인가 | 판정 |
|---|---|---|---|---|
| 드래프터 o_proj·down ×10 (`drafter.py` `_attn_rows`·`block_rows`) | W4 GEMM | `_taps` (Triton `tap_mix`) | 아니다(일반 launch) | 안전 |
| 드래프터·타깃 embed | `token_embedding.lookup` | Triton `_norm` / torch `expand().contiguous()` | 아니다 | 안전 |
| 보조층(`direct_mhc.decode_direct`) | Triton `moe_output._finish` | Triton `mhc_contract._contract` 또는 TileLang `mhc_post`. 그다음 층의 `_hc_post_pre` → `mk_mhc_kernel` | 바로 뒤는 아니다. `mk_mhc_kernel` 은 PDL 이지만 `launch_dependents; griddepcontrol.wait` 뒤에만 읽는다 | 안전 |
| 마지막 층 | Triton `moe_output._finish` | `finish` → contract 또는 TileLang `mhc_post` | 아니다 | 안전 |

- C=1 은 같은 자리들을 이미 consumer 로 서빙한다. 행 수에 따라 달라지는 후속은 MHC 하나다. n ≤ 8 이면 `mk_mhc_ar_kernel`(가중치만 읽고 대기), 16행이면 `mk_mhc_kernel`(대기 먼저)이다.
- 09-13 트레이스에서도 consumer 의 `next_gap` 은 +0.6 µs 다. 후속이 조기 출발하지 않았다는 뜻이다(아래 표).

### 3. 단일 GPU (loopback 프록시, 프로덕션 옆 레인)

`tests/test_engine_oneshot_consumer_cuda.py`. 오라클은 프로덕션 `dsv4_oneshot_ar.cu` 를 그대로 포함한다. 그래서 일반 커널이 같은 빌드의 대조군이다. CPU 스레드가 NIC 대신 세 피어의 서로 다른 패킷을 착지시킨다.
- 대상: 네 rank 위치 전부. 1/7/8/**16**/17/32/64행과 5, 65,533, 98,311 원소(꼬리·빈 레인·미완 보폭).
- 입력: 부팅 소거 세트 한 판과, 크기를 2⁻¹²~2¹² 로 섞어 BF16 반올림 경계를 치는 무작위 두 판.
- 판정: consumer == 일반 == 독립 rank 순서 FP32 fold, BF16 바이트 단위.
- 캡처 사슬: 늦은 PDL 생산자(`staged_copy`, 출발시킨 뒤 약 2 ms 기다렸다가 입력을 씀) → 합 → PDL 후속(`staged_copy`).
  - 재생마다 입력·합 출력·후속 출력을 NaN 으로 독을 넣는다. 프록시는 착지를 3 ms 늦춘다.
  - 합이 생산자 전에 읽었다면 게시된 페이로드가 틀리고, 프록시가 잡는다. 후속이 합 전에 읽었다면 출력이 틀린다.
- 한 그래프 안에 C=2 consumer 합 → 17행 일반 합 → C=1 consumer 합을 섞어 재생한다. 링 티켓은 int64 MAX 와 이어지고, 끝에 `tickets == published × 48` 를 확인한다.
- **결과: (GPU 티켓 대기 중 — 아래 "GPU 티켓" 절)**

### 4. 부팅 자체 시험

16행 합 == NCCL(두 커널), 16행 소거 시험, 16행 캡처 재생은 TP4 부팅에서만 돈다. **아직 실행하지 않았다.**

## 크기 추정 — 이득은 작을 것이다 (측정 아님)

4랭크 CUPTI 디코드 트레이스를 분해했다. 원천은 srv2 `/home/choiceoh/expert-capture/onepass-runs/20260913T230809-af6162bd0e38/`(`hybrid-k7-readback`, 소스 131d7a24, 2K)의 `diagnostic-c1-2000` 스텝 2~5 와 `diagnostic-c4-2000` 스텝 13~16 이다. 해시는 `traces.sha256`. #944(두 번째 레일) **이전** 빌드라 레일이 하나다. C=2 트레이스는 없다.
- 도구는 `overlap_trace.py` 다. #944 의 `decompose_trace.py` 에 일반 `k_oneshot` 과 후속 간격을 더했다.
- lead = 생산자 끝 − 집합통신 시작. 양수면 생산자와 겹쳤다는 뜻이다.
- floor = 랭크 시계를 완료 시각으로 맞춘 뒤, 마지막 랭크가 준비된 시점부터 착지까지(중앙).

| 집합통신 | 페이로드 | 개/스텝 | lead 중앙 µs | next_gap 중앙 µs | floor 평균 µs | 원천 |
|---|---|---|---|---|---|---|
| `k_oneshot_consumer` 전체합 | 64 KiB (8행) | 13 | **+17.2** | +0.6 | 40.9 | C=1 |
| `k_oneshot` 일반 전체합 | 256 KiB (32행) | 7 | **−0.6** | +1.4 | 87.5 | C=4 |
| `k_publish_packets` | 64 / 256 KiB | 43 / 48 | +31.4 / +41.4 | −32.6 / −90.3 | 29.6 / 85.3 | C=1 / C=4 |
| `k_oneshot_moe_packets` | 64 / 256 KiB | 36 / 36 | (생산자가 스트림 7 밖) / +15.7 | −36.0 / −90.2 | 31.5 / 76.7 | C=1 / C=4 |

- **consumer 의 PDL 겹침이 전체합에서 사주는 벽시계는 거의 없다.**
  - 일반 합도 생산자가 끝난 0.6 µs 뒤에 출발한다. 조기 출발로 아낄 launch 지연이 CUPTI 로는 1 µs 도 안 된다.
  - 전체합의 후속은 전부 PDL 이 아니다(Triton·torch). 발행 시점에 후속이 미리 출발하는 이득(패킷 경로의 `next_gap` −90 µs)이 여기엔 없다.
- 남는 기전은 소유 CTA 축소다. 16행에서 48 → 32 CTA 만 링 가드·펜스·피어 대기를 한다. 크기는 모른다.
  - 시사점: 256 KiB 에서 CTA 48개가 전부 기다리는 `moe_packets` 의 floor(76.7)가 1 CTA 인 `publish`(85.3)보다 낮다. 기다리는 CTA 수가 지배 비용은 아니다.
- C=2 프로파일(#965)에서 전체합은 스텝당 14.6회다. 회당 수 µs 라면 **스텝당 0.1 ms 이하**다. C=2 스텝 55.3 ms 의 0.2% 이하라 onepass step/s 로는 가려내기 어렵다.
- 권하는 판정: 기준 부팅과 후보 부팅에서 C=2 `diagnostic` 트레이스를 떠 `overlap_trace.py` 로 비교한다. 16행 `ordinary` 대 16행 `consumer` 의 tail·floor 가 직접 맞선다.

## 패킷 커널 — 16행 고정비·직렬화 절감이 있는가 (코디네이터 추가 과제)

C=2 프로파일(#965, 프로파일러 하)의 스텝당 값을 회당으로 나누면 다음과 같다.
- `k_publish_packets`: 38.7 → 75.6 µs (+36.9). `k_oneshot_moe_packets`: 43.0 → 80.1 µs (+37.1).
- 전체합: C=1 consumer 60.3 µs(8행) → C=2 일반 67.1 µs(16행) (+6.8).

이 표만으로는 전송이 얼마나 늘었는지 가를 수 없다.
- PDL 커널의 CUPTI 구간은 생산자와 겹친 시간을 품는다(#944). 패킷은 C=1·C=2 모두 PDL 이다. 그래서 16행에서 길어진 생산자(W4 GEMM 직접 출력, MoE 마감)의 몫도 함께 늘어난다.
- 전체합은 반대다. C=1 은 PDL(09-13 트레이스에서 lead 중앙 17 µs)이고 C=2 는 일반(겹침 없음)이라 증가가 작게 보인다.
- 가르려면 현재 main 의 C=2 4랭크 트레이스가 필요하다. 아직 없다.

GPU 쪽에서 행 수에 비례하는 일(소스 기준):
- `k_publish_packets`: 1 CTA × 1 스레드다. 원자 연산·펜스·발행·플래그 대기뿐이라 행 수와 무관하다. 행 수에 비례하는 것은 페이로드 전송뿐이다.
- `k_oneshot_moe_packets`: 두 가지가 행에 비례한다.
  - 소유 CTA 16 → 32 의 가드·펜스·피어 대기.
  - 스레드당 1 trip 의 마감 복사(4,096 → 8,192 벡터). CTA 들이 병렬로 한 번씩 돈다.

전송은 페이로드에 비례한다(09-13 트레이스, 한 레일, #944 이전).
- floor: publish 29.6 → 85.3 µs, moe 31.5 → 76.7 µs (64 → 256 KiB). 기울기는 0.24~0.29 µs/KiB 다.
- 256 KiB 에서 CTA 48개가 전부 기다리는 moe_packets 의 floor 가 1 CTA 인 publish 보다 낮다. 기다리는 CTA 수는 지배 비용이 아니다.
- 현재 main 은 두 레일·inline 플래그다. C=1 publish 구간이 트레이스 70 µs, 프로파일 39 µs 로 절대값이 훨씬 작다. 그러니 이 기울기는 상한으로만 읽는다(128 KiB 에서 회당 +15~19 µs 이하).

**결론: 두 패킷 커널에서 정확성을 지키며 뺄 16행 고정비나 직렬화 비용은 찾지 못했다.** 행에 비례해 느는 것은 PCIe x4·200G 위의 페이로드 바이트다. 검토하고 넣지 않은 후보는 넷이다.
- **패킷에서 블록 0 만 피어 대기**: 다른 소유 블록은 대기 뒤에 아무 일도 없으니 산술적으로는 정확하다. 그러나 위 floor 비교가 기다리는 CTA 수가 지배적이지 않음을 보인다. 이득 근거가 없어 넣지 않았다.
- **moe_packets 격자 축소**(24 CTA × 2 trip, 티켓 2): launch 비용과 복사 병렬도를 맞바꾼다. 방향을 모른다.
- **TX 를 일반 저장으로**: MHC 가 로컬 입력을 TX 에서 다시 읽는다. 대신 발행 펜스가 L2 를 비워야 한다. 방향을 모른다.
- **피어별 페이로드를 두 PCIe function 에 반씩(균형 레일)**: 가장 바쁜 레일이 256 → 192 KiB 가 된다. 둘이 공유하는 200G 포트(384 KiB ≈ 16 µs)에 묶여 약 −3 µs/회로 추정한다. QP·CQ 크기·프록시 프로토콜을 바꾸는 일이라 TP4 없이 검증할 수 없다.
- 커널 표에 보이는 C=2 mHC 증가(`mk_mhc_packets_kernel<false>` 82 µs/회, `mhc.py` 의 `small = n <= 8`)는 #965 가 이미 짚었고 transport 가 아니다.

## CPU 결과 (소스 `4d50fd37`, 이미지 `sha256:b45454b5…`, CUDA 숨김)

| 항목 | 결과 | 파일 |
|---|---|---|
| 프로덕션 확장 4변형(rails 1/2 × inline 0/1) 컴파일·로드 | PASS. torch 2.13.0+cu132, nvcc 13.2.78. 확장 이름이 #957 기록과 같다(소스가 같다) | `compile.json`·`compile.log`·`toolchain.log` |
| 단일 GPU 오라클 컴파일·로드 | PASS. `oneshot_ar`·`oneshot_ar_consumer`·`staged_copy` 바인딩 | `oracle.json`·`oracle.log` |
| 관련 CPU 시험 17 모듈 | 106 개 중 98 통과, 8 skip(전부 GB10 필요), 실패 0 | `cpu.log` |
| g++ 오라클 요약 줄 | 20,917 크기, MAXEL 에서 stash 3/3 trip, 상한에서 소유 CTA 32, 혼합 발행 20,000 PASS | `mutation.log` |
| 오라클 변이 4종 | 4/4 잡음, 원본 PASS | `mutation.log` |

## GPU 티켓

(작성 중)

## 검증하지 않은 것

- TP4 부팅: 16행 자체 시험의 실제 RDMA 실행, `oneshot_sum_16rows_us` 게이지 값.
- C=1/C=2 onepass(step/s·품질·수락률)와 C=2 트레이스 분해. 캠페인 리드의 판정이다.
- 단일 GPU 시험은 CPU 스레드가 NIC 를 대신한다. RDMA 타이밍·레일·실제 피어 지연은 재지 않는다. 타이밍은 재지 않았다. 파이썬 프록시 지연이 커널 차이를 덮기 때문이다.

## 재현

CPU(srv4, 이미지 `st-engine:bracket-9c45086a0622` = `sha256:b45454b5fdc138bfaafa6470cf9cb49bbcd74783ceffb343b0a7b1d0cf87fcd5`, CUDA 숨김, 무거운 작업 잠금 아래):

```sh
flock -w 7200 /tmp/c2opt-heavy.lock docker run --rm --runtime=runc --network=none --cpus 4 --memory 24g \
  -e CUDA_VISIBLE_DEVICES= -e CUTE_DSL_ARCH=sm_121a -e PYTHONPATH=/repo -e MAX_JOBS=2 \
  -v "$REPO":/repo:ro -v "$OUT":/out -w /repo --entrypoint bash st-engine:bracket-9c45086a0622 -c '
    python3 probes/engine_oneshot_cpu_check.py --output /out/compile.json
    python3 measurements/st_c2_oneshot_consumer_20260915/oracle_compile.py --output /out/oracle.json
    python3 -m unittest -v tests.test_engine_oneshot_consumer tests.test_engine_oneshot_latency \
      tests.test_engine_kernel_shape tests.test_engine_oneshot_integer tests.test_engine_oneshot_sum \
      tests.test_engine_oneshot_rails tests.test_engine_oneshot_modes tests.test_engine_oneshot_proxy \
      tests.test_engine_oneshot_setup tests.test_engine_oneshot_health tests.test_fleet_onepass tests.test_fleet_single \
      tests.test_engine_kda_ring_bench tests.test_engine_oneshot_consumer_cuda tests.test_engine_moe_output_transport \
      tests.test_engine_oneshot_gather_cuda tests.test_engine_direct_producer_cuda'
```

단일 GPU 레인(srv2 동결 체크아웃에서):

```sh
ST_IMAGE=sha256:b45454b5fdc138bfaafa6470cf9cb49bbcd74783ceffb343b0a7b1d0cf87fcd5 ST_PROBE_TREE=c2cons-<sha8> \
  bash bench/fleet.sh run --gpu --detach c2cons-gpu-<sha8> 15 '<note>' -- \
  bash probes/run_engine_probe.sh probes/engine_kernel_check.py --lanes oneshot_consumer --output /cache/c2cons-<sha8>.json
```

트레이스 분해:

```sh
python3 overlap_trace.py <run>/diagnostic-c1-2000 2 3 4 5
python3 overlap_trace.py <run>/diagnostic-c4-2000 13 14 15 16
```

## 파일

- `compile.json`·`compile.log`·`toolchain.log`: 프로덕션 확장 4변형(rails 1/2 × inline 0/1) 컴파일·로드. 소스 해시가 main 과 같다.
- `oracle.json`·`oracle.log`: 단일 GPU 오라클 컴파일·로드와 바인딩 목록.
- `cpu.log`: CPU 단위 시험. `mutation.log`: 오라클 변이 시험.
- `overlap_trace.py`·`trace-c1-131d7a24.txt`·`trace-c4-131d7a24.txt`·`traces.sha256`: 트레이스 분해.
- GPU 영수증(티켓 완료 뒤).
