# Qwen3.8 랭크 패킷을 leave 가 접기 (carry H5)·MoE 패킷 (X2) — 바이트는 같고 GPU 쪽은 이득 없음, 기각 (2026-09-19, srv4 단일 GPU 레인)

> 그대로 두는 기록 — 캐리 H5·X2 의 GB10 판정과 원시 로그다. 뒤 실행은 이 파일을 고치지 않고 새 기록을 남긴다.

**커널 컴포넌트 시간이다. 엔진·플릿 속도 주장은 없다(D17).** 코드는 머지하지 않았다. 판정에 쓴 트리는 기록 브랜치
`record/qwen38-h5-x2-rank-packets` 에 있다: H5 는 `83ebfcb6`, X2 는 `bd165087`. 프로브는 `probes/engine_qwen38_rank_packets.py`
(레인 `qwen38_rank_packets`·`qwen38_moe_packets`)와 `probes/engine_direct_producer_timing.py`(레인 `direct_producer_timing`)다.
단일 GPU one-shot 오라클(`probes/oneshot_producer_oracle.cu`, CPU 스레드가 NIC 과 세 피어를 대신함) 위에서 쟀다. 장치 GB10,
이미지 `st-engine:glm53`. 세 티켓 모두 18:08–18:11 에 돌았다. 다른 세션의 운영자 창이 열려 srv4 에 자리가 났을 때다.

## 무엇을 만들었나

- **H5:**
  - `oneshot_packets` 가 바운드 폭(2560)의 행을 받게 했다(.cu 의 4096 고정 검사 제거).
  - `gated_residual.leave_norm(packets=)` 는 교환의 설명자에서 네 랭크 주소를 대기 뒤에 읽는다. 그리고 랭크 순서 0,1,2,3 으로
    FP32 로 접은 뒤 한 번 반올림한다(consumer 의 `osar_sum_rank_order` 와 같은 산술).
  - net: 캡처 스텝의 합은 `_sum` → `exchange` 로 나가고, 다음 leave 가 소비한다.
- **X2 MoE:** 패킷 grid 의 복사 단계가 Qwen 의 게이트 합 BF16(routed + shared·gate) 를 TX 에 바로 쓴다(`MOE_GATED` 템플릿,
  `exchange_moe_gated`, net `_finish`).
- **X2 dense:** 운영자 결정 3 에 따라 코드를 짓기 전에 GLM 형상의 기존 직접 생산자(#826)를 먼저 쟀다.

## 결과

### 바이트 — 모두 같다

- **`q38packets-0919a` [원시](q38packets-0919a.log):**
  - 네 랭크 × 1·4·16 행에서 소거 fixture(순서가 틀리면 값이 바뀜)를 썼다.
  - consumer 합 → leave 대비, 패킷 → 접는 leave 가 바이트까지 같다. 보통·PDL·PDL+프리페치 모두, eager 와 캡처 재생 6회(링 한 바퀴 넘김),
    피어 3 ms 늦은 착지에서 같다. 발행 표 개수도 맞다.
  - GLM 의 패킷 시험 5건도 같은 전송 소스로 통과했다(운영자 결정 Q8).
- **`q38moepk-0919a` [원시](q38moepk-0919a.log):** 같은 조건에서 게이트 합 패킷 → leave 가 `gated_sum` → consumer → leave 와 같다.
  TX 에 쓴 바이트 자체도 `gated_sum` 출력과 같다.

### 시간 — GPU 쪽만(피어 선착지, 40 µs PDL 생산자 뒤, warm, 중앙값 µs)

| 행 | consumer → leave | 패킷 → 접는 leave (H5) | MoE: gated_sum → consumer → leave | gated_sum → 패킷 → 접는 leave | 게이트 합 패킷 → 접는 leave (X2) |
|---:|---:|---:|---:|---:|---:|
| 1 | 193.0 | 194.1 (+1.1) | 192.6 | 191.9 (−0.7) | 190.3 (−2.3) |
| 4 | 195.8 | 199.2 (+3.4) | 195.9 | 197.0 (+1.1) | 195.9 (0.0) |
| 16 | 220.5 | 227.6 (+7.1) | 220.5 | 225.4 (+4.9) | 223.8 (+3.3) |

- **H5 는 행이 늘수록 진다.** leave 의 (행, 스트림) 프로그램마다 네 패킷을 매핑된 호스트 메모리에서 읽는다. 그래서 같은 행을
  스트림 수(4)만큼 다시 읽는다. consumer 는 rx 를 한 번만 읽는다.
- **X2 MoE 도 이득이 없다.** 피니셔 발사를 없앤 몫(약 1–2 µs)을 같은 접기 비용이 상쇄한다. 서빙 폭인 4 행(C=1 K=3)에서 0,
  16 행(C=4)에서 +3.3 µs 다.
- **evicted 팔은 쓸 수 없다.** 중앙값 400–530 µs 에 최소가 190–500 µs 로 흩어진다(원시 로그에 남김).
- **다시 열 길:** 행마다 한 프로그램이 패킷을 한 번만 읽고 네 스트림을 도는 접기 모양. 이번 목록 밖이다(Q9).

### X2 dense 의 문 — 이 오라클로는 판정할 수 없다 (`q38direct-0919a` [원시](q38direct-0919a.log))

- GLM 형상(4096 × 2048·3072·4096, 8·16 행)에서 "GEMM → 패킷 커널 → 접기"는 합당 187–226 µs, "예약 → GEMM 이 TX 에 직접 → 발행
  → 접기"는 34–57 µs 였다. 바이트는 같다.
- 이 차이(합당 약 150 µs)는 **오라클의 인공물**이다.
  - 오라클에서는 48-CTA one-shot 커널이 호출당 약 150 µs 걸린다. #967 기록도 "절대 µs 는 믿지 않는다, 쉬는 CTA 하나에 약 1.4 µs"라고
    적었다.
  - 반면 GLM 프로덕션 프로파일에서 48-CTA `k_oneshot_moe_packets` 는 피어 대기를 포함해 호출당 36–44 µs, 1 스레드
    `k_publish_packets` 는 29–39 µs 다.
  - 직접 생산자와 GEMM+패킷을 가르는 것이 바로 그 48-CTA 커널이므로, 이 비교는 무효다.
- 결정 3 의 문("합당 1 µs 이상 이기면 N=2560 으로 옮김")을 충족하는 유효한 근거가 없으므로 옮기지 않는다.
- 재측정 길: 오라클의 Ctrl 을 프로덕션처럼 `aligned_alloc` + `cudaHostRegister` 로 두는 것. 지금 오라클은 `cudaHostAlloc(Mapped|Portable)`
  이고, 이것이 원인인지는 가리지 않았다. 또는 GLM 플릿에서 `direct_mhc` 를 켜고 끈 A/B.

## 판정

- **H5 기각.** 서빙 행 수에서 GPU 쪽이 같거나 느리다.
- **X2 기각.** MoE 쪽은 이득 없음, dense 쪽은 유효한 근거 없음.
- 두 항목 모두 바이트는 같았다. 코드는 기록 브랜치에만 둔다.
