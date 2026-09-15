# PR #895: 일반 프리필의 패킷 직접 소비만 유지

> 그날의 조사 — **2026-09-14 의 조사다.** 그날 참이었던 것이고 유지되지 않는다 — 이후 무엇이 바뀌었는지는 `MEASUREMENTS.md` 가 안다.

2026-09-14 범위를 **P — packet FFN**으로 좁힌다. 혼합 decode/prefill 경로 M은
종료하고, compact KDA S도 이번 PR의 실행 코드에서 제외한다. 일반 디코드,
KDA 상태·cache·graph·prefix 수명은 통합 main `87304780`의 구현을 사용한다.
main의 deferred FP32 KDA 기본값, drafter QK 정규화 및 C1 MoE scale 개선도 유지한다.

## 남는 실행 경로

`TokenShards.all_gather_packets()`에서 토큰 소유 rank가 라우팅을 한 번 계산한 뒤
FP8 입력과 top-8 ID/FP32 가중치를 같은 패킷으로 보낸다. 패킹할 때 자기 shard의
FP8→BF16 복원 값을 함께 쓰며, 기존 긴 prefill GEMM으로 라우팅한다. 수신 측
routed expert와 shared gate/up은 패킷을 직접 읽고 라우터를 다시 실행하지 않는다.
기존 `Glm53Net.forward()`의 FFN 분기와 all-gather 한 번, 최종 reduce-scatter 순서를
유지한다. 송신 라우팅을 포함한 새 측정이 필요하며, 과거 수신 이후 v8 시간과
직접 비교하지 않는다.

적용 범위는 native eager TP4, chunk-ordered prefill, H4096/E288/I512/top-8,
SF6 M128 및 `8192 < rows <= 32768`이다. reader 지원 여부를 rank가 먼저 합의한다.
짧은 prefill/decode, observer와 지원되지 않는 pack은 기존 BF16 입력 경로를 쓴다.
FP8→FP32 scale 곱→BF16 반올림과 expert별 group-16 양자화를 유지한다.

32,256행의 전체 BF16 입력 252 MiB/rank를 없앨 수 있다. 이는 제거한 중간
버퍼 크기이며 속도나 순 메모리 트래픽 감소를 뜻하지 않는다. router/expert/shared의
패킷 읽기와 역양자화 비용이 더 크면 채택하지 않는다. 송신 라우팅 v2의 component 검증 후 사용자 요청으로 GB10 기본값을 ON으로 바꿨다.

## 제외한 코드와 보존한 증거

혼합 전용 커널·frontend·C++/Python planner·metadata·ticket·profile API/binding,
compact 전용 상태·cache·graph 연결 및 전용 probe/test를 제거했다. 혼합 static
V4 분기와 static dispatch 함수도 `6522564a`와 동일하게 복원했다. 현재 fleet은
폐기된 혼합 probe를 새로 허용하지 않는다.

과거 소스와 원시 측정은 동결 커밋 및 `measurements/`에 남긴다. 마지막 전체
S/P/M 구현은 [`3ac2b17f`](https://github.com/choiceoh/stkernel/tree/3ac2b17f34a21eebd9f2947dda61461adca4bad8)다.
과거 probe와 CPU runner는 각 manifest의 동결 revision에서만 재현한다.
혼합 경로는 같은 실행의 일반 경로보다 32K 완료가 7–12% 느렸으며, 디코드도
prefill 준비를 기다렸다. [판정 근거](../measurements/mixed_latency_20260914/README.md)는
실패를 포함해 보존하며 M3 후속 서빙은 진행하지 않는다.

CPU fact/budget 조회를 위한 drafter의 지연 import, explicit host budget 처리와
기존 CI의 procfs 종료 race 수정은 호환성 보조 변경으로 남는다. 실제 drafter
함수·정밀도·K·수락 로직은 바꾸지 않는다. 정리 과정에서 `ExecutionPlan.active`의
tuple 반환을 bool로 복구했으며, packet-only와 decode-only 활성 판정을 검사한다.

## 채택 판단

먼저 실제 L3 rank 가중치와 동일 입력에서 기존 BF16 unpack FFN과 packet FFN을
B/A/A/B로 비교한다. router/routes, expert FP4/SFA와 shared 출력 바이트를 검증한
뒤 전체 FFN device/wall 시간을 잰다. 수치 실패 셀·단계와 바뀐 component/expert를
기록한다. 이 component는 받은 패킷부터 시작해 pack/NIC/all-gather/reduce-scatter를
제외한다. 단일 GB10의 TP4 한 shard 결과이며 serving 성능으로 발표하지 않는다.

기존 P 예약 `st-ffn-packets0914v2`는 720분 대기시간 초과로 실행 전에 끝났다.
이는 GPU 수치 실패나 성능 탈락이 아니다. 현재 P의 채택에는 새 동결 source의
component 검증과 matched 32K/128K C1/C4 onepass 품질·수락률·TTFT·output tok/s가
필요하다. 앞서 미달한 52 ms는 D8/D32 + 32K complete FFN 목표였으며, 단독
prefill kernel 시간이나 준비를 제외한 시간으로 바꾸지 않는다.

최종 `b29b4083`의 GPU 수치 검증은 통과했다. 32K 중앙값은 약 3.51%
짧지만 평균 벽시계는 135.712 → 135.822 ms로 사실상 같고, B/A/A/B
4개 묶음 중 1개만 빨랐다. 따라서 이전 v8의 판정은 **정확성 통과 / 속도 우위
미확정**이다. 당시 기본 OFF를 유지했으며 중앙값만으로 채택 가치를 주장하지 않았다.

## 송신 소유 라우팅의 검증 범위

32K에서 rank당 router 행을 32K→8K로 줄인다. 송신 roundtrip은 64 MiB이며
all-gather 전에 해제한다. 라우팅 메타데이터는 토큰당 48바이트다. 동일 점수의
top-k, FP8 복원 반올림, ragged padding과 expert별 scale을 유지해야 한다.
단일 GPU probe는 한 rank의 송신 준비부터 FFN 끝까지 측정한다. 나머지 세
송신자는 사전 계산하며 `torch.cat`으로 수신을 모사하므로 NIC/4노드 성능이 아니다.
구조 개선의 새 수치·성능 판정은 해당 동결 소스에서 별도로 기록한다.

송신 라우팅 v2(`894a16f4`)는 GPU 검사 20개와 실제 L3 가중치 5개 셀의
수치 검증을 통과했다. 송신 준비를 포함한 32K FFN 평균 벽시계는
**63.886 → 58.451 ms (-8.51%)**, 네 비교 묶음 모두 개선됐다. 전체 셀의
20개 비교 묶음도 모두 개선됐다. 4노드 NIC/TTFT/tok/s 측정은 아니며,
서빙 품질·수락률 검증은 남아 있지만 사용자 요청에 따라 기본 ON으로 채택한다.
[원시 증거](../measurements/ffn_packets_20260914/packet_only/gpu-sender-v2.json).
main `87304780` 통합 후 실제 측정한 송신/router/expert/shared 코드는 유지된다.
