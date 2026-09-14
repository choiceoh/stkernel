# PR #895: 일반 프리필의 패킷 직접 소비만 유지

2026-09-14 범위를 **P — packet FFN**으로 좁힌다. 혼합 decode/prefill 경로 M은
종료하고, compact KDA S도 이번 PR의 실행 코드에서 제외한다. 일반 디코드,
KDA 상태·cache·graph·prefix 수명은 통합 main `6522564a`의 구현을 사용한다.

## 남는 실행 경로

`TokenShards.all_gather_packets()`가 소유한 기존 FP8 패킷을 router, routed expert,
shared gate/up이 직접 읽는다. 기존 `Glm53Net.forward()`의 FFN 분기에 들어가며
별도 혼합 스케줄러, CPU route planner, ticket, decoder 대기열을 만들지 않는다.
기존 FP8 all-gather와 최종 reduce-scatter의 순서를 유지한다.

적용 범위는 native eager TP4, chunk-ordered prefill, H4096/E288/I512/top-8,
SF6 M128 및 `8192 < rows <= 32768`이다. reader 지원 여부를 rank가 먼저 합의한다.
짧은 prefill/decode, observer와 지원되지 않는 pack은 기존 BF16 입력 경로를 쓴다.
FP8→FP32 scale 곱→BF16 반올림과 expert별 group-16 양자화를 유지한다.

32,256행의 전체 BF16 입력 252 MiB/rank를 없앨 수 있다. 이는 제거한 중간
버퍼 크기이며 속도나 순 메모리 트래픽 감소를 뜻하지 않는다. router/expert/shared의
패킷 읽기와 역양자화 비용이 더 크면 채택하지 않는다. 기본값은 계속 OFF다.

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
