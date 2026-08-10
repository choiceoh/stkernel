# 정비 창 런북 — 2026-08-11 원장의 "운영자 결정 대기" 일괄

대상: MEASUREMENTS.md ★★UMA 챕터의 잔여 후속 + 플릿 위생 관찰 + 부수 확정의
RoCE GID 항목. **모든 적용 명령은 사람이 실행한다** — 자동화는 읽기 전용
감사(`launchers/fleet-audit.sh`, srv2에서 실행)까지만.

실행 전/후로 감사를 떠서 diff로 남길 것:

```bash
bash launchers/fleet-audit.sh | tee /tmp/fleet-audit-$(date +%m%d-%H%M).txt
```

## 항목별 절차

### 1. UMA 컴팩션 sysctl 영구화 (4노드) — 근거: 프리필 간헐 −15% 요동, pgmigrate 1:1 인과

현재 srv3·srv4에 **런타임으로만** 적용돼 재부팅 시 증발한다. 4노드 전부
영구화한다 (srv1·srv2는 현재 건강하지만 같은 워크로드라 예방 동일 적용):

```bash
for ip in 10.10.10.2 10.10.10.3 10.10.10.1 10.10.10.4; do
  ssh choiceoh@$ip 'echo "vm.compaction_proactiveness = 0" | sudo tee /etc/sysctl.d/99-uma-compaction.conf && sudo sysctl --system | grep compaction_proactiveness'
done
```

롤백: 파일 삭제 후 `sudo sysctl -w vm.compaction_proactiveness=20`(커널 기본).

### 2. THP defrag 정책 — **판정: A안 채택 (2026-08-11 정비 창, 위임 실행)**

현상 유지 + `fleet-audit.sh` 카운터 감시. thp_fault_alloc/compact_stall 증가
재개 시 B(defrag=never 영구화 + 프리필 무회귀 A/B) 재론. 원 결정 박스:

`defrag=[madvise]`가 살아 있어 direct-compaction 스톨로 재발할 수 있다
(원장: "thp_fault_alloc 증가 재개 시"). 선택지:

| 선택 | 명령(부팅 시 적용 필요 — sysctl 아님) | 트레이드 |
|---|---|---|
| A. 현상 유지 + 감시 | 없음 — `fleet-audit.sh`의 `thp_fault_alloc`/`compact_stall` 카운터 주기 확인 | 재발 리스크 잔존, 조치 최소 |
| B. defrag=never 영구화 | `/etc/systemd/system/thp-defrag.service` (oneshot, `echo never > /sys/kernel/mm/transparent_hugepage/defrag`) 4노드 | direct 스톨 원천 차단 · THP 성립률 하락(UMA에서 TLB 영향은 미실측) |

B를 고르면 A/B 재기동 1회로 프리필 무회귀 확인 후 확정 (원장 등재).

### 3. RoCE GID 정리 (IPv6로 인한 노드별 인덱스 상이) — 포럼 #378890 지목 취약점

현상: srv1=3, srv2·srv4=4 (IPv6 활성 탓에 GID 테이블 구성이 노드/부팅마다
다름). 런처 자동 탐지로 동작엔 문제없으나, 탐지 실패 시 NCCL이 잘못된
GID로 폴백할 수 있는 표면이다. 정리 = RoCE 인터페이스의 IPv6 비활성 →
테이블이 IPv4 RoCEv2 항목으로 수렴, 인덱스가 노드·부팅 불변이 된다:

```bash
for ip in 10.10.10.2 10.10.10.3 10.10.10.1 10.10.10.4; do
  ssh choiceoh@$ip 'printf "net.ipv6.conf.enp1s0f0np0.disable_ipv6 = 1\nnet.ipv6.conf.enP2p1s0f0np0.disable_ipv6 = 1\n" | sudo tee /etc/sysctl.d/99-roce-noipv6.conf && sudo sysctl --system | grep disable_ipv6'
done
```

주의: **엔진 가동 중 적용 금지** (GID 재배열 → 활성 NCCL QP 문맥과 어긋남).
정지 상태에서 적용 → 감사로 4노드 GID 인덱스 동일 확인 → 기동. 런처의
자동 탐지 로직은 그대로 두는 게 정답 (벨트+서스펜더). 롤백: 파일 삭제 +
재부팅(또는 disable_ipv6=0).

### 4. srv1 gpu-clock-cap.service 설치 — 근거: 2398–2444MHz 무캡 가동 (나머지 1976–1989)

TP 동기라 성능 이득 0, 열·의도 위반만 남는다:

```bash
ssh choiceoh@10.10.10.2 'systemctl cat gpu-clock-cap.service' > /tmp/gpu-clock-cap.service
# [Service] 내용 확인 후:
scp /tmp/gpu-clock-cap.service choiceoh@10.10.10.1:/tmp/
ssh choiceoh@10.10.10.1 'sudo cp /tmp/gpu-clock-cap.service /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl enable --now gpu-clock-cap.service && nvidia-smi --query-gpu=clocks.sm --format=csv,noheader'
```

기대: srv1 SM 클럭 ~2000 수렴. (2200 재도전은 운영자 거부권 유지 — 재론 금지.)

### 5. 잔존물 정리

- **좀비 vllm-tp2** (srv1·srv2, ray GCS 연결 실패 루프 — restart 정책이 살아
  있어 미래 메모리 경합 위험):
  ```bash
  for ip in 10.10.10.2 10.10.10.1; do ssh choiceoh@$ip 'docker rm -f vllm-tp2 vllm-tp2-worker 2>/dev/null; true'; done
  ```
  (컨테이너가 다른 프로젝트 소유물이면 rm 대신 `docker update --restart=no`로 무해화.)
- **bluetoothd CPU 폭주** (srv2·srv3, 누적 2–3h — GPU 무관 확인, 코어 낭비):
  ```bash
  for ip in 10.10.10.2 10.10.10.3; do ssh choiceoh@$ip 'sudo systemctl disable --now bluetooth.service'; done
  ```
- **srv4 stash 드랍** (main에 전량 흡수 확인된 구본):
  ```bash
  ssh choiceoh@10.10.10.4 'git -C ~/stkernel stash drop stash@{0}'
  ```

### 6. 플릿 재부팅 — 고차 페이지 재건 (proactiveness=0은 억제일 뿐 치유 아님)

순서:

1. 위 1(+2B 선택 시)·3을 먼저 배포 — 재부팅이 곧 적용·검증 기회.
2. `systemctl --user stop dsv4-tp4` (srv2) — 슈퍼바이저가 재부팅 중 헛돌지 않게.
   **★함정(08-11 실사고): enabled user unit이라 재부팅이 정지를 무효화한다 —
   srv2 재부팅 후 자동 재기동돼 기준 스택을 자동 발사, 이후 A/B와 경합.
   재부팅 직후 반드시 재정지하고, A/B 셀은 컨테이너 env 의도검증을 내장할 것.**
3. 워커(srv3·srv1·srv4) 재부팅 → srv2 재부팅.
4. **링크 트레이닝/FEC 정착 2분 이상 대기** 후 감사 (★원장 함정: 40초 뒤
   측정한 169.9는 아티팩트).
5. 합격 기준: `fleet-audit.sh`에서 proactiveness=0 persisted ×4 · GID 인덱스
   4노드 동일 · np0 speed=200000Mb ×4 · buddy order10 > 0 (재건 확인) ·
   clock-cap 4노드 enabled.
6. `systemctl --user start dsv4-tp4` → 워밍업 후 검증: `bench-tp4` ×3
   (기대 대역 2,866–2,872), `check-quality.py` 9/9, 필요시 ib_write_bw 196
   스팟 체크. KV 수치 비교 시 GPU_MEM 명기 룰 준수.

## 기록

적용 결과(감사 전/후 diff + bench/품질 수치)는 MEASUREMENTS.md에 등재하고,
이 런북의 결정 박스(2번 THP)는 선택지를 지우고 판정으로 치환할 것.
