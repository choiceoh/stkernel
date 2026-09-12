# Mooncake 에서 배워올 것 — 조사 (2026-09-12)

> 그날의 조사 — **2026-09-12 의 조사다.** 그날 참이었던 것이고 유지되지 않는다 — 이후 무엇이 바뀌었는지는 `MEASUREMENTS.md` 가 안다.

`kvcache-ai/Mooncake` `2ccf4cb`(2026-09-11, 하루 전)를 sparse clone 으로 읽고 ST 의 접두사 캐시·NVMe 티어와
대조했다. 읽은 곳: `docs/source/design/{nvme-kv-backend,hicache-design}.md`, `mooncake-store/include/`
(`eviction_strategy.h`, `count_min_sketch.h`, `master_service.h`), 그리고 ST 쪽
`base/{prefix,kv_tier,runner,serve}.py`.

Mooncake 를 고른 이유: ST 가 가장 손으로 지은 부분이 접두사 캐시와 NVMe 티어이고(§32 가 바로 오늘 들어갔다),
그게 Mooncake 가 논문부터 지금까지 하는 일 그 자체다.

---

## 1. 발견 — 56비트 티어 키를 읽을 때 대조하지 않는다

ST 의 접두사 티어는 **경계 해시의 앞 7바이트**로 색인한다:

```python
# base/runner.py
def tier_key(h: bytes) -> int:
    return int.from_bytes(h[:7], "big")     # the tier indexes by int; 56 bits of the boundary's hash
```

그리고 복원은 그 56비트로 읽는다 — `promote(self.tier_key(h), ...)`. **레코드는 전체 해시를 갖고 있는데**
(`record = {"hash": h.hex(), "tokens": e.tokens}`, 부팅 때 `load_prefix_tier` 가 그걸로 `tier_keys` 를 재구성한다)
**읽을 때 그 해시를 요청한 `h` 와 비교하지 않는다.** 두 경계가 56비트에서 충돌하면:

- 쓰기: 나중 것이 앞 것의 블록·스냅샷 위에 발행된다(같은 `seq` 슬롯).
- 읽기: `h1` 을 복원하면 `h2` 의 KV 가 돌아온다. **조용히.**

Mooncake 는 **같은 문제를 정면으로 푼다**. NVMe KV 명령은 16바이트 물리 키인데 논리 키는 가변 길이라,
서로 다른 씨앗의 XXH64 둘로 물리 키를 만들고 넷으로 **신원 검증 해시**를 만들어 루트 헤더에 저장한다.
쓰기는 전부 store-if-not-exists 이고, 물리 키가 이미 있으면 **값을 읽어 바이트를 비교**한다 — 같으면 멱등 성공,
다르면 충돌로 보고 **다음 슬롯(최대 64개)** 을 쓴다. 읽기도 슬롯을 훑으며 저장된 논리 신원이 맞을 때까지 간다.

**확률은 희박하다** — 56비트의 생일 한계는 항목 2^28(약 2.7억) 근처이고 접두사 티어는 수천 개를 든다.
**그런데 실패 모드가 다른 테넌트의 KV 를 조용히 내주는 것**이고, ST 는 **바로 오늘**(§32, PR #609) 테넌트가
서로의 경계를 못 보게 하려고 테넌트 솔트를 넣었다. 솔트는 두 테넌트의 **해시를 다르게** 만들지만, 티어는
그 해시의 **56비트만** 본다. 검사에 필요한 데이터는 이미 디스크에 있고, 드는 값은 비교 한 번이다.

**제안**: `restore_begin` 이 `prefix_tier.record(key)` 를 읽어 `record["hash"] != h.hex()` 면 거부하고 보통
입장으로 보낸다. 실패 경로는 이미 있다(아래 §2.2). Mooncake 의 64 슬롯까지는 필요 없다 — 우리 규모에서
충돌은 "이 항목을 버린다"로 충분하다.

---

## 2. 이미 맞는 것 (확인함)

| 항목 | Mooncake | ST |
|---|---|---|
| **반쯤 쓴 것이 보이면 안 된다** | 청크를 전부 쓴 **뒤에만** 루트 매니페스트를 쓴다. 루트가 가시성 표식이다 | 같다. `kv_tier._save_manifest` 가 tmp + `fsync` + `os.replace` — *"The manifest rename is the sole publication point"*. 파일시스템으로 같은 성질을 얻는다 |
| 실패한 쓰기의 뒷정리 | 이번 시도가 만든 키만 best-effort 삭제, 이미 있던 것은 절대 안 지운다 | 같다. 발행된 세대는 write/fsync/manifest 실패를 지나도 안 건드리고, 부분 unlink 는 툼스톤이 넘긴다 |
| 축출 정책 | `LRUEvictionStrategy` / `FIFOEvictionStrategy` — **평범한 LRU** 다 | ST 는 등급 있는 자유 목록 + LRU(§32). **여기서는 우리가 더 다듬어져 있다** — 배울 것이 없었다 |
| L3 메타데이터를 동기화하지 않는다 | HiRadixTree 는 L3 메타데이터를 들지 않고 접근할 때 백엔드에 **실시간 질의**한다 | ST 는 `tier_keys` 를 메모리에 들고 부팅 때 재구성하며 네 랭크가 `have` 로 투표한다. **실시간 질의가 애초에 불가능하다** — 랭크는 메시지 없이 합의해야 한다(그게 해시를 쓰는 이유다) |

---

## 3. 가져올 것

### 3.1 복원에 마감이 없다 (HiCache 의 세 전략)

ST 의 복원은 티어 스레드가 읽고, 문이 `transfer_done` 을 랭크마다 투표해 전부 끝났을 때 `restore_finish` 를
부른다. **스텝 루프는 디스크를 안 기다린다**(D10 ✓). 하지만 **포기하지도 않는다** — `future.result()` 에
타임아웃이 없고, 느리거나 멈춘 NVMe 읽기는 행과 그 행이 예약한 블록과 스냅샷 하나를 **무기한** 붙잡는다.

HiCache 는 이 문제를 정면으로 다룬다. 프리페치 종료 전략이 셋이고(`best_effort` / `wait_complete` / `timeout`),
**실전에서 쓰는 것은 `timeout`** 이라고 적어 뒀다 — 이유가 둘이다: 프리페치 지연은 본질적으로 예측 불가이고,
마감은 **SLO 가 정한다**. 그리고 마감이 크기를 안다:

```
timeout = prefetch_timeout_base + prefetch_timeout_per_ki_token * num_token_to_fetch / 1024
```

**ST 에는 복구 경로가 이미 있다**: `restore_finish` 가 실패하면 행을 놓고, 스냅샷을 돌려주고, 못 읽는 티어
사본을 버리고, 요청은 보통 입장으로 간다. **시계만 연결하면 된다.**

### 3.2 복원할 가치의 하한

HiCache 는 L3 히트가 **256 토큰을 넘을 때만** 프리페치한다. 그보다 짧으면 읽는 값이 다시 계산하는 값보다 크다.
ST 는 경계면 크기와 무관하게 복원한다. 경계는 블록 단위라 아주 짧지는 않지만, **하한이 없다는 사실 자체가
한 번도 판정된 적 없다** — 블록 하나를 NVMe 에서 읽는 값과 그 블록을 프리필하는 값의 교차점이 어디인지.

---

## 4. 의도적으로 다른 것

**L2(호스트 메모리) 티어가 없고, 있을 수 없다.** HiCache 는 CPU 캐시를 본떠 L1 = GPU 메모리, L2 = 호스트 메모리,
L3 = 분산 저장으로 나눈다. GB10 은 **CPU 와 GPU 가 같은 119.7 GiB 를 쓴다**(헌장 §0) — 호스트로 옮기는 것은
같은 메모리 안에서 옮기는 것이라 층이 아니라 복사다. ST 의 층은 아레나와 NVMe 둘뿐이고, 그건 하드웨어가
정한 것이지 빠뜨린 것이 아니다.

**분산 저장이 없다.** Mooncake 의 L3 는 클러스터가 공유하고 마스터가 복제·테넌트 쿼터·HA·핫스탠바이를 든다
(`master_service.h` 하나가 2,300줄이다). ST 는 **한 배치, 네 랭크, 로컬 NVMe** 다. 공유할 다른 인스턴스가 없다.

---

## 5. 안 가져올 것

**NVMe KV 명령 집합** — Mooncake 는 파일시스템을 건너뛰고 `io_uring`/`ioctl` 로 NVMe KV Store/Retrieve/Delete 를
직접 낸다. ST 의 티어는 보통 파일 I/O 이고, 그 차이는 우리 규모에서 측정된 적이 없다. **가져오려면 먼저 재야 한다** —
지금 티어 읽기가 병목이라는 증거가 없다.

**count-min sketch·핫키 복제·테넌트 쿼터** — 전부 여러 인스턴스가 한 저장소를 나눠 쓸 때의 기계다.

**CacheLib 할당자** — ST 의 아레나는 D1 이 정한 대로 명시적 예약이고, 범용 할당자는 그 반대 방향이다.
