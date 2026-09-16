# 접두사 캐시는 이미 프롬프트의 절반을 덜고 있었다 — 병목은 블록이 아니라 스냅샷이다 (2026-09-16, srv2)

운영자: "티어나 캐시 개선할게 있나."

앞선 기록(`st_deneb_tier_kv_20260916`)은 "100 질의에 17 적중" 을 낮은 적중률로 읽었다.
**그 지표가 틀렸다.** 질의는 프롬프트 하나고, 적중 하나가 수만 토큰을 덮는다.

## 1. 실측 — 라이브 카운터 (11:09 부팅, 티어 off · `ST_KV_GIB=7.0`, 148 요청)

| | 값 | |
|---|---|---|
| `vllm:prompt_tokens_total` | 1,796,682 | |
| `st:prefix_reused_tokens_total` | **920,832** | **51.3 %** |
| `vllm:prefix_cache_queries_total` / `hits_total` | 149 / 19 | |
| `st:prefix_entries` | **96 / 96** | 포화 |
| `st:prefix_cache_evictions_total` | 1,109 | |
| `st:prefix_snapshot_self_evicts_total` | **258** | |
| `st:prefix_snapshot_denials_total` | 0 | |
| `st:prefix_cache_fades_total` | 0 | 티어가 없으니 페이드가 불가능 |
| `st:kv_blocks_cached` / `free` | 111 / 1,398 | |
| `st:prefix_tier_entries` | **0** | 예산 16 GiB |
| `st:state_slots_total` | 2 | 동시 행 둘 |

평균 프롬프트 **12,140 토큰** (1,796,682 ÷ 148) ≈ 블록 경계 **16 개**.

## 2. 판정 — 블록은 남고 체크포인트가 모자란다

- **블록이 아니다.** 캐시가 쥔 블록은 1,398 중 **111**. `denials` 0.
- **스냅샷이다.** `self_evicts` **258** — 긴 프롬프트 하나가 자기 **뒤쪽** 경계를 만들려고
  자기 **앞쪽** 체크포인트를 버린다. `prefix.py` 가 이미 그 자리에 이름을 붙여 뒀다:
  *"a long prompt has more block boundaries than there are slots (a 128K prompt: 170 against 96)
  ... the checkpoints it throws away were still computed."*
- **`fades` 0** 이 티어의 값어치다. 티어가 없으면 희생자는 페이드(블록 유지, 스냅샷 양보,
  상태는 NVMe)가 아니라 그냥 죽는다.

## 3. 그런데 티어를 켜면 상주가 반이 된다

`PREFIX_SNAPSHOT_GIB 2.125`(48 장) vs `PREFIX_UNTIERED_SNAPSHOT_GIB 4.25`(96 장). 티어와
압축 RAM 캐시가 상주를 대신한다는 설계였는데, **측정은 상주가 병목이라고 말한다.** 그러면
켜는 순간 병목이 반으로 준다. 48 + 티어 ≈ 30 이 96 보다 낫다는 근거는 없었다.

## 4. 바꾼 것

- **B — 한가할 때 쓴다.** 스필이 `spill_low_water`(8) 아래에서만 걸렸다. 티어를 켜면
  스냅샷이 48 장이니 첫 바이트가 디스크에 닿으려면 경계가 40 개 상주해야 하고, 하루에 세 번
  다시 올라가는 플릿은 거기 닿지 못한다 — `prefix_tier_entries` 0 이 그 결과다. 스케줄러가
  계획할 것이 없으면(`nothing_to_step`) 미룰 이유가 없다. 스필은 티어 제 스레드에서 돌고
  D10 이 이미 스텝을 막지 않음을 보장한다.
  - 같이 고침: `spill_end` 가 `version` 을 올린다. 안 그러면 하나 쓰고 나서 캐시에 무관한
    변화가 생길 때까지 다음을 안 찾는다 — 압박 경로에서도 그랬다.
- **A — handover 가 경계도 남긴다.** `_hand_over` 는 대화를 파킹하지만 경계는 두고 갔다.
  `flush_prefix(deadline)` 가 기한 안에서 기다리며 쓴다. `maintain_prefix` 는 절대 안
  기다린다(스텝이 기다리면 안 되니까) — 그래서 **아무도** 안 기다렸다.
- **D — 티어를 켜도 96 장.** `PREFIX_SNAPSHOT_GIB` 2.125 → 4.25.

**A 를 SIGTERM 으로 하지 않은 이유.** `base/record.py` 의 SIGTERM 핸들러는 링을 쓰고 **즉시
같은 신호로 죽는다**. 그 신호의 의미가 earlyoom 이기 때문이다(D12: 여섯 번의 죽음이 전부
earlyoom 의 SIGTERM). 상자가 OOM 으로 가는 중에 NVMe 에 몇 초를 쓰는 것은 정확히 반대다.
그래서 graceful 종료는 신호가 아니라 **요청**이어야 하고, 이 리포에는 그 기구가 이미 있다 —
리스의 yield 요청과 `_hand_over`.

## 5. 예산 확인 (배포 트리 `e223ae4222f8`, KV 14.0, 장치 없이)

| snapshots | KV remainder | declared paged KV | unassigned |
|---|---|---|---|
| 48 (전) | 45.06 GiB | 13.16 GiB | +31.90 GiB |
| **96 (후)** | **42.94 GiB** | 13.16 GiB | **+29.78 GiB** |

여유 안이다. 덧붙여 — remainder 42.94 GiB 에 13.16 만 선언돼 있으니 **KV 는 14.0 보다
훨씬 키울 수 있다.** 이 PR 이 다루는 것은 아니다.

## 6. 미측정

- **tiered-48 대 tiered-96.** D 는 "반으로 줄여도 된다"는 가정을 **거부**한 것이지, 96 이
  옳다는 측정이 아니다. 자르려면 재고 자른다.
- **프로덕션에서 티어가 차는 속도와 그 뒤의 복원.** B 가 들어간 뒤 `prefix_tier_entries` 와
  `prefix_tier_restores_total` 이 답한다.
- **재기동을 건너뛴 적중률.** A·B 가 지키려는 것이 그것이고, 아직 한 번도 안 지켜졌다.
- **`stop` 은 여전히 SIGKILL 이다**(`docker rm -f`). 아픈 플릿도 반드시 멈춰야 하므로
  그대로 뒀다. A 가 덮는 것은 yield 를 통한 handover 뿐이다.
