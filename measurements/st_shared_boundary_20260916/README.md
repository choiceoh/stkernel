# 가장 많이 공유되는 경계가 디스크에 갈 수 없었다 (2026-09-16, 재현)

운영자: "캐시나 티어 더 개선할게 있나."

## 1. 재현 — 리프만 티어에 간다

공유 프리픽스(블록 2개) 뒤로 갈라지는 대화 A·B 를 태우고, 한가할 때 스필을 다 돌린 뒤
(`tests/test_engine_prefix.runner`, BLOCK=4):

```
A's boundaries: [4, 8, 12, 16, 20]
leaf:           [20]
on the tier:    [20]
공유 경계 8:  entry=True  leaf=False  on the tier=False

B (8 에서 갈라짐) tier_lookup_chain → None
B 가 메모리에서 찾은 것 → 8 토큰
```

`is_leaf` 는 `children == 0` 이다. 공유 시스템 프롬프트 경계는 **누군가 이어 쓰는 순간
리프가 아니게 되고**, 그때부터 영원히 티어에 못 간다. 메모리에만 살고 재기동하면 사라진다.

`_victim` 은 그것을 메모리에서 가장 늦게 버리도록 이미 보호하고 있다 — *"one long prompt's
forty fresh boundaries cannot flush the system prompt every conversation shares"*. 그런데
**디스크로는 내보낼 수가 없었다.** 09-16 하루에 세 번의 재기동이 그것을 가져갔다.

## 2. 왜 리프만이었나, 그리고 왜 그것으로 부족한가

주석이 든 이유는 **블록**이다: *"a restored leaf brings every block of its chain back, an
inner boundary would bring the same ones."* 맞는 말이고, 요점이 아니다.

- 귀한 것은 블록이 아니라 스냅샷이다 — 같은 날 실측: `snapshot_denials` 0,
  `kv_blocks_cached` 111 / 1,398, `snapshot_self_evicts` **258**.
- **갈라지는 체인은 리프를 아예 못 쓴다.** 둘이 갈라지는 자리의 상태가 필요하고, 그건
  내부 경계다.

## 3. 고침 — 채택된 내부 경계도 후보다

`spill_candidates` 가 **리프 또는 `hits > 0` 인 내부 경계**를 내놓는다. `hits` 는 캐시가
이미 세고 있는 값이고(`lookup_chain` 이 채택할 때 올린다), 다른 프롬프트가 **실제로 여기서
시작했다**는 뜻이다. 중복은 그 값을 치른 경계로만 한정된다. 아무도 시작한 적 없는 내부
경계는 예전 규칙 그대로 리프의 블록을 다시 쓰는 것이므로 제외된다.

재현 뒤:

```
A 만 돌았을 때 on the tier: [20]          <- 8 은 아직 hits=0
B 가 8 을 채택한 뒤:        [8, 20]
메모리가 8 을 잃은 뒤 C 가 티어에서 찾는 것: (8, ...)   <- 예전 규칙에선 None
```

## 4. 고침 — 부팅 뒤 아무것도 캐시를 데우지 않았다

기구는 처음부터 다 있었다 — `POST /v1/prefix/warm`(messages / prompt / ids, `pin` 옵션)과
`probes/st_prefix_warm.py`. **부르는 데가 없었다.** 런처도 슈퍼바이저도 안 불렀다.

슈퍼바이저가 `wait_for_health` 가 **성공한 뒤에** 부른다(문만 열린 게 아니라 채팅이 답한
뒤). `--pin` 으로 데워서 `_victim` 이 그것들을 맨 뒤에 둔다. 워밍 실패는 부팅 실패가 아니다
— 그 시점에 플릿은 이미 건강하고, 안 데워진 캐시는 느린 것이지 고장난 것이 아니다.

- `ST_WARM_FILE` 기본값 `/home/choiceoh/glm53-logs/st-warm.jsonl`. 파일이 없으면 조용히
  건너뛴다. **`ST_WARM_FILE=` (빈 값)은 끈다** — `${x-d}` 이지 `${x:-d}` 가 아니다. 같은 날
  티어에서 배운 함정이다(PR #1042).
- `ST_WARM_TIMEOUT` 기본 600 초, `timeout(1)` 으로 감싼다.

**워밍은 대화를 남기면 안 된다.** `/v1/prefix/warm` 은 지금까지 불린 적이 없어서 드러나지
않았는데, 그 요청은 보통 턴처럼 처리된다 — 그리고 쓸모 있는 워밍 프롬프트는 한 블록 이상,
즉 `PARK_MIN_TOKENS = 128` 을 한참 넘으므로 **부팅마다 워밍 프롬프트 수만큼 파킹된 대화가
생긴다.** 슬롯 상태 통째(랭크당 ~260 MiB)가 64 GiB 대화 티어에 쓰이고 진짜 대화를 LRU 에서
밀어낸다 — 09-13 에 17 토큰 헬스 핑이 하던 것과 같은 일이, 30 초마다가 아니라 부팅마다.

`retain: false` 로는 못 고친다. 그 플래그는 **두 가지를 같이** 한다 — 행을 놓아주는 것과
`runner.transient` 로 **경계를 티어에 못 가게 하는 것**. 워밍은 앞은 원하고 뒤는 정반대다.
그래서 축이 둘이 됐다: `_warm` 은 `_retire` 에서 놓아주되 경계는 건드리지 않는다.

격리 확인(Git Bash 로 `warm_cache` 만 떼어 실행):

| | 결과 |
|---|---|
| 파일 있음 | `prefix cache warmed -- warmed 2 prompts`, argv 에 `--pin --url --timeout` |
| 프로브 실패 | `prefix warm did not finish (rc=3)`, 함수 rc=0 |
| 파일 없음 | 조용히 건너뜀, rc=0 |
| `ST_WARM_FILE=` | 조용히 건너뜀, rc=0 |

그 확인이 버그도 하나 잡았다: 처음 쓴 `warm_cache` 는 `$what` 을 인자로 받지 않고 호출자의
`local` 에 기대고 있었다. 슈퍼바이저는 `set -u` 다.

## 5. 아직 안 한 것

- **워밍 파일의 내용.** `/home/choiceoh/glm53-logs/st-warm.jsonl` 은 아직 없다. 무엇을
  데울지는 데네브가 실제로 보내는 프롬프트를 아는 사람의 결정이라 짐작하지 않았다. 파일이
  생기기 전까지 4 는 아무 일도 하지 않는다.
- **티어 예산 16 / 64 (경계 / 대화).** 경계 몫은 페이드가 드물다는 전제로 정해진 값인데
  `fades_total` 은 아직 0 이다. 티어가 실제로 도는 것을 보고 정할 문제다.
- **프로덕션 효과.** 1·4 가 지키려는 것은 "재기동을 건너뛴 적중" 이고, 한 번도 관찰된 적이
  없다. `prefix_tier_entries`·`prefix_tier_restores_total`·`prefix_cache_fades_total` 이
  답한다 — 지금 셋 다 0 이다.
