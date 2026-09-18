# Qwen3.8 QSA 접기 — GB10 단일 GPU 레인 판정 (2026-09-18)

> 그대로 두는 기록 — 이 날 srv4 단일 GPU 레인에서 돈 캐리 Q8 · Q10 · Q11(`engine/QWEN38_CARRY.md`) 서빙 커널 케이스의 결과와 원시 로그다.
> 뒤 실행은 이 파일을 고치지 않고 새 기록을 남긴다.

속도 주장은 없다(헌장 D17). 전부 **정합 판정**이다: 새 발사가 기존 발사와 바이트 단위로 같은가.
체크포인트 없이 `probes/qwen38_config.json` 의 형상으로, 합성 가중치와 입력을 썼다.

| 티켓 | 트리 | 프로브 | 결과 | 로그 |
|---|---|---|---|---|
| `qwen38-qsa-folds2-0918` | `af6032de`(PR #1196 의 머리, #1193 위) + 큐 수정 #1198 체리픽 = `660abb4a` | `engine_kernel_check.py --lanes qwen38_cells` | qualify 통과, GPU 케이스 **51 건 중 50 통과, 1 실패**(테스트의 단언이 설계보다 셌다 — 아래 2) | [cells-660abb4a.log](cells-660abb4a.log) |

장치: NVIDIA GB10(sm_121a), 이미지 `st-engine:glm53`, 프로덕션 `st-glm53` 옆(예산 8 GiB, 플릿 임대 없음). 테스트 410 s.

## 1. 통과한 것 — 새 발사는 GB10 에서도 기존 발사의 바이트다

- **캐리 Q8** `_qsa_mqa_paged_group_kernel`(한 요청의 연속 행 최대 4 개가 인덱스 키 타일을 한 번 읽는다): `GroupScoreTests` 4 건 전부.
  캡처 스텝 모양(행당 2·3·4 토큰, 묶음 안에서 행들의 horizon 이 그룹 경계를 사이에 두고 다름), 마지막 묶음이 짧은 프리필 세그먼트,
  32 행 발사 기하를 넘는 행 수, 두 선택 진입점 — 행 발사 대비 logits·visible 수·선택이 바이트 동일.
- **캐리 Q10** `_qsa_covered_paged_gqa_kernel`(예산이 덮는 스텝의 dense causal 어텐션): `CoveredAttentionTests` 3 건 전부.
  모델 실폭(6 헤드 × 256, 예산 2048, 2,051 행까지)에서 세그먼트 여럿인 스텝, 행 수가 닿는 모든 split 프로필(64·32·8·4·1), 묶음 1..4,
  KV 헤드 1·2, 게이트 유무 — sparse 발사(`qsa_sparse_paged_attention_blocks`) 대비 `torch.equal` + raw bits.
- **캐리 Q11 덮인 절반**: `test_engine_qwen38_covered_blocks.ServedSelectionTests` — 덮인 스텝에 대한 `qsa_select_paged_blocks` 의 점수 선택을
  어텐션이 읽는 순서로 놓으면 위치만으로 만든 id 와 바이트 동일.
- 같은 실행의 나머지 42 건(KDA decay, MHCV41·MLA 글루, `gated_sum`, `swiglu(pad_to)`, GDN 링 게이트, QSA 입력 융합, 어텐션 게이트,
  블록 어텐션, MoE 라우트, K3 의 네이티브 chunk)도 전부 통과. qualify 값은 `qwen38_lane_20260917` 의 것과 같다
  (게이트 잔차 max 0.0063, QSA norm+rope 6×256 0.0062 / 4×128 0.0069).

## 2. 실패 1 건 — 발견: radix select 의 행 안 순서는 행의 것이 아니라 발사의 것이다

`test_engine_qwen38_query_shards.ServedSelectionTests.test_the_split_selection_is_the_whole_ones_bytes`(캐리 Q11 랭크 절반).
GB10 에서만 도는 넓은 스텝(400 행, 도달 범위 11 행 전에서 시작)이다: 분할하지 않은 호출과 네 랭크의 100 행 호출이 **양쪽 모두 네이티브
radix select**(`engine/kernels/prefill_topk`)를 탄다 — 다른 박스에는 그 빌드가 없어 이 케이스를 돌릴 수 없다.

- **통과한 단언:** 양쪽을 어텐션이 읽는 순서(오름차순, 캐리 Q6)로 놓으면 같다 → 모든 행의 **집합**이 같다. 분할 규칙이 주장하는 것이 이것이고,
  어텐션이 읽는 것도 이것뿐이다(블록을 자기 프로그램 안에서 정렬한다, #1110).
- **실패한 단언:** 덮인 행을 지난 뒤로는 id 대 id 로(선택기가 남긴 순서 그대로) 같다 — 이건 내 테스트가 설계보다 세게 단언한 것이다.
  radix select 는 승자를 bin 이 차는 대로 쓴다: 같은 행의 id 가 랭크의 100 행 발사와 스텝의 400 행 발사에서 **다른 순서**로 나왔다.
  `torch.topk`(인터프리터, 다른 GPU, 64 행 이하)는 행을 그 행만으로 정렬하므로 거기서는 id 대 id 비교가 성립한다 — WSL 인터프리터와
  RTX 5050 에서 이 단언이 통과했던 이유다.

엔진 코드에는 영향이 없다: 선택기가 남긴 순서대로 블록을 읽는 곳은 없다. 테스트를 고쳤다(`fa725053`): id 대 id 비교는 `torch.topk` 경로에서만,
GB10 의 넓은 스텝은 집합으로. 재실행은 아래 3.

## 3. 재실행

`qwen38-qsa-folds3-0918`(트리 `adb2a624` + #1198 체리픽 = `9d784bc7`)을 21:04 에 제출했다. 이 기록을 쓰는 시점에는 레인이 **방을 기다리는 중**이다:

```
waiting: pos 1/1, srv4: no room beside production -- MemAvailable 24.0 GiB, this check's budget 8.0 GiB, floor 16.0: 16.0 GiB would be left
```

프로덕션 옆이라 예산(`ST_PROBE_GIB`)을 낮춰 밀어 넣지 않았다. 결과는 새 기록으로 남긴다.

## 4. 이 실행이 찾은 큐 결함

첫 제출(`qwen38-qsa-folds-0918`)은 접수 전에 죽었다: `bench/fleet_prepare.py` 의 `create` 가 #1152 에서 사라진 `approve_deploy` 를
`prepare()` 에 넘겨 TypeError. main 의 체크아웃에서는 `fleet.sh run` 으로 어떤 티켓도 제출할 수 없었다 → #1198.
시작에 실패한 세션명은 중복 제거 키로 남아 같은 이름의 재제출은 옛 실패 기록만 돌려준다 — 새 이름(`-folds2-`)으로 냈다.
