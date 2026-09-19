# 파킹은 슬롯 통째가 아니라 이어 쓰는 데 필요한 바이트만 — GLM-5.3 286 → 48.0 MiB, Qwen3.8 109 → 28.3 MiB (2026-09-19)

## 질문

NVMe 대화 티어(랭크당 64 GiB)는 프로덕션에서 GLM-5.3 대화 223개로 꽉 차 있었다(63.80 GiB). 대화 하나가 쓰는 것의 대부분은
무엇이고, 압축 없이 줄일 수 있나? 줄이면 이어 쓰기(재개)의 결과가 그대로인가? 같은 날 #1298 로 Qwen3.8 도 같은 티어에 파킹한다.

## 트리와 장소

- 비교: main `7b0248c3`(#1298 뒤, 슬롯 통째 파킹) 대 이 PR.
- 슬롯 구성: 저장소의 `layout()`·`snapshot_layout()` 으로 계산. GLM-5.3 은 서빙 메타의 `config.json`(srv2
  `~/st-main-d569a915/build/st-glm53-meta/`)을 넣고 프로덕션 매니페스트의 슬롯 바이트(299,932,672)와 맞춰 봤다(드래프터 링은 그 차이).
  Qwen3.8 은 저장소가 고정한 설정(`probes/qwen38_config.json`)으로, 서빙 K=3 슬롯 114,645,248 B 는 #1298 의 숫자와 같다.
- 프로덕션 티어: srv2 rank 0, 2026-09-19 20:30:20, 매니페스트와 파일 크기만 읽음(`measurements/parked_record_bound_20260919`).
- 테스트·커널: `stk-test`(Mac 의 docker, CPU). 커널은 Triton 인터프리터(`TRITON_INTERPRET=1`)로 실제 KDA·GDN 링 커널을 돌렸다.

## 어떻게 (저장소 뿌리에서, stk-test 안)

```
python3 measurements/park_live_state_20260919/slot_breakdown.py <GLM config.json> 299932672
python3 measurements/park_live_state_20260919/qwen38_slot_breakdown.py 3
TRITON_INTERPRET=1 python3 measurements/park_live_state_20260919/wrong_cell.py
python3 -m unittest -v tests.test_engine_park_live_state
TRITON_INTERPRET=1 python3 -m unittest -v tests.test_engine_park_live_state.KdaRingTests tests.test_engine_park_live_state.GdnRingTests
```

원시 출력은 [`raw.txt`](raw.txt).

## 결과

**1. 슬롯의 대부분은 초안을 되돌리려고 둔 상태다.**

| 슬롯(랭크당) | 바이트 | `rec`(K+1 칸) | 그 밖 |
|---|---:|---:|---|
| GLM-5.3 (K=7, KDA 34층 × 8칸 × 16헤드 × 128 × 128 × fp32) | 299,932,672 | 285,212,672 (95.1%) | 드래프터 링 10,485,760, conv 4,177,920, 꼬리 56,320 |
| Qwen3.8 (K=3, GDN 층 × 4칸) | 114,645,248 | 113,246,208 (98.8%) | conv 1,105,920, PLE conv 266,240, QSA 키 26,624, PLE id 64 |

링은 위치로 주소를 매긴다. 스텝은 `(context-1) % (K+1)` 한 칸만 읽고(`engine/kernels/state._read_rec`,
`kda/fused_recurrent`: `(max(context-1, 0) % RING_SIZE)` — KDA·GDN 링 커널이 같은 커널, 실험용 `kda/deferred` 의 `previous` 도 같은 칸,
두 넷의 프리필은 `rec[(ctx-1) % wr]`), 계산하는 위치마다 쓴다. prefix 경계 복원(`StateRings.load_rings`)이 이미 이 규칙에 기댄다 —
`open` 이 슬롯을 0으로 지우고 한 칸만 되돌린다.

**2. 이어 쓰는 데 필요한 것만 담으면**: 링마다 살아 있는 한 칸 + 나머지 필드 통째.

| | 통째 | 필요한 것만 | 비 |
|---|---:|---:|---:|
| GLM-5.3 | 299,932,672 | 50,371,584 (48.0 MiB) | 5.95배 |
| Qwen3.8 | 114,645,248 | 29,710,400 (28.3 MiB) | 3.86배 |

prefix 스냅샷(GLM 45.2 MiB, Qwen 29,049,088 B)과 거의 같다(스냅샷은 conv 를 conv-1 칸만, 경계라 꼬리·키 링이 비어 있음).

**3. 같은 결과인가 — 실제 링 커널(인터프리터).** 파킹한 링(모든 칸)과 재개한 링(살아 있는 칸 + 나머지 0)에 같은 스텝을 돌리면 출력과
쓴 칸이 바이트까지 같다: KDA 한 행 `recurrent_kda_ring`(토큰 1·3, 위치 1·7·8·13), 여러 행 `recurrent_kda_ring_rows`, GDN
`recurrent_gdn_ring`(4칸, 토큰 1·3, 위치 1·3·4·9). 틀린 칸이나 빈 링으로 되돌리면 달라진다(`wrong_cell.py`: 살아 있는 칸 True, 다음 칸
False, 없음 False) — 테스트가 차이를 잡는다.

**4. 용량과 옮기는 양 (계산).** GLM-5.3 프로덕션의 223개(평균 KV 7.3 MB)는 63.80 GiB → 11.97 GiB, 같은 평균이면 64 GiB 에
223.7 → 1,192개(5.3배); 파킹·재개마다 슬롯 바이트 286 → 48 MiB, 드라이브 순차 5.1 GB/s 로 58.8 → 9.9 ms. Qwen3.8 대화는 한 블록이면
125.5 → 40.6 MB(3.09배), 10K 토큰(14 블록)이면 266.7 → 181.8 MB(1.47배: KV 가 커진다).

## 못 잰 것

- **GPU·플릿에서 돌리지 않았다.** GLM-5.3 은 플릿 창의 `probes/engine_full_check.py`(서버를 통한 파킹·재개의 바이트·시간·답 품질)나
  랭크 파일 넷이 있는 노드의 `PYTHONPATH=. python3 engine/profiles/glm53/boot.py --local --layers 0-4 --park`(실캐시 파킹 → 재개 →
  4토큰이 한 번에 돌린 것과 같은지). Qwen3.8 은 플릿의 파킹·재개가 아직 한 번도 GPU 에서 돌지 않았다(#1298 도 GPU·플릿 미검증).
- 실제 파킹·재개 시간(9.9 ms 는 계산), 티어에 실제로 들어가는 대화 수(1,192 는 지금 평균 길이로 계산).
- fp32 상태에 zlib 같은 코덱을 더 얹었을 때의 비율(아무도 재지 않았다; #783 압축 캐시에도 비율 기록 없음).
