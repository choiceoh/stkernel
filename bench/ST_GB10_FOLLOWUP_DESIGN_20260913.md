# GB10 × 4 실행 형상의 후속 설계

설계 기준: PR #895의 `56e3f3d8`와 그 기반 `71306bda`. [상위 설계](ST_GB10_ARCHITECTURE_20260913.md)의 후속 작업을 코드 경계와 검증 단위로 구체화한다. **S의 cache·eager·graph 연결과 P의 packet-native FFN은 기본값이 꺼진 실험 옵션으로 구현했다. M/I는 후속 작업이다.** 커널의 GPU 검증도 예약 당시 대기 상태이며, 아래 설계를 성능 결과로 해석하지 않는다.

S 구현: `STK_compact_kda=1` → `ExecutionPlan.compact_kda` → `Facts.kda_state_layout="committed_boundary"`. deferred 검증을 함께 선택하며 native FP32, 분할 없는 decode, `prefill_tiles=1`을 요구한다. `rec_meta[2]`에는 current/boundary의 문맥 위치를 저장하고 물리 slot과 함께 이동한다. eager transaction이 미확정인 동안 재검증·snapshot·restore·slot 재사용을 거부한다. graph 수명과 slot 재사용의 stream fence는 기존 runner가 소유한다. [compact_state.py](../engine/profiles/glm53/compact_state.py), [CPU 검증](../measurements/kda_compact_20260913/serving_cpu.json). 아래 상세 설계의 이름은 구현 API와 다를 수 있다.

목표는 단일 요청의 품질·수락률·출력 tok/s를 유지하면서, 정해진 TP4 모델의 불필요한 상태 저장과 프리필 입력 이동을 없애는 것이다. 이후 디코드가 이미 읽는 expert 가중치에 프리필 계산을 함께 실어 본다. 기존 커널 선택·수치 경계·통신 순서를 함께 소유하므로 가능한 변경이다. 알고리즘 자체의 독점성을 주장하지 않는다.

## 결정과 구현 순서

| 단계 | 바꾸는 경계 | 첫 구현의 범위 | 완료 기준 |
| --- | --- | --- | --- |
| S | KDA 상태 → cache/prefix/graph | compact ABI를 모든 상태 생산·복원 경로에 연결 | 기존 FP32 상태·출력과 일치, 같은 KV 용량에서 실제 arena 감소 |
| P | FFN all-gather → 세 입력 소비자 | 전체 FP8 패킷 한 번을 router·expert pack·shared gate/up이 직접 소비 | 전체 BF16 hidden 제거, 같은 통신 횟수에서 실제 TTFT 개선 |
| M | decode expert 타일 → 준비된 prefill route | 기존 M32 타일 수를 늘리지 않는 혼합 계산 | C=1 지연을 보호하면서 완료된 프리필 작업 증가 |
| I | 위 경로 → 부팅 실행 이미지 | 기존 shape/cache/budget 선언에 물리 ABI와 소스 의존성 연결 | 필요한 형상만 빌드, 관련 없는 수정의 재컴파일 방지 |

S와 P는 개별 채택할 수 있다. M의 첫 component probe는 BF16 입력으로 시작할 수 있으므로 P 구현을 기다릴 필요가 없다. M의 실제 서빙에는 별도의 층 단위 스케줄러와 수치 검증이 필요하다. I는 각 단계가 안정되는 순서대로 기존 실행 계획에 반영한다.

## S. Compact KDA를 서빙 상태로 연결

### S1. 슬롯 배치와 예산

현재 [caches.py](../engine/profiles/glm53/caches.py)의 `layout()`은 `rec[K+1,H,D,D]`를 만들고, 별도 stage에도 recurrent 상태 한 장을 둔다. `cache_capacity()`는 FP32 기준 KV 블록·snapshot 개수를 고정한다. 새 저장 형상도 **그 기준 용량을 먼저 계산한 뒤 실제 바이트만 줄인다.** 절감분을 자동으로 더 큰 KV나 추가 동시 요청으로 채우지 않는다.

제안 인터페이스는 `layout(..., recurrent_layout="ring"|"committed_boundary")`다. dtype과 저장 형상을 분리하며 첫 compact 구현은 FP32만 허용한다.

| 슬롯의 필드 | 제안 형상 | 소유자·쓰기 시점 |
| --- | --- | --- |
| `rec` | `[1,H,D,D]`, FP32 | prefill 완료·restore 또는 accepted commit |
| `rec_boundary` | `[H,D,D]`, FP32 | commit이 prefix 경계를 통과할 때 |
| conv, indexer tail, drafter | 기존 형상 | 기존 위치·rollback 규약 유지 |
| boundary position/validity | 슬롯 세대와 연결한 작은 metadata | recurrent·conv 경계 데이터가 모두 준비된 뒤 게시 |

`rec`와 `rec_boundary`를 같은 slot-major arena에 carve해 #895 `Batch(..., boundaries=...)`의 stride·정렬·비중첩 검사를 그대로 사용한다. compact의 `_stage["rec", L]`은 `rec_boundary`를 참조하고, stage의 별도 할당에는 conv taps만 남긴다. stage 관련 바이트 산출·메모리 admission도 함께 고친다. 기존 `stage_bytes()`만 둔 채 중복 recurrent stage를 할당하면 통합이 끝난 것이 아니다.

한 상태의 본체는 `34 × 16 × 128 × 128 × 4 = 34 MiB`다. K=7에서 링 본체 272 MiB가 두 레코드 68 MiB로 바뀐다. 기존 별도 recurrent stage 34 MiB까지 포함한 비교는 `272 + 34 → 68 MiB`지만, 이것도 요청·rank당 recurrent 필드만의 계산이다. null slot, conv, KV, drafter, snapshot, factor와 정렬은 별도로 보고한다.

prefix snapshot의 논리적 내용은 여전히 한 recurrent 상태와 conv taps다. 반면 `slot_bytes()`로 이동하는 전체 슬롯의 물리 포맷은 달라진다. tier의 저장·복원 식별자에 layout 버전을 포함하고, 과거 ring 슬롯 바이트를 compact 슬롯에 그대로 복사하지 않는다. 다른 포맷으로 이어서 실행하려면 명시적인 변환이나 같은 논리 prefix snapshot에서의 복원이 필요하다.

### S2. 확정·경계·복원의 순서

현재 [pipeline.py](../engine/profiles/glm53/pipeline.py)는 sampler의 최종 `count`를 만든 후 `materialize()` → `stage_boundaries()`를 호출한다. 이 순서를 유지한다. `real_slot`을 사용해야 하며, EOS 때문에 이미 0으로 바뀐 다음 replay용 `slot`을 commit 주소로 쓰지 않는다.

```text
verify: current를 읽고 출력/factor만 생산
  → sampler: EOS·생성 한도를 적용한 최종 count
  → compact commit: current 갱신 + 통과한 경계의 rec_boundary 저장
  → stage: 같은 경계의 conv taps 저장 + boundary metadata 게시
  → drafter 관측 / 다음 replay 또는 host의 prefix 저장
```

`count=0`은 no-op이다. count·slot·context의 유효성은 graph 진입 계약으로 보장한다. component 커널이 invalid count를 쓰지 않는 동작을 서빙 오류의 정상 처리로 사용하지 않는다. 이전 factor의 commit이 끝나기 전에 같은 workspace를 새 verify로 덮어쓰지 않는다.

[bounded_loop.py](../engine/profiles/glm53/bounded_loop.py)의 경계 통과 후 rank 합의 중단을 유지한다. 한 commit의 `T <= block` 제약만으로 여러 commit 동안 단일 boundary 레코드를 보호할 수는 없다. 기존 중단 규약을 바꿀 때는 host가 snapshot을 소비하기 전 다음 경계를 덮어쓰지 않는 별도의 증명이 필요하다.

| 진입 경로 | 필요한 수정·불변 조건 |
| --- | --- |
| 큰 prefill | `net._kda()`의 chunk 결과를 `rec[0]`에 저장. chunk 내부 marks는 현재의 `mark_kda()` side output을 유지; 마지막 경계 한 장으로 여러 marks를 대체하지 않음 |
| 작은 eager step | 기존 direct-ring 선택을 우회해 compact verify 후 실제 확정 count를 commit. 비투기 입력은 전체 길이, 투기 입력은 최종 clipped count 사용 |
| graph decode | `DecodeGraphs.make_inputs()`가 동일 arena의 current/boundary를 `Batch`에 전달. factor는 현재처럼 `(행 수, 검증 폭)` 소유이며 context capacity graph 사이에 직렬로 공유 |
| `checkpoint(position)` | current가 정확히 그 position을 나타내면 current, 해당 boundary가 유효하면 boundary 사용. 임의 과거 position의 modulo 접근 금지 |
| `checkpoint_from_stage()` | compact boundary와 해당 conv stage의 position·slot generation이 일치한 뒤 snapshot으로 복사 |
| `restore()` | snapshot의 recurrent를 current에 복원하고 context를 함께 바인딩. 오래된 boundary를 invalidate; conv·drafter 복원은 기존 규약 |
| slot 반환·재사용 | 진행 중 graph, prefix 복사, tier 전송이 끝난 뒤 반환. 재사용 세대에서 current/validity를 초기화하며 이전 완료 통지가 새 요청을 갱신하지 못하게 함 |

`net.rec_ring`은 현재 speculative width 선택에도 쓰인다. 저장 폭 1과 검증 가능한 `K+1`을 별도 값으로 분리해야 한다. `rec_ring=1`만 대입하면 작은 decode 분기·history·write helper가 서로 다른 상태 의미를 갖게 된다. [net.py](../engine/profiles/glm53/net.py), [decode_graphs.py](../engine/profiles/glm53/decode_graphs.py), [state.py](../engine/kernels/state.py)

S 검증은 기존 compact 커널 검사에 실제 cache 객체의 prefill → checkpoint → decode → EOS → restore → slot 재사용 연쇄를 추가한다. C=1/C=4와 같은 arena의 여러 context bucket, bounded 4회 실행, 경계 직전·정확한 경계·직후를 포함한다. FP32 상태·출력 불일치는 허용하지 않는다. 서빙 채택에는 별도로 같은 빌드의 실제 품질·수락률·출력 tok/s가 필요하다.

## P. 전체 FFN이 FP8 패킷을 직접 소비

구현 상태 (2026-09-14): `STK_prefill_ffn_packets=1`은 native, `prefill_tiles=1`, eager TP4의 `8192 < N <= 32768`에서 동작한다. `PacketGeometry`/`PacketBatch` → `TokenShards.all_gather_packets()` → `Glm53Net._moe_packets()`에 연결했다. prefill step마다 한 번 control-group vote로 모든 층의 세 reader 가용성을 합의하며, 데이터 all-gather 횟수는 FFN당 한 번을 유지한다. shared calibration observer나 지원되지 않는 expert pack이 있으면 해당 층은 모든 rank에서 기존 경로를 쓴다.

실제 expert selector는 현재 main의 M128 `MoEGatedDynamicKernelSF6Prefill`이다. 그 전체 producer의 소스 해시와 입력 복사 외 AST를 고정하고, 기존 32 KiB/CTA BF16 stage에 packet을 역양자화하는 입력 variant만 추가했다. histogram, expert별 group-16 양자화, task publication, MMA, BF16 atomic scatter는 그대로 상속한다. baseline과 같은 eager workspace를 쓴다. shared GEMM에는 패딩을 제외한 N행 Q/S를 전달한다.

첫 구현은 별도 side stream이나 packet buffer 재사용이 없다. invocation마다 새 tensor owner를 만들고 producer와 모든 reader를 같은 current stream에 제출한다. 아래 event/epoch descriptor는 향후 buffer 재사용·side stream 도입 시의 설계이며 현재 구현 API가 아니다. 현재 상태와 실제 수행한 검증은 [P 구현 증거](../measurements/ffn_packets_20260914/README.md)를 기준으로 본다.

### P1. 첫 버전은 통신 한 번을 유지

현재 [TokenShards.all_gather()](../engine/modules/token_shards.py)는 [PrefillCollectives.all_gather()](../engine/kernels/prefill_collectives/__init__.py)의 FP8 패킷을 전체 BF16 hidden으로 풀고 real rows만 반환한다. [net.forward()](../engine/profiles/glm53/net.py)의 FFN에서는 그 텐서를 router, routed expert frontend, shared expert gate/up이 읽는다. 이 세 소비자를 모두 바꿔야 전체 hidden 할당이 사라진다.

첫 경로는 **기존 FFN의 전체 all-gather를 정확히 한 번 수행**한다. 이미 있는 [TiledProjection](../engine/kernels/prefill_collectives/tiles.py)의 작은 통신 여러 번을 이 단계에 합치지 않는다. 전송 크기·횟수·순서를 고정해야 packet consumer 효과를 분리할 수 있다.

제안하는 owner와 입력 계약:

```text
PacketBatchV1:
  received: contiguous uint8[4 * packet_stride]  # 한 invocation이 소유
  real_rows=N, local_rows=ceil(N/4), padded_rows=Np=4*local_rows
  hidden=4096, block_elements=2048, wire_format=기존 FP8-v3
  packet_stride=align128(local_rows*4096 + 4*(local_rows*4096/2048))
  layer_id, invocation_epoch, source_generation
  ready_event, last_reader_event

begin_ffn_packets(local_x, shards, agreed_plan) -> PacketBatchV1
router_from_packets(batch, layer)              -> logits[N,288]
experts_from_packets(batch, route_ids, weights)-> routed[N,4096]
shared_from_packets(batch, layer)              -> shared[N,4096]
shards.reduce_scatter_pair(routed, shared)     -> local_ffn
```

이름은 설계용이다. 기존 `reduce_scatter_pair()`의 BF16 합산 후 FP8 pack 경로는 이미 구현되어 있으므로 새 성과로 세지 않는다. `fuse_sum`이 꺼진 비교에서는 기존 add·reduce 순서를 그대로 사용한다. 최종 hidden/aux의 `gather_result()`도 기존 lossless 통신을 유지한다.

첫 지원 범위는 eager TP4, hidden 4096, 현재 token sharding 가능 조건, **`8192 < N <= 32768`**의 MoE FFN과 지원되는 tiled SF6 expert frontend다. 통신의 FP8 선택 기준은 `Np >= 2048`, router backend의 기준은 **real rows N**이다. transport padding을 실제 토큰 수로 사용하지 않는다. 128K 요청은 해당 범위의 기존 prefill chunk를 통해 검증하며, chunk 정책 자체를 이 변경으로 바꾸지 않는다.

관측·calibration이 BF16 입력을 요구하거나 세 소비자 중 하나라도 지원되지 않으면 해당 FFN 전체가 기존 경로를 선택한다. 모든 rank가 같은 plan을 통신 전에 확정한다. collective가 시작된 뒤 로컬 예외로 다른 종류의 collective에 fallback하지 않는다.

### P2. 세 소비자의 수치 계약

공통 입력 복원은 `float32(fp8_value) * fp32_transport_scale` 뒤의 **BF16 반올림**을 포함한다. BF16 메모리 저장을 제거해도 이 반올림을 레지스터에 남긴다. transport scale은 2048개 원소 단위이며 다른 소비자의 양자화 scale과 같지 않다.

| 소비자 | 구현 진입점 | 보존할 계산 |
| --- | --- | --- |
| router | [prefill_router.py](../engine/kernels/prefill_router.py)의 `_router_gemm` A load만 packet 주소 계산으로 교체 | BM64/BN64/BK64, 기존 K 누산 순서·FP32 accumulator·fusion 설정. 뒤의 sigmoid/bias/top-8/정규화·동점 선택은 기존 `net.route()` 유지 |
| routed expert | [moe_dispatch.py](../engine/kernels/b12x/moe_dispatch.py)의 선택된 dynamic frontend에 packet source variant | expert별 `input_gs`, BF16 입력 반올림, 기존 FP4 group-16 scale/rounding, route histogram·SFA 배치·publish 순서 유지 |
| shared gate/up | [DenseLinear._project_packets()](../engine/kernels/dense/__init__.py)와 [quantize_gather()](../engine/kernels/prefill_collectives/consumer.py) 재사용·확장 | BF16 roundtrip 뒤 group-128 FP8 및 power-of-two scale 재계산. activation과 down projection의 기존 dtype/rounding 유지 |

shared projector는 현재 padded rows 전체를 계산한다. FFN 버전은 `real_rows`를 받아 Q·scale의 앞 N행만 `FP8Linear.project_quantized()`에 넘겨 기존 full-hidden FFN과 GEMM shape를 맞춘다. 단순히 출력만 N행으로 자르면 입력 M에 따른 backend 선택 차이를 놓칠 수 있다. `PaddedDenseLinear`처럼 packet projector가 없는 객체는 첫 버전에서 fallback한다.

expert frontend에는 이미 분리된 범용 `packA` API가 있는 것이 아니다. persistent kernel 내부가 accumulator 초기화, route histogram, 입력 양자화, task publication을 함께 수행한다. P에서는 선택한 body의 **입력 load만 교체**하고 이 protocol을 유지한다. [moe_dispatch.py](../engine/kernels/b12x/moe_dispatch.py), [_prefill_q0_batch8.py](../engine/kernels/b12x/_prefill_q0_batch8.py)

특히 `input_gs.numel()==1`인 경우와 expert별 scale인 경우를 구분한다. 후자는 같은 토큰도 선택된 expert에 따라 packed activation이 달라질 수 있으므로 전역 scale 하나로 토큰당 한 번 pack하는 설계는 채택하지 않는다. `share_input_across_experts`를 켜면 Q0 eligibility도 달라진다. 작은 N의 `_tp_sf6_q0_eligible()`와 별도로 현재 main에는 long-prefill M128 route-cache producer가 있다. P의 첫 구현은 그 producer의 per-expert scale 경로만 지원하며, scalar global-scale/share-input 모드는 fallback한다. 작은 N으로 확장하려면 기존 router backend도 별도로 packet화해야 하며, 각각 실제 selector·소스 식별자를 남기고 검증한다. decode 입력 경로는 이 P 변경의 대상이 아니다.

### P3. 패딩·수명·바이트의 해석

row 주소는 `rank = row // local_rows`, `local_row = row % local_rows`다. 실제 route 생성은 `row < N`에서만 수행한다. pad row를 histogram에 넣거나 유효하지 않은 expert id를 기존 histogram kernel에 넘기지 않는다. 패킷은 세 소비자의 마지막 GPU read까지 살아 있어야 한다. side stream을 쓰면 ready/release event와 storage의 stream 수명을 함께 관리하며, 예외 정리도 마지막 reader까지 fence한다.

32,256행의 바이트 계산은 다음과 같다. 단위는 rank당 MiB다.

| 항목 | 계산값 | 의미 |
| --- | ---: | --- |
| 기존 full BF16 hidden | 252 | 모든 소비자가 전환되면 제거할 수 있는 중간 할당 |
| 그 텐서의 store + read 1회 | 504 | 제거 대상 연산의 바이트 합; 전체 트래픽 순감소 측정값이 아님 |
| 네 rank의 수신 FP8 payload + scale | 126.24609375 | 기존과 동일, 이 형상에서는 packet 정렬 추가분 없음 |
| shared projection의 Q + group-128 scale | 129.9375 | 기존 quantizer 재사용 시 여전히 필요한 출력 workspace |

세 소비자의 packet 재읽기, expert route별 역양자화 명령, logits·route metadata·shared Q/S의 동시 생존 비용이 남는다. 따라서 504 MiB를 메모리 대역폭으로 나눈 값을 TTFT 개선으로 보고하지 않는다. arena 전체와 외부 workspace의 최대 동시 생존량을 [budget.py](../engine/profiles/glm53/budget.py)의 현행 admission에 반영하고, 실제 allocated/reserved peak도 측정한다.

P1 이후에만 통신 tile 소비를 검토한다. 기존 `TiledProjection`의 두 slot ready/released protocol을 재사용하되, tile 내 행 `(rank,j)`의 원래 위치는 `rank*parent_local_rows + local_begin + j`다. tile-major 단순 concat은 토큰 순서를 바꾼다. router eligibility도 작은 tile 길이가 아니라 부모의 N으로 결정해야 한다. 통신 메시지 수가 바뀌므로 이 단계는 새로운 A/B다.

### P4. 검증과 실패 판정

1. 동일 FP8 packet의 기존 unpack 결과를 기준으로 router logits, top-8 id/weight, expert packed A/SFA, shared Q/S 및 FFN 출력을 대조한다. 8192/8193, 32768/32769 경계, N mod 4, zero·underflow·scale 경계·router 동점을 포함한다. 첫 두 경계의 unsupported 행 수는 fallback 선택을 검사한다.
2. full BF16 fallback과 packet 후보를 **같은 body·수치 옵션·실제 N**으로 비교한다. 관측기가 켜진 경우 기존 입력이 계속 전달되는지 확인한다. 선택한 body에서 입력 변환이 일치하지 않으면 속도 측정으로 넘어가지 않는다.
3. 먼저 full FFN component에서 router+pack+shared+reduce 전체 시간, 임시 메모리, 패킷 수·바이트를 측정한다. 기존 router 또는 projection만의 단독 속도는 채택 근거가 아니다.
4. 같은 런타임의 32K/128K cache-off one-pass에서 prefill tok/s·TTFT와 이후 C=1 decode 품질·수락률·출력 tok/s를 비교한다. C=4도 동등 문맥에서 요청별 지표와 전체 처리량을 함께 남긴다. 작은 2K 요청은 fallback 회귀를 확인한다.

## M. 디코드 타일에 프리필 route를 함께 계산

2026-09-14 구현 상태: M0 admission과 **M1a hot component**를 추가했다.
`engine/modules/mixed_experts.py`가 M16/M32의 기존 타일 안에 들어가면서 M128 tail 하나를 제거할 수 있는 route만 고른다. expert별 최소 tail 길이, expert ID 순으로 선택하며 추가 route는 전체 128개로 제한한다. decode route 순서와 모든 cold route의 원래 `(row, slot)`은 보존한다.

`PreparedMixedExperts`는 서로 다른 BF16 입력 두 개, immutable invocation identity, per-expert scales를 소유하고 명시적인 source map을 별도 CuTe producer에 전달한다. producer는 runtime row extent를 사용한다. prepared V5는 기존 frontend 초기화/재라우팅을 건너뛰고 동일한 MMA body를 실행한다. 전체 ordinary kernel AST가 조건문 삽입 전과 같음을 검사한다. source/weight mutation, 다른 stream/capture, stale layer/epoch/slot/source generation은 실행 전에 거부한다. route-owned FP32 partial은 BF16 down 및 weighted rounding을 유지하며, probe의 최종 reducer 비용/오차는 별도로 기록한다.

**M1 전체 및 서빙은 아직 구현하지 않았다.** 현재 component는 decode와 선택된 hot route만 반환한다. cold M128 재묶기·실행, shared 완료 합산, 취소/slot 회수, TP4 합의와 scheduler/graph 연결은 남아 있다. 따라서 `removed_prefill_tiles`는 잔여 route를 M128로 다시 묶을 때의 조건부 work accounting이며 실제 제거/속도 측정이 아니다. CPU planning·metadata allocation 비용도 별도 기록하며 이것을 decode hot path에 넣지 않는다. 실행 knob나 기본 selector는 추가하지 않았다. [M evidence](../measurements/mixed_experts_20260914/README.md)에 범위와 검증을 남긴다.

### M1. 빈 행과 실제 절감되는 타일을 구분

현재 [static V5](../engine/kernels/b12x/moe_static_kernel_v5.py)는 [V4](../engine/kernels/b12x/moe_static_kernel_v4.py)의 계산을 사용한다. 현재 `t,r,sf6`는 전체 decode 1..8행에서 M16, 9..32행에서 M32이며, 대상 SF6 dynamic prefill은 M128 형상이다. 따라서 단순히 두 입력을 concat한 기존 launch 호출로는 원하는 스케줄이 되지 않는다. 동일한 층의 FFN 입력과 route가 이미 준비된 프리필만 후보가 된다.

expert e의 decode route 수를 `d_e`, 준비된 prefill route 수를 `p_e`라 하자. 첫 실험에서는 decode가 방문한 expert의 **실제 M16/M32 타일 수를 늘리지 않는다**.

```text
m = 16 (decode rows <= 8), otherwise 32
a_e = ceil(d_e / m)
spare_e = m*a_e - d_e                  (d_e > 0, 그 외 0)
0 <= h_e <= min(p_e, spare_e)           # 함께 계산할 prefill route 수
removed_prefill_tiles_e = ceil(p_e/128) - ceil((p_e-h_e)/128)
```

마지막 식은 나머지 prefill이 동일 M128 body로 재묶인다는 조건의 work-tile 계산이다. GPU의 실제 DRAM read 측정값은 아니다.

| d_e | p_e | h_e | decode M32 타일 | 잔여 prefill M128 타일 | 제거한 prefill 타일 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 100 | 30 | 1 → 1 | 1 → 1 | 0 |
| 2 | 140 | 12 | 1 → 1 | 2 → 1 | 1 |
| 32 | 140 | 0 | 1 → 1 | 2 → 2 | 0 |

첫 행처럼 빈 행을 채워도 다른 prefill 타일이 사라지지 않으면 가중치 읽기 절감 없이 pack/scatter 비용만 늘 수 있다. 따라서 구현된 첫 admission은 `removed_prefill_tiles_e > 0`인 후보만 선택하며, 이후 여러 decode 방문을 묶을 때도 **최종 cold work까지 포함한 전체 비용**으로 결정한다. 입력 pack·추가 metadata·partial output·cold 재묶기 비용이 절감보다 크면 그 후보를 실행하지 않는다.

### M2. 입력·완료 상태를 명시적으로 소유

첫 component probe는 동일 층의 준비된 BF16 decode/prefill 입력과 확정 route를 사용한다. graph 간 대기나 통신을 먼저 넣지 않는다. 기존 decode 행 순서를 유지한 뒤 선택된 prefill 행을 각 expert의 tail에 배치한다. 가중치·SF6 scale·expert별 input/down scale·split 정책·gate/up 반올림의 호환성을 명시적으로 확인한다.

기존 static frontend의 `pair_idx // topk`는 단일 입력 텐서의 토큰 주소다. 혼합 body는 `(decode|prefill, 원래 행, route 번호)`를 담는 명시적 source map으로 입력·출력 주소를 선택한다. expert 타일이 M32라는 사실이 전체 입력 토큰 수도 32 이하라는 뜻은 아니다. 기존 static 입력 길이 제한을 풀고 두 텐서를 concat하는 방식으로 우회하지 않으며, route/SFA/partial-output capacity를 별도로 선언한다.

제안 descriptor와 ticket:

```text
MixedExpertPlan:
  layer_id, invocation_epoch, layout_id, source_generation
  decode_rows, prefill_ticket_id, prefill_global_row_ids
  per_expert_decode_count, selected_prefill_routes, cold_work_quota

PrefillLayerTicket:
  request_id, slot_generation, layer_id, tile_id, real_rows
  immutable_source_owner, route_ids, route_weights, expert_input_scales
  route_done[real_rows, 8], partial_outputs, shared_output
  ready_event, last_reader_event, completion_event
```

프리필의 top-8 route를 변경하거나 cold expert를 생략하지 않는다. 각 토큰의 여덟 route와 shared output이 모두 준비되어야 해당 FFN이 완료된다. 첫 서빙 버전은 ticket의 모든 real rows가 완료된 뒤 다음 층으로 진행한다. 일부 행만 다음 층으로 보내는 최적화는 나중에 별도 설계한다.

현재 dynamic frontend는 launch마다 output accumulator와 route/task counters를 초기화한다. 그러므로 부분 작업을 이어 간다고 기존 `launch_sm120_dynamic_moe()`를 반복 호출하면 안 된다. 혼합 body에는 **ticket당 1회 초기화 → 준비된 route 게시 → 제한된 work dispatch → 완료 합산**의 분리가 필요하다. 일반 body를 그대로 재호출하는 경로는 허용하지 않는다. row completion 표시는 결과 저장이 보인 뒤 게시하며, 마지막 kernel/통신 consumer가 끝나기 전에 ticket이나 slot을 반환하지 않는다.

### M3. 반올림과 우선권은 별도 문제

현재 static frontend는 expert별 행을 atomic으로 배치한다. 고정 GLM TP4 경로는 BF16 down/weighted partial을 FP32 accumulator에 atomic 합산하고 마지막에 BF16으로 변환한다. prefill 행 추가는 atomic 실행 순서와 work 순서를 바꿀 수 있다. MMA의 행별 계산이 같아도 최종 decode 출력이 byte-exact라고 가정하지 않는다. M1a의 route-owned partial과 별도 reducer도 기본 atomic 합산과 구별해 검증한다.

검증은 route별 기여 값, 최종 합산 순서, 실제 생성 품질을 구분한다. 동일한 prepared route를 사용한 분리 실행/혼합 실행에서 먼저 route 기여를 비교한다. 최종 출력은 baseline 반복 간 변동도 기록한다. 정확한 합산 순서 보존이 불가능하면 route/part별 임시 출력과 결정적인 reducer를 독립 후보로 설계하고 **그 메모리·시간·수치 변경 전체를** 비교한다. 오차 tolerance만으로 디코드 품질 통과를 대신하지 않는다.

디코드 우선권은 stream priority만으로 구현하지 않는다. 실행 중인 persistent work가 즉시 양보한다고 가정할 수 없다. 첫 혼합 kernel은 준비된 hot work만 받고, 추가 decode M 타일을 만들지 않으며, 종료 후 decode 후속 경로가 진행할 수 있게 한다. cold work는 측정된 tile 수/시간 quota로 잘라 실행한다. 모든 SM을 점유한 grid가 아직 실행되지 않은 prefill producer를 기다리는 구조는 만들지 않는다.

[SharedOverlap](../engine/kernels/dense/shared_mlp.py)은 C=1에서 shared expert를 side stream에 실행하고 main callback을 router+MoE로 제한한다. 이 callback에서 prefill shared GEMM까지 실행하면 기존 scratch를 충돌시킬 수 있다. prefill shared는 사전 준비·fence하거나 별도 소유 workspace를 선언한다. C=1 shared overlap을 잃는 구현이면 그 손해도 최종 baseline 비교에 포함한다. C=4의 현재 별도 선택을 임의로 통일하지 않는다.

### M4. TP4 서빙으로 확장할 때

rank마다 local timing을 보고 서로 다른 mixed work나 collective 순서를 선택하면 안 된다. rank가 공유하는 `MixedExpertPlan`을 graph 진입 전에 확정하고, layer·epoch·ticket·quota가 일치한 경우만 실행한다. 입력이 준비되지 않은 경우도 모든 rank가 decode-only를 선택한 뒤 진입한다. rank 간 준비 합의·descriptor 전달 비용을 디코드 시간에서 제외하지 않는다.

초기 서빙 실험은 `decode_iterations=1`로 admission 경계를 명확히 한다. 대조군도 같은 설정으로 비교하되, **기존 생산 설정의 bounded 4회 경로와도 따로 비교**한다. 1회 대조군에만 이겼다는 이유로 채택하지 않는다. 4회 graph 안에 넣으려면 고정된 descriptor 버퍼와 미리 준비된 유한 혼합 형상, ordered completion, rank 합의 중단이 필요하다. 실행 중 host가 임의 시점에 prefill 포인터를 교체하는 API는 제공하지 않는다.

prefix/slot 수명은 S의 compact 채택 여부와 무관하게 보호한다. 취소된 prefill ticket도 이미 제출된 kernel의 reader fence까지 source·partial output을 유지한다. 같은 slot을 다른 요청이 재사용한 경우 이전 generation의 완료 통지는 폐기한다.

### M5. 실험 순서와 판정

| 순서 | 실험 | 남길 결과 |
| --- | --- | --- |
| M0 | 실제 route trace로 가능한 tail-fill과 M128 타일 감소 계산 | 층·expert별 d/p/h, 제거 타일 수, cold 잔량. oracle 입력이며 속도 결과는 아님 |
| M1 | 단일 GPU, 준비된 입력의 분리/혼합 component | route 기여·최종 출력, pack부터 cold 완료까지 총시간, hot decode 반환 시각, scratch peak |
| M2 | 동일 층 ticket scheduler와 취소/재사용 | 중복·누락 route 0, 모든 cold work 완료, 유한 quota, source/event 수명 |
| M3 | TP4 동시 도착 one-pass | 실제 decode tok/s·토큰 지연 분포, prefill TTFT·완료 시간, 품질·수락률, rank별 collective epoch |

M3은 기존 decode 요청에 32K 또는 128K prefill이 도착하는 같은 스케줄로 비교한다. baseline decode-only 구간, 기존 scheduler의 동일 혼합 부하, 후보의 동일 혼합 부하를 구분한다. 측정 종료 시 미완료 prefill을 버리고 throughput을 계산하지 않는다. 프리필 없이도 C=1 hot path에 고정 비용이 붙는지 확인하고, C=4에서도 요청별 지연을 확인한다.

채택 조건은 baseline 반복 분산으로 정한 C=1 비열등 한도와 품질·수락률을 지키면서 **완료된 프리필의 TTFT/완료 시간 또는 처리량이 개선되는 것**이다. 허용 지연 한도·반복 수·work quota는 paired 실행 전에 고정해 기록한다. cold backlog가 계속 자라거나, route trace에서 대부분 제거 타일이 0이거나, 실제 생산 graph 대비 손해면 해당 혼합 후보를 보류한다.

## I. 유한한 실행 이미지와 재컴파일 경계

새 범용 엔진을 하나 더 만들지 않고 [ExecutionPlan](../engine/profiles/glm53/execution.py), [Arena](../engine/base/arena.py), 기존 커널 shape/cache 체계를 확장한다. profile이 지원하는 C·검증 폭·context capacity·prefill bucket과 각 경로의 물리 ABI를 선택한다. 다음 세 식별자는 역할을 분리한다.

| 식별자 | 포함하는 내용 | 재사용·무효화 기준 |
| --- | --- | --- |
| kernel artifact | 해당 body와 실제 소스 의존성, compile-time shape, 수치 옵션, SM121, compiler/CUDA identity | 관련 소스·ABI·도구 체계 변경만 해당 artifact 무효화 |
| execution layout | cache 필드·slot stride, packet/weight layout version, route/완료 protocol, 지원 graph 집합 | 다른 physical layout의 slot·준비된 입력 혼용 금지 |
| runtime receipt | 위 ID, checkout, 실제 weight hash, 선택 body, image, rank topology, 런타임 옵션 | 성능 비교의 동일성 확인; compile key와 구분 |

전체 repo SHA만 바뀌었다고 모든 kernel을 재컴파일하지 않는다. 반대로 Python만 바뀌었다고 항상 artifact가 유효한 것도 아니다. ABI나 code generator의 실제 의존성 변경은 무효화한다. 실제 weight byte hash는 실행·정확도 식별자이며, weight 값을 specialization하지 않는 kernel의 compile key에는 불필요하게 넣지 않는다.

첫 이미지에는 채택된 형상과 body만 준비한다. 사용하지 않는 C/K 조합을 전부 warmup하지 않는다. 최초 빌드의 wall time·cache hit/miss·graph 준비 시간은 각각 기록한다. CUDA graph의 절대 포인터, NIC 등록 키·GID·통신 epoch는 프로세스 부팅 때 바인딩하며 캐시 파일에서 복원하지 않는다. 다른 rank의 layout/protocol 불일치는 첫 collective 전에 탐지한다.

## 구현 작업과 공통 검증 인수 조건

| 작업 묶음 | 주 변경 위치 | 먼저 끝낼 검증 |
| --- | --- | --- |
| S1 배치·예산 | `glm53/caches.py`, `budget.py`, tier의 slot layout 식별자 | CPU layout/alias/예산, 같은 KV·snapshot 용량 |
| S2 상태 수명 | `net.py`, `decode_graphs.py`, `pipeline.py`, `kernels/state.py` | GPU 상태 연쇄·prefix·EOS·bounded·재사용 |
| P1 packet owner/router | `prefill_collectives`, `token_shards.py`, `prefill_router.py` | N/Np/행 순서/rounding/관측 fallback |
| P2 expert/shared | `b12x/moe_dispatch.py`와 선택 body, `dense/__init__.py`, `net.py` | packed 입력·FFN 출력·전체 workspace와 component 시간 |
| M0–M1 work accounting/body | route trace, `moe_static_kernel_v4.py`/`v5.py`를 기준으로 한 별도 후보 body | 실제 타일 감소 가능성, prepared-input 수치·총비용 |
| M2–M3 ticket/TP4 | `execution.py`, prefill 진입, graph/통신 ordering | 모든 route 완료, 취소·rank 합의, 생산 baseline 대비 지표 |
| I 의존성·이미지 | 기존 kernel cache/shape와 boot identity | 최소 artifact 집합, cache hit/miss, 재바인딩 |

커널/protocol 변경은 각각 선택 가능한 후보로 시작한다. 판정 전에는 serving 기본값을 바꾸지 않는다. S, P, M을 한꺼번에 켠 결과로 개별 이득을 추정하지 않고, 개별 검증 뒤 필요한 조합을 별도 비교한다. S의 메모리 절감이 P/M workspace로 소비되면 최종 peak를 다시 기록한다.

실제 one-pass 기록에는 base → candidate → base 순서, 모든 rank의 소스·가중치·이미지·선택 body, 예약 identity, compile/graph 준비, prefix/KV 재사용, profiler 여부를 포함한다. C=1 품질·한국어 출력·수락 길이 분포·출력 hash와 실제 tok/s, C=4의 동일 문맥 요청별 TTFT·generation rate·완료 시간 및 aggregate throughput을 남긴다. packet byte 수, 제거할 수 있는 work tile 수, simulator/oracle 예측과 실측 consumer 결과를 다른 항목으로 기록한다.

현재 #895의 [커널 검증·등록 기록](../measurements/kda_compact_20260913/README.md)은 compact component만 대상으로 한다. 이 후속 설계의 추가가 그 예약의 실행 대상을 바꾸지는 않는다. S/P/M의 serving 수치·성능은 각 구현 후 해당 gate를 통과해야 한다.
