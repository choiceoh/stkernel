# stkernel

DeepSeek-V4-Flash-0731 · **TP=4** 프로덕션 오버레이 스택
(4× NVIDIA DGX Spark GB10 · CRS812 스위치드 패브릭 · vLLM eldritch/b12x 포크 이미지)

베이스 이미지 `aidendle94/sparkrun-vllm-ds4-gb10:production-hybrid-1.6`(2026-06-26 계열)은
서드파티라 소스 트리가 없다. 이 리포는 그 이미지를 **재빌드 없이** 개선하기 위한
전체 스택이다 — 파이썬 파일을 `site-packages` 위에 read-only 바인드 마운트하는
오버레이 방식이며, 모든 수정부에 `# deneb fork: port of upstream PR #NNNNN` 마커가 있다.

```
-v <overlay>/attention.py:/opt/venv/.../vllm/models/deepseek_v4/attention.py:ro
```

## 구성

| 디렉터리 | 내용 |
|---|---|
| `overlay/` | **프로덕션 오버레이 4파일** (업스트림 PR 5건 포팅 + TP4/GB10 전용 슬리밍) |
| `launchers/` | 프로덕션 런처 + 슈퍼바이저 + systemd 유닛 |
| `bench/` | 검증·측정 도구 |
| `probes/` | 계측 빌드 (CUDA 이벤트 단계 분해, `DENEB_ATTN_PROF=1`) — 오버레이와 동일 코드 + 계측 |

## overlay/ — 프로덕션 채택 5건

| PR | 파일 | 내용 | 실측 |
|---|---|---|---|
| [#51209](https://github.com/vllm-project/vllm/pull/51209) | `attention.py` | IndexCache — C4A 레이어가 이전 top-k 재사용 (`index_topk_freq=4`) | 장문 디코드 32K **+13.6%** / 128K **+16.0%** |
| [#51042](https://github.com/vllm-project/vllm/pull/51042) | `flashinfer_sparse.py` + `sparse_swa_dsv4.py` | DSpark 비인과 decode SWA 버퍼 폭을 메타데이터로 전달 (window_size 오판독 수정) | 정합성 |
| [#51252](https://github.com/vllm-project/vllm/pull/51252) | `indexer.py` | 프리필 청커 예산을 `compress_ratio`로 나눠 소비 버퍼와 단위 일치 | 정합성 · 프리필 무손실 |
| [#49059](https://github.com/vllm-project/vllm/pull/49059) | `flashinfer_sparse.py` | 빈 프리필 청크(0-element reshape) 크래시 가드 | 예방 |
| [#51202](https://github.com/vllm-project/vllm/pull/51202) | `flashinfer_sparse.py` | prefill/decode 게이트를 요청수 대신 토큰수로 | 예방 |

검증: 프리필 2,437–2,533 tok/s(무회귀) · 장문 리트리벌 9/9(2K/32K/128K) · traceback 0.

## TP4/GB10 전용 슬리밍 (2026-08-09)

오버레이는 업스트림 범용 코드의 포팅이라 이 배포에서 **구조적으로 도달 불가능한**
경로를 담고 있었다. 아래를 제거했고, 전제가 어긋나면 조용히 오동작하는 대신
기동 시점에 죽도록 fail-fast assert를 심었다.

| 제거 항목 | 근거 |
|---|---|
| SM100(B200급) 어텐션 클래스 + plain-row bf16/per-tensor-fp8 KV 경로 | GB10=SM121은 항상 `DeepseekV4FlashInferSM120Attention` + `fp8_ds_mla`(uint8). 클래스명은 import 안전 스텁으로 보존 |
| breakable-cudagraph forward 분기 (`VLLM_USE_BREAKABLE_CUDAGRAPH`) | 확정 스위치 0 고정 — 켜면 장문 디코드 5배 저하 + 엔진 사망 |
| b12x WO projection 경로 + 커스텀 op 2종 | 런처가 env를 노출하지 않음. `setup_b12x_wo_projection()`은 외부 호출 대비 no-op 스텁으로 보존 |
| DCP/PCP(context parallel) 분기·커널, PP(IndexCache 랭크 계산) | 런처에 CP/PP 플래그 없음. 메타데이터의 `dcp_*` 필드는 소비자(`sparse_attn_indexer`) 계약상 1/0/1 고정으로 유지 |
| FlashMLA 타일 스케줄러 (`build_tile_scheduler`) | SM120은 b12x 경로라 스케줄러 미사용, `_flashmla_C`도 sm_121a 미빌드 — 기존에도 항상 스킵. `tile_sched_*` 필드는 None 고정으로 유지 |
| FP4 인덱서 캐시, `index_topk_pattern`, compress_ratio==1(V3.2/GLM) 인덱서 빌더 경로 | FP4는 SM10x 전용, pattern은 미노출(freq만 사용), V4-Flash 인덱서 캐시는 전부 C4A(ratio 4) |
| ROCm/XPU 분기, 업스트림 DSpark parallel-drafting 2배 threshold | CUDA 4노드 고정, fork DSpark impl 사용 |
| `experimental/` 5파일 (#51430, #47474 포팅) | p430은 breakable CG 전용(프로덕션 fatal), p474는 성능 중립 미채택 — 복원은 git history(`e95f82f`) |
| 런처 `SP`/`EPFLAG` 노브 | b12x MoE TP 전용 → enable_sp(#46789)·EP 계열 구조적 불가 |

최적화: `VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD`·`VLLM_DSV4_INDEXER_SP`·
`B12X_PAGED_INDEX_SUPERTILE_K` env 조회를 핫패스에서 init/import 시점 1회로
호이스팅, `_indexer_sp_owned_ranges`의 per-call import/TP 조회 캐싱, 인덱서
디코드 빌드의 `seq_lens.max().item()` 동기화 폴백 제거(ratio>1 상시라 호스트
상한으로 대체).

**유지한 가변 경로** (런처 노브라 상수가 아님): `IDXSP`(indexer SP on/off),
`IDXFREQ`, `V2RUNNER`(batch_topology 유무), `SPEC_TOKENS`(flatten vs native
디코드 — next_n∈{1,2}는 native deepgemm), C128A 레이어 분기, B12X vs DeepGEMM
인덱서 스케줄 분기.

주의: 오버레이 소스가 바뀌었으므로 다음 기동에서 torch.compile/AOT 해시가
갈려 **1회 장시간 재컴파일 워밍업**이 발생한다 (커널 캐시 마운트는 그대로).
롤백은 기존과 동일 — 해당 `-v ...:ro` 마운트 제거 또는 git으로 이전 오버레이
복원.

## 배포 레이아웃

| 노드 | 오버레이 경로 |
|---|---|
| srv2(head)·srv3 | `~/hybrid-stack/overlay-b12x/` |
| srv1·srv4 | `~/hybrid-stack-port/overlay-b12x/` |

- 기동: `launchers/start-hy4-tp4.sh` (srv2에서; 워커 3대는 ssh로 원격 기동)
- 상시 운영: `dsv4-tp4.service`(user unit) → `dsv4-tp4-supervisor.sh`
  — 부팅 시 자동 기동, 실생성 헬스프로브 3연속 실패 시 포렌식 덤프 후 재기동
- **롤백**: 런처의 해당 `-v ...:ro` 마운트 한 줄 제거 → 재기동. 오버레이가 없으면 이미지 원본 그대로.

## bench/ — 측정 도구와 함정

| 도구 | 용도 | 함정 |
|---|---|---|
| `bench-tp4.py` | 프리필+짧은 디코드 (매회 고유 프롬프트 = 프리픽스캐시 무효) | 디코드는 dspark 수용률 탓 73–110 진동 — 구성 비교엔 프리필 사용 |
| `bench-dec.py` | **디코드 비교 전용** — 768토큰 생성으로 수용률을 평균화 | 짧은 벤치로는 10% 효과도 판별 불가 |
| `bench-ctx.py` | 장문 TTFT/디코드 분리 (스트리밍) | 비스트리밍이면 프리필이 디코드에 섞여 오측 |
| `check-quality.py` | 2K/32K/128K에 사실 3개를 20/50/85% 깊이로 심고 리트리벌 | 인덱스 stride 버그가 산문 열화가 아닌 **검색 실패**로 드러남 |
| `bench-conc.py` | 동시성 스윕 C=1/2/4 | 한 스윕 내 C값들은 상관 표본 — 단독 스윕으로 판정 금지 |

## probes/ — 계측 빌드

`DENEB_ATTN_PROF=1` + 해당 파일을 마운트하면 단계별 CUDA 이벤트 타이밍이 로그로 나온다
(`[deneb-prof]`/`[deneb-prep]`/`[deneb-fi]`). 캡처 가드(`is_current_stream_capturing`) 필수 —
없으면 `cudaErrorStreamCaptureInvalidated`로 기동 실패.

실측 트리 (프리필, 정상상태): 어텐션 전체는 스텝 벽시계의 **~16%**
(prep 32% — 인덱서 활성 레이어 크리티컬패스 / attn 68% — 그중 FlashInfer 커널 93%,
q-prep 0.1%). 나머지 ~84%(MoE·mHC·norm·comms)는 미계측 — 다음 표적.

## 미채택 포팅 (파일은 제거, 근거 보존 — 복원: git history `e95f82f`)

| 파일(구 `experimental/`) | PR | 미채택 사유 |
|---|---|---|
| `attention_p430.py` | #51430 | eager 구간 축소는 **breakable CG 전용** — 프로덕션은 `VLLM_USE_BREAKABLE_CUDAGRAPH=0`이라 죽은 코드. 켜면 이 포팅은 CUDA illegal access(전제인 #49236 `eager_scratch.py`가 이미지에 없음), 켜는 것 자체도 장문 디코드 5배 저하+엔진 사망 |
| `*_p474.py` | #47474 | `token_to_req_indices` 3중 계산 캐시 — 포팅·검증 완료했으나 **성능 중립**(우리 포크에 이미 빠른 경로 2개 존재), 최광역 파일(`v1/attention/backend.py`)이라 미채택 |

## 확정 스위치 (재시험 불필요)

- `VLLM_DSV4_INDEXER_SP=1` — 유일하게 켬 (+5.4%, bit-exact)
- `VLLM_USE_BREAKABLE_CUDAGRAPH=0` **고정** — 1이면 장문 디코드 ~5배 저하 + 엔진 사망. 해당 forward 분기는 오버레이에서 제거됨
- `B12X_INDEXER_STREAM`/`B12X_KV_STREAM` off — 켜면 디코드 반토막
- b12x MoE는 **TP 전용** — EP/DP 불가 → SP(#46789)·EP 계열 전부 구조적 불가. 런처의 `SP`/`EPFLAG` 노브도 제거됨

## 라이선스

`overlay/`·`probes/`의 vLLM 파생 파일은 원본 SPDX 헤더 그대로
**Apache-2.0**이며, 리포 전체가 같은 라이선스를 따른다 (`LICENSE`).
