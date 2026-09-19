# Qwen3.8 — GB10 단일 GPU 레인: Q7 판정, 한 번 죽은 qualify, 임포트로 죽은 headroom (2026-09-19)

> 그대로 두는 기록 — 이 날 srv4 단일 GPU 레인에서 돈 세 티켓의 결과와 원시 로그다. 뒤 실행은 이 파일을 고치지 않고 새 기록을 남긴다.

속도 주장은 없다(헌장 D17). 장치: NVIDIA GB10(sm_121a), 이미지 `st-engine:glm53`, 프로덕션 `st-glm53` 옆, 플릿 임대 없음.
체크포인트 없이 `probes/qwen38_config.json` 의 형상, 합성 가중치와 입력.

| 티켓 | 트리(main) | 프로브 | 결과 | 로그 |
|---|---|---|---|---|
| `qwen38-cells-0919` | `3ba51e9c` | `engine_kernel_check.py --lanes qwen38_cells` | **실패 — qualify 에서 죽음**(6.2 s), GPU 케이스는 돌지 않음 | [cells-3ba51e9c-qualify-failed.log](cells-3ba51e9c-qualify-failed.log) |
| `qwen38-cells-0919b` | `51a332a4` | 같은 프로브 | qualify 통과, GPU 케이스 **57 건 전부 통과**(118 s) | [cells-51a332a4.log](cells-51a332a4.log) |
| `qwen38-hc-headroom-0919` | `51a332a4` | `engine_qwen38_hc_mix_fused.py`(예산 4 GiB) | **실패 — 임포트**(4 s): 레인은 `engine/`·`probes/` 만 보내는데 프로브가 `bench.probe_report` 를 임포트 | [headroom-51a332a4-import-failed.log](headroom-51a332a4-import-failed.log) |

두 트리 사이에 엔진 코드 차이는 없다(`51a332a4` = `3ba51e9c` + #1206: `bench/fleet_onepass.py` 한 줄과 표).

## 1. 캐리 Q7 — 디코드 블록 선택 한 발사: GB10 에서 통과

`tests.test_engine_qwen38_qsa_select.SelectTests` 6 건 전부(#1202, `engine/kernels/qsa_select.py`):
규칙 자체(안정 내림차순 정렬의 앞 k 개, k 512, 700 · 1,024 · 5,000 · 32,768 열, visible 0 / 1 / k−1 / k / 전체)에 대조,
동률이 k 번째 자리를 가로지를 때 낮은 블록부터, 보이는 블록 밖 열의 inf·NaN 은 고르지 않음, 음수 점수의 순서,
**동률 없는 점수에서 torch.topk 와 같은 집합**, 행 뷰와 거부 케이스.
같은 실행의 나머지 51 건(M2 의 라우터, K3 의 네이티브 chunk, Q8 · Q10 · Q11, M1, S1, K1, K4, Q3 – Q6, MHCV41 · MLA 글루)도 통과.

판정하지 않은 것: `qsa_select.WIDEST`(32,768 열)의 교차점. 캡처 버킷의 실제 열 수는 2 의 거듭제곱이 아니어서(1152 … 32,832; 다른 세션의 지적)
131K 버킷은 지금 torch 형태로 간다 — 교차점의 GB10 실측은 캐리 Q9 의 스윕 몫이다.

## 2. 발견 — 부팅의 qualify 가 한 번 죽었고, 재현되지 않았다

첫 티켓은 `lanes.qualify` 의 셋째 항목에서 죽었다:

```
RuntimeError: QSA norm and partial rotation drift from engine/modules beyond max 0.05 / rms 0.02:
{'norm_rope_4x128': (0.9379310607910156, 0.017086993902921677)}
```

- 같은 코드·같은 시드로 **3 분 뒤의 재실행은 통과**했고 값은 `(0.00690, 0.00232)` — 2026-09-18 의 GB10 실행(`qwen38_qsa_folds_20260918`)과도,
  개발 PC 의 RTX 5050 과도 소수점까지 같다. 5050 에서 `norm_rope_partial` 을 두 셀 × 네 행 수로 2만 번 넘게 발사해도 어긋난 원소는 0 이었다.
- 크기로 보면 (max 는 크고 rms 는 작다) 300 행 케이스에서 **헤드 하나의 회전 절반 32 채널** 정도가 틀린 양이다.
- 세운 가설 하나는 **기각**: 커널이 앞 64 채널을 두 번 쓰지만(정규화 전체 저장, 그 위에 회전 저장) 경합은 아니다 — 컴파일된 TTGIR 에서 32 원소 저장은
  모든 워프의 스레드가 `tid % 32` 로 중복 실행하고, 주소 `out + 32 + i` 에 회전 전 값을 쓰는 스레드는 같은 스레드가 뒤이어 회전 값을 쓴다.
- 비교의 **양쪽이 모두 GPU 계산**이다(`norm_rope_partial` 대 torch 의 `rmsnorm_unit_offset` → `rope_tables` → `apply_rope`): 어느 쪽이 틀렸는지 이 기록은 가르지 못한다.
- 그 실행의 조건: 프로덕션 옆, MemAvailable 22.4 – 23.4 GiB 에서 10 분 대기 뒤 입장, 바로 앞뒤로 다른 세션의 레인 티켓이 순차 실행(겹치지 않음).

**의미:** 이 qualify 는 플릿 부팅도 돈다(D3: 어긋나면 부팅이 죽는다). 간헐적이면 부팅이 이유 없이 한 번씩 죽을 수 있고, 같은 발사가 서빙에서 어긋나면
조용한 품질 저하다. 원인은 모른다. 다음에 볼 것: qualify 가 죽을 때 어긋난 (행, 헤드, 채널)과 양쪽 값을 덤프하게 하기, 같은 티켓을 여러 번 돌려 빈도 재기.

## 3. headroom 프로브는 임포트에서 죽었다 — 고침

`probes/run_engine_probe.sh` 는 레인의 박스로 `engine/` 과 `probes/` 만 rsync 한다. `engine_qwen38_hc_mix_fused.py` 가 모듈 머리에서
`bench.probe_report` 를 임포트해서 4 초 만에 죽었다(내 누락: 로컬에서는 리포 전체가 있어 드러나지 않았다). 이 PR 이 보고 계약 임포트를 선택 사항으로
바꾸고 테스트로 고정한다. 머지 뒤 티켓을 다시 넣는다.
