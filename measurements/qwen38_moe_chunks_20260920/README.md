# Qwen3.8 MoE 9~16행을 두 번의 micro 계산으로 — 기본 끔 (2026-09-20)

질문: K=3에서 C=3/4 타깃 검증의 12/16행이 타는 static MoE 대신 균등한 두 micro 계산이 유리한가? **단일 GPU 검사 완료, 플릿 품질·속도 채택 판정 전, PR 미개설**.

`--moe-decode-chunks` / `ST_MOE_DECODE_CHUNKS=1`로 9~16행만 `(ceil(M/2), floor(M/2))`로 나눈다. C=1/2의 4/8행과 프리필 경로는 그대로다. 한 행씩 계산하지 않는다. 동일 라우팅을 한 번 remap하고 두 micro 결과를 이어 붙인다. sentinel 128 입장 검사를 함께 적용한다.

## 검사와 시간

- srv4 NVIDIA GB10, image `st-engine:glm53`, PyTorch 2.13.0+cu132 / CUDA 13.2, TP 통신 없음.
- synthetic Qwen geometry E=128, H=2560, I=640, top-k=10, scale search=2. 실제 체크포인트 출력 품질을 검사한 것이 아니다.
- 최종 `qwen-moe-chunks-0920f`, ticket `17898316833969039`, source `2c306ef4`. 28/28 검사 통과. 독립 activation-quantized oracle 상대 오차 최대 0.0083682(<0.02), finite/zero, router 일치, replay/새 입력, scratch 격리 확인.
- 9..16과 4/8행 모두 두 입력에서 기존 결과와 **바이트 동일**. 이 입력 범위의 증거이며 모든 모델 출력이 같다는 주장은 아니다.
- 64MiB cold scrub, 8개 route 표본 × 4개 교대 순서. `cat` 포함, router는 양쪽에서 제외. 모든 시간 표본은 `probe-f.log`에 있다.

| 행 | 기존 cold µs | 후보 cold µs |
|---:|---:|---:|
| 16 | 719.01 | 681.84 |
| 15 | 697.41 | 602.83 |
| 14 | 654.22 | 562.21 |
| 13 | 689.70 | 550.37 |
| 12 | 674.99 | 493.18 |
| 11 | 614.48 | 524.30 |
| 10 | 598.91 | 463.65 |
| 9 | 594.30 | 447.38 |
| 8 | 323.10 | 321.78 |
| 4 | 260.13 | 257.63 |

4/8행은 같은 코드이므로 작은 차이는 성능 개선으로 보지 않는다. 12행 시간 −26.9%, 16행 −5.2%는 컴포넌트 수치다.

재현: `ST_PROBE_GIB=8 REPO=$PWD bash bench/fleet.sh run --gpu --detach SESSION 10 NOTE -- bash probes/run_engine_probe.sh probes/engine_kernel_check.py --lanes qwen38_moe_chunks --output /cache/SESSION.json`.

## 실패도 남기는 경로

| 세션 / ticket | 소스 | 원시 / 결과 |
|---|---|---|
| qwen-moe-chunks-0920a / 17898302823813945 | 02f20225 | `probe-a.log`, 첫 두 micro 확인 |
| qwen-moe-chunks-0920b / 17898304433832891 | c7e093e6 | `probe-b.log`, static M64 시도에서 illegal memory access. M32 미실행. 변경 폐기 |
| qwen-moe-chunks-0920c / 17898308443878134 | 63e9473e | `probe-c.log`, 서빙 연결 후 28검사 통과 |
| qwen-moe-step-0920d / 17898309643891618 | 63e9473e | `step-d.log/json`, 4GiB cap에서 두 자식 모두 OOM. 이전 harness rc=0은 잘못된 성공이었고 유효 그래프 없음 |
| qwen-moe-step-0920e / 17898312043919464 | 6d6e6dd4 | `step-e.log/json`, cap 증가 후 실제 rank3 층4..7 + MTP 검사 완료 |
| qwen-moe-chunks-0920f / 17898316833969039 | 2c306ef4 | `probe-f.log`, 기존 결과와 바이트 동일 검사 추가, 위 최종 표 |

step-e는 `OneRankComm`으로 통신을 대체한 단일 GPU 축소 그래프다. target wall µs: C1 4040.0→4301.7, C2 4282.5→4331.5, C3 5976.7→4643.0, C4 6291.6→5616.2(6블록), 6655.8→5419.7(43블록). 한 방향 순서뿐이며 drafter 시간도 크게 흔들렸다. **C1 +6.48%를 통과로 판정하지 않는다.** 실제 C1 tok/s의 5% 하락 제한은 TP4에서 다시 검증해야 한다.

실패한 자식, 누락 그래프, 실패한 수치 arm을 성공으로 넘기지 않도록 probe를 수정했다. 최종 옵션은 기본 꺼짐이며 플릿 비교 계획은 [별도 기록](../qwen38_decode_ab_20260920/README.md)에 있다.
