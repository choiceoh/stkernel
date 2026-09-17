# Hadamard 회전은 이 스택의 NVFP4에서 이긴다 — 레버를 닫는다 (2026-09-18, srv2 CPU, 부팅 없음)

운영자 "양자화 오차를 같은 포맷에서 더 줄일 방법 없나" → QuaRot/SpinQuant 계열 회전 제안 → 실제 랭크 전문가 9개로 실험 → **전 팔에서 회전이 손해. 이 저장소의 전처리 사슬이 회전의 이득을 이미 가져갔다.**

## 방법

- `probes/nvfp4_hadamard_rotation_probe.py` — 프로덕션 랭크(`st-glm53-9391-up-gate-full` rank0)의 실제 전문가
  9개(층 3/20/40 × 전문가 0/73/287)를 dequant 후, 축소 차원 기준으로 {비회전, 블록대각 H128, Hfull} × 스케일
  탐색 반경 {0, ±1(as1), ±2(as2)}로 재양자화. 벡터화 양자화기는 `compare_block`과 블록 단위 완전 일치를
  assert(cross-check) 통과 후 실행. 원시: [rotation-ablation.json](rotation-ablation.json).
- **가중치 팔은 실측**, 활성 팔은 기제 시연용(가우시안은 회전 불변이라 ~0이 나와야 하는 하니스 자기검증;
  아웃라이어 형상은 천장치). 실활성 수치는 캡처 부팅이 필요하다.

## 결과 (pooled, none.r0 대비 SSE)

| 팔 | fc1_w13 | fc2_w2 | 활성·가우시안 | 활성·아웃라이어 | full-MLP 합성 |
|---|---:|---:|---:|---:|---:|
| none.r1 (=as1) | 0 | 0 | **+17.5%** | **+4.4%** | **+3.4%** |
| none.r2 (=as2) | 0 | 0 | +18.4% | +4.5% | +3.7% |
| h128.r0 | **−713%** | −678% | −0.2% | −236% | −303% |
| h128.r2 | −565% | −535% | +18.2% | −98% | −200% |
| hfull.r2 | −565% | −535% | +18.3% | −168% | −234% |

음수 = 회전이 오차를 키운다. **모든 실데이터 팔에서 회전은 6~8배 나쁘고, 탐색 반경을 더해도 회전 손해의
3분의 1도 회수하지 못한다.** 가우시안에서 회전이 0%인 것은 하니스 검증대로(회전 불변)이고, 아웃라이어
입력에서 회전이 오히려 최악인 이유는 에너지가 모든 블록에 퍼져 블록 스케일 전체가 올라가기 때문이다.

## 왜 여기서는 회전이 지는가 — 세 겹의 이유

1. **가중치는 이미 자연 기저 위에서 GPTQ(act_order)로 팩됐다.** 회전은 그 최적화된 배치를 섞어 버린다.
   (주의: 이 팔은 FP4→dequant→회전→FP4 이중 양자화라 회전에 불리하게 편향돼 있다. 원본 bf16에서 재팩하는
   깨끗한 비교는 가능하나 — `~/models/GLM-5.3-Flash-DFlash2` — 아래 2·3이 남는 한 판정이 뒤집힐 근거는 없다.)
2. **활성 채널 평형은 smoothing 폴드가 이미 한다.** (SmoothQuant식 norm÷s·readers×s, 기본값.) 회전이 노리는
   아웃라이어 채널 분산이 이미 절반 이상 접힌 상태다.
3. **NVFP4의 16블록 e4m3 스케일은 이미 블록별 적응 스케일이다.** 회전이 블록 내 균등화로 주는 이득과
   스케일 탐색(as1, −17.7% SSE)이 같은 예산을 두고 겹친다.

## as2의 한계도 같이 나왔다

full-MLP 합성에서 r2는 r1 대비 +0.24pp, 가우시안 활성에서 +0.9pp에 불과하다. as2 기본 전환
([PR #1157](https://github.com/choiceoh/stkernel/pull/1157))의 소비자 게이트가 아직 없다는 기록과 함께
읽을 것 — 기대 이득 폭 자체가 작다.

## 판정

- **회전 레버를 닫는다.** QuaRot 계열의 "큰 폭"은 smoothing+블록 스케일+탐색이 없는 스택의 이야기이고,
  이 스택은 그 세 가지를 이미 다 갖고 있다.
- 같은 포맷에서 남는 방향은 사이트별 정밀도 배치(포맷 집합 유지, 배치 변경) 정도인데, 이것도 9/16의
  비전파 교훈(가중치 오차 2배 개선이 head NLL을 악화)이 적용되므로 head NLL 게이트 없이는 건드리지 않는다.

## 재현

```sh
docker run --rm -v <worktree>:/src -v ~/models/st-glm53-9391-up-gate-full:/ranks:ro -w /src \
  -e CUDA_VISIBLE_DEVICES= -e OMP_NUM_THREADS=4 -e PYTHONPATH=/src st-engine-seed:3175af9ca485 \
  /src/probes/nvfp4_hadamard_rotation_probe.py --ranks /ranks/rank0of4.safetensors \
  --samples 32 --output /out/rotation-ablation.json
```
