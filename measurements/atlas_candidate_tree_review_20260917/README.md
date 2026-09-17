# Atlas 기법 2·3 상세 검토 — 2026-09-17

**2번은 기존 후보 병합 계약을 유지하는 작은 변경으로 검토할 가치가 있다. 3번의 고정 spine+leaf는 보유한 t=0 자료에서 수용 길이 손해다. t=1은 이 결론의 대상이 아니며 미판정이다.**

ST 검토 기준은 `6b588418b47bfb73383ec3d755f6486c1ca9e518`, Atlas는
`95f674951d6ab8f491f7907804a462c170f9c048`이다. 엔진 코드 변경·GPU 실행·부팅은 하지 않았다.
`audit.py`가 원본 파일 해시와 정수 합계를 확인하고 모든 고정 형태를 열거한다.
결과는 [audit.json](audit.json)에 보존한다. 모델 실행 결과를 새로 만든 것이 아니다.

## 2. 후보 64개를 전체 vocabulary로 복원하지 않는 병합

현재 경로는 `local logits → rank별 16개 packed key → TP4 64개 → dense 복원 → top16 → selector`다.
[vocab.py](../../engine/modules/vocab.py)와 [draft_select.py](../../engine/kernels/draft_select.py)에서
로컬 선택, 통신량 축소, selector 점수와 greedy walk 융합은 이미 구현돼 있다.
Atlas의 [GPU selector](https://github.com/Atlas-Inf/atlas/blob/95f674951d6ab8f491f7907804a462c170f9c048/kernels/gb10/common/dflash2_candidate_selector.cu)는
top4/rank64이며 ST의 top16/rank256과 다르다. Atlas 코드의 동점 선택은 큰 token ID를 선호하므로 직접 이식하지 않는다.

C=1, K=7에서 gathered packet은 7×64×8 = **3,584 B**, dense FP32는
7×154,880×4 = **4,336,640 B (4.136 MiB)**다. 단, `CandidateBuffer`는 이미
상주 버퍼에서 이전 후보만 지우고 새 후보만 쓴다. **매번 4.136 MiB 전체를 채우는 경로라는 앞선 설명은 정정한다.**
남은 비용은 큰 주소 공간을 읽는 radix top-k와 그 중간 작업이다. head GEMM과 TP collective는 별도다.

[9월 15일 component 기록](../st_vocab_selection_20260915/README.md)의 RTX 5050 C=1 값은
로컬 packet 8.053 µs, simulated peers를 합친 전체 top-k 경로 91.669 µs다.
실통신·head·모델 forward가 없고 GB10도 아니므로 이 차이를 현재 서비스 절감 시간으로 쓰면 안 된다.
수용률 향상 기법이 아니라, 같은 제안을 더 싸게 만드는 기법이다.

### 동점 순서를 유지할 구체적 경로

단순히 64개 key를 정렬하면 후보 집합은 맞아도 동점 순서는 바뀐다. 실제로 §87에서 이 때문에 병합 최적화를 철회했다.
selector는 같은 최종 점수에서 첫 열을 선택하고, sampled 경로는 후보 열 순서에 맞춰 확률과 난수를 사용한다.
따라서 후보 집합만 맞는 구현은 기존 계약을 충족하지 않는다.

기록된 Torch `2.13.0+cu132`의 git `cf30153c4c131c8164ee7798e5022d810682e2cb`를 추적했다.
해당 [CUDA top-k](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/cuda/TensorTopK.cu)는
이 형상에서 multi-block 선택 후, **cutoff보다 큰 값은 원래 vocabulary ID 순으로, cutoff와 같은 값은 그 뒤에 ID 순으로** 수집한다.
[top-k wrapper](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/cuda/TensorTopK.cpp)와
[작은 정렬](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/cuda/Sort.cu)은
그 16개에 unstable SmallBitonicSort를 적용한다. 이 마지막 순열까지 재현하면 dense 공간은 필요하지 않다는 소스 기반 설계다.

1. 64개 packed 후보에서 현재와 같은 ordered-FP32 cutoff와 상위 16개 집합을 구한다.
2. 선택된 16개를 위의 두 그룹과 원래 token ID로 배치한다.
3. 먼저 기존 Torch의 동일한 16개 unstable sort와 ID gather를 재사용한다. 이 단계까지 통과한 다음에만 작은 정렬 자체의 융합을 고려한다.
4. `-inf` cutoff에서는 packet 밖의 dense 배경 열도 기존 답에 포함된다. 최저 16개 vocabulary ID를 가상 배경 후보로 보충하고 packet에 존재하는 ID는 그 실제 값으로 덮어야 한다. sentinel padding과 중복 ID는 제거한다.

NaN의 ordered key, ±0, ±inf, decodable tail, 후보가 이동하는 graph replay와 inactive row 재사용도 보존해야 한다.
일반 score 비교와 radix의 bit ordering은 같지 않을 수 있으므로 cutoff 분류에도 기존 ordered key를 쓴다.
Torch의 일반적인 API 동점 보장이 아니라 **해당 빌드 소스의 동작에 의존하는 설계**다.
실제 재생 이미지의 `torch.version.git_version`과 CUDA 출력 확인은 아직 안 했으며, 소스 분석을 GPU 동등성 증명으로 부르지 않는다.

최소 검증은 기존 CUDA top-k의 values/IDs 일치, 같은 상태에서 7개 draft 전부 일치,
t=1에서는 같은 난수의 sampled IDs·q 확률·검증 결과 일치다. 그 뒤 동일 런타임의 전체 proposal 시간을 비교한다.
기존 캡처의 drafter/head 재생으로 full target 부팅 없이 수행 가능한 작업이다. 엔진 tok/s 주장은 이후 같은 빌드 C=1 비교가 필요하다.

## 3. 검증 8행 안의 spine+leaf

Atlas의 [형태 정의](https://github.com/Atlas-Inf/atlas/blob/95f674951d6ab8f491f7907804a462c170f9c048/crates/spark-model/src/speculative/tree_shape.rs)는
root 1행과 최대 7개 draft node, 깊이당 최대 4개 후보다. 주경로 이외는 자식 없는 잎이다.
확인한 Rust 호출부에서는 형태 정의/테스트까지만 있고, 이 구조가 production verification에 연결돼 있다는 근거는 찾지 못했다.
[형태 검색 스크립트](https://github.com/Atlas-Inf/atlas/blob/95f674951d6ab8f491f7907804a462c170f9c048/scripts/tree_shape_search.py)는
조건부 coverage를 쓰는 점이 유용하다. 다만 0.95 haircut, 미관측 깊이의 마지막 확률 반복, 행당 1.2% 비용은 모델 가정이다.
검색 루프도 spine 길이 1..5만 열거해 유효한 `spine6+leaf1`을 놓친다. 이 수치/검색을 ST 판정기로 가져오면 안 된다.

### 보유한 t=0 자료의 산술

기존 주경로를 그대로 두고 길이를 L로 자르면 잃는 토큰은 `max(A-L,0)`다.
첫 기각 깊이에 정확한 대안 잎이 있으면 최대 **1토큰**을 더 얻는다. 그 잎에는 후속 경로가 없기 때문이다.
따라서 t=0 고정 상태에서:

`ΔA = 첫 기각을 잎으로 구제한 횟수 / N - Σ max(A-L,0) / N`

아래는 모든 깊이 배치를 열거한 **완벽한 잎의 낙관적 상한**이다. 각 자료의 원래 spine은 유지하며,
실제 후보에 정답이 없어도 맞힐 수 있다고 가정한다. 괄호 안은 bonus를 포함한 예상 emitted tokens/step이다.

| 고정 8행 | 9/16 캡처 16사례 | 9/15 캡처 6,285스텝 |
|---|---:|---:|
| chain7 기준 | 2.8750 (3.8750) | 3.6625 (4.6625) |
| spine6 + leaf1 상한 | 2.8750 (3.8750) | 3.5313 (4.5313) |
| spine5 + leaf2 상한 | 2.7500 (3.7500) | 3.3376 (4.3376) |
| spine4 + leaf3 상한 | 2.5625 (3.5625) | 3.0662 (4.0662) |

9/16의 spine4는 완전 수용 5사례에서 15토큰을 잃는다. 가장 유리한 세 깊이를 모두 구제해도 10토큰만 회수한다.
총 46→최대41, 평균 2.875→최대2.5625다. tok/s 본전을 내려면 전체 step이 최소 **8.06%** 빨라져야 한다.
16사례뿐이므로 일반 성능 판정은 아니지만, 이 자료에서 해당 고정 형태가 수용 길이를 높인다는 주장은 배제할 수 있다.

별도 빌드/문항인 [9/15 원본](../st_draft_rank_overlap_20260915/README.md)에서도 부호는 같다.
실제 기록된 unary 순위에서 walk를 제외한 형제를 쓰면 최선 spine6+leaf1은
7번째 체인에서 1,738개를 잃고 첫 위치에서 429개를 구제한다. emitted tokens/step은
**4.6625→4.4543**, 약 **4.47%** 감소다. bilinear 점수로 정렬한 대안의 실측이 아니며,
동기 진단 일부는 draft rank를 1순위로 대신 기록했다는 원본 한계도 이어받는다.
그 한계와 무관한 완벽한 잎 상한조차 기준보다 낮다. 두 자료를 합쳐 통계를 내지는 않았다.

### 같은 8행이어도 비용은 같지 않다

ST는 이미 [일반 tree 실험](../../engine/profiles/glm53/tree_decode.py), 부모에 조건부인 DFlash selector,
ancestor closure와 예측 expert bytes 비용 선택, FP32 KDA factors, branch-private DSA bank를 갖고 있다.
단순히 tree를 추가하는 것이 새 기법은 아니다. [최근 검증 범위](../st_tree_bank_20260914/README.md)는
CPU/interpreter/오프라인 컴파일이며 production full-step graph와 실제 tree tok/s는 미검증이다.

- **KDA:** 현재 spine-first DFS는 형제로 되돌아갈 때 조상 factors를 재적용한다.
  chain7은 full-state pass 8회, 깊이 1·2·3에 잎이 있는 spine4는 14회다.
  이는 [현재 state_updates 산술](../../engine/modules/speculative_tree.py)이며 시간 1.75배라는 뜻은 아니다.
- **DSA:** 형제는 주경로와 다른 position/조상 mask/완성 pool을 가진다. 같은 8행이라도 주소 준비와 commit이 다르다.
  기존의 paged/private KV 직접 읽기와 sibling 격리를 유지해야 한다.
- **MoE:** 9/15의 올바른 형제 경로는 기존 8행에 더할 때 expert 읽기 약 +7.3%였다.
  이것은 9번째 행의 관측 proxy다. 삭제되는 체인 행과 추가 형제를 함께 고려하지 않고 고정 8행의 순비용으로 쓰면 안 된다.
  전문가 bytes가 시간에 선형이라는 가정도 실측 하한이 아니다.
- **호스트:** 현재는 CPU lattice 선택, object broadcast, topology/transaction 준비가 있다.
  고정 8행만 선언한다고 기존 CUDA graph의 비용이 되는 것은 아니다.

재검토할 형태는 **chain7을 기본으로 유지하고, 끝 체인 1행과 앞쪽 잎 1행의 가치가 역전되는 상태에서만 바꾸는 정책**이다.
예를 들어 row7을 depth7/parent6 또는 depth1/parent0으로 고르는 두 형태부터 제한한다.
이를 위해 실제 `unary + alpha × predecessor-conditioned edge` 점수와 첫 기각 시 대안 coverage를 수집하고,
요청 단위 holdout에서 tail 수용 손실·추가 verification/commit 비용까지 합쳐 평가해야 한다.
낮은 margin 자체가 낮은 타깃 수용률이라는 보장은 없다. q의 softmax mass는 관측 acceptance가 아니다.
새 형태 선택과 부모/깊이/DSA mask 생성이 GPU에 남는 8행 graph가 필요하므로, 현 eager 실험의 단순 설정 변경으로 끝나지 않는다.

## t=1에 대한 추가 확인

위 두 수용 자료는 모두 **temperature=0**이다. [9/16 수집기](../st_draft_sensitivity_20260916/README.md)는
C=1 greedy만 받고, [기존 기각 진단](../../engine/profiles/glm53/draft_diagnostics.py)은
`temps <= 0`만 기록한다. t=1 평가·성능 자료는 존재하지만
[성능 분석](../st_tool_eval_bundle_20260916/with-perf/performance-analysis.json)은
speculative acceptance counters를 수집하지 않았다고 명시한다.
따라서 t=1의 깊이별 수용, 대안 확률과 비용에 대한 위와 같은 판정 자료는 확보하지 못했다.

더구나 ST sampled 경로의 [block_verify](../../engine/base/sampler.py)는
`P_i = min(P_{i-1} p_i(x_i)/q_i(x_i), 1)`과 다음 위치의 residual mass로
수용 임계값을 정한다. 마지막 위치는 다른 threshold를 쓴다.
**K를 줄이면 단순히 기존 수용 길이를 L에서 자르는 것과 달라질 수 있다.**
타깃 argmax와 후보의 일치 여부로 t=1을 평가해서도 안 된다.

t=1의 별도 판단에는 실제 sampled prefix, proposal support/정규화 q, target p,
uniform의 위치별 키, block thresholds와 실제 수용 길이 및 top_p 등 요청 조건이 필요하다.
새 형제 아래 p를 구하려면 해당 branch의 target 실행도 필요하다. 현재 greedy replay 캡처만으로는 복원할 수 없다.
또한 기존 tree verifier는 temperature!=0을 거부한다. sampled tree의 수용·잔여분 재샘플링 규칙이
목표 분포를 보존하도록 별도 설계/검증돼야 한다. 지금의 greedy tree를 켜는 것은 t=1 해결책이 아니다.

2번은 후보 values/순서와 sampled q/난수 대응을 그대로 유지하면 t=1에도 적용할 수 있는 비용 절감이다.
3번은 **t=0 고정 형태는 비추천, t=1 및 상태별 선택은 미판정**으로 구분한다.

재현: `python3 measurements/atlas_candidate_tree_review_20260917/audit.py`
