# GLM-5.3 출력 붕괴 사후 조사 — 2026-09-01

- Date: 2026-09-01
- Scope: GLM-5.3 TP=4의 EP·비-EP 공통 무의미 출력, temperature 0에서의
  동일 토큰 반복, speculative acceptance 0%
- Broken deployment ancestry: `556e7a8` (#143)
- Code fixes already merged: `3e16dc2` (#150), `27b5dc3` / `987dfd4` (#153)

방법: ① 깨진 배포 ancestry의 one-shot AllReduce·FP8 lm-head·vocab mask·V2
sampler를 최종 합성 상태로 정독 ② 설치된 V2 rejection sampler와 부팅 로그 대조
③ EP 전용 경로와 EP·비-EP 공통 hidden-state 경로 분리 ④ `origin/main`의
#150·#153 수정과 깨진 소스를 줄 단위 비교. 이 문서는 코드 귀속과 현재 로그
증거를 고정하며, 생성 품질의 matched runtime A/B는 별도 종료 게이트로 남긴다.
조사 과정에서는 배포·재기동·설정 변경·추가 API 요청을 하지 않고 기존 로그만
읽었다.

## 판정

깨진 `556e7a8`에는 **EP·비-EP 공통 hidden state를 망가뜨릴 수 있는 확정
correctness 결함이 두 개 동시에 존재했다.**

1. #90 one-shot AllReduce의 payload publish race
2. #133 MHC ONEPASS 제어흐름이 기본-off small-M split을 48로 덮어쓴 회귀

#90은 각 thread의 payload copy/fence가 끝났다는 block-wide 합류 없이 thread 0이
그 block의 완료를 선언했다. #133은 T≤16 decode에서 정상 split 4/8을 generic
planner의 48로 바꿔 256-thread MHC 커널의 hidden loop를 0회로 만들었다. 둘 다
EP dispatch 바깥에서 target hidden state를 훼손할 수 있다.

MHC 결함은 조건이 맞으면 매 layer에서 결정적으로 0-iteration이 되므로
prompt-independent collapse의 **더 직접적인 1순위 후보**다. one-shot 결함도
실제 memory-ordering 위반이지만 발현은 scheduling에 의존한다. 깨진 부트에는
둘이 함께 있었고 한 축씩 고정한 생성 A/B가 없으므로 사건을 하나에만 귀속하지
않는다.

| 축 | 판정 | 근거 |
|---|---|---|
| MHC small-M #133 | **확정 코드 결함·사건 1순위 후보** | ONEPASS off가 정상 4/8 split을 48로 덮어씀; H=4096에서 256-thread hidden loop 0회 |
| one-shot AR #90 | **확정 코드 결함·공동 후보** | copy/fence와 completion atomic 사이 block barrier 부재; real mode로 모든 작은 BF16 TP AR을 대체 |
| vocab mask #132 | **배제** | target은 154,880행 중 suffix 24행만 차단; local mask는 draft candidate 전용 |
| V2 sampler #141–#143 | **배제** | temp=0 첫 거절에서 target argmax 방출; #141은 thinking-budget predicate뿐, #142 V1 코드는 비활성 후 #143에서 revert |
| FP8 lm-head | **이번 원인으로 미귀속** | 과거 #129 dead-head는 #131/#132/#135에서 제거·fail-closed; 현행에는 BF16 수치 비교 부재라는 별도 방어 공백이 남음 |
| b12x EP fixed pair output #146 | **별개 EP 결함** | EP 경로의 미초기화 행 문제는 실재했지만 비-EP 붕괴를 설명할 수 없음 |

두 코드 결함 자체는 정적으로 확정된다. 다만 #150과 #153을 한 축씩만 적용한
동일 프롬프트 생성 A/B는 완료되지 않았다. 최신 main의 정상 출력은 복구를
증명해도 어느 수정이 사건의 실제 발화점이었는지는 분리하지 못한다.

## 관측 증상

비-EP, temperature 0에서 서로 무관한 두 요청이 같은 출력을 냈다.

```text
1부터 5까지 세어줘        -> Theapolapolapolapolapola...
대한민국의 수도는?         -> Theapolapolapolapolapola...
```

반복 토큰은 `The`(id 785)와 `apol`(id 16640)이고 둘 다 TP rank 0의
vocab shard 범위에 있다. speculative acceptance는 0.0%, 평균 acceptance
length는 1.00이었다. EP에서도 문맥과 무관한 토큰열이 나왔다.

이 지문만 보면 “rank 0 이외 어휘가 마스킹됐다”는 가설이 자연스럽다. 그러나
실제 마스크 bound와 sampler 방출 계약을 대조하면 그 가설은 성립하지 않는다.

## #90의 정확한 publish race

#90(`b907eb4`)은 one-shot AR을 3 launch에서 고정
`<<<ARGRID=256, 256>>>` 1 launch로 합쳤다. 깨진 배포 소스의 순서는 다음과
같다.

```cpp
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; ...)
    c->tx[slot][i] = src[i];
if (owns)
    __threadfence_system();
if (threadIdx.x == 0)
    last = atomicAdd(&c->done_ctr, 1) % ARGRID == ARGRID - 1;
__syncthreads();
if (last && threadIdx.x == 0)
    c->tx_seq = nxt;
```

근거:
[깨진 `556e7a8` 소스](https://github.com/choiceoh/stkernel/blob/556e7a8/overlay/modules/tp_oneshot_ar/dsv4_oneshot_ar.cu#L164-L183),
[#90](https://github.com/choiceoh/stkernel/pull/90).

`__threadfence_system()`은 호출한 thread 자신의 write 순서만 보장한다. block의
다른 warp가 copy와 fence를 끝냈다는 뜻이 아니다. 따라서 warp 0의 thread 0은
자기 element를 쓴 직후 completion atomic을 실행할 수 있다. 모든 block의 thread
0이 atomic에 도달하면 마지막 block이 `tx_seq`를 publish하지만, 다른 warp들의
payload write는 아직 진행 중일 수 있다.

atomic 뒤의 `__syncthreads()`는 `last` 공유값을 같은 block에 전달할 뿐이다.
이미 completion counter가 증가한 뒤이므로 payload 완료 계약을 복구하지 못한다.

고정 256-block geometry가 race window를 더 키웠다.

- plain decode T=1: `n=1*4096`, 16 data-owning blocks + 240 empty blocks
- C=1, k=7 verify T=8: `n=8*4096`, 128 data-owning + 128 empty blocks
- C=2/C=4 verify: T=16/32, 최대 `MAXEL=131072`

empty block도 completion counter를 증가시킨다. payload를 쓰는 warp들의 완료와
무관한 atomic이 빠르게 누적되어 부분 payload publish 가능성을 높인다.

## #133의 MHC small-M split 회귀

#133(`61459ec`)은 opt-in ONEPASS early-return을 추가했다. ONEPASS 자체는
profile에 선언되지 않아 기본 off였지만, early-return 뒤 generic planner를
`else`에 넣은 제어흐름이 stock small-M 경로까지 바꿨다.

```python
use_small_fma = num_tokens <= 16
if use_small_fma:
    n_splits = 8 if num_tokens < 8 else 4

if use_small_fma and onepass_enabled:
    return onepass_result
else:
    if use_deep_gemm:
        n_splits = compute_num_split(...)

mhc_fused_tilelang(..., n_splits=n_splits)
```

근거:
[깨진 `556e7a8` dispatcher](https://github.com/choiceoh/stkernel/blob/556e7a8/overlay/modules/glm53_mhc_tilelang/tilelang.py#L664-L739),
[#133](https://github.com/choiceoh/stkernel/pull/133).

GB10·GLM53 decode 값을 대입하면 덮어쓴 값은 48이다.

```text
num_tokens <= 16
hc_hidden_size = 4 * 4096 = 16384
grid_size = ceil(num_tokens / 64) = 1
n_sms = 48
compute_num_split = min(48, ceil(16384/64)/4) = 48
```

small-M 진입 전에 있는 `n_splits in (1,2,4,8)` assert는 덮어쓰기보다 앞이라
48을 잡지 못한다. `mhc_fused_tilelang`는
`h_per_split = 4096 // 48 = 85`,
`h_iters = 85 // 256 = 0`을 계산한다. 따라서 핵심 hidden loop가 한 번도 돌지
않고 `residual_out`은 `empty_like`의 미초기화 상태로 남으며 FMA/squared-sum
출력은 0 accumulator에서 기록된다.

근거:
[small-M kernel](https://github.com/choiceoh/stkernel/blob/556e7a8/overlay/modules/glm53_mhc_tilelang/tilelang_kernels.py#L544-L659),
[`compute_num_split`](https://github.com/choiceoh/stkernel/blob/556e7a8/overlay/modules/glm53_mhc_tilelang/tilelang_kernels.py#L50-L59).

이 경로는 T≤16 decode의 layer 사이 MHC post/pre마다 실행된다. race가 아니라
shape로 결정되는 0-iteration이므로, 서로 다른 prompt가 같은 소수 argmax로
수렴한 지문을 one-shot보다 더 직접적으로 설명한다.

#153(`27b5dc3`)은 ONEPASS가 return하지 않은 뒤 generic planner를
`if not use_small_fma`로 제한해 stock 4/8-way split ownership을 복구했다.
[#153 수정](https://github.com/choiceoh/stkernel/blob/27b5dc3/overlay/modules/glm53_mhc_tilelang/tilelang.py#L664-L743),
[#153 PR](https://github.com/choiceoh/stkernel/pull/153).

## 두 결함이 EP와 비-EP에 공통인 이유

GLM53 profile은 다음 설정으로 one-shot을 실제 serving 경로에 넣는다.

```text
VLLM_DSV4_ONESHOT_AR=1
VLLM_DSV4_ONESHOT_SHADOW=0
```

[`glm53.env`](https://github.com/choiceoh/stkernel/blob/3e16dc2/profiles/glm53.env#L243-L264)의 `shadow=0`은 self-test 뒤
NCCL을 관측만 하는 모드가 아니라 one-shot 결과를 실제 forward에 사용한다.
[`CUDACommunicator.all_reduce`](https://github.com/choiceoh/stkernel/blob/3e16dc2/overlay/modules/glm53_oneshot_wiring/cuda_communicator.py#L275-L295)는
TP collective보다 먼저 shim을 호출하고,
[`_eligible`](https://github.com/choiceoh/stkernel/blob/3e16dc2/overlay/modules/tp_oneshot_ar/dsv4_oneshot_shim.py#L257-L291)은
CUDA·BF16·contiguous·131072 elements 이하 tensor를 모두 받는다.

GLM target의 attention output과 row-parallel projection은 이 TP all-reduce를
반복해서 지난다. EP는 MoE expert 배치 방식만 바꾸므로 이 공통 residual 경로를
우회하지 않는다.

MHC fused post/pre도 expert dispatch가 아니라 모든 target layer 사이의 공통
residual 경로다. T≤16이면 EP 여부와 무관하게 같은 48-split·0-iteration 결함을
지난다. 따라서 두 결함 모두 lm-head와 sampler보다 앞에서 prompt 신호를 훼손할
수 있다. 소수 토큰 반복은 “어휘를 막은 것”이 아니라 “어휘에 들어가기 전 hidden
state가 무너진 것”으로 설명된다.

## vocab mask 배제

부팅 로그는 다음 bound를 직접 남겼다.

```text
[vocab-mask] masking ids 154856..154879
(24 rows the tokenizer has no token for) out of 154880
```

target processor에는 local `valid_vocab_size`가 전달되지 않는다.
[`compute_logits`](https://github.com/choiceoh/stkernel/blob/3e16dc2/overlay/modules/glm53_model_wiring/glm5next_model.py#L1068-L1101)가
gather된 target logits의 `[..., 154856:]`만 `-inf`로 바꾼다. 따라서
154,856/154,880행, 즉 **99.9845%**가 살아 있다.

#132의 local mask는
[`DFlash2Qwen3ForCausalLM.candidate_logits_processor`](https://github.com/choiceoh/stkernel/blob/3e16dc2/overlay/modules/glm53_dflash2_fp8_head/qwen3_dflash2.py#L283-L290)에만
`valid_vocab_size`를 전달한다. TP4 local width 38,720에서 생존 행은
`[38720, 38720, 38720, 38696]`이다. 마지막 shard의 끝 24행만 차단된다.

rank 0 어휘만 남기려면 `VLLM_GLM53_DECODABLE_VOCAB=38720`처럼 잘못 작은
override가 필요하다. profile은 tokenizer 자동 검출을 사용했고 로그가
`154856`을 확정하므로 이번 부트에는 해당하지 않는다.

## acceptance 0%와 sampler 배제

설치된 V2 rejection sampler
`vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py`를 확인했다.
temperature 0에서는 target logits의 block argmax를 계산하고, draft와 다르면
첫 거절 위치에 target argmax를 기록한다. greedy non-bonus 경로는 residual
resample도 하지 않는다.

따라서 acceptance 0%이면 출력은 매 step **target argmax**다. draft candidate
mask가 acceptance를 낮출 수는 있어도 `The/apol`을 방출할 수는 없다.
이 증거는 오염 지점을 sampler 뒤가 아니라 target logits 또는 그 앞 hidden
state로 올려 보낸다.

#141의 실제 sampler 변경은 thinking-budget predicate뿐이다.
#142는 이미 profile에서 빠진 V1 `glm53_sparse_q` proposer를 바꿨고 #143에서
전부 revert됐다. 이 세 PR은 일반 temperature 0 방출 법칙을 바꾸지 않았다.

## FP8 경로 판정

#129에는 실제 dead-head 버그가 있었다. post-requant scale 9,696/9,696을 0으로
flush해 head를 죽였고 #131의 live 관측이 이를 확인했다. 그러나 깨진 부트가
사용한 최종 트리에서는:

1. #131이 zero-flush 변형을 제거했고,
2. #132가 unsafe UE8M0 scale을 pack 전에 예외 처리해 BF16으로 fallback하며,
3. #135가 실패한 online quantization 재시도를 막는다.

현재 FP8 head는 scale 형식/배치 계약만 검사하고 BF16 logits 대비 수치 divergence
gate 없이 cache를 공개한다. 이 방어 공백은 별도 수리 대상이지만, 코드에서
`The/apol` collapse를 직접 만드는 현행 경로는 찾지 못했다.

dense FP8의 과거 early-build/stale-copy 버그도 acceptance 0% 전력이 있다. 이번
부팅 로그에는 checkpoint 전체 load 뒤 180개 projection의 최종 rebuild가 다시
기록되어 그 특정 stale-copy 사건은 배제된다. copy probe 예외가 unchecked로
남을 수 있는 감사 공백은 별도다.

## 수정 상태와 현재 부팅의 한계

### #150: one-shot publish 계약

#150(`3e16dc2`)은 incident와 맞는 두 핵심 수정을 이미 main에 병합했다.

1. 각 thread의 `__threadfence_system()` 뒤, thread 0의 completion atomic 전에
   block-wide `__syncthreads()`를 추가했다.
2. fixed grid를 256에서 GB10의 48 SM과 같은 48 block으로 줄이고, SM121a·48 SM
   device contract를 bootstrap에서 강제했다.

수정 소스:
[#150 CUDA ordering](https://github.com/choiceoh/stkernel/blob/3e16dc2/overlay/modules/tp_oneshot_ar/dsv4_oneshot_ar.cu#L165-L189),
[#150 PR](https://github.com/choiceoh/stkernel/pull/150).

#150은 setup·RDMA connect·self-test를 all-rank vote로 바꾸고, real mode commit
뒤 rank-local NCCL fallback을 fatal로 전환했다. 이는 일부 rank만 NCCL로
빠져 collective가 갈라지는 별도 hang 위험도 닫는다.

### #153: MHC small-M split ownership

#153(`27b5dc3`, merge `987dfd4`)은 ONEPASS early-return 이후의 generic
planner를 non-small 경로에만 한정했다. small-M은 ONEPASS off일 때 #133 이전
stock 4/8 split을 그대로 유지한다. AST 회귀 테스트는 small-M assignment와
generic assignment가 같은 제어흐름 body에 다시 들어가는 것을 막는다.

### 관측한 재부팅

현재 재부팅 로그는 수정된 source fingerprint를 확인했다.

```text
[osar] source md5=3b951172 kernels=1
[osar] connected rank=1 world=4 shadow=False
[osar] self-test PASS div=0 (real=True)
```

`556e7a8`의 깨진 CUDA source fingerprint는 `c7295685`, #150 source는
`3b951172`다. 따라서 현재 컨테이너가 수정 소스를 compile했다는 것은 확인된다.
다만 uniform `(6,4096)` 1회 self-test는 race가 우연히 발현되지 않아도 통과할
수 있고, 생성 의미 정합성을 증명하지 않는다.

더 중요하게, 이 부팅은 호스트 시각 00:24에 시작했고 #153은 00:39에 병합됐다.
따라서 관측한 부팅은 #150 one-shot 수정은 포함하지만 #153 MHC 수정은 포함할 수
없다. 이 부팅의 API 200·self-test PASS나 이후 출력은 최신 main의 두 수리가 모두
적용된 복구 증거로 사용할 수 없다.

## 남은 종료 게이트

### 프로덕션 복구 판정

1. #150과 #153을 모두 포함한 revision을 배포한다.
2. 네 rank의 one-shot source fingerprint가 `3b951172`인지 확인하고, 합성된
   `tilelang.py`가 #153의 `if not use_small_fma` 계약을 포함하는지 확인한다.
3. 비-EP, temperature 0에서 사건의 두 exact prompt를 다시 실행한다.
4. 서로 다른 정상 답변, `The/apol` 반복 없음, target/draft acceptance가
   0%에 고정되지 않음을 확인한다.
5. 같은 revision에서 `VLLM_DSV4_ONESHOT_SHADOW=1` 또는
   `VLLM_DSV4_ONESHOT_AR=0` NCCL bracket도 정상인지 확인한다.

latest-main real OSAR와 NCCL이 모두 정상일 때 **프로덕션 복구**는 판정할 수 있다.
그러나 두 수정이 함께 들어가므로 이것만으로 incident를 #90 또는 #133 하나에
귀속할 수는 없다.

### 원인 분리 — production 밖에서만

단일 원인 귀속이 필요하면 알려진 결함을 serving production에 재도입하지 말고
격리 GPU에서 두 축을 독립적으로 비교한다.

| MHC | AllReduce | 판정 목적 |
|---|---|---|
| #153 fixed | NCCL | 정상 기준 |
| #153 fixed | #150 fixed OSAR | one-shot 수정 후 정합성 |
| #133 broken dispatcher | NCCL | MHC split 회귀 단독 재현 |
| #153 fixed | #90 broken OSAR | one-shot race 단독 재현; 반복·비균일 payload 필요 |

마지막 두 cell은 known-bad 코드라 production에서 실행하지 않는다. 이 matrix 없이
사건 인과는 “MHC 1순위, one-shot 공동 후보”로 남긴다.

latest main + NCCL에서도 실패하면 다음 격리 순서는 target FP8 lm-head off →
dense FP8 off다. vocab mask와 draft sampler부터 다시 보는 것은 위 코드·로그
증거와 맞지 않는다.

## 후속 방어

- one-shot self-test를 rank별 상수 1회가 아니라 비균일 payload와
  T=1/8/16/32 ring 반복으로 확장한다.
- boot gate와 별도로 temperature 0 semantic canary 두 개를 배포 종료 조건으로
  둔다. health/API 200과 CUDA assert 0은 출력 정합성 증거가 아니다.
- FP8 target head에 sampled BF16-vs-FP8 argmax/divergence gate를 추가한다.
- vocab bound override는 작은 양수도 허용하므로 tokenizer 자동 bound와 현저히
  다르면 fail-closed하도록 별도 강화한다.
- incident 로그에는 source fingerprint, profile knobs, exact prompts, token IDs,
  target argmax와 acceptance를 함께 보존한다.

## 결론

#140의 EP micro 분해 성공과 #146의 EP 미초기화 행 수정은 실제 문제였지만,
비-EP까지 같은 방식으로 깨진 production 사건의 공통 원인은 아니다. 이번 조사에서
EP·비-EP 공통 target hidden state를 훼손할 수 있는 확정 결함은 두 개였다:
#133 MHC small-M 48-split·0-iteration과 #90 one-shot publish race. 전자는
결정적으로 매 layer를 건드려 사건의 더 직접적인 1순위 후보이고, 후자는 실제
memory-ordering 위반인 공동 후보다.

#150과 #153은 각각 두 코드 계약을 수리했다. 관측한 재부팅은 #150만 포함하므로
복구 증거가 아니다. 최신 두 수리를 함께 배포한 뒤 exact prompt + NCCL bracket으로
production 정합성을 닫고, 단일 원인 귀속은 격리 matrix 전까지 보류한다.
