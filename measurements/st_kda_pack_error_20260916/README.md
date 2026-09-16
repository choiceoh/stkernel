# KDA 투영의 팩 오차 — W4A8 디코드 레인과 FP8 프리필 레인 (2026-09-16)

원장 항목: [MEASUREMENTS.md](../../MEASUREMENTS.md) 의 `2026-09-16 — KDA 투영은 bf16 이 아니라 이미 fp8 이다`.
**오프라인 프로브다 — 판정이 아니다**(원장 규칙 1·6). tok/s·수용률·품질은 재지 않았고 부팅도 없다.

운영자 질문은 "KDA 를 W4A16+GPTQ 로 내리면 오차가 심각한가"였는데, 전제가 틀렸다. `kernels/dense/DenseLinear`
은 `L*.kda.in_proj`·`L*.kda.o_proj` 를 이미 양자화해서 돌린다 — ≤32 행은 W4A8, 그 위는 FP8. 그래서 잰 것은
"내릴까"가 아니라 **지금 두 레인이 각각 무엇을 잃고 있나**다.

## 파일

| 파일 | 무엇 |
|---|---|
| `kda_pack_error.py` | 두 레인의 오차. 저장소 패커(`engine/kernels/dense/packing.py`)를 임포트한다 — 재구현이 아니다 |
| `kda_pack_error_L1.log` · `kda_pack_error_L20.log` | 그 출력 (1층 · 20층) |
| `kda_weight_bits.py` | bf16 그릇에 무엇이 들었나 — 가수 후행 0비트 분포 |
| `kda_weight_bits.log` | 그 출력. 같은 파일 안의 대조군(`embed`·`head`·`in_norm`)과 별도 모델 대조군(DFlash2) 포함 |

## 왜 이렇게 쟀나

- **실덤프 두 벌을 서로 다른 스트림으로.** GPTQ 는 `calib-v2-fit`(329,580 토큰)에서 교정하고
  `calib-v2-heldout`(86,620 토큰)에서 채점한다. 33차는 같은 질문에 합성 헤시안으로 −13%, 실덤프로 −69~74%
  라는 다른 답을 냈고 그 차이가 프로브 설계였다. 교정 분포에서 채점하면 GPTQ 를 띄워 주고, 무관한 분포에서
  채점하면 깎아내린다. 표의 `in-sample` 열이 heldout 열 옆에 있어 그 간격이 보인다.
- **팩은 저장소의 것.** 검증은 W4 RTN 가중치 오차 `8.164e-2` 가 33차의 기록(~8.3%)을 재현하는 것이다.
  이게 맞지 않으면 나머지 숫자는 못 읽는다.
- **운영 설정.** `[served]` 표시가 붙은 행이 실제로 도는 것이다 — `act_order=True`(`GPTQ_ACT_ORDER`,
  45차 §23 GPU 판정 7차), per-row shift. 채널 스무딩은 걸지 않았다: 인자가 POW2 라 W4A8·FP8 레인에서
  정확히 무영향이다(`kernels/dense/smoothing` 의 기록, 45차 §23 조사 9차). per-row shift 도 33차 레버 3 이
  0 으로 판정했고, 이번에 per_row True/False 가 유효숫자 4자리까지 같아 재확인됐다.
- **채점은 표본이 아니라 2차 모멘트.** `sqrt(tr(D H Dᵀ)/tr(W H Wᵀ))`, D = W − deq(pack(W)) — heldout 분포
  에서의 기대 상대 출력오차다. 표집 잡음이 없고 행별로 분해돼 in_proj 행군이 한 번의 곱에서 나온다.

## 재현

가중치와 헤시안을 한 디렉터리에 모은다(`<dir>`), 층 L 에 대해:

```
kda_l<L>.npz          {"L<L>.kda.in_proj", "L<L>.kda.o_proj"}, uint16 = bf16 비트
L<L>.fit.<h>.pt       calib-v2-fit   /mkcalib/rank<r>/Glm5NextForCausalLM/model.layers.<L>.self_attn.<h>.pt
L<L>.heldout.<h>.pt   calib-v2-heldout/ 같은 경로
```
`<h>` 는 `in_proj_qkvbfg_a`(in_proj)와 `o_proj`. 가중치 추출은 랭크 파일에서:

```python
import numpy as np, torch
from safetensors import safe_open
with safe_open("~/models/st-glm53-nvidia-tp4-9391/rank3of4.safetensors", framework="pt", device="cpu") as f:
    np.savez("kda_l1.npz", **{k: f.get_tensor(k).view(torch.int16).numpy().astype(np.uint16)
                              for k in ("L1.kda.in_proj", "L1.kda.o_proj")})
```

실행(ost-97x 의 RTX 5050, x86_64 sm_120 체크 이미지 — 패커가 도는 CUDA 박스면 아무 데나 된다.
srv4 GPU 는 부팅이 쥐고 있었고 플릿 큐는 쓰지 않았다):

```bash
docker run --rm --gpus all -v ~/kda-err:/work -v /path/to/repo:/repo:ro \
    st-engine:glm53-sm120-x86 /repo/measurements/st_kda_pack_error_20260916/kda_pack_error.py \
    --layer 1 --dir /work --repo /repo
```

비트 테스트는 체크포인트만 있으면 된다(GPU 불필요):

```bash
python3 kda_weight_bits.py ~/models/st-glm53-nvidia-tp4-9391/rank3of4.safetensors \
    L1.kda.in_proj L1.kda.o_proj L1.in_norm embed head
python3 kda_weight_bits.py ~/models/st-glm53-nvidia-tp4-9391/rank3of4.safetensors --rows L1.kda.in_proj
```

## 읽을 때 조심할 것

- **바이트 이야기로 넘어가지 말 것.** 랭크당 KDA 투영은 랭크 파일에서 2,248 MiB 의 bf16 이지만 그건 부팅이
  소비하는 형태다(`consume_weight`). 상주하는 건 W4 팩과 FP8 팩이다. "bf16 을 fp8 로 내리면 1.1 GiB 를
  번다"는 계산은 틀렸다 — 서빙은 이미 bf16 을 지나갔다. 남는 건 디스크·로드 시간뿐이다.
- **GPTQ 는 가중치 오차를 키우면서 출력 오차를 줄인다.** `||D||/||W||` 열이 RTN 보다 GPTQ 에서 나쁘다.
  그게 목적함수다 — 가중치 오차로 팩을 고르면 거꾸로 고른다.
- **두 층만 쟀다**(1·20층, rank3of4). 비트 패턴은 두 층 다 같지만 오차 크기는 다르다(20층이 더 나쁘다).
- 게이트 272행(`beta` 16·`fa` 128·`ga` 128)은 유일하게 진짜 bf16 이고, 그래서 FP8 오차가 q/k/v 행의 10배다.
