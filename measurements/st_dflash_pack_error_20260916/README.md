# DFlash2 양자화 오차의 부팅 없는 분석 (2026-09-16)

**이 분석으로 정밀도를 높일 층의 우선순위를 결정할 수 없다.** TP4 rank 0에서 실제 캐시된 W4 GPTQ를 BF16 원본과 비교하면, 1·2·3층 `down_proj`의 상대 RMS 오차가 가장 크다. 같은 가중치를 FP8 RTN으로 바꾸는 오프라인 후보는 이 연산들의 오차 에너지를 약 80~83% 줄였다. 그러나 수용률에 대한 민감도와 추가 실행 비용을 측정하지 않았으므로, 앞서 제시한 2층 우선 추천은 철회한다. 아래 표는 층별 수치 오차이며 교체 우선순위가 아니다. 층 번호는 checkpoint와 같은 0-based다.

엔진 부팅·재시작·GPU 실행 없이 srv2의 CPU 4개, 메모리 한도 6 GiB로 계산했다. 모델과 캐시는 읽기 전용으로 마운트했다. 실행 전후 `torch.cuda.is_initialized()`는 false였다. 제품 코드나 기본값은 변경하지 않았다.

## 범위와 출처

- 코드 기준: `a40046f42954e8cc6cb04a5cc0cc3fd9ce755ab1`.
- PR #1003 (`27c03dd2c03dc4b85f726b159557ffe23493e447`) 이후의 `outputs-5-14-24-33-42` namespace만 사용했다. 수정 전 aux layer 보정 데이터는 사용하지 않았다.
- 모델: srv2 `/home/choiceoh/models/GLM-5.3-Flash-DFlash2/model.safetensors`.
- 보정: `/home/choiceoh/glm53-cache/mkcalib/rank0/DFlash2Qwen3ForCausalLM/outputs-5-14-24-33-42/`.
- 실제 팩: `/home/choiceoh/glm53-cache/st-dense-packs/`. 현재 가중치·TP shard·보정 행렬·스무딩·패커 알고리즘의 해시와 기본 damping 0.01이 일치하는 팩만 선택했다. 파일 시각이나 이름만으로 선택하지 않았다.
- 5개 block의 W4 dense reader 30개와 committed-decode FC FP8 1개. FC는 `SERVING_POLICY`의 FP8 및 완료된 decode 보정에 해당하는 캐시를 평가했다. 현재 서빙 프로세스에 이 팩들이 장착됐다는 증명은 아니다.
- 스무딩은 실제 norm fold, alpha 0.5, POW2 및 전체 비분할 reader의 column peak를 재현했다. W4는 **per-tensor** 설정이다. 타깃 KDA의 per-row 설정을 가져오지 않았다.
- 이미지, 명령, CPU·메모리 한도, GPU device request 부재: [full-runtime.json](full-runtime.json).
- 실행·소스 해시·검산 결과: [full-run.json](full-run.json). 각 reader의 가중치 및 보정 해시, 캐시 경로, 에너지 합: [full-results.json](full-results.json). 원시 실행 로그: [full.log](full.log).

## 계산 방법

원본 BF16 가중치 `W`, 실제 팩을 역양자화한 `Q`, 입력 Gram 행렬 `H`에 대해 다음 값을 계산했다.

```
D = W - Q
relative RMS = sqrt(sum_r D[r] H D[r]^T / sum_r W[r] H W[r]^T)
```

전체 출력 행과 입력 열, 모든 비대각 상관항을 포함했다. FP64로 누적하되 FC의 20480×20480 행렬 전체를 FP64로 복사하지 않도록 타일로 계산했다. W4의 nibble→E2M1→group scale→E4M3 반올림→row scale을 생산 코드의 역양자화 함수로 그대로 읽었다. FP8 후보는 생산 코드와 같은 128×128 block 및 power-of-two scale을 쓰는 RTN이다. FP8 GPTQ 후보는 아니다.

검산은 (1) 작은 입력의 명시적 matmul 오차와 Gram 방식의 일치, (2) 선택 행의 W4 역양자화와 생산 함수 전체 출력의 bitwise 일치, (3) FP8 RTN 후보와 생산 `fp8_rtn`의 bitwise 일치다. 이 검산과 실제 데이터 계산은 같은 CPU 프로세스에서 수행했다.

## 해석과 다음 측정

오차가 작아도 후보 토큰의 순위를 바꾸는 민감한 연산일 수 있고, 오차가 커도 뒤 연산에서 영향이 약해질 수 있다. FC의 상대 오차 0.676%만으로 FC의 수용률 영향이 작다고 판단할 수 없다. 여러 conv에서 FP8 RTN의 수치 오차가 더 컸다는 관측도 실제 수용률 비교를 대신하지 않는다.

다음 측정은 연산별 정밀도 변경의 **연속 수용 길이 증가와 추가 step 시간**을 함께 비교해야 한다. 아래는 아직 수행하지 않은 측정 설계다.

이후 [민감도·비용 재생 하니스](../st_draft_sensitivity_20260916/README.md)를 구현하고 CPU 검증을 마쳤다. 실제 입력·상태 캡처와 native GPU 재생은 아직 수행하지 않았다.

1. 같은 실제 drafter 입력·상태에서 연산 하나의 정밀도만 바꾸고, 이후 block·최종 head·selector까지 다시 실행한다. TP4의 해당 연산을 함께 바꿔 평가한다. 한 번 확보한 입력·상태 묶음으로 후보들을 재생하면 후보마다 엔진을 부팅할 필요는 없다. 현재 저장물에는 그 재생에 필요한 입력·상태가 부족하다. FC처럼 context를 바꾸는 후보는 타깃 aux 이력으로 후보의 context K/V도 다시 구성해야 한다.
2. 최종 후보의 위치별 수용 여부와 **첫 기각 전 연속 수용 길이**를 측정한다. 뒤 위치의 일치가 늘어도 앞에서 기각되면 출력 토큰은 늘지 않는다. 원래 후보에 대한 타깃 확률을 새 후보의 검증 확률로 재사용하면 안 된다. greedy는 같은 타깃의 greedy continuation을 기준으로 prefix 일치를 확인할 수 있고, sampling은 새 후보 prefix에 대한 타깃 검증이 추가로 필요하다. 저장된 prefix 재생은 최종 live 판정을 대신하지 않는다.
3. 실제 C=1 형상과 실행 lane에서 추가 지연과 메모리를 측정한다. CPU 계산 시간이나 가중치 바이트 수를 GPU의 추가 step 시간으로 간주하지 않는다. 미니 replay로 선별한 뒤에는 동일 빌드의 전체 step 시간과 출력 tok/s로 확인한다.
4. 후보 우선순위는 동일 workload에서의 `출력 tokens/step ÷ seconds/step` 증가로 정한다. 품질을 유지하고 메모리 예산에 들어오는 후보만 채택한다. 후보를 조합할 때는 효과를 단순 합산하지 않고 조합을 다시 측정한다.

예를 들어 한 `down_proj`의 FP8 가중치·scale 저장량은 W4 대비 약 **5.24 MiB/rank** 증가한다(4096×3072 기준). 이 값은 저장량 계산이며 지연 측정이 아니다. 실제 추가 메모리와 지연은 lane의 팩 보존·커널 dispatch에 따라 달라진다. 수용에 따른 출력 토큰 증가율이 step 시간 증가율을 넘어야 tok/s가 좋아진다.

한계: **rank 0, 보정에 사용한 동일 분포(in-sample), weight-only** 분석이다. activation A8 반올림, 중간 BF16 반올림, all-reduce, residual·attention·conv의 비선형 전파, selector 및 실제 수용률은 평가하지 않았다. 보정 파일에는 `weights_id`가 없어 이를 수집한 타깃 checkpoint의 동일성을 추가로 증명할 수 없다. 현재 파일 해시와 팩 identity 일치는 확인했다. 서로 다른 연산의 에너지나 감소율을 더해 전체 모델 개선율로 쓰면 안 된다.

현재 저장된 selector hidden/features에는 타깃 aux와 모든 층의 원입력이 없어 전체 drafter의 BF16/후보 replay까지 복원할 수 없다. 실제 채택에는 같은 빌드의 C=1 비교에서 품질, 수용률, tokens/step, tok/s와 step 시간이 필요하다. 이번 결과만으로 런타임 변경을 채택하지 않는다.

## 재현

`probe.py` 옆에 기준 커밋의 아래 파일을 복사한다. engine 패키지를 import하거나 boot 모듈을 호출하지 않는다.

```sh
AUDIT_DIR=/home/choiceoh/expert-capture/dflash-pack-error-0916-7e62
mkdir -p "$AUDIT_DIR"
cp engine/kernels/dense/packing.py engine/kernels/dense/smoothing.py "$AUDIT_DIR/"
cp engine/kernels/dense/__init__.py "$AUDIT_DIR/dense_init.py"
cp measurements/st_dflash_pack_error_20260916/probe.py "$AUDIT_DIR/"
```

원격 실행 당시 `$AUDIT_DIR`는 `/home/choiceoh/expert-capture/dflash-pack-error-0916-7e62`였다. 위 파일을 둔 srv2에서 실행한다.

```sh
docker run --rm --cpus 4 --memory 6g --network none \
  -e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES= \
  -e OMP_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=4 \
  --entrypoint python3 \
  -v /home/choiceoh/glm53-cache:/cache:ro \
  -v /home/choiceoh/models/GLM-5.3-Flash-DFlash2:/model:ro \
  -v "$AUDIT_DIR:/work" \
  sha256:1b2b41d014c59caa52d81d73bf9d359f8047f51284da2b9da88b7fc801397040 \
  -u /work/probe.py --out /work/full --rows 0 --threads 4
```

`--rows 128`은 고정 seed의 초기 선별용이며, 아래 최종 결과는 `--rows 0` 전체 행 계산이다. `--only layers.2.mlp.down_proj`로 특정 reader만 다시 계산할 수 있다. 캐시가 현재 입력 해시와 맞지 않으면 `unmatched`로 기록하며 RTN baseline으로 조용히 대체하지 않는다.

## 전체 행 결과

31/31 reader가 일치했다. 전체 93,696 출력 행을 평가했다. 보정 행 수는 reader별 33,034–71,992개다. 아래 수치는 모두 상대 RMS 오차(%)이며, 현재 캐시 오차의 내림차순일 뿐 수용률 영향이나 비용 대비 효과의 순위가 아니다. FC baseline만 FP8 GPTQ이며 나머지는 W4 GPTQ다.

| Reader (0-based) | 현재 캐시 | FP8 RTN 후보 | 오차 에너지 감소 |
|---|---:|---:|---:|
| `layers.2.mlp.down_proj` | 5.913% | 2.444% | 82.9% |
| `layers.1.mlp.down_proj` | 5.721% | 2.387% | 82.6% |
| `layers.3.mlp.down_proj` | 5.554% | 2.457% | 80.4% |
| `layers.2.self_attn.o_proj` | 5.438% | 3.024% | 69.1% |
| `layers.0.mlp.down_proj` | 5.205% | 2.296% | 80.6% |
| `layers.3.self_attn.o_proj` | 4.916% | 2.818% | 67.1% |
| `layers.1.mlp.gate_up_proj` | 4.898% | 2.473% | 74.5% |
| `layers.2.mlp.gate_up_proj` | 4.653% | 2.443% | 72.4% |
| `layers.3.mlp.gate_up_proj` | 4.571% | 2.472% | 70.8% |
| `layers.4.mlp.down_proj` | 4.542% | 2.536% | 68.8% |
| `layers.0.mlp.gate_up_proj` | 4.517% | 2.418% | 71.3% |
| `layers.4.mlp.gate_up_proj` | 3.893% | 2.286% | 65.5% |
| `layers.1.self_attn.o_proj` | 3.828% | 2.357% | 62.1% |
| `layers.4.self_attn.o_proj` | 3.415% | 2.012% | 65.3% |
| `layers.0.self_attn.o_proj` | 3.363% | 2.267% | 54.6% |
| `layers.4.self_attn.qkv_proj` | 3.322% | 1.976% | 64.6% |
| `layers.3.self_attn.qkv_proj` | 2.776% | 1.577% | 67.7% |
| `layers.1.self_attn.qkv_proj` | 2.674% | 1.626% | 63.0% |
| `layers.2.self_attn.qkv_proj` | 2.461% | 1.553% | 60.2% |
| `layers.0.self_attn.qkv_proj` | 2.137% | 1.679% | 38.2% |
| `layers.0.attention_conv.kernel_projection` | 1.835% | 1.252% | 53.5% |
| `layers.2.attention_conv.kernel_projection` | 1.148% | 1.159% | -1.9% |
| `layers.3.attention_conv.kernel_projection` | 0.813% | 1.364% | -181.3% |
| `layers.1.attention_conv.kernel_projection` | 0.780% | 1.164% | -122.6% |
| `layers.3.mlp_conv.kernel_projection` | 0.739% | 1.734% | -450.9% |
| `layers.1.mlp_conv.kernel_projection` | 0.730% | 1.560% | -356.7% |
| `layers.2.mlp_conv.kernel_projection` | 0.701% | 1.731% | -510.2% |
| `fc.committed-decode-v1` | 0.676% | 1.668% | -509.4% |
| `layers.4.attention_conv.kernel_projection` | 0.582% | 1.051% | -226.0% |
| `layers.0.mlp_conv.kernel_projection` | 0.488% | 1.527% | -881.1% |
| `layers.4.mlp_conv.kernel_projection` | 0.487% | 1.694% | -1108.6% |
