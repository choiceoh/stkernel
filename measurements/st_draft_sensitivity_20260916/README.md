# DFlash 층별 수용 민감도와 추가 비용을 비교하는 하니스

**구현·실제 캡처·TP4 GPU 비교 완료.** C=1 요청 8개에서 수집한 16사례(32K·128K 포함)에 대해 dense 연산 30개를 각각 FP8/BF16으로 교체했다. 60개 후보 모두 사례별 수용 길이 개선이 없고 지연·가중치 비용만 증가해 이번 자료에서는 W4를 유지한다. [비교 결과와 원본 데이터](comparison/README.md), [캡처 결과와 보관 경로](live/README.md). 전체 엔진 tok/s와 완결 출력 품질은 별도 미측정이다.

이 하니스는 한 번 확보한 실제 입력·상태에서 블록 연산 하나씩 FP8 RTN 또는 BF16로 바꿔, 뒤의 block·head·selector까지 다시 실행한다. 실제 수용 길이의 민감도와 추가 지연·가중치 저장량을 같이 본다. 이전 [층별 양자화 오차](../st_dflash_pack_error_20260916/README.md)는 교체 순위로 사용하지 않는다.

## 구현

- `engine/profiles/glm53/draft_replay.py`: 준비된 실제 W4/FP8 팩, 스무딩된 norm, conv·selector 가중치, 실제 FP8 head, kernel shape와 collective 설정을 rank별로 저장한다. 사례마다 proposal 직전 context ring 및 anchor/mask embedding을 저장한다. 원본 drafter checkpoint와 실행 소스, 상태·사례 파일의 SHA-256을 남긴다.
- `engine/profiles/glm53/boot.py`: 사용자 요청으로 이 측정 작업본의 `DRAFT_REPLAY_CASES = 16`을 켰다. warmup 이후 붙으며, 캡처 중 host decode를 사용한다. **캡처 요청의 지연을 성능 baseline으로 사용하면 안 된다.** fleet 세션 `draftreplay4-0916`에서 수집 후 측정용 서버를 종료했다. 별도 FC 자동 수집은 BF16 source 부재로 부팅을 실패시켜 이 측정 arm에서 껐다.
- `probes/draft_sensitivity.py`: 전체 타깃 모델을 띄우지 않고 캡처된 drafter와 head만 구성한다. TP4 전체에서 같은 연산 하나를 교체한다. 기존 GPU kernel과 collective 설정으로 실행하고 baseline draft를 정확히 재현하지 못하면 분석을 거부한다.

캡처는 C=1, temperature=0, 추가 샘플링 제약이 없는 요청에 한정한다. 기본 간격은 요청별 eligible step 16개마다, 요청당 최대 2사례, 전체 최대 16사례다. 긴 요청 하나가 예산을 모두 사용하지 않는다. 최대 사례 수는 128로 제한한다. 설정한 사례를 모으고 pending label을 모두 확보하면 비동기 경로를 되돌린다. 파일 쓰기 실패는 rank 간 공유해 한 rank만 다음 collective로 진행하지 않게 한다.

캡처 비용은 아직 실측하지 않았다. 시작 시 준비된 drafter 팩과 head 등을 CPU로 복사하고 저장·해시하며 원본 checkpoint도 해시한다. 사례마다 embedding collective, context ring 등의 GPU→CPU 복사, 동기 파일 쓰기·해시와 rank 간 확인이 추가된다. 특히 수집 기간 전체에 비동기 decode를 끄므로 캡처하지 않는 step과 캡처 대상이 아닌 요청도 영향을 받는다. 적격 요청이 부족하면 16사례가 모이지 않아 비동기 경로가 자동 복구되지 않는다. 상시 서빙용 경량 계측으로 취급하면 안 된다.

## 정답과 수용 길이

정답은 **해당 anchor 이후 실제로 확정된 greedy continuation**이다. 원래 draft를 검증하던 타깃의 첫 기각 뒤 logits는 잘못된 prefix에 조건부이므로 사용하지 않는다. 후속 decode에서 정답 K개가 모일 때까지 기다린다. 종료·취소·출력 한도로 더 모을 수 없으면 짧은 정답을 그대로 보존한다.

- 첫 불일치가 관측되면 연속 수용 길이를 정확히 계산한다.
- 정답이 짧고 관측 구간이 모두 일치하면 `[확인된 길이, K]` 범위로 남긴다.
- 뒤 위치의 토큰 일치가 늘어도 앞에서 기각됐으면 수용 길이에 더하지 않는다.
- 후보가 제안한 draft가 원래 baseline과 달라도 동일한 greedy continuation의 첫 불일치까지 비교할 수 있다. 이는 **고정된 prefix/state의 오프라인 민감도**이며, 후보가 장기간 만든 context와 분포를 포함한 live acceptance 측정은 아니다.
- sampling은 새 후보 prefix에 대한 타깃 확률이 필요하므로 이 버전에서 받지 않는다.

## 비용과 선별

같은 사례의 baseline/candidate를 B/A/A/B 순서로 GPU graph replay한다. 각 구간은 rank 중 가장 긴 시간을 사용하고 표본·중앙값을 함께 저장한다. 양쪽 모두 캡처된 embedding을 입력으로 받으므로 **block/head/selector와 그 collective의 비용**이다. 타깃 verification, embedding lookup/collective, observe와 전체 decode step은 포함하지 않는다.

JSON에는 다음이 남는다.

- 평균 연속 수용 길이와 변화의 하한·상한, 위치별 prefix 생존율, 실제 사례·요청 수.
- 추가 GPU replay 시간, 추가 토큰당 replay 비용, rank별 활성 가중치·scale 바이트 증가량.
- `max_extra_step_fraction_bounds`: EOS/출력 제한으로 clipping하지 않은 tokens/step 증가율. 전체 step 지연 증가율이 이보다 작아야 tok/s가 개선된다. 전체 step의 실제 시간이나 tok/s를 추정해 채워 넣지는 않는다.
- `cost_screen`: 수용 증가가 더 작고 지연·메모리 비용은 더 큰 후보를 구분한다. 측정한 비용과 효과 사이에 tradeoff가 있으면 후보를 여러 개 남긴다. 시간 표본의 통계적 불확실성과 전체 step 비용이 해결되지 않았으므로 `deployment_winner`는 항상 null이다.

재생 프로세스는 비교를 위해 baseline과 candidate 팩을 동시에 보유한다. 가중치 바이트 차이는 활성 reader payload의 차이이며 실제 서빙 peak memory 감소/증가의 실측이 아니다. 최종 채택에는 동일 빌드의 C=1 live 품질·tokens/step·step 시간·tok/s 검증이 필요하다.

## 실행 순서

1. 이 작업본에는 `DRAFT_REPLAY_CASES = 16`이 설정되어 있다. 실제 C=1 greedy 요청을 여러 개 보내며 32K·128K 등 목표 context를 포함한다. 모델 출력 품질용 corpus와 요청 단위 holdout도 분리한다. 저장 위치는 `<dump-dir>/draft-replay/rankN/`이며 매 실행마다 새 디렉터리를 사용한다. 이번 캡처는 완료되어 [별도 보관 위치](live/README.md)에 있다.
2. 4개 rank의 파일을 보존한다. 각 rank에는 `state.pt`, `manifest.json`, `case-NNNNN.pt`, 대응하는 `.json` 정답이 있어야 한다. 같은 소스·PyTorch·GPU architecture와 원본 drafter checkpoint를 사용한다.
3. GPU 실행은 `bench/draft_replay.py`를 fleet admission에 등록한다. [완료된 비교와 launcher 명령](comparison/README.md)을 참고한다. launcher가 캡처 당시 engine source를 고정하고 TP4 환경을 준비해 서버 엔진 없이 아래 probe를 실행한다. world=1로 rank 0만 실행하는 것은 거부한다. 아래는 rank별 내부 probe 명령이며 fleet 등록을 대체하지 않는다.

```sh
python3 probes/draft_sensitivity.py \
  --capture /path/to/draft-replay \
  --checkpoint /home/choiceoh/models/GLM-5.3-Flash-DFlash2/model.safetensors \
  --reader all --precision fp8-rtn --rounds 10 \
  --output /path/to/draft-sensitivity-fp8.json
```

`--reader layers.2.mlp.down_proj.weight`처럼 범위를 좁힐 수 있으나 이번 실제 비교에서도 해당 reader의 수용 이득은 0이었다. `--precision bf16`은 해당 연산의 고정밀 민감도 비교다. 한 arm에서 여러 연산을 동시에 바꾸지 않는다.

## 범위와 남은 검증

고정된 context ring에서 재생 가능한 **5개 block의 dense 연산 30개**가 대상이다. FC/context projection을 바꾸려면 타깃 aux 이력부터 context K/V를 다시 구성해야 하므로 `fc.weight`는 명시적으로 거부한다. FC 영향이 작다고 결론내린 것이 아니다. native GPU proposal 비용은 측정했지만 Selector 자체의 정밀도 변경, sampling, 후보를 적용한 장기 generation, 전체 엔진 성능·품질 판정은 완료 항목이 아니다.

CPU 테스트는 실제 요청의 개선 수치를 만들지 않는다. 작은 DFlash의 block→head→selector 경로에서 reader 변경이 전파되고 원상복구되는지, 준비된 팩 전체의 저장·복원, TP shard 및 스무딩, 정답의 지연 수집, censored bounds, 파일·rank 불일치 거부와 비용 선별을 확인한다.

검증 환경: srv2 CPU-only Docker, 이미지 `sha256:1b2b41d014c59caa52d81d73bf9d359f8047f51284da2b9da88b7fc801397040`, CPU 2개, 메모리 4 GiB, `--network none`, `NVIDIA_VISIBLE_DEVICES=void`, `CUDA_VISIBLE_DEVICES=`. **18개 테스트 통과**, CUDA 미초기화 확인. [CPU 결과](cpu-tests.log), [소스 해시와 검증 기록](validation.json).

위 `validation.json`은 캡처 설정을 0에서 16으로 켜기 전 소스에 대한 기록이다. 이후 main 변경을 병합하고 관련 CPU 테스트 18개를 다시 통과했다([로그](cpu-tests-after-merge.log)). 실제 캡처 커밋은 `4e6698a1a47ffe447c4293d3cfc3122065bc1c69`이며 [실제 파일 검증 기록](live/audit.json)을 별도로 남겼다. 이전 테스트 기록의 해시는 덮어쓰지 않았다.

```sh
python3 -m unittest tests.test_draft_sensitivity -v
```
