# ST와 기존 vLLM 성능 차이 — 2026-09-12 조사

ST의 실제 디코드 스텝은 약 91ms이며, 동일한 HTTP 요청을 사용한 9월 9일 vLLM 기록은 약 44ms다. 초안 수용률 차이보다 스텝 실행 시간 차이가 크다. 코드와 부팅 로그를 대조하면 기존 vLLM의 저정밀 dense GEMM, one-shot AllReduce, prefill sequence parallelism 경로 일부가 ST 실행 경로에 연결되지 않은 것이 확인된다. 각 경로가 현재 지연에서 차지하는 정확한 ms는 아직 측정하지 않았다.

## 동일 요청 비교

기준선은 `../glm53_ep_local_20260908/onepass2-completed/onepass.jsonl`의 `EPONEPASS2B1`이다. 이번 ST의 다섯 요청 모두 request SHA-256, 실제 prompt token 수가 기준선과 일치한다. 동일한 한국어 본문·질문·thinking·temperature·출력 상한을 사용했고 두 실행 모두 트래픽 검사를 통과했다. 같은 날 교차 부팅한 A/B 실험은 아니므로 아래 값은 역사적 관측 비교다. 런타임·수치 경로·출력 내용과 길이·캐시 상태의 차이는 남아 있다.

| 지표 | vLLM 9월 9일 | ST 9월 12일 |
| --- | ---: | ---: |
| 디코드 중앙값 step/s | 22.789 | 10.963 |
| 위 값의 역수, ms/step | 43.882 | 91.220 |
| 2K 세 요청 디코드 중앙값 tok/s | 77.857 | 36.101 |
| 32K 디코드 tok/s | 76.784 | 35.600 |
| 128K 디코드 tok/s | 84.738 | 33.836 |
| 전체 초안 수용률 | 49.375% | 46.189% |
| K=5로 계산한 tokens/step | 3.469 | 3.309 |
| 128K TTFT | 41.602초 | 64.413초 |
| 128K 입력/TTFT | 3,090 tok/s | 1,996 tok/s |

ST/vLLM의 step/s 비율은 0.481, 계산한 tokens/step 비율은 0.954다. 수용률을 기준선 수준으로 올려도 같은 ST 스텝 속도에서는 약 38.0 tok/s라는 단순 계산이 나온다. 이것은 실험으로 얻은 회복 예측이 아니며, 수용률만으로 두 배 가까운 격차를 설명할 수 없다는 분해다. 문맥별 수용률을 별도로 수집하지 않았으므로 전체 수용률을 128K 속도에 직접 적용하지 않는다.

32K TTFT는 첫 중단 시도의 캐시 영향 가능성 때문에 프리필 비교에서 제외한다. 128K도 cache-hit 토큰 수와 반복 표본이 없으며 단일 관측으로 제시한다.

## 확인한 실행 경로 차이

1. **본체와 드래프터의 dense GEMM.** 기준선의 실제 부팅 로그는 본체 213개 FP8 dense 선형층과 213개 MK W4 pack, 드래프터 31개 FP8 선형층과 31개 MK W4 pack을 기록한다. 본체에는 NVFP4 prefill pair 213개도 준비되어 있고 해당 경로 진입 로그가 있다. ST의 `engine/profiles/glm53/specs.py`와 `drafter.py`는 해당 dense 가중치를 BF16으로 보유하고, `net.py`와 `drafter.py`는 `torch.nn.functional.linear`로 실행한다. 따라서 두 엔진의 행렬 계산 경로와 읽는 가중치 형식이 다르다. ST의 routed MoE 자체는 여전히 NVFP4이며, 모델 전체가 BF16이라는 뜻은 아니다. 저정밀 경로를 옮길 때는 출력 수치와 정답·문자 게이트를 다시 검증해야 한다.

2. **TP 통신과 mHC 연계.** 기준선 로그는 실제 4-rank one-shot AllReduce 연결·self-test·consumer PDL capture를 확인한다. ST의 `engine/base/comm.py:Comm.all_reduce`는 `torch.distributed.all_reduce`를 호출하며, 모듈 문서도 legacy one-shot 경로를 사용하지 않음을 명시한다. 본체는 각 attention과 FFN의 row-parallel 출력에서 통신하고, mHC는 별도 TileLang 호출이다. 이 차이는 현재 코드로 확인되지만 통신 및 rank 대기 비용의 정확한 비중은 현재 프로덕션 trace가 필요하다.

3. **드래프터 실행 비용.** ST 드래프터는 BF16 선형층 외에도 GQA K/V를 `repeat_interleave`로 펼치고 FP32 einsum·softmax·einsum으로 attention을 계산한다. 제안 토큰은 `.tolist()`로 호스트에 읽은 후 본체 검증 입력을 다시 장치에 만든다. 현재 그래프 모드는 켜져 있지만, 캡처가 이 계산·메모리 이동을 자동으로 전용 fused 커널로 바꾸지는 않는다. 직전 전체 모델 replay 진단(`../engine_decode_replay_20260912/full-fixed-rank0.log`)은 target 67–70ms, draft 18–19ms, observe 약 2–3ms를 기록했다. 이는 짧은 입력과 별도 동기화를 사용한 이전 진단이므로 이번 프로덕션 91ms의 정확한 세부 프로파일로 취급하지 않는다.

4. **프리필 분산과 정밀도 경로.** 기준선의 실제 로그에 prefill sequence parallelism, FP8 v3 transport, MHC token shard 선택이 있다. 현재 ST `net.forward`는 각 rank에서 전체 입력 토큰의 mHC를 수행하고 BF16 dense 선형층을 사용한다. 이는 긴 입력 차이를 설명하는 구체적인 후보이며, 순수 프리필 반복 계측 없이 각각의 이득을 단정하지 않는다.

현재 프로덕션 부팅 로그도 `moe_static=stock`, `mla_prefill=stock`, `lanes=served`, `decode_eager=0`을 확인했다. SF6 같은 후속 MoE 설정 차이도 있지만, 기존 SF6 채택 기록의 스텝 차이는 약 0.37%여서 이것 하나로 현재 약 2배의 스텝 시간 차이를 설명하지 않는다. GPU/CPU 부하가 모든 시점에 동일했다고 주장하지 않으며, 이번 확인에서 head 컨테이너의 CPU quota 설정은 0(무제한)이었다.

## 후속 우선순위

현재 전체 모델의 본체 dense GEMM, drafter, 통신 시간을 먼저 분리하고, 기존 저정밀 GEMM·드래프터 attention 경로를 ST 자원 소유권 및 수치 계약에 맞춰 연결하는 것이 우선이다. 그 다음 one-shot 통신/mHC 연계와 prefill token sharding을 비교한다. 각 변경의 채택 기준은 같은 원패스의 실제 step/s와 출력 tok/s, 검색·문자·최종 답변 완료 상태다. 이 조사에서는 서비스 설정 변경, 컨테이너 재시작, 추가 GPU 부하 실험을 하지 않았다.

`vllm-comparison.json`에 다섯 요청의 해시와 비교 수치를, `vllm-baseline-excerpts.txt`에 기준선 원본 로그의 줄 번호와 압축 파일 SHA-256을 보존했다. 현재 경로를 확인한 소스는 하네스 커밋 `cc6163c78613777225a4e48ac4ae7ac78871f97b`이며 프로덕션 엔진 release는 `5734b29fde84`다.
