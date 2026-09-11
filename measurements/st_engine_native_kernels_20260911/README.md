# ST native kernel migration — 2026-09-11

GLM53의 7개 served 레인을 `engine/kernels`로 옮긴 뒤 srv1의 GB10에서 검증했다.
vLLM 패키지를 제거한 `st-engine:9391` 이미지를 사용했다. 기존 서빙 컨테이너와
실가중치 랭크 파일은 변경하지 않았다. 작업 기준 커밋은
`3c9622b7d5eb8b326322c8933a20668381015c2f`이며, 변경 후 소스는
`source-sha256.json`에 기록했다. 이미지 ID는 `image-inspect.json`에 있다.

## 판정

- `gpu-unittests.log`: 엔진 회귀 테스트 **64개 통과, skip 없음**. 캐시·롤백·러너·
  로더 검사와 새 up|gate 샤딩/FP4 반올림/구형 랭크 거부 검사를 포함한다.
- `cpu-tests.log`: PyTorch 없는 개발 환경에서 **47개 통과, 17개 skip**.
- `gpu-kernels.log`: 7개 레인 통과. ST 모듈 51개 임포트, served 표 구성,
  실행 전 과정에서 vLLM 임포트 차단, 최종 `vllm_loaded=false`.
- `gpu-moe-288.log`: **288 experts, top-k 8, H=4096, rank I=512**에서
  토큰 수 1/8/129, zero-route, CUDA Graph 3회 재생 통과.
- `gpu-kda-strided.log`: `ST_GLM53_KDA_PREFILL_QK_NORM=1`에서도 6개 KDA 사례 통과.
- `runtime.json`: 저장소 마운트 없이 이미지의 `/opt/st/engine`을 임포트하고
  vLLM 미설치, 독립 DeepGEMM, 라이브러리 버전과 GPU를 확인했다.
- `cpu-imports.log`: 새 ST 실행기의 GPU 없는 모듈 검사 통과.
- `boot-help.log`: 이미지의 기본 entrypoint로 엔진 부팅 CLI를 로드했다.

프로브의 상대 오차는 `max(abs(actual-reference)) / max(abs(reference))`다.
MLA 내부 부팅 검사는 별도의 L2 상대 오차를 사용한다.

| 검사 | 관측 결과 |
| --- | --- |
| conv, 초기 상태 유/무 × 1/6/96 토큰 | 출력·상태 일치 |
| KDA, 초기 상태 유/무 × 1/6/64 토큰 | chunk 출력 최대 0.6494%, recurrent 모든 상태 최대 2.04e-7, 초기 상태 보존 |
| mHC, 1/6/8/65 토큰 | pre 최대 0.4831%, post 최대 0.1516% |
| indexer | 유효 로짓 상대 오차 1.07e-7 |
| kpool, 1/3/17 pools | FP8 바이트와 스케일 일치 |
| MLA | 6개 부팅 사례 통과, graph 3회 재생 오차 0.6290% |
| b12x, 8 experts | 이식 전 커널 대비 최대 0.8929%, graph 재생 일치 |
| b12x, 288 experts | 이식 전 커널 대비 최대 1.9608%, graph 재생 일치 |

b12x의 이식 판정 기준은 **이식 전 FlashInfer 커널 대비 2% 이하**다. 원본과 ST는
별도의 JIT 모듈 이름을 사용한다. 원본의 BF16 atomic scatter는 실행 순서에 따른
반올림 차이가 있으므로 바이트 동일성을 요구하지 않는다. PyTorch 참조와는 최대
약 6.95% 차이가 남아 있으며 진단값으로만 기록했다. FP4 양자화와 누적 순서의 영향을
분리하는 추가 분석이 필요하다. 이 검사로 PyTorch 참조와의 수치 일치를 주장하지 않는다.

## 수정된 가중치 계약

b12x FC1 입력은 **up | gate**다. 기존 ST preshard의 gate | up을 수정했으며,
packed 가중치와 접힌 스케일이 같은 순서를 따른다. 새 파일에는
`weight_layout=st-glm53-b12x-up-gate-v1`을 기록한다. 부팅과 실가중치 검사는
이 메타데이터가 없는 전문가 파일을 아레나 할당 전에 거부한다.
기존 GLM 랭크 파일은 새 preshard로 다시 생성해야 한다.

## 재현

동일한 seed가 있는 GB10 호스트의 저장소 루트에서 실행한다. seed ID와 ABI 버전은
[`dependencies.json`](../../engine/runtime/dependencies.json)에 고정되어 있다.

```bash
ST_IMAGE=st-engine:9391 bash engine/runtime/build.sh
ST_IMAGE=st-engine:9391 ST_PROBE_NO_GPU=1 bash probes/run_engine_probe.sh probes/engine_kernel_check.py --imports-only
ST_IMAGE=st-engine:9391 bash probes/run_engine_probe.sh probes/engine_kernel_check.py
ST_IMAGE=st-engine:9391 bash probes/run_engine_probe.sh probes/engine_kernel_check.py --lanes moe --moe-experts 288
ST_IMAGE=st-engine:9391 ST_GLM53_KDA_PREFILL_QK_NORM=1 bash probes/run_engine_probe.sh probes/engine_kernel_check.py --lanes kda
docker run --rm --gpus all --network none -i --entrypoint python3 st-engine:9391 - < measurements/st_engine_native_kernels_20260911/verify-runtime.py
docker run --rm --gpus all --network none -w /repo -e PYTHONPATH=/repo --mount "type=bind,src=$PWD,dst=/repo,readonly" --entrypoint python3 st-engine:9391 -m unittest discover -s tests -p 'test_engine_*.py' -v
```

검증 컨테이너에는 CPU 4개·메모리 16 GiB·JIT 빌드 작업 2개를 허용했다.
캐시는 격리된 `/tmp/st-engine-9391/cache`를 사용했다. 로그의 행 끝 공백만 제거했다.
GPU 커널 로그는 최종 코드의 `/repo` 마운트에서, 런타임 검사는 빌드된 이미지에서
실행했다. 288-expert 검사는 MoE 독립 seed를 명시하기 전 실행했지만 단독 실행의
seed 29와 입력은 동일하다.

전체 45층의 실가중치 품질, TP4 전체 모델 부팅, 실제 DFlash2 수용률,
처리량·ITL은 이번 커널 이식 검사에 포함하지 않았다.
