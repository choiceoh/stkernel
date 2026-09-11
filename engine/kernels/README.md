# ST 커널 패키지

`engine.profiles.glm53.lanes.served()`가 사용하는 실행 코드를 이 패키지가 소유한다.
vLLM의 임포트, `torch.ops.vllm` 등록, FlashInfer 패키지 내부로의 커널 마운트는 없다.
라이브러리·컴파일러가 없거나 MLA 수치 판정이 실패하면 오류를 전달한다.

| 레인 | ST 구현 | 외부 라이브러리 |
| --- | --- | --- |
| KDA chunk / recurrent | `kda/`의 기존 커널 두 파일과 FLA 보조 7파일(`op.py` 포함) | PyTorch, Triton |
| causal conv | `causal_conv.py`의 prefill / update Triton 커널 | PyTorch, Triton, NumPy |
| 상태 링 | `state.py`에서 물리 슬롯·위치로 필요한 이력을 읽고 변경된 위치만 쓰기 | PyTorch, Triton |
| mHC pre / post | `mhc/`의 TileLang 혼합 커널과 작은 M의 prenorm 패딩 | PyTorch, TileLang, Triton, DeepGEMM |
| 인덱서 로짓 | `deep_gemm.py`에서 `deep_gemm.fp8_fp4_mqa_logits` 직접 호출 | DeepGEMM |
| 인덱서 query 양자화 | `kpool.py`의 Hadamard-128·FP8 커널, GB10 행 수별 1/8/32행 tile | PyTorch, Triton |
| kpool | `kpool.py`의 1워프 반환 전용 압축·회전·FP8 변환, 별도 캐시 쓰기 진입점 | PyTorch, Triton |
| 인덱서 슬롯 | `indexer.py`의 풀 ID 정렬·토큰 확장·페이지 주소 변환·유효 개수·출력 쓰기를 한 커널에서 처리 | PyTorch, Triton |
| MLA | `mla/`의 전용 Python 드라이버, warp max reduction과 DSMEM split 병합을 적용한 `glm53_megakernel.cu` | PyTorch, CUDA 13 nvcc |
| b12x MoE | `b12x/`의 API·디스패치·CuTe 커널·내부 보조 모듈 | PyTorch, CUTLASS DSL, CUDA bindings, FlashInfer 유틸/JIT |

`SOURCES.json`은 이식 전 파일의 경로와 SHA256을 기록한다. 저장소의 기존 overlay가
소유하는 구현을 우선했고, 없는 FLA/b12x 보조 파일과 conv/kpool은 같은 플릿 이미지에서
가져왔다. b12x의 stock `_moe_dynamic/gated.py`는 바이트를 유지해 기존 소스 검증도
그대로 유효하다. KDA의 L2norm 소스 검증 값은 임포트가 바뀐 로컬 파일의 SHA256이다.
MLA CUDA의 `sha256`은 이식 원본을 보존하고, `local_sha256`과 `local_modifications`는
ST의 하드웨어 최적화가 반영된 파일과 변경 내용을 검증한다.
라이선스와 출처는 `THIRD_PARTY_NOTICES.md`에 있다.

mHC는 GLM 레인이 호출하는 pre/post를 직접 제공한다. vLLM의 CustomOp 등록, 모델 후크,
플랫폼 디스패치와 DeepGEMM 미설치 대체 경로는 포함하지 않는다. KDA의 보조 RMSNorm은
일반 `torch.nn.Module`이다. MLA의 Python 드라이버는 sparse MLA에 필요한 빌드·작업공간·
수치 판정만 보존하며, 다른 모델의 GEMM/가중치 포장 후크를 가져오지 않는다.

GB10 플릿 이미지의 PDL 정책을 유지한다: conv·TileLang·DeepGEMM에서 끈다.
DeepGEMM 설정은 첫 실행에 한 번 적용해 CPU에서의 패키지 검사와 장치 배정 전 임포트가
CUDA 문맥을 만들지 않게 한다. MLA는 첫 eager 호출에 JIT와 수치 판정을 끝내야 한다.
그래프 캡처 중 처음 부르면 명시적으로 실패한다.

MLA는 GB10에서 실측한 thread-block cluster와 distributed shared memory(DSMEM)를
사용한다. 기본 split 계획과 합산 순서를 유지하면서 `32 <= T <= 64`, `1 <= W <= 2176`,
split 2 또는 3인 호출만 cluster로 실행한다. 각 split의 기존 shared memory를 재사용해
FP32 부분값을 합치므로 이 경로는 전역 partial 버퍼와 grid 전체 ticket barrier를 사용하지
않는다. 다른 형상은 기존 경로를 사용한다. `ST_GLM53_MK_MLA_CLUSTER=0`으로 cluster를
끄고 비교할 수 있으며, 부팅 시 실제 커널의 cluster 수용량과 수치 결과를 확인한다.
측정에서 느렸던 4~8-block cluster는 기본 디스패치에 포함하지 않았다.
결과와 재현 절차는 [GB10 MLA 측정](../../measurements/st_gb10_mla_20260911/README.md)에 있다.

인덱서 query 양자화는 회전·BF16 반올림·FP8 scale 계산을 유지하면서 launch 크기를
선택한다. 1,024행 이하는 1행·1 warp, 1,025~65,536행은 8행·1 warp를 사용하며,
더 큰 입력은 기존 32행·2 warp를 사용한다. GLM의 인덱서 head는 32개이므로 행 수는
토큰 수의 32배다. [GB10 양자화 측정](../../measurements/st_gb10_indexer_quant_20260911/README.md)에
레지스터·shared memory, 실제 가중치 검사와 형상별 시간을 기록했다.

이식한 Python 실험 변수는 `ST_GLM53_*` 이름을 사용한다. 예를 들어 KDA strided norm은
`ST_GLM53_KDA_PREFILL_QK_NORM`, mHC big-fuse 설정은 `ST_GLM53_MHC_BIGFUSE`다.
MLA는 필수 레인이므로 이전 `VLLM_GLM53_MEGAKERNEL`/`VLLM_GLM53_MK_MLA` 활성화 변수가
필요하지 않다. 원본 CUDA 파일 안의 진단용 `VLLM_GLM53_*` 문자열은 출처 보존을 위해
유지되지만 ST 실행기는 전달하지 않는다. FLA·FlashInfer 라이브러리 변수는 원래 이름이다.

## 런타임 이미지

저장소 루트에서 `bash engine/runtime/build.sh`로 `st-engine:glm53`을 만든다.
빌드 스크립트가 seed 이미지 ID를 확인하고 그 ID에 고정한 로컬 태그를 사용한다.
`ST_IMAGE`로 결과 이미지 이름을 지정할 수 있다.

seed는 기존 플릿과 같은 PyTorch/CUDA/CuTe ABI를 제공한다. 빌드 중 포함되어 있던
DeepGEMM의 확장 바이너리와 JIT 헤더 879개 파일을 `deep_gemm` 독립 경로로 복사하고
모든 SHA256을 확인한 다음, vLLM 패키지와 남은 overlay 파일을 제거한다. 결과 이미지에서
`find_spec('vllm') is None`과 두 DeepGEMM API를 검사한다. 실행 시 seed의 프레임워크
패키지에 접근하지 않는다. DeepGEMM 추출 기록은 설치 경로의 `ST_SOURCE.json`에 있다.

새 라이브러리 빌드를 채택할 때는 `runtime/dependencies.json`의 버전과 두 DeepGEMM
API, 전체 수치 검사를 다시 검증한다. 이미지 빌드·프로브는 기존 서빙 컨테이너를 변경하지 않는다.

    bash engine/runtime/build.sh
    ST_PROBE_NO_GPU=1 bash probes/run_engine_probe.sh probes/engine_kernel_check.py --imports-only
    bash probes/run_engine_probe.sh probes/engine_kernel_check.py
    bash probes/run_engine_check.sh --layers 0-4

GPU 검사는 사용 가능한 GB10에서 실행한다. JIT 캐시는 기본 `$HOME/.cache/st`에 두며
`ST_CACHE`로 변경한다. `--lanes conv,kda,mhc`처럼 일부 레인을 골라 재현할 수 있다.
b12x는 `--lanes moe --moe-experts 288`로 실제 TP4 형상(288 experts, top-k 8,
hidden 4096, rank intermediate 512)을 추가 검사한다. 이 검사에서만 seed의 원본
FlashInfer API를 호출해 이식 전후를 비교한다. 엔진은 항상 자체 b12x를 호출한다.
일반 이식 형상은 원본 커널 대비 상대 오차 2%로 검사한다. GLM TP4 고정 레인은 아래의
실가중치 반복 검사와 독립 반올림 참조를 추가로 사용한다.
원본과 ST의 JIT 캐시는 서로 다른 모듈 이름으로 저장한다.

사전 샤딩의 routed FC1은 b12x의 **up | gate** 순서이며 스케일도 같은 순서다.
이전 ST의 gate | up 파일은 사용할 수 없다. 새 preshard는
`weight_layout=st-glm53-b12x-up-gate-v1` 메타데이터를 기록하고, 부팅과 실가중치 검사는
이 값을 아레나 할당 전에 확인한다. 기존 파일은 다시 생성해야 한다.
참조의 FC1 누적 정밀도와 FP4 중간값 반올림도 FP32 및 ties-to-even으로 수정했다.
실가중치 분석에서는 PTX 근사 역수·FP8 반올림으로 생기는 FP4 활성값 차이와 BF16 합산의
실행 간 차이를 분리했다. GLM TP4의 micro/static 디코드는 BF16으로 반올림한 전문가 기여분을
미리 할당한 FP32 버퍼에 더한 뒤 한 번만 BF16으로 변환한다. 36개 조건·각 64회 반복에서
일반 실행과 그래프의 반복 차이는 0%, 독립 역수 반올림 참조 대비 최대 차이는 0.663%였다.
큰 프리필의 dynamic 커널은 유지되며, 전체 모델 품질·성능 판정은 아직 완료되지 않았다.
실험 원본과 한계는 `measurements/st_engine_completion_20260911/README.md`에 있다.

GPU가 없는 개발 환경에서는 아래 검사로 임포트 경로와 전이 의존 파일, 소스 보존을 확인한다.

    python3 -m unittest discover -s tests -p 'test_engine_*.py'
