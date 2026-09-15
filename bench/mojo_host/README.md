# Mojo host commit 실험

> 그날의 조사 — **2026-09-14 의 조사다.** 그날 참이었던 것이고 유지되지 않는다 — 이후 무엇이 바뀌었는지는 `MEASUREMENTS.md` 가 안다.

2026-09-14 결론: **이 함수의 Mojo 전환은 보류한다.** 실제로 컴파일한
Mojo 1.0.0 확장을 Python에서 호출했지만, Apple M5 CPU에서는 기존 Python보다
느렸다. 이 디렉터리에는 재현 가능한 실험만 있으며 엔진의 import, 기본값,
이미지 의존성을 추가하지 않는다.

## 무엇을 비교했나

`engine/profiles/glm53/burst_decode.py::BurstDecode._apply_outcome`의 호스트 처리를
대상으로 삼았다. 기준 함수는 AST로 **현재 소스에서 직접 읽으므로** torch나
Triton을 import하거나 손으로 옮긴 기준 구현을 사용할 필요가 없다.
랭크 합의 호출은 양쪽 모두 같은 no-op으로 대체한다.

Mojo 후보는 결과 한 건의 C=1..4 행을 한 번에 처리한다. 행의 생존 여부와
count/before를 고정 크기 네이티브 배열에 캐시하고, 기존 Python 토큰 이력과
문맥 길이·수용 카운터·히스토그램·블록 경계에 결과를 반영한다. 실제로 연결할
경우 필요한 Python 래퍼 호출, PythonObject 변환, 리스트 갱신, 결과 기록까지
측정한다. Mojo 1.0 바인딩의 일반 Exception은 알려진 세 가지 가드 오류에만
RuntimeError로 변환하며 그 비용도 포함한다.

검증은 현재 Python 함수와 비교한다. C=1..4, K=1/3/5/7, 문맥·블록 경계,
해제된 요청, EOS/zero count, 이미 완료된 행, 수용 카운터, 중복/문자열 ID,
잘못된 count와 stale context, no-progress 오류 우선순위를 다룬다.
랭크 합의 실패는 두 구현 모두 상태 변경과 결과 공개 전에 전파해야 한다.

CPU 실험의 정수는 디코드에서 사용하는 범위 안의 값이고 컨테이너는 built-in
dict/list/tuple이다. 임의의 Python 서브클래스, 정수 오버플로, GPU 메모리 공개
순서, 실제 랭크 통신·취소 경합을 검증한 범용 네이티브 라이브러리는 아니다.

## 로컬 기록

Apple M5 / macOS 26.6 arm64 / CPython 3.12.13 / Mojo 1.0.0 (ed45d567).
기준 소스는 `119f881d`의 burst decode다. 원본
[JSON](../../measurements/mojo_host_20260914/macos_arm64.json)에 소스·라이브러리·
하네스 SHA-256, Python ABI, 도구 버전, 각 표본, 최종 상태 해시를 저장했다.

| 초기 토큰 이력 | 동시 행 | Python µs/결과 | Mojo µs/결과 | Mojo 지연 증가 |
|---|---:|---:|---:|---:|
| 32K (32,760) | 1 | 1.030 | 1.519 | +47.4% |
| 32K (32,760) | 4 | 2.596 | 3.742 | +44.1% |
| 128K (131,064) | 1 | 1.021 | 1.591 | +55.9% |
| 128K (131,064) | 4 | 2.628 | 4.289 | +63.2% |

조건마다 준비된 합성 결과 4,096건, 워밍업 2쌍, 기록 12쌍을 사용했다.
Python→Mojo / Mojo→Python 순서를 번갈아 실행하고 모든 쌍에서 최종 상태가
같음을 확인했다. `pending.outcomes`는 실제 burst처럼 4건마다 비우며, 이 공통
루프 비용도 포함한다. 준비·컴파일·import·검증·해시 계산은 시간에서 제외한다.
표의 값은 중앙값이고, 비율은 두 중앙값으로 계산했다. JSON에는 쌍별 비율의
중앙값도 별도로 남긴다.

이는 **호스트 함수의 CPU 마이크로벤치**다. 32K/128K는 Python 토큰 이력의
초기 길이이며, 모델의 해당 문맥 서빙 측정이 아니다. readback은 합성 데이터이고
통신·GPU·토큰화·TTFT·출력 tok/s·수용률·답변 품질을 측정하지 않았다.
플릿 큐에 실험을 넣거나 서버를 부팅하지 않았다.

현재 해석은 작은 산술 루프의 이득보다 Python 객체 경계의 비용이 크다는 것이다.
언어 전체의 우열이나 GB10의 성능 결론은 아니다. 이 경로는 Python 자체가
약 1~3µs이므로, 더 복잡한 네이티브 상태 소유권/버퍼 변환을 도입할 우선순위도
낮다. 이후 CPU 프로파일에서 충분히 큰 배열 연산이 확인되면, 연속 버퍼를
한 번 건네는 다른 후보를 이 방식으로 측정할 수 있다.

## 재현

컴파일은 별도 SDK 환경에서 명시적으로 한 번 수행한다. 아래 명령은 CPU만 쓴다.

```sh
uv venv build/mojo-sdk --python 3.12
uv pip install --python build/mojo-sdk/bin/python mojo==1.0.0
build/mojo-sdk/bin/python bench/mojo_host.py build --compiler build/mojo-sdk/bin/mojo
ST_MOJO_HOST_DIRECTORY=build/mojo-host build/mojo-sdk/bin/python -m unittest tests.test_mojo_host -v
build/mojo-sdk/bin/python bench/mojo_host.py run --output build/mojo-host/cpu.json
```

`run`은 미리 만든 `.so`만 읽는다. manifest의 소스, 바이너리, 컴파일러,
플랫폼, Python ABI/버전이 맞지 않으면 실패하고 다시 빌드하도록 안내한다.
자동 컴파일이나 Python 후보로의 조용한 대체는 없다. 바이너리는 플랫폼 간에
옮기지 말고 해당 CPU 환경에서 새로 빌드한다.

일반 unittest는 SDK 없이 하네스 검사만 실행하고 네이티브 6개 검사를 skip한다.
전용 `Mojo host CPU experiment` CI는 고정 버전 SDK를 설치하고 실제 확장을
컴파일한 뒤 **네이티브 검사를 필수 실행**한다. 모듈 누락은 실패다.
공유 CI 러너의 속도는 합격 문턱으로 삼지 않고 JSON artifact로만 보관한다.

공식 자료: [Mojo에서 Python 확장 만들기](https://mojolang.org/docs/manual/python/mojo-from-python/),
[Python 오류 변환](https://mojolang.org/docs/std/python/bindings/raise_python_exception/).
