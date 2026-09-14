# B12X 세부 최적화 이식 — 2026-09-14

운영자의 “전부 가져와”와 “13.2 전체 마이그레이션해”를 함께 적용했다.
비교 기준 커널은 main `60b4c6e7`이다. B12X `12b4eb2574416c524eef0da273e2c063d35347d3`의
`b12x/_lib/intrinsics.py`와 Apache-2.0 고지를 따라 ST의 주소·배리어 계약에 맞게 이식했다.

## 적용

- SF6의 같은 공유 메모리 워드를 반복 읽고 shift/mask하던 부분을 u16/u8 직접 읽기와
  컴파일 시점 immediate displacement로 바꿨다. volatile, side effects, 링 소유권과 shuffle 순서는 유지한다.
- MLA의 연속·strided E4M3 두 값을 한 번에 BF16x2로 넓힌다. CUDA 13.2의
  `cvt.rn.bf16x2.e4m3x2`를 사용하고 이전 컴파일러는 거부한다.
- query 양자화와 BF16 MMA/FP32 누적 순서는 바꾸지 않았다.

## CPU 및 컴파일 증거

`sf6-operands.json`: 실제 CuTe helper 네 종류를 컴파일하고 독립 CPU 인코더와
24,576 operand word를 비교했다. 주소 정렬·경계·canary를 검사했다. GPU 수치 검사는 아니다.

`sf6-baseline.json` / `sf6-native.json`: 같은 CuTe 4.6.2에서 실제 전체 커널을 컴파일했다.

| 경로 | 정적 SASS 명령 위치, 기준 → 후보 | 레지스터, 기준 → 후보 |
|---|---:|---:|
| SF6 M1 | 3,646 → 3,566 | 119 → 115 |
| SF6 M7/M8 | 3,670 → 3,590 | 119 → 115 |
| M8 계측용 변형 | 3,764 → 3,663 | 96 → 115 |
| M8 register-scale 미사용 | 3,526 → 3,526 | 121 → 121 |
| M16/M32 기존 경로 | 6,148/6,164, 동일 | 117, 동일 |

M8 register-scale 미사용과 M16/M32의 전체 native binary SHA256은 기준과 같다.
일반 SF6 경로의 shared memory는 91,136 bytes로 같고 stack/local은 0이다.
계측용 변형은 레지스터가 늘었으므로 모든 변형의 자원이 개선됐다고 해석하지 않는다.

최종 NVCC 13.2.78의 MLA 증거는
[`../st_cuda132_20260914/mla-instructions.json`](../st_cuda132_20260914/mla-instructions.json)에 있다.

| MLA 경로 | 정적 SASS 명령 위치, 기준 → 후보 | 레지스터, 기준 → 후보 |
|---|---:|---:|
| decode | 1,360 → 1,264 | 97 → 97 |
| cluster | 1,568 → 1,480 | 63 → 62 |
| prefill32 | 1,112 → 936 | 128 → 128 |
| conversion | 48 → 40 | 14 → 16 |

같은 13.2.78 컴파일러, 같은 추출 harness, 같은 SM121a 대상이다. 네 경로 모두 spill은 0이다.
최종 CUDA 소스 SHA256은 `c70fade02943a0b17b93107ef4b619e40b2ceca8e3c71c3d0a480676860cfd26`이다.

이 폴더의 `mla130-*`와 `mla132-*`는 전체 이전을 결정하기 전, cu130 Torch 위에서
13.0.88 / 13.2.51 컴파일러를 분리한 탐색 기록이다. 당시에는 이전 변환 분기를 남겼다.
JSON의 소스 해시가 그 시점을 구분한다. 최종 13.2 전용 소스의 증거로 대체해서 읽지 않는다.

## 재현과 범위

CPU 전용 컨테이너에서 `probes/engine_moe_sf6_check.py --cpu`,
`probes/engine_moe_sf6_compile.py --register-scales --sass`,
`probes/engine_mla_microopt_compile.py --baseline <기준 .cu> --cuda-root /usr/local/cuda-13.2`
각각에 `--output`을 지정한다. 실제 일곱 확장의 compile/dlopen은
`probes/engine_cuda132_native_compile.py`로 검사했다.

**실행 시간·step/s·tok/s·수용률의 개선을 주장하지 않는다.** 정적 명령 개수는 런타임 속도가 아니다.
운영자의 “큐 태우지마”에 따라 큐/onepass/GPU 수치·그래프 검사를 하지 않았다.
최종 런타임과 네 노드 이미지 기록은 [CUDA 13.2 이전 기록](../st_cuda132_20260914/README.md)에 있다.
