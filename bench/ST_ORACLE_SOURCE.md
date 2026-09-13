# 개발 중인 코드로 예측하기

```sh
python3 bench/storacle.py predict --base origin/main
# 같은 명령의 별칭; staged/unstaged/untracked engine 소스까지 포함
python3 bench/storacle.py compare --base HEAD --json
# 특정 기준·후보 커밋, 다른 체크아웃, 실행 설정도 지정 가능
python3 bench/storacle.py compare --tree /path/to/stkernel --base BASE_SHA --candidate CANDIDATE_SHA
python3 bench/storacle.py compare --base HEAD --set prefill_tiles=2
```

현재 지원하는 소스 프로브는 native GLM53이다. 기본 비교 범위는 2K/32K/128K와 C=1/C=4다.
`--ctx 32000,128000 --width 1,4 --acc 0.45`로 범위를 정한다. `--acc`는 양쪽에 적용하는
평균 수락률 **가정**이다. 코드 수정으로 품질이나 수락률이 좋아졌다고 간주하지 않는다.
출력의 `output_rate_assumption`은 요청당·전체 예상 tok/s를 구분한다.

`predict`의 기존 모델 레지스트리/범용 레인 모드는 그대로이고, `--base`를 주면 이 소스 비교로 들어간다.
기존 `sim --compose`는 #838 계측 형상을 재현하는 모드다. 현재 개발 코드의 형상은 이 비교에서 읽는다.

## 실제로 읽고 실행하는 것

- 기준 커밋의 `engine/` 소스와 현재 작업트리를 각각 고정된 사본으로 만든다. 원본 체크아웃·인덱스를 바꾸지 않는다.
- `facts`, `kernel_shape`, `ExecutionPlan`, `scheduler.Contract`, `chunk_for`, `caches.layout`·`snapshot_layout`·`stage_bytes`는 **각 사본의 코드**를 별도 CPU 프로세스에서 읽거나 실행한다. 모델 가중치·CUDA 그래프는 로드하지 않는다.
- `boot.py`의 production 기본값, 실제 `ExecutionPlan`/`Contract` 생성식과 token budget 식을 읽는다. 지원하지 않는 생성식으로 코드가 바뀌면 오래된 숫자로 폴백하지 않고 오류를 낸다.
- K·청크·검증 폭은 같은 오라클 시간 모형에 넣고, 캐시/상태 바이트는 실제 layout 함수로 다시 계산한다. 시간 계수는 양쪽 모두 동일한 #838 계측값을 출발점으로 쓴다.
- 기본 geometry는 GLM53의 기존 참조 토폴로지(45층, DSA 11층)와 해당 소스의 `kernel_shape.MEASURED`다. `--config config.json`을 주면 해당 소스의 `facts.architecture`로 실제 체크포인트 형상을 검증한다. drafter geometry는 소스의 참조값이며 drafter 체크포인트는 로드하지 않는다.

현재 기본값에서 청크는 32,256토큰, decode reservation은 2,310토큰이다. 이는 과거
오라클의 9,216토큰 prefill 형상과 다르다. `TOKEN_BUDGET`을 바꾸거나 `chunk_for`의
계산을 수정하면 변경된 코드가 구한 청크와 prefill 예상 시간이 다음 호출에 반영된다.
상태 layout을 바꾸면 슬롯/스냅샷/경계 stage 바이트도 바뀐다. 할당 바이트를 줄인 비율을
GPU 시간 절감 비율로 사용하지 않는다.

## 커널 개선의 계측값 연결

커널 코드의 해시 변경만으로 새 GPU 시간을 알아낼 수는 없다. 변경 파일은 MoE,
비MoE/KDA, attention, communication, drafter, prefill 비용 항목에 연결한다. 공용 코드처럼
범위가 불명확한 변경은 넓게 표시한다. 미계측 항목이 있으면 `modeled_delta`에는 기존
계수로 계산한 부분 변화가 남고, 전체 `delta`는 `null`이다. 주석만 바뀐 Python 코드는
미계측 커널 변경으로 취급하지 않는다. 다른 profile의 전용 코드는 GLM 비용에 넣지 않는다.

같은 runtime에서 기준/후보의 해당 비용 항목을 측정할 때 먼저 입력을 고정한다:

```sh
python3 bench/storacle.py compare --base BASE_SHA --ctx 32000 --width 1 \
  --write-profile-template /tmp/st-paired-profile.json
```

템플릿의 `base`/`candidate` fingerprint는 소스 내용·실제 실행 설정·geometry/layout에
묶인다. `inputs`에 커밋과 소스 SHA도 남는다. ref의 이름이나 commit 여부만 바뀌고 소스와
설정이 같으면 fingerprint는 유지된다. 새 코드나 다른 실행 설정의 계측값을 재사용하지 않는다.
현재 구현은 원본 소스 바이트를 fingerprint에 포함하므로 주석 수정 후에도 프로파일을 새로 고정해야 한다.

템플릿의 `runtime`에 공통 GPU/이미지/드라이버 식별자, `evidence`에 원본 측정 파일이나
실행 기록을 적고 각 component의 `base_ms`, `candidate_ms`, `samples`를 채운다.
`samples`는 양쪽에 확보한 paired 표본 수다. 빈 값·0개 표본은 소비할 수 없다.
decode 시간은 해당 ctx/width에서 **스텝 전체의 해당 구성요소** 시간이고, 단일 레이어의
커널 시간은 아니다. 반복 호출 수를 반영해야 한다. prefill 시간은 C=1의 전체 입력 compute 시간이다.

```sh
python3 bench/storacle.py compare --base BASE_SHA --ctx 32000 --width 1 \
  --profile /tmp/st-paired-profile.json --json
```

측정한 비용 항목만 양쪽의 시간 모형에서 교체한다. 다른 미계측 변경이 남으면 전체
변화는 계속 미확정이다. 기록은 사용자가 제공한 계측 자료이며 도구가 GPU 측정을 대신하지 않는다.
profile은 같은 소스에서 다른 컨텍스트/폭으로 자동 외삽하지 않는다.

실제 TTFT에는 큐·토큰화·JIT·접두사 재사용이 추가된다. 이 모드의 prefill은 compute 예측이고,
속도/품질 판정은 해당 소스로 실행한 consumer onepass 기록에서 한다.
