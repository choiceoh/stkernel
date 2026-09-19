# 에이전트 개발 도구

> 살아 있는 참조 — **도구 검색·실행·관리의 계약.** 실제 동작과 다르면 그건 버그다.

에이전트는 `./dev` 하나로 시작한다. 검색, 환경 상태, 인자 설명, 실행 결과는
`schema_version: 1`인 JSON이다. 명령 도움말 `--help`만 일반 CLI 텍스트다.
기존 실행기는 그대로 판정하며, 통합 입구는 발견·환경 선택·호출·결과 포장을 맡는다.

```bash
./dev
./dev search '검사'
./dev describe feedback
./dev status
./dev run feedback -- engine/kernels/mhc_contract.py
./dev run check -- --pattern test_engine_dev_tools --list
./dev run --dry-run env.sync -- --verify
./dev audit
```

## 에이전트의 호출 계약

| 명령 | 반환하는 것 |
|---|---|
| `./dev` | 현재 저장소·선택한 Python·도구 요약·다음 작업별 명령 |
| `list` | 관리되는 도구의 ID·목적·효과·실행 위치 |
| `search <말> [--limit 20] [--offset 0]` | 관리 도구와 실제 소스에서 발견한 스크립트. `next_offset`으로 다음 페이지 |
| `describe <id 또는 경로>` | 인자 계약·예시 argv·원본 도움말·효과·필요 패키지·문서 |
| `status` | 선택된 Python의 패키지 버전과 `versions.env` 비교, 실행 파일 위치, 누락·복구 명령 |
| `run [--dry-run] [--timeout 초] [--max-output 바이트] <id> -- <인자>` | 기존 도구 실행 결과. 옵션은 도구 ID 앞에 둔다 |
| `audit` | 사라진 실행 파일·문서·중복 ID와 아직 어댑터가 없는 스크립트 수 |

검색은 `tools/`, `bench/`, `probes/`, `launchers/`, `engine/runtime/`의 실제 파일을
읽는다. Python의 `__main__` 진입점과 셸 스크립트를 찾으며, import나 실행을 하지
않는다. 새 파일도 바로 검색되므로 등록을 잊었다고 사라지지 않는다.
`managed: false`는 검색·소스 설명만 가능하다. 실행 경로와 필요한 효과를 검토한 뒤
`tools/dev_catalog.py`에 `Tool`을 추가하면 `run`에서도 사용할 수 있다.

## 실행 환경

Python은 `ST_DEV_PYTHON` 지정 → 체크아웃 `.venv/bin/python` → Mac의
`~/.venvs/stkernel/bin/python` → 호출한 Python 순서로 고른다. 지정한
`ST_DEV_PYTHON`이 없으면 다른 환경으로 조용히 바꾸지 않고 `blocked`를 반환한다.
선택된 Python의 bin을 **자식 프로세스** PATH 앞에 두므로 Python을 다시 부르는
기존 셸 실행기도 같은 환경을 사용한다. `.zshrc`를 읽거나 전역 PATH를 바꾸지 않는다.

`status`는 패키지 메타데이터를 읽는다. torch import, GPU 할당, SSH, 인증 정보
읽기, 설치를 하지 않는다. 실행 파일이 있다는 사실과 로그인·서비스·GPU가 준비됐다는
사실은 다르므로 후자는 미확인으로 명시한다. 누락된 시스템 도구는 호스트 패키지
관리자로 관리하고, Python 패키지와 노드 도구는 `tools/devenv`를 따른다.

`env.setup`과 `env.sync`를 실행하면 기존 설치 스크립트가 실제로 환경을 바꾼다.
둘의 `--dry-run`은 통합 입구에서 **호출할 명령만 보여 주는 것**이며 설치 계획을
미리 계산하거나 설치 검증을 대신하지 않는다. 다른 조회 명령이 설치를 유발하지 않는다.

## 결과와 증거

`run`은 `shell=True`를 사용하지 않는다. 원래 인자는 그대로 argv로 전달하고,
stdin도 전달한다. stdout·stderr는 기본 각각 마지막 12,000바이트까지 반환한다.
생략한 출력은 `truncated`로 표시한다. JSON 원문 전체를 받은 경우 `data`로 제공하고
중복 출력을 줄이기 위해 `stdout`은 null로 둔다.
호출이 끝나면 임시 출력은 지운다. 보존할 실험 증거와 로그는 기존 실험 도구가 관리한다.

| 결과 | 종료 코드 | 뜻 |
|---|---|---|
| `completed` | 0 | 호출이 끝남. 기본 `validation: not_assessed`는 검증 통과 주장이 아님 |
| `failed` | 자식 코드 | 실행기 실패. `process_exit_code`와 stderr 확인 |
| `blocked` | 2 | 인자·실행 위치·의존성 문제. `recovery` 확인 |
| `incomplete` | 3 | `check`가 0으로 끝났어도 실행 불가·skip·0개 검사가 남음 |
| `timeout` | 124 | 제한 시간 경과, 로컬 프로세스 그룹 종료 |
| `cancelled` | 130 | 로컬 호출 취소 |

`check`의 원래 종료 코드는 바꾸지 않고 JSON에 보존한다. 통합 입구는 그 요약에서
검증 완결성을 별도로 판정한다. CPU 검증은 GPU 정밀도·속도·품질의 증거가 아니다.

## 공유 GPU 도구

`fleet.*`는 `FLEET_CONTROLLER`(기본 `srv2`)의 공개 실행기
`~/glm53-logs/fleet.sh`를 사용한다. 보조 파일을 찾는 저장소와 작업 디렉터리는
컨트롤러의 `~/stkernel`로 명시한다(`ST_DEV_FLEET_REPO`로 절대 경로 지정 가능).
공개 실행기는 복사본이므로 파일 위치만으로 저장소를 계산하면 잘못된 경로가 된다.
컨트롤러에서는 직접 호출하고, 다른 컴퓨터에서는 기존 SSH 설정으로 접속한다.
원격 인자는 각각 셸 인용한다. 새 큐를 만들거나,
로컬 체크아웃을 배포하거나, 기존 큐의 승인·리스 검사를 대신하지 않는다.

```bash
./dev run fleet.status
./dev describe fleet.submit
./dev run fleet.submit -- agent-session /controller/path/spec.json
./dev run fleet.result -- EXPERIMENT_ID
./dev run fleet.inbox -- agent-session --after 0
```

모든 manifest·소스 경로는 **컨트롤러 기준**이며 상대 경로는 위 저장소에서 시작한다.
큐 조회가 0으로 끝났어도 보조 파일이나 리스를 읽지 못했으면 `incomplete`를 반환한다.
제출한 후보 SHA와 준비 영수증의
계약은 `bench/EXPERIMENTS.md`를 따른다. 원격 호출이 timeout이면 접속만 끝났을 수
있다. 다시 제출하기 전에 `fleet.show`·`fleet.jobs`·`fleet.result`로 상태를 확인한다.

## 유지보수

에이전트 시작 안내는 `AGENTS.md`에 있고 `CLAUDE.md`에서도 연결한다. 도구 설명을
문서마다 복제하지 않고 카탈로그와 실제 스크립트에서 읽는다. CI의
`tests/test_engine_dev_tools.py`가 어댑터 경로, 검색, 실제 argv 전달, 잘못된 환경,
원격 인용, 시간 초과, 검증 미완료 판정을 확인한다.
