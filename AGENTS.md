# 에이전트의 개발 도구 입구

> 살아 있는 참조 — **에이전트가 도구를 찾고 쓰는 순서.** 실제 도구와 다르면 그건 버그다.

새 세션에서는 먼저 `./dev`를 실행한다. Python 표준 라이브러리만 필요하며,
셸 초기화나 패키지 설치 없이 JSON으로 도구와 작업 순서를 반환한다.

- 도구를 찾거나 비슷한 스크립트를 만들기 전: `./dev search "<목적>"`.
- 도구의 인자·효과·실행 위치·예시 확인: `./dev describe <id>`.
- 설치·버전·Python 경로 확인: `./dev status`.
- 실행: `./dev run <id> -- <원래 인자>`. 실제 명령만 확인하려면
  `./dev run --dry-run <id> -- <원래 인자>`.
- 파일을 바꾼 뒤 검사 선택: `./dev run feedback -- <바꾼 파일들>`.
- 새 도구는 검색에 자동으로 나타난다. 관리되는 실행기로 연결하려면
  `tools/dev_catalog.py`에 어댑터를 추가하고 `./dev audit`로 확인한다.

`run`의 `process_exit_code`는 자식 프로세스 결과다. `validation`과 실제 증거를
따로 읽는다. `incomplete`·`CANNOT RUN`·skip을 검사 통과로 쓰지 않는다.
GPU 작업은 기존 `fleet.*` 도구를 사용한다. 큐의 승인·소스 고정·증거 규칙은
[bench/EXPERIMENTS.md](bench/EXPERIMENTS.md)가 정한다. `fleet.*`의 파일 경로는
컨트롤러 기준이며 로컬 파일을 자동으로 복사하지 않는다.

PR·측정 기록·코드 지도에 관한 저장소 공통 규칙은 [CLAUDE.md](CLAUDE.md)를 읽는다.
도구 계약과 환경 선택의 자세한 설명은 [docs/DEVTOOLS.md](docs/DEVTOOLS.md)에 있다.
