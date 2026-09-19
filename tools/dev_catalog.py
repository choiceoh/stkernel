"""Adapters for the agent entrypoint. Execution stays with each existing tool."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Tool:
    id: str
    summary: str
    command: tuple[str, ...] = ()
    effects: str = "read"
    docs: str = "README.md"
    examples: tuple[tuple[str, ...], ...] = ((),)
    packages: tuple[str, ...] = ()
    platforms: tuple[str, ...] = ()
    timeout: int = 60
    execution: str = "local"
    notes: str = ""


TOOLS = (
    Tool("env.status", "환경·설치 도구·버전 차이와 복구 명령 확인 / environment inventory", execution="builtin"),
    Tool("env.doctor", "선택된 Python에서 CPU 개발환경 진단 / missing dependencies",
         ("{python}", "tools/dev_doctor.py", "--strict", "--json")),
    Tool("env.setup", "개발환경 설치·복구: 현재 컴퓨터를 버전 목록에 맞춤 / environment install repair",
         effects="environment-write", execution="setup", timeout=1800,
         notes="Mac은 mac.sh, Linux는 node.sh. 설치·사용자 PATH·git 설정·기본 체크아웃을 바꿀 수 있다."),
    Tool("env.sync", "서버와 5050의 개발환경을 버전 목록에 맞춤 / synchronize nodes",
         ("bash", "tools/devenv/sync.sh"), effects="remote-environment-write", timeout=1800,
         examples=((), ("--verify",)), notes="기본 대상 srv1..srv4, ost-97x. --verify도 먼저 설치를 수행한다."),
    Tool("check", "CPU 검사 실행 / tests verification", ("{python}", "tools/check.py"),
         effects="local-test", packages=("numpy", "safetensors"), timeout=1800,
         examples=(("--pattern", "test_engine_dev_tools", "--list"), ("--pattern", "test_engine_*")),
         notes="CANNOT RUN·skip은 검증 미완료다. GPU 검사는 fleet.run을 사용한다."),
    Tool("feedback", "변경 파일에 필요한 검사와 가장 싼 실행 레인 선택 / test routing",
         ("{python}", "bench/feedback.py", "--json"), docs="bench/EXPERIMENTS.md",
         examples=(("--base", "origin/main"), ("engine/kernels/mhc_contract.py",))),
    Tool("regress", "기준 커밋과 현재 트리의 검사 결과 비교 / regression",
         ("{python}", "tools/regress.py", "--json"), effects="local-test-and-fetch", timeout=3600,
         packages=("numpy", "safetensors"), examples=(("--ref", "origin/main"),)),
    Tool("push.check", "이미 병합된 브랜치인지 확인 / before push PR",
         ("{python}", "tools/push_check.py"), effects="git-fetch", docs="CLAUDE.md"),
    Tool("ledger.rebase", "측정 원장 충돌 정리 / MEASUREMENTS merge",
         ("{python}", "tools/ledger_rebase.py"), effects="working-tree-write", docs="CLAUDE.md",
         examples=(("--dry-run",),), notes="기본 실행은 원장과 참조를 수정한다. --dry-run으로 먼저 확인한다."),
    Tool("graph.build", "engine/base 코드 지도 갱신 / graphify",
         ("bash", "tools/graphify_engine_base.sh"), effects="working-tree-write", timeout=600,
         docs="graphify-out/README.md"),
    Tool("graph.check", "코드 지도와 실제 소스 일치 검사 / graph validation",
         ("{python}", "tools/validate_graphify_engine_base.py"), docs="graphify-out/README.md"),
    Tool("model.onboard", "체크포인트 형상·지원 경로·해야 할 일 확인 / model checkpoint",
         ("{python}", "tools/onboard.py", "--json"), examples=(("--ckpt", "/path/to/checkpoint"),)),
    Tool("runtime.inventory", "런타임 이미지에 설치된 커널과 도구 조회 / image inventory",
         ("{python}", "tools/inventory.py"), effects="container-run", docs="engine/INVENTORY.md", timeout=660,
         examples=(("--image", "st-engine:glm53"),), notes="--write는 engine/INVENTORY.md도 수정한다."),
    Tool("runtime.build", "고정한 시드로 ST 런타임 이미지 빌드 / Docker",
         ("bash", "engine/runtime/build.sh"), effects="image-write", docs="engine/runtime/README.md",
         platforms=("Linux",), timeout=3600),
    Tool("observe", "실행 중 엔진의 metrics 관측 / live observation",
         ("{python}", "bench/step_peek.py"), effects="network-read", docs="bench/EXPERIMENTS.md",
         examples=(("--seconds", "5"),), notes="관측값은 성능 채택 증거가 아니다. --out은 파일에 append한다."),
    Tool("fleet.status", "실제 컨트롤러의 GPU 큐 상태 / queue waiting",
         ("status",), execution="fleet", docs="bench/EXPERIMENTS.md"),
    Tool("fleet.submit", "기존 비동기 큐에 실험 제출 / async experiment",
         ("submit",), execution="fleet", effects="queue-write", docs="bench/EXPERIMENTS.md",
         examples=(("agent-session", "/controller/path/spec.json"),),
         notes="manifest와 그 안의 경로는 컨트롤러 기준. 소스·승인·GPU 리스는 기존 큐가 검증한다."),
    Tool("fleet.run", "정식 CPU/GPU 실행기에 작업 제출 / CPU single 5050 fleet",
         ("run",), execution="fleet", effects="queue-write", docs="bench/EXPERIMENTS.md",
         examples=(("--gpu", "--check", "--detach", "session", "--", "bash", "probes/run_engine_check.sh"),),
         notes="GPU는 --detach 또는 fleet.submit으로 제출하고 결과를 조회한다. 경로는 컨트롤러 기준."),
    Tool("fleet.result", "실험의 보존된 결과와 증거 조회 / result evidence",
         ("result",), execution="fleet", docs="bench/EXPERIMENTS.md", examples=(("EXPERIMENT_ID",),)),
    Tool("fleet.inbox", "에이전트가 구독한 실험 결과 조회 / inbox cursor",
         ("inbox",), execution="fleet", docs="bench/EXPERIMENTS.md",
         examples=(("agent-session", "--after", "0"),)),
    Tool("fleet.jobs", "실험 목록과 진행 상태 조회 / jobs active",
         ("jobs",), execution="fleet", docs="bench/EXPERIMENTS.md", examples=(("--active",),)),
    Tool("fleet.show", "예약 명령과 대기 이유 조회 / reservation inspect",
         ("show",), execution="fleet", docs="bench/EXPERIMENTS.md", examples=(("session",),)),
    Tool("fleet.logs", "실험의 보존된 로그 조회 / logs debug",
         ("logs",), execution="fleet", docs="bench/EXPERIMENTS.md", examples=(("session",),)),
    Tool("rg", "코드·문서 내용 검색 / text search", ("rg",), examples=(("-n", "pattern", "engine"),)),
    Tool("fd", "파일 이름 검색 / file discovery", ("fd",), examples=(("pattern", "tools"),)),
    Tool("jq", "JSON 조회와 변환 / JSON query", ("jq",), examples=((".", "path.json"),)),
    Tool("ruff", "Python 린트와 포맷 / lint format", ("ruff",), effects="arguments-dependent",
         examples=(("check", "tools/dev.py"),)),
    Tool("git", "변경·브랜치·이력 관리 / version control", ("git",), effects="arguments-dependent",
         examples=(("status", "--short", "--branch"),)),
    Tool("gh", "GitHub PR·검사·이슈 / pull request CI", ("gh",), effects="arguments-dependent",
         examples=(("pr", "status"),), notes="인증 상태 확인은 gh auth status. 토큰을 출력하거나 기록하지 않는다."),
    Tool("wt", "Worktrunk 작업 트리 관리 / worktree", ("wt",), effects="arguments-dependent",
         examples=(("list",),)),
    Tool("uv", "Python·가상환경·패키지 관리 / package environment", ("uv",), effects="arguments-dependent",
         examples=(("python", "list", "--only-installed"),)),
    Tool("docker", "로컬 컨테이너·이미지 관리 / container", ("docker",), effects="arguments-dependent",
         examples=(("ps",),), notes="GPU 실험은 docker로 직접 시작하지 않고 fleet.run에 제출한다."),
    Tool("hyperfine", "명령 실행 시간 반복 비교 / CLI benchmark", ("hyperfine",),
         effects="executes-arguments", examples=(("--help",),)),
)

BY_ID = {tool.id: tool for tool in TOOLS}

# Discovery remains available for tools that have not yet gained an execution adapter.
DISCOVERY_DIRS = ("tools", "bench", "probes", "launchers", "engine/runtime")
