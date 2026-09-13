#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""ST 오라클(ST Oracle) — 하나의 문.

플릿 없이 스텝과 레이턴시를 답하는 이 리포의 시뮬레이션 도구군의 이름이다. 엔진의 실제
스텝 루프를 돌리고(step_sim), 스텝을 바이트로 조립하며(step_kernels), 살아있는 부팅을
관측하고(step_peek), 저장된 증거를 다시 판다(step_replay). 이 파일은 얇은 문: 하위
명령을 각 도구로 건네고, 모델 레지스트리의 상태를 한 장으로 보여준다.

    python3 bench/storacle.py models                 # 모델별 바이트 상태(측정/구조/결측)
    python3 bench/storacle.py sim --compose ...      # step_sim 으로
    python3 bench/storacle.py kernels --model ...    # step_kernels 로
    python3 bench/storacle.py peek|replay ...        # 관측·재분석으로

D17 그대로: 오라클은 부팅 수를 줄이지, 판정을 대신하지 않는다.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOLS = {"sim": "step_sim.py", "kernels": "step_kernels.py",
         "peek": "step_peek.py", "replay": "step_replay.py"}


def cmd_models() -> int:
    sys.path.insert(0, str(HERE))
    import step_kernels as kern
    print("ST-Oracle · 모델 레지스트리 — 측정됨/구조/결측:")
    for name in sorted(kern.MODEL_REGISTRY):
        b = kern.EngineBytes.for_model(name)
        miss = b.missing()
        have = []
        if b.kv_bytes_per_token:
            have.append(f"KV {b.kv_bytes_per_token / 1024:.2f} KiB/토큰")
        if b.state_ring_bytes:
            have.append(f"상태 {b.state_ring_bytes / (1 << 20):.1f} MiB/행")
        if b.experts:
            have.append(f"전문가 {b.experts}×top{b.topk}")
        status = "조립 가능" if not miss else "결측: " + ", ".join(miss)
        print(f"  {name:<8} {'·'.join(have) or '(레지스트리 항목 없음)'} → {status}")
    print("\n결측은 한 번의 체크포인트 읽기(config.json·랭크 파일 크기) 또는 프로브가 채운다 —")
    print("채워지면 --compose 가 그 모델의 사다리를 같은 검증 경로로 만든다.")
    return 0


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return 0
    sub = sys.argv[1]
    if sub == "models":
        return cmd_models()
    if sub not in TOOLS:
        print(f"알 수 없는 하위 명령 {sub!r} — models | sim | kernels | peek | replay", file=sys.stderr)
        return 2
    return subprocess.run([sys.executable, str(HERE / TOOLS[sub]), *sys.argv[2:]]).returncode


if __name__ == "__main__":
    sys.exit(main())
