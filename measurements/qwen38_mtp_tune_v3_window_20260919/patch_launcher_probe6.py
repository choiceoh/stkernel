"""Windows 5 and 6 (operator 2026-09-19: "경합일어나도 상관없으니까 ... 같이하던가"): the measurement tree's launcher leaves
the single-GPU lane's st-probe-* containers out of its busy check, so a boot proceeds beside a lane probe on srv4.
A local edit of ~/st-worktrees/q38mtp-w4 and -w6 -- never committed."""
from pathlib import Path

p = Path.home() / "st-worktrees/q38mtp-w6/launchers/start-st-qwen38.sh"
s = p.read_text()
old = """  busy=$(node_sh "$ip" "docker ps --format '{{.Names}}' | grep -E '^(glm53|q38|vllm|st-)' || true")"""
new = """  busy=$(node_sh "$ip" "docker ps --format '{{.Names}}' | grep -E '^(glm53|q38|vllm|st-)' | grep -v '^st-probe-' || true")"""
if new not in s:
    assert s.count(old) == 1, "the busy check moved"
    s = s.replace(old, new)
    p.write_text(s)
print("patched" if new in p.read_text() else "not patched")
