#!/usr/bin/env python3
"""Summarize preserved fleet evidence without making any new GPU requests."""
import json
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
ansi = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
report = {"source_commit": (root / "source-commit.txt").read_text().strip(), "arms": {}}
health = root / "health-wall-seconds.tsv"
if health.exists():
    for line in health.read_text().splitlines():
        arm, seconds = line.split()
        report["arms"].setdefault(arm, {})["health_wall_s"] = int(seconds)
for path in sorted(root.glob("*-srv[1234].log")):
    arm, node = path.stem.rsplit("-", 1)
    text = ansi.sub("", path.read_text(errors="replace"))
    result = {"phase_s": {}, "source_loads": [], "fp8": [], "fold": [], "rank": []}
    for label, seconds in re.findall(r"\[boot-stamp\] (.*?) took ([\d.]+)s", text):
        result["phase_s"].setdefault(label, []).append(float(seconds))
    for tensors, gib, read, rate, apply in re.findall(r"weights: (\d+) tensors, ([\d.]+) GiB \| read ([\d.]+)s \(([\d.]+) GiB/s\) \| apply ([\d.]+)s", text):
        result["source_loads"].append(dict(tensors=int(tensors), gib=float(gib), read_s=float(read), apply_s=float(apply)))
    for label, enabled, hit, miss, errors, times in re.findall(r"\[fp8-cache\] (\S+) enabled=(\w+) hit=(\d+) miss=(\d+) errors=(\d+) host-seconds=([^\n]*)", text):
        result["fp8"].append(dict(label=label, enabled=enabled == "True", hit=int(hit), miss=int(miss), errors=int(errors), times_s={k: float(v) for k, v in re.findall(r"(\w+)=([\d.]+)", times)}))
    for label, stats, seconds in re.findall(r"\[fp8-dense\] (\S+) megakernel (.*?) in ([\d.]+) s", text):
        result["fold"].append(dict(label=label, seconds=float(seconds), stats=stats))
    for kind, rank, size, seconds in re.findall(r"\[rank-cache\] (hit|saved) rank=(\d+)(?: bytes=(\d+))? in ([\d.]+)s", text):
        result["rank"].append(dict(kind=kind, rank=int(rank), bytes=int(size) if size else None, seconds=float(seconds)))
    result["cache_warnings"] = [line for line in text.splitlines() if ("[rank-cache]" in line or "[fp8-cache]" in line) and any(x in line for x in ("unavailable", "rejecting", "skipped", "another rank"))]
    result["copy_disarmed"] = [int(n) for n in re.findall(r"(\d+) disarmed by the copy check", text)]
    report["arms"].setdefault(arm, {}).setdefault("nodes", {})[node] = result
onepass = root / "onepass.jsonl"
if onepass.exists():
    for line in onepass.read_text().splitlines():
        row = json.loads(line)
        report["arms"].setdefault(row["name"], {})["onepass"] = row
responses = {}
for path in sorted(root.glob("*-responses.jsonl")):
    arm = path.name.removesuffix("-responses.jsonl")
    responses[arm] = [json.loads(line) for line in path.read_text().splitlines()]
base = next((name for name in responses if name.endswith("BASE")), None)
if base:
    control = {r["prompt_sha256"]: r for r in responses[base]}
    for arm, rows in responses.items():
        report["arms"].setdefault(arm, {})["response_comparison"] = {
            "count": len(rows), "matching_prompts": sum(r["prompt_sha256"] in control for r in rows),
            "exact_response_matches": sum(r["prompt_sha256"] in control and r["response_sha256"] == control[r["prompt_sha256"]]["response_sha256"] for r in rows),
            "completion_tokens": [r["completion_tokens"] for r in rows],
        }
report["cold_warm_response_comparison"] = {}
for cold, rows in responses.items():
    if not cold.endswith("COLD"):
        continue
    warm = cold.removesuffix("COLD") + "WARM"
    if warm in responses:
        control = {r["prompt_sha256"]: r for r in rows}
        report["cold_warm_response_comparison"][warm] = {
            "count": len(responses[warm]),
            "matching_prompts": sum(r["prompt_sha256"] in control for r in responses[warm]),
            "exact_response_matches": sum(r["prompt_sha256"] in control and r["response_sha256"] == control[r["prompt_sha256"]]["response_sha256"] for r in responses[warm]),
        }
exit_path = root / "exit-code"
report["exit_code"] = int(exit_path.read_text()) if exit_path.exists() else None
resources = root / "host-resources.jsonl"
report["host_resource_samples"] = {}
if resources.exists():
    samples = [json.loads(line) for line in resources.read_text().splitlines()]
    for node in (1, 2, 3, 4):
        rows = [r for r in samples if r["node"] == node and "error" not in r]
        if rows:
            report["host_resource_samples"][f"srv{node}"] = {
                "n": len(rows), "first_epoch": rows[0]["epoch"], "last_epoch": rows[-1]["epoch"],
                "min_available_gib": min(r["memory_kib"]["MemAvailable"] for r in rows) / 1024**2,
                "min_disk_available_gib": min(r["disk_available_bytes"] for r in rows) / 1024**3,
                "swap_used_increase_mib": (rows[0]["memory_kib"]["SwapFree"] - rows[-1]["memory_kib"]["SwapFree"]) / 1024,
            }
(root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
lines = ["# GLM 5.3 Flash startup cache fleet trial", "", f"Source: `{report['source_commit']}`", "", "All arms use the same code, PREFILL_WARMUP=0 and the canonical Korean onepass workload at 2K and 32K. BASE disables both caches; COLD creates artifacts; WARM reuses them. Timings are observed single boots, not medians.", "", "| Arm | Health wall (s) | Node | Load model (s) | Target source read + apply (s) | FP8 fold sum (s) | Rank cache | FP8 hit / miss / error |", "|---|---:|---|---:|---:|---:|---|---|"]
for arm, data in report["arms"].items():
    for node, row in sorted(data.get("nodes", {}).items()):
        phase = row["phase_s"].get("load-model", [])
        target_source = max(row["source_loads"], key=lambda s: s["gib"], default=None)
        source = "skipped" if any(r["kind"] == "hit" for r in row["rank"]) else (f'{target_source["read_s"] + target_source["apply_s"]:.1f}' if target_source else "none")
        fold = sum(s["seconds"] for s in row["fold"])
        rank = "; ".join(f"{r['kind']} {r['seconds']:.3f}s" for r in row["rank"]) or "none"
        counts = " / ".join(str(sum(r[k] for r in row["fp8"])) for k in ("hit", "miss", "errors"))
        load = ", ".join(map(str, phase)) or "pending"
        lines.append(f"| {arm} | {data.get('health_wall_s', 'pending')} | {node} | {load} | {source} | {fold:.1f} | {rank} | {counts} |")
for arm, data in report["arms"].items():
    if data.get("onepass"):
        row = data["onepass"]
        lines.extend(["", f"{arm} onepass: quality `{row.get('quality')}`, Korean `{row.get('korean')}`. Exact response comparison: `{data.get('response_comparison')}`.", ""])
if report["host_resource_samples"]:
    lines.extend(["", "Host resources sampled every 10 seconds across the recorded trial interval (sampled minima, not CUDA peak allocations):", "", "| Node | Samples | Minimum available RAM (GiB) | Minimum disk available (GiB) | Swap use increase (MiB) |", "|---|---:|---:|---:|---:|"])
    for node, row in report["host_resource_samples"].items():
        lines.append(f"| {node} | {row['n']} | {row['min_available_gib']:.2f} | {row['min_disk_available_gib']:.2f} | {row['swap_used_increase_mib']:.1f} |")
if report["cold_warm_response_comparison"]:
    lines.extend(["", f"Exact generated responses compared with COLD: `{report['cold_warm_response_comparison']}`. These are transcript comparisons, distinct from artifact-byte integrity checks.", ""])
lines.extend(["", f"Trial exit: `{report['exit_code']}` (null means still in progress).", "", "See report.json for per-node phase breakdowns, cache warnings, identity and full onepass records. Original logs and exact responses are preserved beside this report. The source column is the target checkpoint iterator; drafter iterator timings remain in report.json. Target source-iterator time is absent on rank-cache hits. Host phase timers include existing synchronization and should not be treated as isolated GPU kernel timings. Recovery boots do not have an independent wall timer; their health is recorded in restore.log.", ""])
(root / "report.md").write_text("\n".join(lines))
print("\n".join(lines))
