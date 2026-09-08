#!/usr/bin/env python3
"""Reconstruct rank-transport startup evidence; never contact a serving process."""
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys

root, repo = map(Path, sys.argv[1:3])
subprocess.run([sys.executable, str(repo/"measurements/glm53_overlay_deploy_20260908/startup-cache-report.py"), str(root)], check=True, stdout=subprocess.DEVNULL)
data = json.loads((root/"report.json").read_text())
ansi = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
images, environments = set(), set()
source = data["source_commit"]
def canonical(name):
    return subprocess.check_output(["git", "-C", str(repo), "show", source+":build/glm53/"+name])
manifest = canonical("manifest.tsv").decode()
expected = {line.split("\t")[0]:hashlib.sha256(canonical(line.split("\t")[0])).hexdigest()
            for line in manifest.splitlines() if line and not line.startswith("#")}
expected["manifest.tsv"] = hashlib.sha256(("# source_commit="+source+"\n"+manifest).encode()).hexdigest()
target_names = ("glm53_rank_cache.py", "glm53_startup_cache.py", "glm53_megakernel.py", "gpu_worker.py", "deneb_boot_stamps.py")
verify = "--verify" in sys.argv
if verify:
    assert data["exit_code"] == 0 and int((root/"runner-exit").read_text()) == 0
    assert list(data["arms"]) == ["RANKPIPE"+s for s in ("PRIME", "BASE1", "FAST1", "FAST2", "BASE2")]
    gpu = json.loads((root/"gpu-exact.json").read_text())
    assert gpu["ok"] and gpu["corrupt_chunk_rejected"] and gpu["failure_drained"]
    assert [r["policy"] for r in gpu["runs"]] == [0,1,0,1]
    assert all(r["exact"] and r["drained"] for r in gpu["runs"])
    assert gpu["source_sha256"] == {name:expected[name] for name in target_names[:2]}
    cpu = json.loads(re.findall(r"CPU_RESULT=(.*)", (root/"cpu-image.log").read_text())[-1])
    assert cpu["tests"] >= 37 and cpu["failures"] == cpu["errors"] == cpu["skips"] == 0 and not cpu["cuda_initialized"]
for arm, row in data["arms"].items():
    fast, prime = "FAST" in arm, arm.endswith("PRIME")
    first = json.loads((root/f"{arm}-first-requests.json").read_text())
    row["first_requests"] = first
    env = dict(s.split("=",1) for s in json.loads((root/f"{arm}-cache-env.json").read_text()))
    row["runtime_environment"] = env
    environments.add(tuple(sorted((k,v) for k,v in env.items() if k != "VLLM_GLM53_RANK_CACHE_PIPELINE")))
    before = json.loads((root/f"{arm}-before-boot.json").read_text())
    after = json.loads((root/f"{arm}-after-boot.json").read_text())
    boot = (root/f"{arm}-boot.out").read_text()
    row["compile_cache"] = [json.loads(v) for v in re.findall(r"\[compile-cache\] (\{[^\n]+\})", boot)]
    if verify:
        assert first["ok"] and len(first["requests"]) == 3 and all(r["ok"] for r in first["requests"])
        assert row["onepass"]["quality"] == dict(ok=6,total=6) and row["onepass"]["korean"]["dirty"] == 0
        assert env["VLLM_GLM53_RANK_CACHE_PIPELINE"] == str(int(fast))
        assert env["VLLM_GLM53_RANK_CACHE_CPU_VOTE"] == env["VLLM_GLM53_SKIP_UNUSED_GRAPH_PROFILE"] == "0"
        assert len(row["nodes"]) == len(before) == len(after) == 4
        assert len(row["compile_cache"]) == 1
        if not prime: assert row["compile_cache"][0]["action"] == "reuse"
    for node, d in row["nodes"].items():
        text = ansi.sub("", (root/f"{arm}-{node}.log").read_text())
        d["io"] = [json.loads(v) for v in re.findall(r"\[rank-cache-io\] (\{[^\n]+\})",text)]
        d["stages"] = [json.loads(v) for v in re.findall(r"\[rank-cache-stage\] (\{[^\n]+\})",text)]
        d["w4"] = [dict(re.findall(r"(\w+)=([\d.]+)", v))
                   for v in re.findall(r"\[mk-pack-io\] ([^\n]+)", text)]
        state = (root/f"{arm}-{node}.state").read_text()
        images.add(state.splitlines()[0].split()[-1])
        if node == "srv2":
            health = text.find('GET /health HTTP/1.1" 200')
            posts = list(re.finditer(r'(\d+\.\d+\.\d+\.\d+):\d+ - "POST /',text))
            row["posts"] = dict(loopback=sum(m[1].startswith("127.") for m in posts), external=sum(not m[1].startswith("127.") for m in posts), before_health=sum(m.start()<health for m in posts))
            if verify: assert health >= 0 and row["posts"] == dict(loopback=7,external=0,before_health=0)
        if verify:
            assert state.startswith("running 0 false sha256:")
            receipt = (root/f"{arm}-{node}.sha256").read_text() if node == "srv2" else state
            actual = {Path(p).name:h for h,p in re.findall(r"([0-9a-f]{64})\s+(\S+\.py)",receipt)}
            assert actual == {name:expected[name] for name in target_names}, (arm,node,"wrong runtime")
            assert {name:v["sha256"] for name,v in after[node]["files"].items()} == expected
            assert before[node]["files"] == after[node]["files"]
            assert not d["cache_warnings"] and not any(d["copy_disarmed"])
            assert not sum(r["errors"] for r in d["fp8"])
            for phase in ("encoder-profile","profile-run","cudagraph-memory-profile","cudagraph-capture","compile+warmup"):
                assert phase in d["phase_s"]
            if not prime:
                assert before[node]["ninja"] == after[node]["ninja"]
                assert len(d["rank"]) == 1 and d["rank"][0]["kind"] == "hit"
                assert sum(r["hit"] for r in d["fp8"]) == 244 and not sum(r["miss"] for r in d["fp8"])
                assert len(d["w4"]) == 2
                assert [int(v["sha_hits"]) for v in d["w4"]] == [223,258]
                assert all(v["fast"] == v["sha256"] == "1" and
                           v["legacy_hits"] == v["md5_fallback"] == v["alias_errors"] == "0"
                           for v in d["w4"])
                assert len(d["io"]) == len(d["stages"]) == 1
                io = d["io"][0]
                assert io["ok"] and io["mode"] == ("pipeline" if fast else "serial")
                assert io["pinned_bytes"] == (2 if fast else 1)*64*1024*1024
                assert io["bytes"] == d["rank"][0]["bytes"] and io["chunks"] > 0
                assert d["stages"][0]["kind"] == "hit"
if verify:
    assert len(images) == len(environments) == 1
    data["verification"] = dict(ok=True,images=sorted(images),canonical_sources=expected)
samples = [json.loads(v) for v in (root/"host-resources.jsonl").read_text().splitlines()]
if verify:
    assert samples and all("error" not in v and v["returncode"] == 0 for v in samples)
for node, resource in data["host_resource_samples"].items():
    rows = [v for v in samples if v["node"] == int(node[-1]) and "error" not in v]
    resource["peak_swap_used_increase_mib"] = max(0, rows[0]["memory_kib"]["SwapFree"] -
        min(v["memory_kib"]["SwapFree"] for v in rows))/1024
    resource["max_sample_gap_s"] = max(b["epoch"]-a["epoch"] for a,b in zip(rows,rows[1:]))
    if verify:
        assert resource["n"] > 100 and resource["min_available_gib"] > 0
        assert resource["max_sample_gap_s"] <= 35
events = [json.loads(v) for v in (root/"fleet-lifecycle.jsonl").read_text().splitlines()]
data["execution_exits"] = json.loads((root/"execution-exits.json").read_text())
recovered = [v for v in events if v["session"] == "rankpiperecover0908" and v["event"] == "handoff-accepted"]
data["recovery_handoff"] = recovered
if verify:
    for session in ("rankpipe0908v3", "rankpiperecover0908"):
        completed = [v for v in events if v["session"] == session and v["event"] == "payload-finished"]
        assert len(completed) == 1 and completed[0]["rc"] == 0
    assert len(recovered) == 1 and data["execution_exits"]["recovery_supervisor"] == 0
    assert '10.10.10.2:' in (root/"recovery-head.log").read_text()
    assert re.search(r'10\.10\.10\.2:\d+ - "GET /health HTTP/1.1" 200', (root/"recovery-head.log").read_text())
summary = {}
for kind in ("BASE", "FAST"):
    rows = [r for a,r in data["arms"].items() if kind in a]
    if rows:
        metrics = {"health_wall_s":[r["health_wall_s"] for r in rows],
                   "head_load_model_s":[r["nodes"]["srv2"]["phase_s"]["load-model"][0] for r in rows],
                   "slowest_rank_restore_s":[max(d["io"][0]["total_s"] for d in r["nodes"].values()) for r in rows]}
        summary[kind] = {k:dict(samples=v,mean=statistics.mean(v)) for k,v in metrics.items()}
data["comparison"] = summary
lines = ["# Rank checkpoint pipeline startup", "", f"Measured source: `{source}`. PRIME excluded; B/A/A/B. CPU readiness vote fixed at 0; loopback API; background prefill bench off; required model/MM/graph warmup retained.", "", "| Arm | Health s | Head model s | Slowest rank restore s |", "|---|---:|---:|---:|"]
for arm,row in data["arms"].items():
    slow = max((d["io"][0]["total_s"] for d in row["nodes"].values() if d["io"]),default=None)
    lines.append(f"| {arm} | {row['health_wall_s']} | {row['nodes']['srv2']['phase_s']['load-model'][0]} | {slow} |")
lines += ["", "Serial mapped_hash_s includes page faults. Reader and DMA timers overlap; they must not be summed as sequential wall time.", "", "| Arm | Node | Restore s | Mapped hash s | Read s | Hash s | Host copy s | Copy wait s | Vote s |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
for arm,row in data["arms"].items():
    for node,d in row["nodes"].items():
        if not d["io"]:continue
        io=d["io"][0]
        values=[io[k] for k in ("total_s","mapped_hash_s","read_s","hash_s","host_copy_s","copy_wait_s")]+[d["stages"][0]["vote_s"]]
        lines.append("| "+arm+" | "+node+" | "+" | ".join(f"{v:.3f}" for v in values)+" |")
lines += ["", "Comparison: `"+json.dumps(summary)+"`", "", "Verification: `"+str(data.get("verification",{}).get("ok",False))+"`"]
(root/"report.json").write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n")
(root/"report.md").write_text("\n".join(lines)+"\n")
print("\n".join(lines[:11]))
