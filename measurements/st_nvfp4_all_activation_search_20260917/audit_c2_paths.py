"""Read completed receipts; never contact the engine or launch GPU work.

Server: python3 audit_c2_paths.py server --root ~/glm53-logs/onepass-runs
Local:  python3 audit_c2_paths.py quality --root /tmp/st-fp4-all-receipts
Config: python3 audit_c2_paths.py config --root /path/to/stkernel
All modes emit JSON to stdout. Canonical quality grades are left unchanged.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import runpy


RUNS = {
    "ss1": "20260917T130020-d15c307f7bb5",
    "as1": "20260917T133200-b3eb2a4aaa98",
}
PHASES = ("measure-c1", "measure-c2-2000-q1", "measure-c2-32000-qall")


def config(root):
    ns = runpy.run_path(str(root / "tests/test_engine_moe_scatter_config.py"))["namespace"]()
    parse, select, reuse, key = (ns[name] for name in (
        "_parse_glm53_static_v2", "_static_v2_decode_config",
        "_static_v2_input_reuse_config", "_static_v2_cache_key"))
    fields = ("decode_reform", "input_reuse", "activation_scale_search", "fc2_scale_search",
              "c2_direct_scatter", "c2_scatter_reuse", "c2_fc2_prefetch", "scatter_vec4",
              "scatter_packed_load", "sync_cleanup")
    cells = []
    for rows in (8, 16):
        configs = {}
        for arm in ("ss1", "as1"):
            cfg = reuse(select(parse("t,r,sf6,batch," + arm), rows),
                        288, 288, rows, 4096, 512, 8, rows * 8)
            configs[arm] = cfg
            cells.append({"arm": arm, "m": rows, "config": {k: cfg.get(k) for k in fields},
                          "effective_fc2_radius": cfg["activation_scale_search"] or cfg["fc2_scale_search"]})
        diff = {k for k in configs["ss1"] if configs["ss1"][k] != configs["as1"][k]}
        assert diff == {"activation_scale_search", "fc2_scale_search"}, diff
        assert key(configs["ss1"], m=rows) != key(configs["as1"], m=rows)
    source = root / "engine/kernels/b12x/moe_dispatch.py"
    return {"scope": "CPU execution of dispatch selectors, not a native numerical test",
            "source": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "cells": cells}


def read(path, receipts, *, lines=False):
    raw = path.read_bytes()
    receipts.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()})
    return [json.loads(line) for line in raw.splitlines()] if lines else json.loads(raw)


def server(root):
    receipts, results = [], []
    for arm, run_id in RUNS.items():
        for phase in PHASES:
            data = read(root / run_id / phase / "server.json", receipts)
            ranks = []
            for rank in data["ranks"]:
                steps = [row for row in rank["rows"] if row.get("kind") == "host_step"]
                prefill = [{key: row.get(key) for key in ("step", "rows", "positions", "tokens")}
                           for row in steps if row.get("phase") == "prefill"]
                widths = Counter(row["tokens"] for row in steps if row.get("phase") == "decode")
                ranks.append({"rank": rank["rank"], "complete": rank["complete"],
                              "prefill": prefill, "decode_rows": dict(widths)})
            assert len(ranks) == 4 and all(rank["complete"] for rank in ranks)
            results.append({"arm": arm, "run_id": run_id, "phase": phase, "ranks": ranks})
    return {"scope": "completed, profiler-off consumer server receipts; no new measurement",
            "receipts": receipts, "results": results}


def quality(root):
    receipts, results = [], []
    totals = defaultdict(lambda: [0, 0])
    for folder in ("base-1", "base-2", "candidate-1", "candidate-2"):
        arm = "ss1" if folder.startswith("base") else "as1"
        requests = read(root / folder / "requests.jsonl", receipts, lines=True)
        grades = read(root / folder / "quality.jsonl", receipts, lines=True)
        cases, groups = [], defaultdict(list)
        for row in grades:
            if row["phase"] != "measure-c1" and not row["phase"].startswith("measure-c2-"):
                continue
            c = 1 if row["phase"] == "measure-c1" else 2
            for result in row["results"]:
                if c == 2 or row["ctx"] in (2000, 32000):
                    total = totals[f"{arm}-c{c}-2k32k"]
                    total[0] += result["score"]
                    total[1] += result["max_score"]
                cases.append({"phase": row["phase"], "ctx": row["ctx"],
                              "client": row.get("client"), "case": result["case"],
                              "score": result["score"], "max_score": result["max_score"],
                              "failed_checks": [k for k, v in result["checks"].items() if not v]})
        for row in requests:
            if row["phase"].startswith("measure-c2-"):
                groups[row["phase"]].append(row)
        waves = []
        for phase, rows in groups.items():
            assert len(rows) == 2
            waves.append({"phase": phase,
                          "unique_workloads": len({row["workload_sha256"] for row in rows}),
                          "unique_outputs": len({row["output_sha256"] for row in rows}),
                          "clients": [{k: row.get(k) for k in (
                              "client", "workload_sha256", "output_sha256", "prompt_tokens",
                              "completion_tokens", "finish_reason", "ttft_s", "elapsed_s")}
                                      for row in sorted(rows, key=lambda r: r["client"])]})
        results.append({"folder": folder, "arm": arm, "cases": cases, "c2_waves": waves})
    return {"scope": "original strict grades; C1 two runs vs C2 two clients in one wave, not equal replication",
            "receipts": receipts, "same_context_totals": dict(totals), "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("server", "quality", "config"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({"server": server, "quality": quality, "config": config}[args.mode](args.root), indent=2))
