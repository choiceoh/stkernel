"""PR760 host-only Runner timing and an optimistic amortization ceiling.

This does not execute BoundedGraph or predict its CUDA latency. The zero-device
model intentionally runs synchronously; it cannot measure existing async hiding.
"""
import ast
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench.step_sim import CostModel, run_once, _drop_ring
from engine.base.scheduler import Contract


def literal(path, name):
    for node in ast.parse((ROOT / path).read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"missing constant: {path}:{name}")


def main():
    budget = literal("engine/profiles/glm53/boot.py", "TOKEN_BUDGET")
    rows = literal("engine/profiles/glm53/boot.py", "MAX_SEQS")
    align = literal("engine/profiles/glm53/facts.py", "CHUNK_ALIGN")
    draft = literal("engine/profiles/glm53/facts.py", "SPEC_K")
    contract = Contract(align, budget, draft, 0., rows,
                        decode_token_budget=align+draft, prefill_tail_multiple=4)
    cost = CostModel(name="host-only; assumed acceptance 0.5", k=draft, acc=.5,
                     seed=7, decode_ms=0, prefill_tok_s={})
    records = []
    for repeat in range(3):
        cases = [(context, concurrency) for context in (2000, 32000, 128000) for concurrency in (1, rows)]
        if repeat % 2:
            cases.reverse()
        for context, concurrency in cases:
            result = run_once([context]*concurrency, 512, contract, cost=cost, can_async=False)
            records.append(dict(context=context, concurrency=concurrency, repeat=repeat,
                                result=_drop_ring(result)))
    summary = []
    for context in (2000, 32000, 128000):
        for concurrency in (1, rows):
            group = [r["result"] for r in records if (r["context"], r["concurrency"]) == (context, concurrency)]
            host_us = statistics.median(r["by_kind"]["decode"]["med_ms"] * 1000 for r in group)
            summary.append(dict(context=context, concurrency=concurrency,
                measured_host_step_median_us=host_us,
                optimistic_saved_us_per_step={str(limit): host_us*(1-1/limit) for limit in (2, 4)},
                observed_decode_widths=group[0]["decode_widths"]))
    files = ("bench/step_sim.py", "engine/base/scheduler.py", "engine/profiles/glm53/boot.py",
             "engine/profiles/glm53/facts.py", "measurements/st_bounded_loop_20260913/simulate.py")
    report = dict(scope="CPU Runner timing only; no bounded GPU execution, async overlap or quality proof",
        ceiling_assumption="All measured host work hypothetically amortizes over L steps; real result consumption still costs work",
        unmodeled=["Torch/kernel submission", "TP4 stop collective and conditional graph control",
                   "actual GPU critical path", "already hidden async host work", "serving burst reservation/readback"],
        runtime=dict(python=sys.version, platform=platform.platform()), contract=asdict(contract),
        workload="C=1/C=4, 2K/32K/128K, 512 generated positions, fixed seed7 acceptance assumption, 3 repeats",
        source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files},
        summary=summary, records=records)
    Path(__file__).with_name("simulation.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
