"""PR760 Runner simulation with this experiment's actual scheduler contracts.

No GPU timings are invented for overlap, early projection or L2 reuse. Device
cost zero measures host overhead; the saved-record fit is a separate exercise.
"""
import ast
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench.step_sim import CostModel, run_once, _drop_ring
from engine.base.scheduler import Contract, chunk_for
from engine.profiles.glm53.execution import ExecutionPlan


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
    tile = chunk_for(align, budget, draft)
    plans = {name: ExecutionPlan(prefill_tiles=n, tile_rows=tile)
             for name, n in (("baseline", 1), ("layer-prefill2", 2), ("layer-prefill4", 4))}
    # Acceptance is a workload assumption, held fixed; NullModel generates no
    # language and cannot qualify acceptance, EOS or output quality.
    cost = CostModel(name="host-only; assumed acceptance 0.5", k=draft, acc=.5,
                     decode_ms=0, prefill_tok_s={})
    records = []
    for context in (2000, 32000, 128000):
        for concurrency in (1, rows):
            for repeat in range(3):
                order = list(plans) if repeat % 2 == 0 else list(reversed(plans))
                for name in order:
                    plan = plans[name]
                    contract = Contract(align, tile*plan.prefill_tiles+draft if plan.prefill_tiles > 1 else budget,
                                        draft, 0., rows, decode_token_budget=align+draft)
                    result = run_once([context]*concurrency, 512, contract, cost=cost, can_async=False)
                    records.append(dict(arm=name, context=context, concurrency=concurrency, repeat=repeat,
                                        contract=asdict(contract), result=_drop_ring(result)))
    summary = []
    for context in (2000, 32000, 128000):
        for concurrency in (1, rows):
            for name in plans:
                group = [r["result"] for r in records if (r["context"], r["concurrency"], r["arm"])
                         == (context, concurrency, name)]
                summary.append(dict(arm=name, context=context, concurrency=concurrency,
                    prefill_steps=group[0]["steps"]["prefill"],
                    decode_host_median_ms=statistics.median(r["by_kind"]["decode"]["med_ms"] for r in group),
                    prefill_host_median_ms=statistics.median(r["by_kind"]["prefill"]["med_ms"] for r in group),
                    observed_decode_widths=group[0]["decode_widths"]))
    report = dict(scope="CPU Runner/queue simulation only; no candidate GPU speed or quality prediction",
        unmodeled={"tp-overlap": "C=4 subgroup kernel time, reduction time and stream contention",
                   "early-observe": "projection/tail critical path and shared memory bandwidth",
                   "layer-prefill": "weight-cache reuse, GPU activation peak and prefix publication latency"},
        workload="C=1/C=4, 2K/32K/128K, 512 generated positions, seed 7, 3 alternating-order repeats",
        source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
                       ("bench/step_sim.py", "engine/base/scheduler.py", "engine/profiles/glm53/execution.py")},
        summary=summary, records=records)
    output = Path(__file__).with_name("simulation.json")
    output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
