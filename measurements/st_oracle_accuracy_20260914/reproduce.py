"""Reproduce exact CPU accounting counterexamples against the pre-change oracle.

Delays and device memory sampling are disabled for BOTH implementations. These
are arithmetic/workload checks, not latency or serving performance benchmarks.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))
from bench import step_sim as current
from bench import oracle_records
from engine.base import instruments
from engine.base.scheduler import Contract

BASE = "2ac7de6f7946356372ddfceb29d9af6b5032fe99"
source = subprocess.check_output(["git", "show", f"{BASE}:bench/step_sim.py"], cwd=ROOT, text=True)
previous = types.ModuleType("oracle_before_accuracy")
previous.__file__ = str(ROOT / "bench/step_sim.py")
sys.modules[previous.__name__] = previous
exec(compile(source, f"git:{BASE}:bench/step_sim.py", "exec"), previous.__dict__)
contract = Contract(16,1024,7,0.,8)
out = dict(scope="exact CPU accounting counterexamples; no timing/serving accuracy claim",
           baseline_commit=BASE, baseline_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
           current_source_sha256=hashlib.sha256((ROOT/"bench/step_sim.py").read_bytes()).hexdigest(),
           gpu_used=False, checks={})

for name, module in (("before", previous), ("after", current)):
    data = {}
    r = dict(requests=[dict(ctx=512)], prefill=[], decode=dict(windows_med=1000.,num_spec=0,acc_raw=0.))
    fitted = module.fit_cost(r)
    data["zero_k"], data["zero_acceptance"] = fitted.k, fitted.acc
    r = dict(requests=[dict(ctx=512,decode_tok_s=100.),dict(ctx=1024,decode_tok_s=50.)],
             prefill=[], decode=dict(windows_med=1000.,tokens_per_step=1.,num_spec=0,acc_raw=0.,
                                      windows_by_ctx={"512":[1000.],"1024":[500.]}))
    fitted = module.fit_cost(r,channel="client")
    data["client_context_ms"] = [1000*fitted.decode_delay(1,c) for c in (512,1024)]
    with patch.object(module,"_delay"), patch.object(instruments,"_dev_free_bytes",return_value=None):
        cost = module.CostModel(k=7,acc=1.,decode_ms=1.,prefill_tok_s={512:512000.})
        one = module.run_once([512],1,contract,cost=cost,can_async=False)
        tail = module.run_once([512,512],[2,10],contract,cost=cost,can_async=False)
        data["one_token_decode_steps"] = one["steps"]["decode"]
        data["tail_decode_tokens"] = tail.get("committed_decode_tokens",tail["tokens_per_wall_step"]*tail["steps"]["decode"])
        cost = module.CostModel(k=0,acc=0.,decode_ms=1.,decode_ms_per_row=3.,decode_ms_by_ctx={1:2.,512:4.})
        wave = module.run_once([512]*4,12,contract,cost=cost,can_async=False)
        service = sum(n*(4.+3.*(int(w)-1))/1000 for w,n in wave["decode_widths"].items())
        data["reported_phase_step_s"] = wave["decode_step_s_phase"]
        data["expected_phase_step_s"] = sum(wave["decode_widths"].values())/service
    out["checks"][name] = data

out["expected"] = dict(zero_k=0,zero_acceptance=0.,client_context_ms=[10.,20.],
                       one_token_decode_steps=0,tail_decode_tokens=10)
real = ROOT/"measurements/c4_scaling_20260913/c4-20260912T232937.json"
artifact = oracle_records.load_records(real)[0]
out["historical_c4"] = dict(path=str(real.relative_to(ROOT)), source_sha256=hashlib.sha256(real.read_bytes()).hexdigest(),
    before_replayable=previous.arrivals_from_record(artifact),
    after_waves=[dict(arm=v["oracle_arm"],requests=len(v["requests"]),
                      groups=oracle_records.workload(v)["groups"],
                      prompt_tokens=oracle_records.workload(v)["prompts"])
                 for v in oracle_records.views(artifact) if v["oracle_arm"].startswith("c4")])
path = Path(__file__).with_name("regressions.json")
path.write_text(json.dumps(out,indent=2)+"\n")
print(path)
