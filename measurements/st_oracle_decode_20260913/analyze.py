"""Use the ST Oracle and preserved observations to budget the next decode work. No GPU."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
from step_kernels import EngineBytes, decode_step, fold_routing_from_timeline

FILES = {
    "partial_c1": "measurements/st_decode_agreement_20260913/consumer-v4.json",
    "stages": "measurements/st_draft_agreement_20260913/prior-device-stages.json",
    "kda": "measurements/st_kda_batch_20260913/gpu-v4.json",
    "mhc": "measurements/st_terminal_decode_20260913/gpu-v2-recovered.json",
    "routing": "measurements/c4_scaling_20260913/decode-timeline-rank3.json",
}
source = {key: json.loads((ROOT/path).read_text()) for key, path in FILES.items()}
rank = source["partial_c1"]["ranks"][0]
base_ms = rank["device_seconds"]*1000/rank["iterations"]
kda = next(x for x in source["kda"]["kda"]["cases"] if x["rows"] == 1
           and x["accepted"] == 3 and x["context"] == 4096 and x["cache"] == "warm")
mhc = next(x for x in source["mhc"]["timing"]["cases"] if x["local_rows"] == 7
           and x["features"] == 5 and x["regime"] == "warm")
saved_ms = ((kda["ordinary_us"]-kda["deferred_us"])
            + (mhc["baseline_us"]-mhc["fused_us"]))/1000
candidate_ms = base_ms-saved_ms
target_ms = 1000/22
scenarios = []
for name, ms in (("observed_partial_prepare_c1", base_ms),
                 ("known_component_savings_transfer_fully", candidate_ms),
                 ("impossible_upper_bound_remove_all_remaining_kda", candidate_ms-kda["deferred_us"]/1000),
                 ("target", target_ms)):
    scenarios.append(dict(name=name, step_ms=ms, steps_per_second=1000/ms,
                          extra_ms_to_remove=max(0, ms-target_ms)))
b = EngineBytes.for_model("glm53")
fold = fold_routing_from_timeline(ROOT/FILES["routing"])
b.routing_gamma, b.routing_scale = fold["routing_gamma"], fold["routing_scale"]
budgets = []
for ctx in (32000, 128000):
    c1, c4 = decode_step(b, ctx, 1), decode_step(b, ctx, 4)
    budgets.append(dict(ctx=ctx, c1_ms=c1.total(), c4_ms=c4.total(),
                        c1_components=c1.ms, c4_components=c4.ms,
                        aggregate_gain_if_acceptance_equal=4*c1.total()/c4.total(),
                        c4_ms_to_remove_for_2_4x=max(0,c4.total()-4*c1.total()/2.4)))
report = dict(scope="CPU ST Oracle sensitivity calculation; no new engine measurement",
              gpu_used=False, source={k:dict(path=v, sha256=hashlib.sha256((ROOT/v).read_bytes()).hexdigest())
                                      for k,v in FILES.items()},
              baseline_status=source["partial_c1"]["status"], baseline_source=source["partial_c1"]["source"],
              context=32000, observed_acceptance=rank["accepted"]/rank["drafted"],
              component_saved_ms=saved_ms, scenarios=scenarios, structural_budgets=budgets,
              constraints=["The timing seed is incomplete prepare-c1, not a qualified consumer baseline.",
                           "Full transfer of component savings is assumed; input shapes and sources differ.",
                           "No latency saving is assigned to the new tiled commit before its GPU result.",
                           "Removing all remaining KDA is an unattainable upper bound, not a candidate prediction.",
                           "C4/context comparisons use historical #838 coefficients and equal acceptance, not current C4 evidence.",
                           "No predicted acceptance change, precision change, or speculative-k change is applied."])
Path(__file__).with_name("forecast.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
print(json.dumps(dict(component_saved_ms=saved_ms, scenarios=scenarios),indent=2))
