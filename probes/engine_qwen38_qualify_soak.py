"""How often does the boot's lane qualify fail on a GB10, and whose failure is it? (Qwen3.8, a lane finding)

On 2026-09-19 the single-GPU lane's boot qualify died once -- `norm_rope_4x128 (0.938, 0.0171)`, about one head's
rotated half wrong at the indexer's 300-row case -- and passed three minutes later with the numbers every other run
has given (measurements/qwen38_lane_20260919). One failure in a handful of runs is not a rate, and the error then could
not say which side was wrong. The same qualify runs at every fleet boot (D3: a drift kills the boot), and the same
launch serves every QSA layer's heads.

This runs engine/profiles/qwen38/lanes.qualify again and again on the card -- the same seed, so the same inputs the
failure had -- and keeps two things:

    failures          each RuntimeError's text, which since gated_residual.blame names the elements past the band,
                      whether the kernel and the torch reference each repeat themselves, and which side leaves the
                      CPU's reference (the first few in full, all of them counted)
    distinct results  the holds that passed return their worst (max, rms) a lane: the same inputs through
                      deterministic launches give ONE such result, so a second one is a launch that did not repeat
                      itself within the band -- a finer detector than the band

    bash bench/fleet.sh run --gpu qwen38-qualify-soak 10 'Qwen3.8 qualify soak: the failure rate of the boot qualify' -- \\
      bash probes/run_engine_probe.sh probes/engine_kernel_check.py --lanes qwen38_qualify_soak:2000

Correctness only: nothing here is timed for a claim. A failure makes the run fail, with the record already printed.

What one process cannot sample is a fault that belongs to a process's FIRST launches (the failure of 09-19 was a fresh
process's first qualify): here that is repeat 0, once. Every qwen38_cells ticket is one more sample of it.
"""
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probes.engine_qwen38_cells import CONFIG_SHA256, facts, held  # noqa: E402

REPEATS = 2000
KEPT = 5                                                   # failures kept in full; the rest are counted


def soak(qualify, repeats: int, kept: int = KEPT) -> dict:
    """`qualify()` called `repeats` times: how many raised, the first `kept` errors' text with the repeat they came at,
    and each distinct result of the calls that held with how many calls gave it (most frequent first)."""
    failures, texts, results = 0, [], {}
    for i in range(repeats):
        try:
            worst = qualify()
        except RuntimeError as exc:
            failures += 1
            if len(texts) < kept:
                texts.append({'repeat': i, 'error': str(exc)})
            continue
        key = json.dumps(held(worst), sort_keys=True)
        results[key] = results.get(key, 0) + 1
    ranked = sorted(results.items(), key=lambda item: -item[1])
    return dict(repeats=repeats, failures=failures, errors=texts, distinct_results=len(ranked),
                results=[dict(calls=calls, worst=json.loads(key)) for key, calls in ranked[:kept]])


def run(output=None, repeats: int = REPEATS):
    import torch
    assert torch.cuda.get_device_capability() == (12, 1), 'requires GB10'
    from engine.profiles.qwen38 import lanes
    F, device = facts(), torch.device('cuda')
    began = time.perf_counter()                            # repeat 0 is the process's first: the compiles are in it
    record = soak(lambda: lanes.qualify(device, F), repeats)
    record.update(lane='qwen38_qualify_soak', config_sha256=CONFIG_SHA256, seconds=round(time.perf_counter() - began, 1),
                  device=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda)
    print(json.dumps(record), flush=True)
    if output is not None:
        Path(output).write_text(json.dumps(record, indent=1) + '\n')
    if record['failures'] or record['distinct_results'] != 1:
        raise RuntimeError(f"the boot's lane qualify is not steady on this card: {record['failures']} of {repeats} calls "
                           f"failed, {record['distinct_results']} distinct results among those that held")
