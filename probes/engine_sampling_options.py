"""Bounded CUDA qualification and component timings for device request options.

No model, fleet lease or serving-rate claim. Use the fleet's normal probe wrapper
to repeat on GB10; the default memory cap also fits a shared desktop GPU.
"""
import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import triton

from engine.base.sampler import process_logits, top_logprobs, top_logprobs_batch
from engine.base.sampling_options import SamplingState, warm_sampling_options


def timing(fn, iterations):
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    wall = time.perf_counter()
    start.record()
    for _ in range(iterations):
        fn()
    stop.record()
    stop.synchronize()
    return {"stream_us": start.elapsed_time(stop) * 1000 / iterations,
            "wall_us": (time.perf_counter() - wall) * 1e6 / iterations}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab", type=int, default=154880)
    parser.add_argument("--spec", type=int, choices=range(1, 8), default=7)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--memory-fraction", type=float, default=.12)
    args = parser.parse_args()
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    torch.manual_seed(17)
    vocab, t = args.vocab, args.spec + 1
    result = {"evidence_scope": "component", "device": torch.cuda.get_device_name(),
              "torch": torch.__version__, "triton": triton.__version__, "cuda": torch.version.cuda,
              "engine_shape": {"profile": "glm53", "vocab": vocab, "spec_k": args.spec, "tp": 1,
                               "model_loaded": False, "dtype": "bfloat16"},
              "iterations": args.iterations, "shapes": [], "source_sha256": {}}
    for path in ("engine/base/sampling_options.py", "engine/kernels/common/sampling_options.py",
                 "engine/profiles/glm53/pipeline.py", "engine/profiles/glm53/adapter.py",
                 "engine/base/draws.py", "engine/base/sampler.py"):
        result["source_sha256"][path] = hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
    for n in (1, 2):
        options = [{"repetition_penalty": 1.2, "presence_penalty": .7, "frequency_penalty": .2,
                    "logit_bias": {7: 1., 13: -.5}, "logprobs": 5} for _ in range(n)]
        tokens = [list(range(2048)) + [7, 7, 13] for _ in range(n)]
        state = SamplingState.from_histories(options, tokens, [2048]*n, [5]*n, vocab, "cuda")
        raw = torch.randn(n*t, vocab, device="cuda", dtype=torch.bfloat16)
        drafts_host = [[7, 13, 7, 29, 31, 43, 53][:args.spec] for _ in range(n)]
        drafts = torch.tensor(drafts_host, dtype=torch.int64, device="cuda")
        generated = torch.full((n,), 3, dtype=torch.int64, device="cuda")
        ends = torch.full((n, 1), vocab-1, dtype=torch.int64, device="cuda")
        before, after = torch.empty(n*t, vocab, device="cuda"), torch.empty(n*t, vocab, device="cuda")
        warm_sampling_options(raw[:, :vocab//4].contiguous(), vocab, t, 0, vocab-2)

        def reference():
            for row in range(n):
                for pos in range(t):
                    process_logits(raw[row*t+pos], options[row], state.seen[row], state.counts[row],
                                   drafts_host[row][:pos], vocab-2, ends[row] if 3+pos < 5 else None,
                                   out=before[row*t+pos])

        def fused():
            state.process(raw, drafts, generated, ends, decodable=vocab-2, out=after)

        reference()
        fused()
        torch.testing.assert_close(after, before, rtol=2e-6, atol=2e-6)
        finite = torch.isfinite(before)
        error = float((after[finite] - before[finite]).abs().max())
        self_picks = before.argmax(-1).tolist()
        if self_picks != after.argmax(-1).tolist():
            raise AssertionError("option transform changed a greedy pick")
        old_lp = lambda: [(i, *top_logprobs(row, i, 5)) for row, i in zip(before, self_picks)]
        new_lp = lambda: top_logprobs_batch(before, self_picks, 5)
        if old_lp() != new_lp():
            raise AssertionError("batched logprobs changed scores or top-token order")
        for _ in range(3):
            reference()
            fused()
            old_lp()
            new_lp()
        samples = {"reference": [], "fused": [], "logprobs_before": [], "logprobs_after": []}
        # A/B/B/A limits drift without pretending a shared GPU is an isolated fleet.
        for order in (("reference", "fused"), ("fused", "reference"),
                      ("fused", "reference"), ("reference", "fused")):
            for name in order:
                samples[name].append(timing(reference if name == "reference" else fused, args.iterations))
        for old, new in ((old_lp, new_lp), (old_lp, new_lp)):
            samples["logprobs_before"].append(timing(old, args.iterations))
            samples["logprobs_after"].append(timing(new, args.iterations))
        result["shapes"].append({"requests": n, "positions": n*t, "max_abs_error": error,
                                 "argmax_equal": True, "samples": samples,
                                 "median_wall_us": {k: statistics.median(s["wall_us"] for s in v)
                                                    for k, v in samples.items()}})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
