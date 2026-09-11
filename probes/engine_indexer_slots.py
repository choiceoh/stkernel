"""Qualify slot finalization against an exact baseline net.py from git.

Use the standalone ST runtime. --baseline-net must be extracted from the
commit being compared; its hash is recorded. Only _indexer is used from it.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from engine.base.comm import LocalTP
from engine.profiles.glm53 import lanes
from engine_decode_overhead import paired
from engine_indexer_lanes import operators, real_indexer


def load_baseline(path):
    spec = importlib.util.spec_from_file_location("st_indexer_baseline", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Glm53Net._indexer


def local_tp(fused):
    tokens = torch.tensor([[31, -1, 0, 16, 16]], device="cuda", dtype=torch.int32)
    table = torch.tensor([7, 2], device="cuda", dtype=torch.int32)
    expected, valid = torch.empty_like(tokens), torch.empty(1, device="cuda", dtype=torch.int32)
    fused.indexer_slots(tokens, table, 16, 512, 32, expected, valid)
    outputs = [(torch.full_like(tokens, -99), torch.full_like(valid, -99)) for _ in range(4)]
    tp = LocalTP(4)
    lanes.bind_tp(tp)
    try:
        tp.run(lambda comm: fused.indexer_slots(tokens, table, 16, 512, 32, *outputs[comm.rank]))
    finally:
        lanes.bind_tp(None)
    assert all(torch.equal(out, expected) and torch.equal(count, valid) for out, count in outputs)
    return {"ranks": 4, "outputs_and_counts_exact": True}


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-net", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--rank-file", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    assert importlib.util.find_spec("vllm") is None
    baseline = load_baseline(args.baseline_net)
    ref, fused = lanes.reference(), lanes.served()
    report = {"scope": "slot finalization and real-weight L3 indexer; no full-model ITL claim",
              "torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(),
              "baseline_net_sha256": hashlib.sha256(args.baseline_net.read_bytes()).hexdigest(),
              "vllm_installed": False, "local_tp": local_tp(fused),
              "protocol": {"rounds": 5, "samples_per_round": 100, "warmup": 40, "order": "alternating AB/BA"}}
    report["real_indexer"] = real_indexer(args.checkpoint, args.rank_file, ref, fused, baseline)
    print("real indexer:", json.dumps(report["real_indexer"]), flush=True)
    report["measurements"] = []
    profiles = []
    g = torch.Generator(device="cuda").manual_seed(441)
    table = torch.randperm(4096, device="cuda", generator=g).to(torch.int32)
    for rows in (1, 6, 24, 256, 512):
        tokens = torch.randint(-1, 4096, (rows, 2051), device="cuda", dtype=torch.int32, generator=g)
        tokens[:, 1500:] = -1
        results = [(torch.empty_like(tokens), torch.empty(rows, device="cuda", dtype=torch.int32)) for _ in range(2)]
        fns = [lambda fn=fn, tokens=tokens, out=out, counts=counts:
               fn(tokens, table, 64, 2112, 512, out, counts)
               for fn, (out, counts) in zip((ref.indexer_slots, fused.indexer_slots), results)]
        for fn in fns:
            fn()
        assert all(torch.equal(a, b) for a, b in zip(*results))
        item = {"tokens": rows, "width": 2051, **paired(fns, rounds=5, samples=100, warmup=40)}
        report["measurements"].append(item)
        profiles.append((item, fns))
        print("measurement:", json.dumps(item), flush=True)
    # CUPTI initialization follows every timed case.
    for item, fns in profiles:
        item["operators"] = {label: operators(fn) for label, fn in zip(("baseline", "optimized"), fns)}
        print("kernel counts:", item["tokens"],
              {k:v["cuda_kernel_count"] for k,v in item["operators"].items()}, flush=True)
    assert not any(n == "vllm" or n.startswith("vllm.") for n in sys.modules)
    report["vllm_loaded"] = False
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
