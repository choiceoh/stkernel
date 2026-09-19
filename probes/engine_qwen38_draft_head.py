"""Qwen3.8's draft argmax from an inverted-file index over the head's rows (dense/ivf_head) on one GB10: what it costs,
what building it costs, and how often it names the full head's argmax (probe, single-GPU lane).

The rank's real head (rank file `head`, BF16 [62,080, 2,560]) is quantised as the served lane does (FP8Linear) and
indexed at each (clusters, probes). The queries are NOT the MTP head's hidden states -- those need the whole model,
which one GB10 beside production does not hold -- but three stand-ins, from easiest to hardest:

    token      a row's own direction, scaled: the argmax is (almost always) that row -- a confident draft
    mixture    a Dirichlet mix of four rows' directions: a draft torn between a few candidates
    gaussian   isotropic noise: no preference at all, the index's worst case

Agreement on `mixture` bounds what a real drafter would see from above only loosely; the served acceptance is the
verdict (a window), and a capture of real draft queries is the next measurement. Timing: the full head (FP8Linear
decode rows + the vocabulary argmax key) against the index's argmax key, CUDA graphs of 8 calls, interleaved.

    python3 probes/engine_kernel_check.py --lanes qwen38_draft_head --ranks /home/choiceoh/models/st-qwen38-tep4 \\
        --output /cache/qwen38-draft-head.json
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GRID = ((512, 8), (512, 16), (512, 32), (1024, 16), (1024, 32), (1024, 64), (2048, 32), (2048, 64))
TIMED = ((1024, 32), (2048, 64))
QUERIES = 4096
CALLS, ROUNDS = 8, 9


def queries(w, kind: str, n: int, gen):
    import torch
    rows, k = w.shape
    unit = w.float() / w.float().norm(dim=1, keepdim=True).clamp_min(1e-6)
    scale = 30.0                                          # a confident logit gap for unit rows of this width
    if kind == "token":
        pick = torch.randint(0, rows, (n,), generator=gen).to(w.device)
        q = unit[pick] * scale
    elif kind == "mixture":
        pick = torch.randint(0, rows, (n, 4), generator=gen).to(w.device)
        mix = torch.distributions.Dirichlet(torch.ones(4)).sample((n,)).to(w.device)
        q = (unit[pick] * mix[..., None]).sum(1) * scale
    else:
        q = torch.randn(n, k, generator=gen).to(w.device)
    return q.to(torch.bfloat16)


def run(output=None, ranks=None, rank: "int | None" = None) -> dict:
    import torch
    from engine.kernels.common.vocab_candidates import argmax_key
    from engine.kernels.dense import FP8Linear, ivf_head
    from engine.profiles.qwen38 import facts
    from engine.profiles.qwen38.fleet import rank_loader
    ranks = Path(ranks or "/home/choiceoh/models/st-qwen38-tep4")
    torch.manual_seed(0)
    if rank is None:
        rank = max(int(p.name[4]) for p in ranks.glob("rank?of4.safetensors"))
    F = facts.load(ranks)
    head = rank_loader(ranks / f"rank{rank}of{facts.TP}.safetensors", expected_layout=F.weight_layout).load(["head"])["head"]
    lane = FP8Linear(head, decode_rows=True)
    wq, ws = lane.weight
    valid, start = head.shape[0], rank * F.vocab_local
    report = {"device": torch.cuda.get_device_name(), "rank": rank, "rows": valid, "queries": QUERIES, "grid": {},
              "timing": {}}

    def full_key(h):
        return argmax_key(lane(h), start, valid)

    gen = torch.Generator().manual_seed(0)
    sets = {kind: queries(head, kind, QUERIES, gen) for kind in ("token", "mixture", "gaussian")}
    # the head's precision as a drafter sees it: how often the verify step's FP8 head (rows quantised too) and the
    # draft's W8A16 head (rows BF16, fp8_rows.project_bf16) name the argmax the checkpoint's BF16 head names
    from engine.kernels.dense import fp8_rows
    report["head_precision"] = {}
    for kind, q in sets.items():
        agree = {"fp8": 0, "w8a16": 0}
        for a in range(0, QUERIES, 16):
            h = q[a:a + 16].contiguous()
            want = (h.float() @ head.float().t()).argmax(1)
            agree["fp8"] += int((lane(h).float().argmax(1) == want).sum())
            agree["w8a16"] += int((fp8_rows.project_bf16(h, lane.weight)[:, :valid].float().argmax(1) == want).sum())
        report["head_precision"][kind] = {name: round(v / QUERIES, 4) for name, v in agree.items()}
    print(json.dumps({"head_precision": report["head_precision"]}), flush=True)
    exact = {kind: torch.cat([full_key(q[a:a + 16]) for a in range(0, QUERIES, 16)]) for kind, q in sets.items()}
    built = {}
    for clusters, probes in GRID:
        if clusters not in built:
            torch.cuda.synchronize()
            began = time.perf_counter()
            built[clusters] = ivf_head.build(lane.weight, clusters=clusters, probes=probes)
            torch.cuda.synchronize()
            report.setdefault("build_s", {})[clusters] = round(time.perf_counter() - began, 2)
        index = ivf_head.IVFHead(lane.weight, built[clusters].centroids, built[clusters].members, probes)
        row = {"cap": index.cap, "read_MB": round(index.read_bytes() / 1e6, 2)}
        for kind, q in sets.items():
            got = torch.cat([ivf_head.argmax_key(index, q[a:a + 16], start, valid) for a in range(0, QUERIES, 16)])
            ids = 0xffffffff - (got & 0xffffffff)
            want = 0xffffffff - (exact[kind] & 0xffffffff)
            row[kind] = round(float((ids == want).float().mean()), 4)
        report["grid"][f"{clusters}/{probes}"] = row
        print(json.dumps({"index": f"{clusters}/{probes}", **row}), flush=True)
    for m in (1, 4):
        h = sets["mixture"][:m].contiguous()
        arms = {"full head": full_key}
        for clusters, probes in TIMED:
            index = ivf_head.IVFHead(lane.weight, built[clusters].centroids, built[clusters].members, probes)
            arms[f"ivf {clusters}/{probes}"] = lambda x, index=index: ivf_head.argmax_key(index, x, start, valid)
        keep, graphs = [], {}
        for name, fn in arms.items():
            keep.append(fn(h))
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(CALLS):
                    keep.append(fn(h))
            graphs[name] = g
        times = {name: [] for name in graphs}
        for _ in range(ROUNDS):
            for name, g in graphs.items():
                g.replay()
                torch.cuda.synchronize()
                began = time.perf_counter()
                g.replay()
                torch.cuda.synchronize()
                times[name].append((time.perf_counter() - began) / CALLS * 1e6)
        report["timing"][m] = {name: round(statistics.median(v), 1) for name, v in times.items()}
        print(json.dumps({"rows": m, "us": report["timing"][m]}), flush=True)
        del graphs, keep
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(report, indent=1) + "\n")
    return report


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
