"""Bounded TP4 graph lifecycle reproduction without model weights.

Capture target, draft, greedy and stochastic graph families, then alternate
their replays with eager collectives as the serving loop does. Each stage is
logged before its synchronization so a hang has a precise last boundary.
"""
import argparse
import json
from types import SimpleNamespace

import torch

from engine.base.comm import Comm
from engine.base.graphs import DecodeGraphs
from engine.modules.vocab import topk
from engine.profiles.glm53.decode_graphs import SamplingGraphs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--collectives", type=int, default=90)
    ap.add_argument("--buckets", type=int, default=9)
    ap.add_argument("--steps", type=int, default=8)
    args = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction((2 << 30)/torch.cuda.get_device_properties(0).total_memory)
    comm = Comm.init(world=4, timeout_s=45)
    families = []

    def stage(name):
        print(json.dumps({"rank": comm.rank, "stage": name}), flush=True)

    try:
        width, vocab, tokens = 4096, 38720, 6
        logits = {}

        def inputs(n, t, capacity):
            key = (n, t)
            if key not in logits:
                logits[key] = torch.empty(n*t, vocab, dtype=torch.bfloat16, device="cuda")
            return torch.zeros(n*t, width, dtype=torch.bfloat16, device="cuda"), logits[key]

        def target_forward(inp):
            x, out = inp
            h = x
            for _ in range(args.collectives):
                h = torch.tanh(comm.all_reduce(h.clone())*.25+.001)
            out.copy_(h[:, :1].expand_as(out))
            return h, None, out

        stage("target capture begin")
        shapes = [(n, tokens, 4096 << c) for c in range(args.buckets) for n in range(1, 5)]
        target = SimpleNamespace(tokens=tokens, net=SimpleNamespace(comm=comm, rank=comm.rank, vp=vocab))
        target.graphs = DecodeGraphs(target_forward, inputs, shapes)
        families.append(target.graphs)
        stage("target captured")

        def draft_forward(inp):
            h = comm.all_reduce(inp.clone())
            candidates = topk(h[:, :1].expand(-1, vocab).contiguous(), comm, comm.rank*vocab, 16)
            return candidates.indices

        draft = DecodeGraphs(draft_forward,
                             lambda n, t: torch.zeros(t, width, device="cuda", dtype=torch.bfloat16), [(1, tokens)])
        families.append(draft)
        stage("draft captured")
        sampler = SamplingGraphs(target, vocab*4, 1.)
        families.extend((sampler.greedy, sampler.stochastic))
        stage("samplers captured")
        for i in range(args.steps):
            n, c = 1+i%4, i%args.buckets
            shape = (n, tokens, 4096 << c)
            stage(f"step {i} eager")
            comm.all_reduce(torch.ones(1, device="cuda")).item()
            stage(f"step {i} draft")
            proposals = draft.run((1, tokens), lambda x: x.fill_(.2))
            proposals.tolist()
            stage(f"step {i} target")
            target.graphs.run(shape, lambda x: x[0].fill_(.1))
            torch.cuda.synchronize()
            stage(f"step {i} sampler")
            selected = sampler.run(shape, [0. if i%2 == 0 else .7]*(n*tokens)).tolist()
            assert len(selected) == n*tokens
            stage(f"step {i} done")
        stage("PASS")
    finally:
        for graph in reversed(families):
            graph.close()
        comm.close()


if __name__ == "__main__":
    main()
