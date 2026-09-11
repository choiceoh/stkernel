"""Numerical context diagnostics with real attention weights on one TP4 rank.

This tests cache plumbing, not full-model language quality or TP collectives.
Activations are synthetic; the final activation is identical across prefixes.
The DSA oracle independently uses every causal latent for short sequences.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as fn

from engine.base.arena import Arena
from engine.base.params import bind, total_bytes
from engine.profiles.glm53 import facts, lanes
from engine.profiles.glm53.caches import Glm53Caches, layout
from engine.profiles.glm53.net import Glm53Net, Step
from engine.profiles.glm53.weights import rank_loader


def error(actual, expected):
    a, b = actual.float(), expected.float()
    return {"relative_l2": ((a-b).norm()/b.norm().clamp_min(1e-12)).item(),
            "relative_max": ((a-b).abs().max()/b.abs().max().clamp_min(1e-12)).item(),
            "rms": a.square().mean().sqrt().item(),
            "cosine": fn.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()}


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--rank-file", type=Path, required=True)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--layers", default="0,3,43,44")
    ap.add_argument("--tokens", type=int, default=33)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction((2 << 30)/torch.cuda.get_device_properties(0).total_memory)
    torch.manual_seed(91333)
    F = facts.load(a.checkpoint)
    assert 8 < a.tokens <= min(128, F.topk)
    layers = [int(x) for x in a.layers.split(",")]

    class IsolatedRank:
        rank, world_size = a.rank, 4
        def all_reduce(self, x): return x

    net = Glm53Net(F, IsolatedRank(), lanes.reference(), layers)
    specs = [s for s in net.specs() if any(
        s.name.startswith(f"L{L}.{part}.") for L in layers for part in ("kda", "mla", "idx"))]
    # A small block exercises page crossings without a long FP32 reference run.
    # We also run the deployed block size, so this override cannot hide a bug.
    arena = Arena(total_bytes(specs) + (128 << 20))
    loader = rank_loader(a.rank_file)
    net.p = bind(specs, loader.load([s.name for s in specs], arena=arena, max_run=32 << 20))
    results = []
    for block in (F.block, 16):
        cf = replace(F, block=block)
        cache = Glm53Caches(arena, cf, layers, 12, 2)
        cache.pool.reserve(1, block)
        cache.pool.reserve(0, block*2)
        cache.pool.release(1)
        cache.pool.reserve(0, max(block, a.tokens-block*2))
        cache.slots.take(0)
        for L in layers:
            x = torch.randn(a.tokens, F.hidden, device="cuda", dtype=torch.bfloat16)*.1
            other = torch.randn_like(x)*.1
            other[-1] = x[-1]
            attention = net._dsa if F.is_dsa(L) else net._kda
            captured = []
            base_lane = lanes.reference()

            def mla(q, latent, slots, valid, scale, kv_scale):
                out = base_lane.mla_sparse(q, latent, slots, valid, scale, kv_scale)
                # Independent dense causal attention: no indexer or selected slots.
                rows = []
                for i, pos in enumerate(current_positions):
                    all_slots = cache.token_slots(L, 0, torch.arange(pos+1, device="cuda")).long()
                    kv = latent[all_slots].float()*kv_scale
                    scores = (q[i].float() @ kv.T)*scale
                    rows.append((scores.softmax(-1) @ kv).to(q.dtype))
                    selected = slots[i, :valid[i]].long().sort().values
                    assert torch.equal(selected, all_slots.sort().values), (L, pos, valid[i].item())
                dense = torch.stack(rows)
                stats = error(out, dense)
                assert stats["relative_l2"] < .01, stats
                captured.append({"first_position": current_positions[0],
                                 "valid_min": valid.min().item(), "valid_max": valid.max().item(),
                                 "dense_causal_error": stats})
                return out

            net.lanes = replace(base_lane, mla_sparse=mla)

            def run(values, lengths):
                nonlocal current_positions
                cache.reset()
                out, pos = [], 0
                for count in lengths:
                    current_positions = list(range(pos, pos+count))
                    step = Step.prefill(torch.zeros(count, device="cuda", dtype=torch.int64), pos, 0, 1)
                    cache.prepare(step)
                    out.append(attention(L, values[pos:pos+count], step, cache))
                    pos += count
                assert pos == len(values)
                return torch.cat(out)

            current_positions = []
            whole = run(x, [len(x)])
            alternate = run(other, [len(x)])
            bare = run(x[-1:], [1])
            sensitivity = error(alternate[-1], whole[-1])
            no_history = error(bare[-1], whole[-1])
            assert sensitivity["relative_l2"] > 1e-3, (L, "prefix ignored", sensitivity)
            assert no_history["relative_l2"] > 1e-3, (L, "history ignored", no_history)
            chunks = []
            for size in (1, 3, 6, 7, 17):
                lengths = [min(size, len(x)-i) for i in range(0, len(x), size)]
                split = run(x, lengths)
                stats = error(split, whole)
                assert stats["relative_l2"] < .03, (L, block, size, stats)
                chunks.append({"chunk": size, "error": stats})
            row = {"layer": L, "kind": F.kinds[L], "block": block,
                   "physical_blocks": list(cache.pool.row(0)),
                   "prefix_change_last_output": sensitivity,
                   "remove_history_last_output": no_history,
                   "chunks": chunks, "dsa_causal_checks": captured}
            results.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "dsa_causal_checks"}), flush=True)
    weights = {}
    with a.rank_file.open("rb") as stream:
        for s in specs:
            lo, hi = loader.header[s.name]["data_offsets"]
            stream.seek(loader.data_base+lo)
            weights[s.name] = hashlib.sha256(stream.read(hi-lo)).hexdigest()
    result = {"passed": True, "device": torch.cuda.get_device_name(), "rank": a.rank,
              "synthetic_activations": True, "collectives": False, "lanes": "reference",
              "tokens": a.tokens, "weights_sha256": weights, "results": results}
    a.output.write_text(json.dumps(result, indent=2)+"\n")
    print(f"PASS: {a.output}", flush=True)


if __name__ == "__main__":
    main()
