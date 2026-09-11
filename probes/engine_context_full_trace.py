"""Stream real GLM weights one layer at a time for a TP4 context trace.

Runs the target algebra directly, without runner, graphs, drafter or tiers.
Memory is bounded by one layer plus caches; this is not a latency benchmark.
"""
import argparse
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from engine.base.arena import Arena
from engine.base.comm import Comm, LocalTP
from engine.profiles.glm53 import facts, lanes
from engine.profiles.glm53.caches import Glm53Caches, layout
from engine.profiles.glm53.net import Glm53Net, Step, rmsnorm
from engine.profiles.glm53.weights import rank_loader


def stats(x):
    x = x.float()
    return {"rms": x.square().mean().sqrt().item(), "max": x.abs().max().item(),
            "finite": torch.isfinite(x).all().item()}


def relation(x):
    a, b, c = x.float()
    return {"prefix_relative_l2": ((a-b).norm()/a.norm().clamp_min(1e-12)).item(),
            "bare_relative_l2": ((a-c).norm()/a.norm().clamp_min(1e-12)).item(),
            "prefix_cosine": torch.nn.functional.cosine_similarity(a, b, dim=0).item()}


@torch.inference_mode()
def main(local_comm=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ranks", type=Path, required=True)
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--kda-o-norm-eps", type=float)
    ap.add_argument("--local", action="store_true", help="four TP ranks on one GPU; no NCCL")
    a = ap.parse_args()
    from engine.profiles.glm53 import net as net_module
    if a.kda_o_norm_eps is not None:
        net_module.O_NORM_EPS = a.kda_o_norm_eps
    if a.local and local_comm is None:
        torch.cuda.set_per_process_memory_fraction((12 << 30)/torch.cuda.get_device_properties(0).total_memory)
        return LocalTP(4).run(lambda comm: main(comm))
    if not a.local:
        torch.cuda.set_per_process_memory_fraction((6 << 30)/torch.cuda.get_device_properties(0).total_memory)
    comm = local_comm or Comm.init(world=4, timeout_s=600)
    try:
        F = facts.load(a.metadata)
        tok = Tokenizer.from_file(str(a.metadata/"tokenizer.json"))
        texts = ["The capital of France is", "The capital of Korea is"]
        ids = [tok.encode(t, add_special_tokens=False).ids for t in texts]
        assert ids[0][-1] == ids[1][-1]
        ids.append([ids[0][-1]])
        table = lanes.reference()
        net = Glm53Net(F, comm, table)
        loader = rank_loader(a.ranks/f"rank{comm.rank}of4.safetensors")
        arena = Arena(layout(F, net.layers).nbytes(6, 3)+(1 << 20))
        caches = Glm53Caches(arena, F, net.layers, 6, 3)
        caches.reset()
        for i, seq in enumerate(ids):
            caches.pool.reserve(i, len(seq))
            assert caches.slots.take(i) == i+1
        chunks = [(torch.tensor(seq, device="cuda", dtype=torch.int64), 0, i, i+1)
                  for i, seq in enumerate(ids)]
        step = Step.decode(chunks)
        caches.prepare(step)
        last = torch.tensor([sum(map(len, ids[:i+1]))-1 for i in range(len(ids))], device="cuda")
        net.p = loader.load(["embed"], device="cuda", max_run=32 << 20)
        x = net.embed(step.ids)
        res = x[:, None, :].expand(-1, F.hc, -1).contiguous()
        records = [{"stage": "embed", "last": stats(x[last]), "relation": relation(x[last])}]
        post = comb = None
        for L in range(F.layers):
            if post is not None:
                res = table.mhc_post(x, res, post, comb)
            # Release the previous layer before allocating the next one.
            net.p = None
            names = [s.name for s in net.specs() if s.name.startswith(f"L{L}.")]
            net.p = loader.load(names, device="cuda", max_run=32 << 20)
            post, comb, x = net._hc_pre(L, res, "attn")
            attn_in = stats(x[last])
            x = net._dsa(L, x, step, caches) if F.is_dsa(L) else net._kda(L, x, step, caches)
            attn_out = stats(x[last])
            attn_relation = relation(x[last])
            res = table.mhc_post(x, res, post, comb)
            post, comb, x = net._hc_pre(L, res, "ffn")
            ffn_in = stats(x[last])
            x = net._moe(L, x) if F.is_moe(L) else net._dense(L, x)
            combined = table.mhc_post(x[last], res[last], post[last], comb[last]).float().mean(1)
            row = {"layer": L, "kind": F.kinds[L], "attn_input": attn_in,
                   "attn_output": attn_out, "attn_relation": attn_relation,
                   "ffn_input": ffn_in, "ffn_output": stats(x[last]),
                   "combined": stats(combined), "relation": relation(combined)}
            assert row["combined"]["finite"]
            records.append(row)
            print(json.dumps({"rank": comm.rank, **row}), flush=True)
        res = table.mhc_post(x, res, post, comb)
        net.p = None
        net.p = loader.load(["norm", "head"], device="cuda", max_run=32 << 20)
        h = rmsnorm(res.float().mean(1).to(x.dtype), net.p["norm"], F.rms_eps)
        logits = net.head(h[last]).float()
        topv, topi = logits.topk(10)
        endings = [[{"id": i, "text": tok.decode([i]), "logit": v}
                    for i, v in zip(ii, vv)] for ii, vv in zip(topi.tolist(), topv.tolist())]
        result = {"rank": comm.rank, "device": torch.cuda.get_device_name(),
                  "reference_lanes": True, "runner": False, "graphs": False,
                  "drafter": False, "tier": False, "layer_streaming": True,
                  "kda_o_norm_eps": net_module.O_NORM_EPS,
                  "collective": "LocalTP" if a.local else "NCCL",
                  "prompts": texts+[tok.decode(ids[2])], "ids": ids,
                  "records": records, "final_hidden": relation(h[last]), "top10": endings}
        output = a.output.with_name(f"{a.output.stem}-rank{comm.rank}.json") if a.local else a.output
        output.write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps({"rank": comm.rank, "final_hidden": result["final_hidden"], "top10": endings}), flush=True)
    finally:
        if not a.local:
            comm.close()


if __name__ == "__main__":
    main()
