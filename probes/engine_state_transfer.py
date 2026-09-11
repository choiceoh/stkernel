"""Paired recurrent-transfer graphs using the real GLM arena strides.

The 34-layer case measures cache transfers only, without model weights or
network traffic. Read+write bytes are algorithmic traffic, not DRAM counters.
"""
import argparse
import gc
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from engine.base.arena import Arena
from engine.profiles.glm53 import facts
from engine.profiles.glm53.caches import Glm53Caches, layout
from engine.kernels.state_cache import gather_ring, commit_ring
from engine_graph_profile import IsolatedRank, capture
from engine_state_cache_bench import paired


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ckpt-meta', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    F = facts.load(args.ckpt_meta)
    report = dict(scope=__doc__, results=[])
    for layers in ([0], list(range(F.layers))):
        p = layout(F, layers)
        for n in (1, 2, 4):
            arena = Arena(p.nbytes(2, n))
            caches = Glm53Caches(arena, F, layers, 2, n)
            fields = [v for (name, _), v in caches._fields.items() if name == 'rec']
            slots = torch.arange(n, 0, -1, device='cuda')
            contexts = 32768 + torch.arange(n, device='cuda')
            old, old_out = capture(lambda: tuple(v.index_select(0, slots) for v in fields))
            new, new_out = capture(lambda: tuple(gather_ring(v, slots, contexts) for v in fields))
            for a,b in zip(old_out, new_out):
                for i in range(n):
                    row = (32768+i-1) % 6
                    assert torch.equal(a[i,row], b[i,row])
            gather = paired([old.replay, new.replay], lambda: None, IsolatedRank(), 5, 30)
            old.reset(); new.reset()
            del old, new, new_out
            commits = {}
            for tokens in (1, 6):
                def before():
                    for src,dst in zip(old_out, fields):
                        dst.index_copy_(0, slots, src)
                def after():
                    for src,dst in zip(old_out, fields):
                        commit_ring(src, dst, slots, contexts, tokens)
                old, _ = capture(before)
                new, _ = capture(after)
                commits[tokens] = paired([old.replay, new.replay], lambda: None, IsolatedRank(), 5, 30)
                old.reset(); new.reset()
                del old, new
            row_bytes = fields[0].stride(1) * fields[0].element_size()
            result = dict(kda_layers=len(fields), active_sequences=n, gather=gather, commit=commits,
                          baseline_read_write_bytes=2 * len(fields) * n * 6 * row_bytes * 2,
                          optimized_read_write_bytes={t: 2 * len(fields) * n * (1+t) * row_bytes
                                                      for t in (1,6)})
            report['results'].append(result)
            args.output.write_text(json.dumps(report, indent=2)+'\n')
            print('transfer', len(fields), n, {k:v['median_us'] for k,v in gather.items()}, flush=True)
            del old_out, fields, caches, arena, slots, contexts, a, b
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
