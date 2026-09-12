"""Build content-addressed GPTQ caches without loading a full serving model.

Each target projection is released before the next one; only the drafter's
small checkpoint is loaded together to exercise its actual TP packing path.
This prepares artifacts, not serving performance evidence.
"""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from engine.base.loader import RankLoader
from engine.kernels.dense.store import PackStore
from engine.profiles.glm53.weights import rank_loader
from engine.profiles.glm53.net import Glm53Net
from engine.profiles.glm53.drafter import Drafter, load


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--rank',type=int,required=True,choices=range(4))
    ap.add_argument('--ranks',required=True)
    ap.add_argument('--drafter-dir',required=True)
    ap.add_argument('--cache',default='/cache')
    args=ap.parse_args()
    torch.cuda.set_device(0)
    store=PackStore(args.cache,args.rank)
    loader=rank_loader(Path(args.ranks)/f'rank{args.rank}of4.safetensors')
    for key,name in Glm53Net.dense_weight_names(loader.keys()).items():
        start=time.monotonic()
        weight=loader.load([key],device='cuda')[key]
        for offset in range(0,weight.shape[1],4096):
            module=name if weight.shape[1]<=4096 else f'{name}.k{offset//4096}'
            pack=store.pack(weight[:,offset:offset+4096].contiguous(),module)
            del pack
        del weight
        torch.cuda.empty_cache()
        store.release_pages()
        print(json.dumps(dict(rank=args.rank,name=key,seconds=round(time.monotonic()-start,3),packs=dict(store.stats))),flush=True)
    facts=load(args.drafter_dir)
    draft=Drafter(facts,SimpleNamespace(comm=SimpleNamespace(rank=args.rank,world_size=4)),0)
    draft.bind(RankLoader(Path(args.drafter_dir)/'model.safetensors').load([s.name for s in draft.specs()],device='cuda'))
    draft.prepare_fast(store)
    store.release_pages()
    print(json.dumps(dict(rank=args.rank,passed=True,packs=dict(store.stats))),flush=True)


if __name__=='__main__':main()
