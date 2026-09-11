"""One rank of a TP=4 DSv4.1, stood up on this fleet.

Everything below the model is ours -- the loader, the kernels, the budget, the
engram SSD path -- and the module tree is still the pinned reference
(CHARTER D13/D14). Launch one of these per node:

    RANK=0 WORLD_SIZE=4 LOCAL_RANK=0 MASTER_ADDR=10.10.10.2 MASTER_PORT=29555 \\
        python3 engine/dist_run.py --tokens 512

The phase table is the point. `budget.py` predicts what a rank should hold and
says out loud which of its lines are still guesses; this prints what the rank
ACTUALLY holds at each phase, so the two can be put side by side. The one line
budget.py cannot measure without a boot -- module construction, carried as an
8.77 GiB upper bound borrowed from GLM -- is exactly what `build-model` here
reports.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

GIB = 1 << 30
HERE = Path(__file__).resolve().parent
RANKS = Path("/home/choiceoh/models/DeepSeek-V4.1-Flash-tp4")
CKPT = Path("/home/choiceoh/models/DeepSeek-V4.1-Flash")


def free_gib():
    import torch
    if not torch.cuda.is_initialized():
        return None
    return torch.cuda.mem_get_info()[0] / GIB


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--seq", type=int, default=8192)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--layers", type=int, default=0,
                        help="truncate the backbone for a smaller trial (0 = all)")
    parser.add_argument("--decode-steps", type=int, default=4)
    parser.add_argument("--world", type=int, default=None,
                        help="TP width to build for; defaults to WORLD_SIZE. Set it "
                             "with WORLD_SIZE=1 to rehearse a rank's shapes and "
                             "memory on one node (collectives are stubbed, so the "
                             "NUMBERS are not the model's -- the shapes are).")
    parser.add_argument("--rank", type=int, default=None)
    args = parser.parse_args()

    launched = int(os.getenv("WORLD_SIZE", "1"))
    rank = args.rank if args.rank is not None else int(os.getenv("RANK", "0"))
    world = args.world or launched
    local = int(os.getenv("LOCAL_RANK", "0"))

    import torch
    import torch.distributed as dist

    torch.cuda.set_device(local)
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    torch.set_num_threads(8)

    sys.path.insert(0, str(HERE))
    import engram_ssd
    import instruments
    import kernels
    import model as bridge

    rec = instruments.Recorder(f"rank{rank}")
    phases = []

    def stamp(name, t0, f0):
        phases.append((name, time.perf_counter() - t0, (f0 or 0) - (free_gib() or 0),
                       free_gib()))

    t0, f0 = time.perf_counter(), None
    if launched > 1:
        dist.init_process_group("nccl", world_size=launched, rank=rank)
        dist.barrier()
    f0 = free_gib()
    stamp("dist-init", t0, None)

    kernels.install()
    ref = bridge.reference(world, rank)
    engram_ssd.prepare(ref)
    margs = bridge.model_args(ref, args.batch, args.seq)
    if args.layers:
        # Truncating the backbone means every list that names a layer id has to
        # be truncated with it; leaving one behind builds a module that indexes
        # past the end, and the failure lands far from here.
        keep = args.layers
        margs.n_layers, margs.n_mtp_layers = keep, 0
        for field in ("kv_source_layers", "index_source_layers", "engram_layer_ids",
                      "dspark_target_layer_ids"):
            if hasattr(margs, field):
                setattr(margs, field, tuple(i for i in getattr(margs, field) if i < keep))
        margs.compress_ratios = tuple(margs.compress_ratios[:keep])
        if margs.candidate_source_layer >= keep:
            margs.candidate_source_layer = -1
        if hasattr(margs, "engram_num_embeddings"):
            margs.engram_num_embeddings = tuple(
                margs.engram_num_embeddings[:len(margs.engram_layer_ids)])

    # The reference says it plainly (model.py:1185): "The tokenizer only feeds
    # the engram token map." Without it NgramHashState falls back, which moves
    # WHICH engram rows a token hashes to -- not how many, not their shape, and
    # not one byte of memory. A run on random ids does not care; a run that
    # compares text to the reference does, so this is loud rather than silent.
    tok = None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(CKPT), trust_remote_code=True)
    except Exception as exc:
        if rank == 0:
            print(f"  no tokenizer ({exc.__class__.__name__}): engram hashing is "
                  "not the reference's. Shapes and memory are unaffected.")

    t0, f0 = time.perf_counter(), free_gib()
    with ref.set_dtype(torch.bfloat16), torch.device("cuda"):
        net = ref.Transformer(margs, tok)
    stamp("build-model", t0, f0)

    t0, f0 = time.perf_counter(), free_gib()
    swapped = engram_ssd.attach(net, rank, world)
    stamp("engram-ssd", t0, f0)

    t0, f0 = time.perf_counter(), free_gib()
    report = bridge.load_full(net, RANKS / f"rank{rank}of{world}.safetensors",
                              world=world, rank=rank, recorder=rec)
    stamp("load-weights", t0, f0)

    ids = torch.randint(0, margs.vocab_size, (args.batch, args.tokens),
                        dtype=torch.long, device="cuda")
    t0, f0 = time.perf_counter(), free_gib()
    with torch.inference_mode(), torch.device("cuda"), ref.set_dtype(torch.bfloat16):
        _out_ids, logits, _hidden = net(ids, 0)
    torch.cuda.synchronize()
    stamp("prefill", t0, f0)
    prefill_s = phases[-1][1]

    decode_s = 0.0
    if args.decode_steps:
        t0, f0 = time.perf_counter(), free_gib()
        with torch.inference_mode(), torch.device("cuda"), ref.set_dtype(torch.bfloat16):
            nxt = _out_ids.view(args.batch, -1)[:, -1:]
            for step in range(args.decode_steps):
                nxt, _lg, _h = net(nxt.view(args.batch, 1), args.tokens + step)
                nxt = nxt.view(args.batch, -1)[:, -1:]
        torch.cuda.synchronize()
        stamp("decode", t0, f0)
        decode_s = phases[-1][1]

    if rank == 0:
        print(f"\n  rank {rank}/{world}: {report['loaded']:,} loaded, "
              f"{len(report['dequantized'])} dequantized, "
              f"{len(report['sharded'])} tp-sharded here, "
              f"MISSING {len(report['missing'])}")
        print(f"  engram on SSD: {swapped}")
        print(f"\n  {'phase':<16}{'seconds':>10}{'GiB used':>11}{'GiB free after':>16}")
        for name, secs, used, free in phases:
            print(f"  {name:<16}{secs:>10.2f}{used:>11.2f}{free:>16.2f}")
        print(f"\n  prefill {args.tokens} tok in {prefill_s:.2f} s "
              f"({args.tokens / prefill_s:,.0f} tok/s)")
        if decode_s:
            print(f"  decode  {args.decode_steps} steps in {decode_s:.2f} s "
                  f"({args.decode_steps / decode_s:.2f} step/s)")
        print(f"  logits {tuple(logits.shape)} finite="
              f"{torch.isfinite(logits).all().item()}")
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
