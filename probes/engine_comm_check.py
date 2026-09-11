"""ST communication primitives on actual GB10s, without model weights.

Launch one process per node with RANK, WORLD_SIZE and an isolated MASTER_PORT.
Use WORLD_SIZE=4 for the full fleet; a smaller fabric diagnostic does not
qualify the model's TP=4 contract.
The check uses less than 16 MiB of tensors per rank. Every rank reports its
own result; a failed collective has a bounded process-group timeout.
"""
from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from engine.base.comm import Comm


def main():
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    if world < 2 or not 0 <= rank < world:
        raise ValueError("the communication check requires at least two ranks")
    torch.cuda.set_device(0)
    if torch.cuda.get_device_capability() != (12, 1):
        raise ValueError("the engine fleet check requires GB10 SM121")
    comm = Comm.init(rank=rank, world=world, timeout_s=60)
    try:
        torch.manual_seed(52)
        x = torch.randn(8, 256, device="cuda")
        w = torch.randn(256, 256, device="cuda")
        start, end = rank * 256 // world, (rank + 1) * 256 // world
        result = comm.all_reduce(x[:, start:end] @ w[start:end])
        expected = x @ w
        torch.testing.assert_close(result, expected, rtol=2e-4, atol=5e-5)
        gathered = comm.all_gather(torch.full((1,), rank, device="cuda", dtype=torch.int32))
        assert gathered.tolist() == list(range(world))
        for _ in range(10):
            reduced = comm.all_reduce(torch.full((1024,), rank + 1., device="cuda"))
            assert bool((reduced == world * (world + 1) // 2).all())
        comm.barrier()
        print(json.dumps({"passed": True, "rank": rank, "world_size": world, "host": socket.gethostname(),
                          "device": torch.cuda.get_device_name(), "torch": torch.__version__,
                          "tp_linear_max_abs": (result - expected).abs().max().item(),
                          "all_gather": gathered.tolist(), "all_reduce_repeats": 10}), flush=True)
    finally:
        comm.close()


if __name__ == "__main__":
    main()
