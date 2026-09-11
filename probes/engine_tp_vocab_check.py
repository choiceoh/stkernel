"""Small TP4/NCCL graph test for exact vocabulary-parallel greedy selection.

No model weights or serving ports. Launch once per rank with an isolated
MASTER_PORT and the fleet's normal RoCE environment. Allocator capped at 512 MiB.
"""
import json
import torch

from engine.base.comm import Comm
from engine.modules.vocab import argmax


def main():
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction((512 * 2**20) / torch.cuda.mem_get_info()[1])
    comm = Comm.init(world=4, timeout_s=45)
    cases = 0
    try:
        width = 38720
        for rows in (1, 6, 24):
            local = torch.empty(rows, width, device="cuda", dtype=torch.bfloat16)
            generator = torch.Generator(device="cuda").manual_seed(31 + comm.rank)
            for decodable in (4*width - 1000, width - 1):
                local.normal_(generator=generator)
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(3):
                        argmax(local, comm, comm.rank*width, decodable)
                torch.cuda.current_stream().wait_stream(side)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    picked = argmax(local, comm, comm.rank*width, decodable)
                try:
                    for mode in ("random", "ties", "special"):
                        local.normal_(generator=generator)
                        if mode == "ties":
                            local[:, 7] = 100
                            local[:, 9] = 100
                        elif mode == "special":
                            local[0].fill_(float("-inf"))
                            if rows > 1:
                                local[1].fill_(-0. if comm.rank % 2 else 0.)
                                local[2, 11] = float("nan")
                                local[3, 13] = float("inf")
                        expected = comm.all_gather(local, dim=-1)[:, :decodable].argmax(-1)
                        eager = argmax(local, comm, comm.rank*width, decodable)
                        graph.replay()
                        torch.cuda.synchronize()
                        if not torch.equal(picked, expected) or not torch.equal(eager, expected):
                            raise AssertionError((comm.rank, rows, decodable, mode))
                        cases += 1
                finally:
                    graph.reset()
        print(json.dumps(dict(passed=True, rank=comm.rank, cases=cases,
                              max_packet_bytes=24*8,
                              peak_reserved_bytes=torch.cuda.max_memory_reserved())), flush=True)
    finally:
        comm.close()


if __name__ == "__main__":
    main()
