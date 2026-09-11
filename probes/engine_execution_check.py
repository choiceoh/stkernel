"""Bound LocalTP lanes versus direct native calls, including a failed run.

This is execution/collective qualification on real kernels with synthetic
inputs. It does not qualify full-model generation or throughput.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from engine.base.comm import LocalTP
from engine.profiles.glm53 import lanes


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    assert importlib.util.find_spec("vllm") is None
    assert torch.cuda.get_device_capability() == (12, 1)
    torch.manual_seed(73)
    def rand(*shape):
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    direct = lanes.served()
    tp = LocalTP(4, timeout_s=30)
    bound = lanes.served(tp=tp)
    conv_x, conv_w = rand(6, 512), rand(512, 4).float() * .1
    kda_args = (*(rand(1, 6, 16, 128) for _ in range(4)), rand(1, 6, 16),
                rand(16).float() * .2, rand(2048).float() * .1, None, -5.)
    res, fn = rand(6, 4, 4096), rand(24, 16384).float() * .01
    mhc_args = res, fn, rand(3).float(), rand(24).float(), 1e-6, 1e-6, 2., 20, rand(4096), 1e-6
    post_x = rand(6, 4096)
    quant_x = rand(192, 128)
    pool_ids = torch.tensor([[0, 2, -1], [2, 1, 0], [-1, -1, -1]], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([13, 16, 3], device="cuda", dtype=torch.int32)
    block_row = torch.tensor([7, 2], device="cuda", dtype=torch.int32)
    score_q, score_k = rand(6, 16, 128).to(torch.float8_e4m3fn), rand(64, 128).to(torch.float8_e4m3fn)
    score_scale, score_w = rand(64).float().abs(), rand(6, 16).float().abs()
    ends = torch.full((6,), 64, device="cuda", dtype=torch.int32)
    pool_k, pool_score, ape = rand(3, 4, 128), rand(3, 4, 128), rand(4, 128).float()

    def kernels(table):
        conv = table.conv_prefill(conv_x, conv_w, None)
        kda = table.kda_recurrent(*kda_args)
        pre = table.mhc_pre(*mhc_args)
        post = table.mhc_post(post_x, res, *pre[:2])
        quant = table.indexer_quant(quant_x)
        slots = torch.empty((3, 15), device="cuda", dtype=torch.int32)
        count = torch.empty(3, device="cuda", dtype=torch.int32)
        table.pool_slots(pool_ids, lengths, 4, block_row, 16, 512, 32, slots, count)
        score = table.indexer_logits(score_q, score_k, score_scale, score_w, ends)[:, :64]
        pooled = table.kpool_compress(pool_k, pool_score, ape)
        return (*conv, *kda, *pre, post, *quant, slots, count, score, *pooled)

    expected = kernels(direct)  # JIT/warmup on the dispatch owner
    torch.cuda.synchronize()
    print("direct native kernels ready", flush=True)
    def rank_main(comm):
        out = kernels(bound)
        reduced = comm.all_reduce(torch.full((128,), comm.rank + 1., device="cuda"))
        gathered = comm.all_gather(torch.full((1,), comm.rank, device="cuda", dtype=torch.int32))
        comm.barrier()
        return out, reduced, gathered
    checks = []
    for repeat in range(3):
        for rank, (actual, reduced, gathered) in enumerate(tp.run(rank_main)):
            for index, (a, b) in enumerate(zip(actual, expected)):
                assert a.shape == b.shape and a.dtype == b.dtype
                assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)), (repeat, rank, index)
            assert torch.equal(reduced, torch.full_like(reduced, 10))
            assert gathered.tolist() == [0, 1, 2, 3]
        checks.append({"repeat": repeat, "ranks": 4, "native_outputs_exact": True, "tensor_collectives_exact": True})
        print(json.dumps(checks[-1]), flush=True)
        if repeat == 0:
            def fail(comm):
                if comm.rank == 2:
                    raise ValueError("injected executor failure")
                comm.barrier()
            try:
                tp.run(fail)
                raise AssertionError("rank error was swallowed")
            except RuntimeError as error:
                assert isinstance(error.__cause__, ValueError)
                assert str(error.__cause__) == "injected executor failure"
    assert not any(n == "vllm" or n.startswith("vllm.") for n in sys.modules)
    report = {"passed": True, "scope": "native execution ownership and LocalTP collectives; synthetic inputs",
              "torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(),
              "vllm_installed": False, "vllm_loaded": False, "failure_propagated_and_next_run_exact": True,
              "lanes": ["conv_prefill", "kda_recurrent", "mhc_pre", "mhc_post", "indexer_quant",
                        "pool_slots", "indexer_logits", "kpool_compress"], "checks": checks}
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
