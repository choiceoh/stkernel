"""Small GPU reproduction gates for the incomplete 20260913 native consumer.

No weights or engine boot: exercise changing MAX tails through the actual
transport kernel, then isolate the production-sized candidate merge. The CPU
proxy stands in for three NIC peers; this is not a real TP4/model verdict.
"""
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace as NS
import unittest

import torch


class DecodeAgreementTests(unittest.TestCase):
    def test_max_tails_with_changing_peers_inside_four_iteration_graphs(self):
        from engine.kernels.bounded_graph import BoundedGraph
        from tests.test_engine_oneshot_gather_cuda import packets, proxy
        for rank in range(4):
            with proxy(self, rank) as (ext, requests):
                for keys in (1, 7, 28, 63):
                    count = torch.zeros(1, dtype=torch.int64, device='cuda')
                    stop = torch.zeros_like(count)
                    inputs = torch.empty(4, keys, dtype=torch.int64, device='cuda')
                    history = torch.empty_like(inputs)
                    graph = torch.cuda.CUDAGraph(keep_graph=True)
                    with torch.cuda.graph(graph):
                        value = inputs.index_select(0, count).reshape(keys)
                        ext.oneshot_max_int64(value)
                        history.index_copy_(0, count, value.unsqueeze(0))
                    loop = BoundedGraph(graph, count, stop, 4, owners=(inputs, history))
                    try:
                        for trial in range(32):
                            # A different winning peer every iteration, with decreasing
                            # scores across ring reuse so stale high words cannot pass.
                            values = torch.arange(4*4*keys, dtype=torch.int64).reshape(4, 4, keys)
                            values += (32-trial) * (1 << 40)
                            for step in range(4):
                                values[step, (trial+step) % 4] += 1 << 32
                            inputs.copy_(values[:, rank])
                            for step in range(4):
                                requests.put(packets(values[step], rank))
                            loop.replay()
                            with self.subTest(rank=rank, keys=keys, trial=trial):
                                torch.testing.assert_close(history.cpu(), values.max(1).values,
                                                           rtol=0, atol=0)
                    finally:
                        loop.close()
                        graph.reset()

    def test_production_vocab_merge_is_identical_across_replays_and_rank_allocations(self):
        from engine.kernels.common.vocab_candidates import pack, select
        from engine.modules.vocab import topk
        width, k = 38720, 16
        torch.manual_seed(91360)
        for rows in (6, 24):
            # BF16 collisions at the actual 154880-column merge geometry.
            full = torch.randn(rows, width*4, device='cuda').bfloat16()
            full[:, 32:64] = 5
            gathered = torch.cat([select(pack(full[:, r*width:(r+1)*width], r*width, width), k)
                                  for r in range(4)], dim=-1)
            comm = NS(world_size=4, all_gather=lambda value, dim=-1: gathered)
            reference = topk(full[:, :width], comm, 0, k)
            expected = (reference.values.clone(), reference.indices.clone())
            for rank in range(4):
                local = full[:, rank*width:(rank+1)*width].clone()
                mismatch = torch.zeros(2, dtype=torch.bool, device='cuda')
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    values, ids = topk(local, comm, rank*width, k)
                    mismatch[0] |= (values != expected[0]).any()
                    mismatch[1] |= (ids != expected[1]).any()
                try:
                    for _ in range(1024):
                        graph.replay()
                    with self.subTest(rows=rows, rank=rank):
                        self.assertEqual(mismatch.tolist(), [False, False])
                finally:
                    graph.reset()


def main():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 1):
        raise RuntimeError('requires an admitted GB10')
    started = time.monotonic()
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(DecodeAgreementTests))
    root = Path(__file__).resolve().parents[1]
    files = ('engine/kernels/oneshot/dsv4_oneshot_ar.cu', 'engine/modules/vocab.py',
             'engine/kernels/common/vocab_candidates.py', 'probes/engine_decode_agreement_check.py')
    report = dict(status='PASS' if result.wasSuccessful() else 'FAIL', tests=result.testsRun,
                  failures=len(result.failures), errors=len(result.errors), seconds=time.monotonic()-started,
                  scope='single GB10; CPU proxy peers and production-sized vocabulary merge; no model/NIC proof',
                  torch=torch.__version__, cuda=torch.version.cuda,
                  source_sha256={p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files})
    Path('/cache/decode-agreement.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    main()
