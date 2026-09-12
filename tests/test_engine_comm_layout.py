"""The fleet gather's output layout and exact small MAX dispatch."""
import unittest
import tempfile
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from engine.base.comm import Comm


def _gather_worker(rank, rendezvous):
    import torch.distributed as dist
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        comm = Comm(2, rank, dist.group.WORLD)
        parts = [(torch.arange(24).reshape(3, 8)+100*r)[:, ::2] for r in range(2)]
        for dim in (0, 1, -1, -2):
            got = comm.all_gather(parts[rank], dim)
            torch.testing.assert_close(got, torch.cat(parts, dim), rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


class GatherTests(unittest.TestCase):
    def test_real_two_process_gloo_with_strided_inputs(self):
        import torch.multiprocessing as mp
        with tempfile.TemporaryDirectory() as temp:
            mp.spawn(_gather_worker, args=((Path(temp)/'rendezvous').as_uri(),), nprocs=2, join=True)

    def test_rank_major_collective_matches_cat_on_every_axis(self):
        for shape in ((5,), (3, 7), (2, 3, 5)):
            parts = [torch.arange(torch.tensor(shape).prod()).reshape(shape)+1000*r for r in range(4)]
            def gather(out, value, group):
                self.assertTrue(value.is_contiguous())
                out.copy_(torch.stack(parts).flatten(0, 1))
            for dim in range(-len(shape), len(shape)):
                with self.subTest(shape=shape, dim=dim), patch('torch.distributed.all_gather_into_tensor', gather), \
                     patch('torch.distributed.all_gather', side_effect=AssertionError('list gather used')):
                    got = Comm(4, 0, None).all_gather(parts[0], dim)
                    self.assertTrue(torch.equal(got, torch.cat(parts, dim)))
                    self.assertTrue(got.is_contiguous())

    def test_small_max_uses_the_exact_transport_and_other_types_keep_nccl(self):
        selected = []
        def maximum(value):
            selected.append('oneshot')
            value.fill_(19)
            return value
        transport = SimpleNamespace(eligible_max=lambda t: t.dtype == torch.int64, reduce_max=maximum)
        comm = Comm(4, 0, None, transport=transport)
        key = torch.zeros(6, dtype=torch.int64)
        with patch('torch.distributed.all_reduce', side_effect=lambda *a, **kw: selected.append('nccl')):
            self.assertIs(comm.all_reduce_max(key), key)
            comm.all_reduce_max(torch.zeros(1))
        self.assertEqual(selected, ['oneshot', 'nccl'])
        self.assertEqual(key.tolist(), [19]*6)


if __name__ == '__main__':
    unittest.main()
