"""The fleet gather's output layout and exact small MAX dispatch."""
import unittest
import tempfile
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from engine.base.comm import Comm, LocalTP


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
        for dtype in (torch.int64, torch.float32):
            value = parts[rank].to(dtype).clone()[:, ::2]
            pointer = value.data_ptr()
            got = comm.broadcast_tensor(value)
            self_expected = parts[0].to(dtype)[:, ::2]
            assert got.data_ptr() == pointer
            torch.testing.assert_close(got, self_expected, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


class GatherTests(unittest.TestCase):
    def test_integer_packets_use_native_gather_after_layout_normalization(self):
        for shape in ((7,), (6, 16), (2, 3, 16)):
            parts = [(torch.arange(2 * torch.tensor(shape).prod()).reshape(*shape[:-1], shape[-1]*2)
                      + 10000*r)[..., 1::2] for r in range(4)]
            def gather(value):
                self.assertTrue(value.is_contiguous())
                self.assertEqual(value.data_ptr() % 16, 0)
                torch.testing.assert_close(value, parts[2], rtol=0, atol=0)
                return torch.stack(parts)
            transport = SimpleNamespace(eligible_gather=lambda t: t.dtype == torch.int64, gather=gather)
            comm = Comm(4, 2, None, transport=transport)
            with patch('torch.distributed.all_gather_into_tensor', side_effect=AssertionError('NCCL used')):
                for dim in range(-len(shape), len(shape)):
                    torch.testing.assert_close(comm.all_gather(parts[2], dim), torch.cat(parts, dim), rtol=0, atol=0)

    def test_real_two_process_gloo_with_strided_inputs(self):
        import torch.multiprocessing as mp
        with tempfile.TemporaryDirectory() as temp:
            mp.spawn(_gather_worker, args=((Path(temp)/'rendezvous').as_uri(),), nprocs=2, join=True)

    def test_native_root_broadcast_preserves_int64_bits_and_inplace_strided_views(self):
        for size in (1, 6, 24, 64):
            for layout in ('aligned', 'offset', 'strided'):
                def rank(local):
                    backing = torch.arange(size * 2 + 1, dtype=torch.int64) + local.rank * 1000
                    value = backing[:size] if layout == 'aligned' else (
                        backing[1:size+1] if layout == 'offset' else backing[1:2*size+1:2])
                    expected = torch.arange(size, dtype=torch.int64) - 10
                    expected[0] = torch.iinfo(torch.int64).min
                    if size > 1:
                        expected[-1] = torch.iinfo(torch.int64).max
                    if local.rank == 0:
                        value.copy_(expected)
                    pointer = value.data_ptr()
                    transport = SimpleNamespace(
                        eligible_max=lambda t: t.dtype == torch.int64 and t.numel() <= 64
                                               and t.is_contiguous() and t.data_ptr() % 16 == 0,
                        reduce_max=local.all_reduce_max)
                    got = Comm(4, local.rank, transport=transport).broadcast_tensor(value)
                    self.assertIs(got, value)
                    self.assertEqual(got.data_ptr(), pointer)
                    torch.testing.assert_close(got, expected, rtol=0, atol=0)
                with self.subTest(size=size, layout=layout), patch('torch.distributed.broadcast',
                        side_effect=AssertionError('native broadcast entered NCCL')):
                    LocalTP(4, timeout_s=20).run(rank)

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
