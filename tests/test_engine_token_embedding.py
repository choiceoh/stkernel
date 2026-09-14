"""Rank boundaries, BF16 payload bits, TP assembly and graph replay for token embedding."""
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, PropertyMock, patch

import torch

from engine.modules.token_embedding import lookup


def reference(ids, weight, start):
    local = ids - start
    invalid = (local < 0) | (local >= weight.shape[0])
    return torch.nn.functional.embedding(local.masked_fill(invalid, 0), weight).masked_fill(invalid[:, None], 0)


class EmbeddingTests(unittest.TestCase):
    def test_rank_edges_and_strides_preserve_every_bf16_payload_bit(self):
        # All 65,536 BF16 bit patterns, including NaN payloads and signed zero.
        bits = torch.arange(65536).to(torch.int16).view(16, 4096)
        padded = torch.zeros(16, 4104, dtype=torch.int16)
        padded[:, 4:4100] = bits
        weight = padded[:, 4:4100].view(torch.bfloat16)
        for start in (0, 38720, 3*38720):
            indices = torch.tensor([start-1, *range(start, start+16), start+16, -(1 << 63), (1 << 63)-1])
            ids = torch.zeros(indices.numel()*2, dtype=torch.int64)
            ids[::2] = indices
            actual = lookup(ids[::2], weight, start)
            self.assertTrue(torch.equal(actual.view(torch.int16), reference(indices, weight, start).view(torch.int16)))
            self.assertTrue(torch.equal(actual[1:17].view(torch.int16), bits))
            self.assertTrue((actual[[0, 17, 18, 19]].view(torch.int16) == 0).all())

    def test_network_preserves_one_collective_and_four_rank_ownership(self):
        from engine.profiles.glm53.net import Glm53Net
        full = (torch.arange(68*16).reshape(68, 16).float() / 128).bfloat16()
        ids = torch.tensor([-1, 0, 16, 17, 33, 34, 50, 51, 67, 68])
        parts = []
        for rank in range(4):
            net = Glm53Net.__new__(Glm53Net)
            net.rank, net.vp = rank, 17
            net.p = {'embed': full[rank*17:(rank+1)*17]}
            net.comm = NS(all_reduce=Mock(side_effect=lambda x: x))
            parts.append(net.embed(ids))
            net.comm.all_reduce.assert_called_once()
            self.assertIs(net.comm.all_reduce.call_args.args[0], parts[-1])
        self.assertTrue(torch.equal(torch.stack(parts).float().sum(0).bfloat16(), reference(ids, full, 0)))

    def test_cuda_dispatch_uses_original_addresses_and_has_no_empty_launch(self):
        from engine.kernels import token_embedding
        launch, kernel = Mock(), Mock()
        kernel.__getitem__ = Mock(return_value=launch)
        ids = torch.arange(16)[::2]
        weight = torch.zeros(17, 136, dtype=torch.bfloat16)[:, 4:132]
        with patch.object(torch.Tensor, 'is_cuda', new_callable=PropertyMock, return_value=True), \
                patch.object(token_embedding, '_lookup', kernel):
            out = lookup(ids, weight, 38720)
            launch.assert_called_once()
            args = launch.call_args.args
            self.assertIs(args[0], ids)
            self.assertIs(args[1], weight)
            self.assertIs(args[2], out)
            self.assertEqual(args[3:8], (2, 136, 17, 128, 38720))
            self.assertTrue(out.is_contiguous())
            launch.reset_mock()
            self.assertEqual(lookup(ids[:0], weight, 0).shape, (0, 128))
            launch.assert_not_called()

    def test_bad_metadata_is_refused_before_a_kernel_launch(self):
        ids = torch.zeros(1, dtype=torch.int64)
        weight = torch.zeros(17, 128, dtype=torch.bfloat16)
        for i, w, start in ((ids.float(), weight, 0), (ids[:, None], weight, 0),
                            (ids, weight[:0], 0), (ids, weight, -1), (ids, weight, 1 << 63),
                            (ids, torch.empty(17, 128, device='meta'), 0)):
            with self.assertRaises(ValueError): lookup(i, w, start)
        with patch.object(torch.Tensor, 'is_cuda', new_callable=PropertyMock, return_value=True):
            for w in (weight.float(), weight.T, weight[:1].expand(17, -1)):
                with self.assertRaisesRegex(ValueError, 'packed hidden columns'):
                    lookup(ids, w, 0)


@unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA')
class EmbeddingCudaTests(unittest.TestCase):
    def test_payload_bits_ragged_widths_and_changed_id_graph_replay(self):
        for rows, hidden in ((1, 127), (8, 128), (32, 257), (65, 4096)):
            raw = torch.arange(17*(hidden+8), device='cuda').to(torch.int16).view(17, hidden+8)
            weight = raw[:, 4:4+hidden].view(torch.bfloat16)
            held_ids = torch.arange(rows*2, device='cuda', dtype=torch.int64)
            ids = held_ids[::2]
            start = 3*38720
            lookup(ids, weight, start)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph): out = lookup(ids, weight, start)
            try:
                for phase in (0, 7, 19):
                    ids.copy_(start + (torch.arange(rows, device='cuda') + phase) % 21 - 2)
                    raw.copy_((torch.arange(raw.numel(), device='cuda') * 257 + phase).to(torch.int16).view_as(raw))
                    before = raw.clone()
                    graph.replay()
                    expected = reference(ids, weight, start)
                    self.assertTrue(torch.equal(out.view(torch.int16), expected.view(torch.int16)))
                    self.assertTrue(torch.equal(raw, before))
            finally:
                graph.reset()


if __name__ == '__main__':
    unittest.main()
