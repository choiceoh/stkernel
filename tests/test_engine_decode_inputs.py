"""Draft input and keyed-draw identity at CPU/native dispatch boundaries."""
import unittest
from unittest.mock import Mock, PropertyMock, patch

import torch

from engine.base import draws
from engine.modules.draft_inputs import build


class InputTests(unittest.TestCase):
    def test_anchor_masks_and_positions_match_for_strided_rows_and_overflow(self):
        for n in (1, 2, 3, 4):
            anchors = torch.tensor([-1, 9, 0, 8, (1 << 63)-1, 7, -(1 << 63), 6])[::2][:n]
            positions = torch.tensor([0, 9, 128*1024, 8, (1 << 63)-3, 7, -10, 6])[::2][:n]
            for k in (0, 1, 5, 7, 31):
                ids, pos = build(anchors, positions, k, 154879)
                self.assertEqual(tuple(ids.shape), (n * (k+1),))
                self.assertTrue(torch.equal(ids.view(n, k+1)[:, 0], anchors))
                self.assertTrue((ids.view(n, k+1)[:, 1:] == 154879).all())
                expected = positions[:, None] + torch.arange(k+1)
                self.assertTrue(torch.equal(pos, expected.flatten()))
        a, p = build(torch.tensor([9]), 128*1024, 7, 12)
        b, q = build(torch.tensor([9]), torch.tensor(128*1024), 7, 12)
        self.assertTrue(torch.equal(a, b) and torch.equal(p, q))

    def test_invalid_inputs_are_refused(self):
        for anchors, positions, k, mask in ((torch.zeros(2), torch.zeros(2), 7, 2),
                                          (torch.zeros(2, dtype=torch.int64), 3, 7, 2),
                                          (torch.zeros(1, dtype=torch.int64), torch.zeros(2, dtype=torch.int64), 7, 2),
                                          (torch.zeros(1, dtype=torch.int64), 3, -1, 2),
                                          (torch.zeros(1, dtype=torch.int64), 3, 7, 1 << 63)):
            with self.assertRaises(ValueError): build(anchors, positions, k, mask)

    def test_cuda_routes_to_one_launch_without_materializing_inputs(self):
        from engine.kernels import decode_inputs
        launch = Mock()
        kernel = Mock()
        kernel.__getitem__ = Mock(return_value=launch)
        anchors, positions = torch.arange(8)[::2], torch.arange(12)[::3]
        with patch.object(torch.Tensor, 'is_cuda', new_callable=PropertyMock, return_value=True), \
                patch.object(decode_inputs, '_draft_inputs', kernel):
            ids, pos = build(anchors, positions, 7, 154879)
        launch.assert_called_once()
        args = launch.call_args.args
        self.assertIs(args[0], anchors)
        self.assertEqual(args[1].data_ptr(), positions.data_ptr())
        self.assertEqual(args[4:8], (2, 3, 8, 154879))
        self.assertEqual(pos.data_ptr() - ids.data_ptr(), ids.numel() * ids.element_size())

    def test_cuda_draw_dispatch_preserves_strides_seed_and_keyed_layout(self):
        from engine.kernels import decode_inputs
        launch = Mock()
        kernel = Mock()
        kernel.__getitem__ = Mock(return_value=launch)
        nonces, generations = torch.arange(8)[::2], torch.arange(12)[::3]
        with patch.object(torch.Tensor, 'is_cuda', new_callable=PropertyMock, return_value=True), \
                patch.object(decode_inputs, '_step_block', kernel):
            result = draws.step_block(-(1 << 80) + 19, nonces, generations, 7)
        launch.assert_called_once()
        args = launch.call_args.args
        self.assertIs(args[0], nonces)
        self.assertIs(args[1], generations)
        self.assertEqual(args[3:7], (2, 3, draws.mix(-(1 << 80) + 19), 7))
        self.assertEqual(result.shape, (4, 15))
        self.assertEqual(result.dtype, torch.float32)


@unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA')
class InputCudaTests(unittest.TestCase):
    def test_input_and_draw_graph_replay_matches_host_bits(self):
        for n in (1, 2, 3, 4):
            for seed in (0, -1, (1 << 80) + 37):
                anchors = torch.arange(2*n, device='cuda')[::2]
                positions = torch.arange(3*n, device='cuda')[::3]
                nonces = torch.arange(2*n, device='cuda')[::2]
                generations = torch.arange(3*n, device='cuda')[::3]
                def call():
                    return *build(anchors, positions, 7, 154879), draws.step_block(seed, nonces, generations, 7)
                call()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph): ids, pos, block = call()
                try:
                    for start in (0, (1 << 31), (1 << 63)-10):
                        anchors.copy_(torch.arange(n, device='cuda') + 13)
                        positions.copy_(torch.arange(n, device='cuda') + start)
                        nonces.copy_(torch.arange(n, device='cuda') - start)
                        generations.copy_(torch.arange(n, device='cuda') + start)
                        graph.replay()
                        a, p = build(anchors.cpu(), positions.cpu(), 7, 154879)
                        expected = draws.step_block(seed, nonces.cpu(), generations.cpu(), 7)
                        self.assertTrue(torch.equal(ids.cpu(), a) and torch.equal(pos.cpu(), p))
                        self.assertTrue(torch.equal(block.cpu(), expected))
                finally:
                    graph.reset()


if __name__ == '__main__':
    unittest.main()
