"""The o_proj producer pack at 8 and 16 rows: its layout size, which writers read it, and who allocates it."""
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch

from engine.kernels import dense


def writer(cols, rows=(8, 16, 24, 32)):
    layer = dense.DenseLinear.__new__(dense.DenseLinear)
    layer.rows, layer.cols = 4096, cols
    layer.packs = [NS(rows=4096, cols=cols, data=torch.empty(1, dtype=torch.uint8),
                      scale=torch.empty(1, dtype=torch.int8), rowscale=torch.ones(4096))]
    layer.observer, layer.executed, layer.workspace = None, 0, None
    layer.decode_input_rows, layer.bound_input_executed, layer.producer_pack_executed = rows, set(), set()
    return layer


class ProducerPackRowsTests(unittest.TestCase):
    def test_layout_sizes_are_the_cells_own_packs(self):
        self.assertEqual(dense.producer_pack_nbytes(8, 2048), 16 * 1024 + 16 * 8 * 4)
        self.assertEqual(dense.producer_pack_nbytes(16, 2048), 16 * 32 * 128 + 16 * 32 * 4)
        for rows in (1, 7, 14, 24, 32):
            with self.assertRaises(ValueError):
                dense.producer_pack_nbytes(rows, 2048)
        with self.assertRaises(ValueError):
            dense.producer_pack_nbytes(16, 2000)

    def test_only_bound_kda_output_writers_read_sixteen_row_packs(self):
        self.assertEqual([r for r in (8, 16, 24, 32) if writer(2048).producer_pack_rows(r)], [8, 16])
        self.assertEqual([r for r in (8, 16, 24, 32) if writer(4096).producer_pack_rows(r)], [8])
        self.assertEqual([r for r in (8, 16, 24, 32) if writer(3072).producer_pack_rows(r)], [8])
        self.assertEqual([r for r in (8, 16) if writer(2048, rows=()).producer_pack_rows(r)], [])

    def test_the_writer_hands_the_pack_to_its_bound_cell_and_refuses_it_elsewhere(self):
        ext = Mock()
        layer = writer(2048)
        address = torch.tensor([1234], dtype=torch.int64)
        with patch.object(dense, 'extension', return_value=ext):
            for rows in (8, 16):
                pack = torch.empty(dense.producer_pack_nbytes(rows, 2048), dtype=torch.uint8)
                x = torch.zeros(rows, 2056, dtype=torch.bfloat16)[:, 4:2052]
                layer._write_slot(x, address, pack=pack)
                call = ext.run_gemm_bound_input.call_args
                self.assertIs(call.kwargs['producer_pack'], pack)
                self.assertIs(call.args[7], address)
            x = torch.zeros(24, 2048, dtype=torch.bfloat16)
            with self.assertRaisesRegex(ValueError, 'producer pack'):
                layer._write_slot(x, address, pack=torch.empty(1, dtype=torch.uint8))
        self.assertEqual(layer.producer_pack_executed, {8, 16})

    def test_net_allocates_the_layout_of_the_step_rows_only_when_the_writer_reads_it(self):
        from engine.profiles.glm53.net import Glm53Net
        net = Glm53Net.__new__(Glm53Net)
        norm = Mock()
        norm.producer_pack = True
        net.producer_packs, net.lanes = True, NS(kda_output_norm=norm)
        asked = []
        project = Mock()
        project.pack_rows = lambda name, rows: asked.append(rows) or rows in (8, 16)
        for rows in (8, 16):
            pack = net._o_proj_pack('L0.kda.o_proj', torch.empty(rows, 16, 128, dtype=torch.bfloat16), project)
            self.assertEqual((pack.dtype, pack.numel()), (torch.uint8, dense.producer_pack_nbytes(rows, 2048)))
        self.assertIsNone(net._o_proj_pack('L0.kda.o_proj', torch.empty(24, 16, 128, dtype=torch.bfloat16), project))
        self.assertEqual(asked, [8, 16])
        project.pack_rows = lambda name, rows: False
        self.assertIsNone(net._o_proj_pack('L0.kda.o_proj', torch.empty(16, 16, 128, dtype=torch.bfloat16), project))
        net.producer_packs = False
        project.pack_rows = lambda name, rows: True
        self.assertIsNone(net._o_proj_pack('L0.kda.o_proj', torch.empty(16, 16, 128, dtype=torch.bfloat16), project))


if __name__ == '__main__':
    unittest.main()
