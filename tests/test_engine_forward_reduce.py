"""C1 forward reduction integration; native bytes are checked on a reserved GPU."""
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch

from engine.kernels import dense
from engine.profiles.glm53.boot import decode_fastpath_report
from tests import test_engine_decode_fastpaths as fastpath_tests


class ForwardReductionTests(unittest.TestCase):
    def test_new_output_cells_use_owned_packs_and_replay_time_direct_descriptors(self):
        for k in (2048, 3072):
            layer = dense.DenseLinear.__new__(dense.DenseLinear)
            layer.rows, layer.cols = 4096, k
            p = NS(rows=4096, cols=k, data=torch.empty(1, dtype=torch.uint8),
                   scale=torch.empty(1, dtype=torch.int8), rowscale=torch.ones(4096))
            layer.packs, layer.observer, layer.executed = [p], None, 0
            layer.decode_input_rows, layer.bound_input_executed = (8, 16, 24, 32), set()
            ext = Mock()
            descriptor = torch.tensor([1234], dtype=torch.int64)
            with patch.object(dense, 'extension', return_value=ext):
                for workspace in (None, torch.zeros(32)):
                    layer.workspace = workspace
                    x = torch.zeros(8, k+8, dtype=torch.bfloat16)[:, 4:k+4]
                    layer(x)
                    layer._write_slot(x, descriptor)
                    regular, direct = ext.run_gemm_bound_input.call_args_list[-2:]
                    self.assertEqual(regular.args[0].data_ptr(), x.data_ptr())
                    self.assertEqual(regular.args[0].stride(), x.stride())
                    self.assertIs(regular.args[1], p.data)
                    self.assertIs(regular.args[6], workspace)
                    self.assertIs(direct.args[7], descriptor)
                    self.assertIs(direct.args[6], workspace)
            self.assertEqual(layer.bound_input_executed, {8})

    def test_boot_rejects_an_output_owner_missing_c1_execution(self):
        net = fastpath_tests.BoundDecodeTests.model()
        net.decode_fastpath_rows = (8, 16, 24, 32)
        net.decode_pairs_executed = {(l, m) for l in net.layers for m in net.decode_fastpath_rows}
        net.dense = {name: NS(packs=[NS(rows=4096, cols=k)], bound_input_executed={8, 16, 24, 32})
                     for name, k in [('L0.kda.o_proj', 2048), ('L0.mlp.down', 3072)]}
        self.assertEqual(set(decode_fastpath_report(net)['dense']), set(net.dense))
        for name in net.dense:
            net.dense[name].bound_input_executed.remove(8)
            with self.assertRaisesRegex(RuntimeError, 'not executed'):
                decode_fastpath_report(net)
            net.dense[name].bound_input_executed.add(8)


@unittest.skipUnless(torch.cuda.is_available(), 'requires a reserved GB10 GPU')
class ForwardReductionGpuTests(unittest.TestCase):
    def test_same_build_bytes_private_scratch_and_rebound_direct_output(self):
        from probes.engine_forward_reduce import check
        check(lambda *args, **kwargs: None, timing=False)
