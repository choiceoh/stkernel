"""Require actual target consumption and ownership of the fused mHC pack."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch

from engine.kernels import dense
from tests.test_engine_producer_pack_rows import writer


class InputPackTests(unittest.TestCase):
    def test_serving_proof_refuses_each_missing_consumer(self):
        path = Path(__file__).resolve().parents[1] / 'engine/profiles/glm53/boot.py'
        fn = next(n for n in ast.parse(path.read_text()).body
                  if isinstance(n, ast.FunctionDef) and n.name == 'fixed_k_cost_report')
        fn.body = [n for n in fn.body if not isinstance(n, ast.ImportFrom)]
        scope = {}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), scope)
        report = scope['fixed_k_cost_report']
        layer = lambda: NS(producer_pack_executed={8}, input_pack_rows=lambda rows: rows == 8)
        net = NS(decode_fastpath_rows=(8, 16),
                 dense={'L0.kda.in_proj': layer(), 'L2.kda.in_proj': layer()},
                 producer_packs=True, mhc_input_packs=True)
        self.assertEqual(len(report(net)['mhc_input_packs']), 2)
        for value in net.dense.values():
            value.input_pack_rows = lambda rows: rows in (8, 16)
        with self.assertRaisesRegex(RuntimeError, '16 rows'):
            report(net)
        for value in net.dense.values():
            value.producer_pack_executed.add(16)
        self.assertEqual(len(report(net)['mhc_input_packs']), 2)
        for value in net.dense.values():
            value.producer_pack_executed.clear()
        with self.assertRaisesRegex(RuntimeError, 'mHC'):
            report(net)
        net.mhc_input_packs = False
        self.assertEqual(report(net)['mhc_input_packs'], {})

    def test_next_candidates_require_the_bound_consumers(self):
        path = Path(__file__).resolve().parents[1] / 'engine/profiles/glm53/boot.py'
        fn = next(n for n in ast.parse(path.read_text()).body
                  if isinstance(n, ast.FunctionDef) and n.name == 'next_k_cost_report')
        fn.body = [n for n in fn.body if not isinstance(n, ast.ImportFrom)]
        mla = NS(ENABLE_MLA_DIRECT_CVT=True, _DECODE_CELLS_EXECUTED={(8, 10), (16, 12)})
        scope = {'mla': mla}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), scope)
        net = NS(decode_fastpath_rows=(8, 16),
                 dense={'L0.kda.in_proj': NS(input_pack_rows=lambda n: n == 8)},
                 mhc=NS(EXPAND_FN=True, expanded_executed={('L0.hc.attn_fn', 8)}))
        report = scope['next_k_cost_report']
        self.assertEqual(report(net)['mla_direct'], [(8, 10), (16, 12)])
        net.mhc.expanded_executed.clear()
        with self.assertRaisesRegex(RuntimeError, 'mHC'):
            report(net)
        net.mhc.EXPAND_FN = False
        mla._DECODE_CELLS_EXECUTED.remove((16, 12))
        with self.assertRaisesRegex(RuntimeError, 'MLA'):
            report(net)
        mla.ENABLE_MLA_DIRECT_CVT = False
        report(net)

    def test_bound_forward_consumes_the_exact_owned_pack(self):
        layer = writer(4096)
        x = torch.zeros(8, 4096, dtype=torch.bfloat16)
        pack = torch.empty(dense.producer_pack_nbytes(8, 4096), dtype=torch.uint8)
        ext = Mock()
        with patch.object(dense, 'extension', return_value=ext):
            layer(x, producer_pack=pack)
        self.assertIs(ext.run_gemm_bound_input.call_args.kwargs['producer_pack'], pack)
        self.assertEqual(layer.producer_pack_executed, {8})
        for changed in (dict(observer=Mock()), dict(decode_precision='fp8'), dict(decode_input_rows=())):
            with patch.multiple(layer, create=True, **changed):
                with self.assertRaises(ValueError):
                    layer(x, producer_pack=pack)
        with self.assertRaises(ValueError):
            layer(x.repeat(2, 1), producer_pack=pack)

if __name__ == '__main__':
    unittest.main()
