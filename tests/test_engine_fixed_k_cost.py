"""Execute the resident scheduler and check ownership at the producer boundary."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch

from engine.kernels import dense, mla
from tests.test_engine_producer_pack_rows import writer
from tests.test_engine_moe_scatter_config import namespace


class ResidentWaveTests(unittest.TestCase):
    def test_actual_scheduler_preserves_work_and_does_not_add_waves(self):
        path = Path(__file__).resolve().parents[1] / 'engine/kernels/b12x/moe_static_kernel_v4.py'
        tree = ast.parse(path.read_text())
        kernel = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'kernel')
        start = next(i for i, n in enumerate(kernel.body) if isinstance(n, ast.Assign)
                     and ast.unparse(n.targets[0]) == 'n_active')
        end = next(i for i, n in enumerate(kernel.body) if isinstance(n, ast.Assign)
                   and ast.unparse(n.targets[0]) == 'split_base')
        code = compile(ast.Module(body=copy.deepcopy(kernel.body[start:end]), type_ignores=[]), str(path), 'exec')
        for ctas in (32, 40, 48):
            for items in range(0, 513):
                owners = []
                for block in range(ctas):
                    scope = dict(Int32=int, self=NS(even=True, split=False),
                                 cutlass=NS(const_expr=bool), gdim_z=ctas, bidz=block, next_item=[items])
                    exec(code, scope)
                    active = scope['n_active']
                    self.assertLessEqual((items + active - 1)//active, (items + ctas - 1)//ctas)
                    owners.extend(range(scope['start_work_idx'], items, active))
                self.assertEqual(sorted(owners), list(range(items)))

    def test_config_is_c1_only_and_control_has_a_distinct_kernel_key(self):
        md = namespace()
        base = md['_parse_glm53_static_v2']('t,r,sf6,batch')
        config = md['_static_v2_decode_config']
        for rows in (1, 7, 8, 16, 32):
            self.assertEqual(config(base, rows)['resident_waves'], rows == 8)
            self.assertFalse(config(dict(base, resident_waves=False), rows)['resident_waves'])
        md['_static_kernel_cache_key'] = lambda **kw: ()
        key = md['_static_v2_cache_key']
        self.assertNotEqual(key(config(base, 8)), key(config(dict(base, resident_waves=False), 8)))


class InputPackTests(unittest.TestCase):
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

    def test_mla_pair_dispatch_is_limited_and_explicit_splits_remain_a_control(self):
        for rows in (8, 16):
            q = torch.zeros(rows, 16, 512, dtype=torch.bfloat16)
            slots = torch.zeros(rows, 33, dtype=torch.int32)
            lens = torch.zeros(rows, dtype=torch.int32)
            with patch.object(mla, 'ENABLE_MLA_DECODE_PAIR', True), \
                    patch.object(mla, 'mla_decode_pair', return_value='pair') as pair:
                self.assertEqual(mla.mla_decode(q, torch.zeros(1, 512, dtype=torch.uint8), slots, lens, 1., 1.), 'pair')
                pair.assert_called_once()
                with patch.object(mla, 'mla_splits', side_effect=RuntimeError('control')):
                    with self.assertRaisesRegex(RuntimeError, 'control'):
                        mla.mla_decode(q, torch.zeros(1, 512, dtype=torch.uint8), slots, lens, 1., 1., splits=1)


if __name__ == '__main__':
    unittest.main()
