"""The drafter's block MLP takes the target's C1 input cells at 8 rows, and the boot proof requires them to run."""
import unittest
from types import SimpleNamespace as NS

from engine.profiles.glm53.boot import drafter_decode_cell_report
from engine.profiles.glm53.drafter import Drafter


def layer(rows, cols, precision='w4'):
    return NS(packs=(NS(rows=rows, cols=cols),), decode_precision=precision, decode_input_rows=(),
              bound_input_executed=set())


def drafter(layers=2):
    dense = {'fc.weight': layer(4096, 20480)}
    for L in range(layers):
        n = f'layers.{L}.'
        dense.update({n + 'self_attn.qkv': layer(1536, 4096), n + 'self_attn.o_proj.weight': layer(4096, 1024),
                      n + 'mlp.gate_up': layer(6144, 4096), n + 'mlp.down_proj.weight': layer(4096, 3072)})
    return NS(F=NS(layers=layers), dense=dense, DECODE_CELL_WEIGHTS=Drafter.DECODE_CELL_WEIGHTS)


class DrafterDecodeCellTests(unittest.TestCase):
    def test_only_the_mlp_projections_bind_and_only_at_eight_rows(self):
        d = drafter()
        bound = Drafter.bind_decode_cells(d, (8, 16))
        self.assertEqual(bound, ['layers.0.mlp.gate_up', 'layers.0.mlp.down_proj.weight',
                                 'layers.1.mlp.gate_up', 'layers.1.mlp.down_proj.weight'])
        self.assertEqual(d.decode_cell_rows, (8,))
        for name, owner in d.dense.items():
            self.assertEqual(owner.decode_input_rows, (8,) if name in bound else ())

    def test_nothing_binds_without_fastpath_rows_or_a_w4_decode_pack(self):
        d = drafter()
        self.assertEqual(Drafter.bind_decode_cells(d, None), [])
        self.assertEqual(d.decode_cell_rows, ())
        d = drafter(1)
        d.dense['layers.0.mlp.gate_up'].decode_precision = 'fp8'
        self.assertEqual(Drafter.bind_decode_cells(d, (8,)), ['layers.0.mlp.down_proj.weight'])

    def test_the_proof_fails_until_every_bound_cell_ran(self):
        d = drafter(1)
        self.assertEqual(drafter_decode_cell_report(d), {})
        Drafter.bind_decode_cells(d, (8, 16))
        with self.assertRaisesRegex(RuntimeError, 'bound drafter C1 cells were not executed'):
            drafter_decode_cell_report(d)
        d.dense['layers.0.mlp.gate_up'].bound_input_executed.add(8)
        with self.assertRaisesRegex(RuntimeError, 'down_proj'):
            drafter_decode_cell_report(d)
        d.dense['layers.0.mlp.down_proj.weight'].bound_input_executed.add(8)
        self.assertEqual(drafter_decode_cell_report(d)['dense'],
                         {'layers.0.mlp.gate_up': [8], 'layers.0.mlp.down_proj.weight': [8]})


if __name__ == '__main__':
    unittest.main()
