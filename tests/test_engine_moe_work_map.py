"""Device-bounded C2 work ownership, including duplicate-route replays."""
import ast
import __future__
from pathlib import Path
import random
from types import SimpleNamespace as NS
import unittest


ROOT = Path(__file__).resolve().parents[1] / 'engine/kernels/b12x'


def helpers():
    path = ROOT / 'moe_static_common.py'
    nodes = [node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef)
             and node.name in ('_compact_static_get_work_tile', '_indexed_static_get_work_tile')]
    for node in nodes:
        node.decorator_list = []
    ns = dict(Int32=int)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec',
                 flags=__future__.annotations.compiler_flag), ns)
    return ns['_compact_static_get_work_tile'], ns['_indexed_static_get_work_tile']


class WorkMapTests(unittest.TestCase):
    def test_single_tile_ownership_has_no_row_count_reads(self):
        scan, indexed = helpers()
        class NoReads:
            def __getitem__(self, index):
                raise AssertionError('direct ownership read row_counts')
        rng = random.Random(916)
        for active in (0, 1, 8, 11, 16, 32, 48, 64, 98, 128):
            counts = [rng.randint(1, 16) for _ in range(active)]
            for slices in (1, 4, 8):
                for grid in (1, 32, 48):
                    for cta in range(grid):
                        cursor, accum = 0, 0
                        for work in range(cta, active*slices+grid, grid):
                            args = dict(tile_m=16, num_tiles_n=slices, cluster_shape_mn=(1, 1),
                                        current_work_linear_idx=work, current_local_expert_idx=cursor,
                                        accum_tile_m=accum, cta_id_in_cluster=(0, 0, 0))
                            old = scan(counts, [active], **args)
                            new = indexed(NoReads(), [active], single_m_tile=1, **args)
                            self.assertEqual(old[1], new[1])
                            if new[1]:
                                self.assertEqual(old, new)
                                self.assertEqual(new[0], (0, work % slices, work // slices))
                            cursor, accum = old[2:]

    def test_overflow_guard_resets_and_uses_the_complete_original_schedule(self):
        _, indexed = helpers()
        source = ROOT / 'moe_static_kernel_v4.py'
        tree = ast.parse(source.read_text())
        kernel = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'kernel')
        reset = next(n for n in ast.walk(kernel) if isinstance(n, ast.If)
                     and ast.unparse(n.test) == 'flat_tid == Int32(0)')
        guard = next(n for n in ast.walk(kernel) if isinstance(n, ast.If)
                     and ast.unparse(n.test) == 'cutlass.const_expr(self.c2_work_map)'
                     and 'atomic_add_global_i32' in ast.unparse(n))
        def code(node):
            return compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec')
        def atomic(address, value):
            array, at = address
            previous = array[at]
            array[at] += value
            return previous
        counter = [999]
        ns = dict(Int32=int, flat_tid=0, next_item=counter, active_expert_count=[99],
                  self=NS(c2_work_map=True, even=False, split=False, tile_m=16),
                  cutlass=NS(const_expr=lambda value: value),
                  get_ptr_as_int64=lambda tensor, index: (tensor, index), atomic_add_global_i32=atomic)
        # Alternate valid and overflowing routes on the SAME counter storage.
        for counts in ([128], [16]*8, [64, 64], [1]*128, [17, 15, 33, 63], [8]*16, []):
            exec(code(reset), ns)
            for count in counts:
                for row in range(count):
                    ns['row'] = row
                    exec(code(guard), ns)
            self.assertEqual(counter[0], sum(count > 16 for count in counts))
            expected = [(m, n, expert) for expert, count in enumerate(counts)
                        for m in range((count+15)//16) for n in range(4)]
            for cta in range(48):
                cursor, accum = 0, 0
                for work in range(cta, len(expected)+48, 48):
                    tile, valid, cursor, accum = indexed(counts, [len(counts)],
                        single_m_tile=int(counter[0] == 0), tile_m=16, num_tiles_n=4,
                        cluster_shape_mn=(1, 1), current_work_linear_idx=work,
                        current_local_expert_idx=cursor, accum_tile_m=accum, cta_id_in_cluster=(0, 0, 0))
                    self.assertEqual(valid, work < len(expected))
                    if valid:
                        self.assertEqual(tile, expected[work])


if __name__ == '__main__':
    unittest.main()
