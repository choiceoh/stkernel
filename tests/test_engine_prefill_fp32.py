"""Long-prefill route sums must clear the whole FP32 plane on every call."""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1] / 'engine/kernels/b12x'


class LongPrefillAccumulatorTests(unittest.TestCase):
    def test_both_producers_zero_each_fp32_word_and_leave_guards_untouched(self):
        for filename in ('moe_dynamic_gated_sf6_prefill.py', 'moe_dynamic_prefill_packets.py'):
            tree = ast.parse((ROOT / filename).read_text())
            fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                      and n.name == 'initialize_route_q0_and_publish')
            def assigns(node, name):
                return isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name
                                                           for t in node.targets)
            columns = next(n for n in fn.body if assigns(n, 'cols_u32'))
            begin = next(i for i, n in enumerate(fn.body) if assigns(n, 'scatter_total_u32'))
            end = next(i for i, n in enumerate(fn.body[begin:], begin)
                       if isinstance(n, ast.If) and ast.unparse(n.test) == 'flat_tid == Int32(0)')
            code = compile(ast.Module(body=[columns, *fn.body[begin:end]], type_ignores=[]), filename, 'exec')
            for rows, cols in ((1, 4096), (257, 4096), (8193, 16), (32256, 16), (32768, 16)):
                words = rows * cols
                # Nonzero poison plus guards detects stale upper-half words,
                # out-of-bounds zeroing, and overlap between producer threads.
                writes = bytearray(words)
                def store(address, *values):
                    self.assertEqual(values, (0, 0, 0, 0))
                    self.assertEqual(address % 16, 0)
                    index = address // 4
                    self.assertTrue(0 <= index <= words - 4)
                    for j in range(index, index + 4):
                        writes[j] += 1
                for flat in range(288 * 3):
                    exec(code, dict(Int32=int, Int64=int, Uint32=int, num_tokens=rows, cols=cols,
                                    flat_tid=flat, flat_stride=288*3, scatter_base=0,
                                    st_global_v4_u32=store, scatter_output_u32=None))
                self.assertEqual(writes, bytearray([1]) * words, (filename, rows, cols))


if __name__ == '__main__':
    unittest.main()
