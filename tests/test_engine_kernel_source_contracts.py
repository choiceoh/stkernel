"""Source contracts for publication ordering and stated thresholds.

These tests read kernel sources instead of executing them. The served MoE lanes
are CuTe-DSL bodies that only run under a GPU, so a missing ordering step is
invisible to anything that cannot launch a kernel -- and the fleet's CPU lanes
never touch this code at all.

The contract is the one the 2026-09-18 FC1/C=2 audit found missing (#1133): the
static decode MoE publishes the FC1 packed input pair (packed A and its block
scales) with generic global stores, publishes that work through the resident
grid barrier, then reads the pair back with TMA. `membar.gl` and the barrier
epoch order ordinary global accesses; TMA reads go through the async proxy, so
the two need `fence.proxy.async.global` between them. Without it a consuming CTA
may read the new packed bits with the previous scales -- an operand that is
neither the search winner nor the baseline, with no NaN and no out-of-bounds
address needed to see it. Answer quality, not memory safety, is what fails.

Scope: these assert the served static decode lane, which is what #1133 fixed. Other
b12x bodies (moe_micro_kernel, the dynamic prefill family) publish packed inputs and
then TMA-read them through the same store -> resident-grid-barrier -> read shape, but
their ordering contracts have not been audited, so nothing is asserted about them
here. A finding there belongs in an audit first and in this file second.
"""
import ast
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / 'engine/kernels/b12x/moe_static_kernel_v4.py'
TILED = ROOT / 'engine/kernels/b12x/moe_static_kernel_v5.py'
CELLS = ROOT / 'engine/kernels/cells.py'
PREFILL = ROOT / 'engine/kernels/prefill_collectives/__init__.py'

# The TMA partitions of the packed input pair the publication produces. Weights
# (`tAgB*`) are static and carry no producer of their own, so they are not here.
PACKED_INPUT = ('tAgA', 'tAgSFA')

FENCE = 'fence_proxy'
BARRIER = '_resident_grid_barrier'
PTX_RULE = ('a generic global store feeding a TMA async-proxy read needs '
            'fence.proxy.async.global between them (see #1133)')


def module(path):
    return ast.parse(path.read_text())


def class_named(tree, name):
    node = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name), None)
    if node is None:
        raise AssertionError(f'{name} is gone; re-point this contract at the lane that replaced it')
    return node


def method_named(cls, name):
    return next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name), None)


def publication_events(fn):
    """(lineno, col, kind, what) for the calls and reads this contract is about, in source order."""
    events = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            called = ast.unparse(node.func)
            if called.endswith(BARRIER):
                events.append((node.lineno, node.col_offset, 'barrier', called))
            elif called.endswith(FENCE) and any(
                    'async.global' in ast.unparse(arg) for arg in node.args):
                events.append((node.lineno, node.col_offset, 'fence', called))
        elif isinstance(node, ast.Subscript) and ast.unparse(node.value) in PACKED_INPUT:
            events.append((node.lineno, node.col_offset, 'read', ast.unparse(node.value)))
    return sorted(events)


def assert_fence_orders_publication(test, fn, where):
    """The fence must sit after the publication barrier and before the first packed read."""
    events = publication_events(fn)
    barriers = [e for e in events if e[2] == 'barrier']
    fences = [e for e in events if e[2] == 'fence']
    reads = [e for e in events if e[2] == 'read']
    test.assertTrue(barriers, f'{where}: the resident grid barrier that publishes the pair is gone')
    test.assertTrue(reads, f'{where}: no TMA read of {" or ".join(PACKED_INPUT)}; re-point this contract')
    test.assertTrue(fences, f'{where}: no fence_proxy("async.global") -- {PTX_RULE}')
    published = max((e for e in barriers if (e[0], e[1]) < (reads[0][0], reads[0][1])),
                    default=None)
    test.assertIsNotNone(published, f'{where}: every barrier sits after the packed read')
    ordered = [e for e in fences if (published[0], published[1]) < (e[0], e[1])
               < (reads[0][0], reads[0][1])]
    test.assertTrue(ordered,
                    f'{where}: the fence is not between the publication barrier (line {published[0]}) '
                    f'and the first packed read (line {reads[0][0]}); {PTX_RULE}')


class StaticMoEProxyFenceTests(unittest.TestCase):
    def test_the_served_static_lane_fences_its_packed_input_publication(self):
        kernel = method_named(class_named(module(STATIC), 'MoEStaticKernelV4'), 'kernel')
        self.assertIsNotNone(kernel, 'MoEStaticKernelV4.kernel moved; re-point this contract')
        assert_fence_orders_publication(self, kernel, 'MoEStaticKernelV4.kernel')

    def test_the_tiled_lane_cannot_outgrow_the_fence_by_leaving_the_v4_body(self):
        v5 = class_named(module(TILED), 'MoEStaticKernelV5')
        bases = {ast.unparse(base).split('.')[-1] for base in v5.bases}
        kernel = method_named(v5, 'kernel')
        if kernel is None:
            self.assertIn('MoEStaticKernelV4', bases,
                          'MoEStaticKernelV5 serves through an inherited kernel that is no longer v4, '
                          f'so it no longer inherits the fenced body; {PTX_RULE}')
            return
        assert_fence_orders_publication(self, kernel, 'MoEStaticKernelV5.kernel')


def module_constant(path, name):
    for node in module(path).body:
        if (isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == name):
            return ast.literal_eval(node.value)
    raise AssertionError(f'{name} is no longer a literal in {path.name}')


class StatedThresholdTests(unittest.TestCase):
    def test_the_prefill_recipe_names_the_fp8_switch_at_its_real_value(self):
        """A recipe is an admission contract; a stale number admits a shape on a false premise."""
        actual = module_constant(PREFILL, 'FP8_MIN_ROWS')
        texts = [n.value for n in ast.walk(module(CELLS))
                 if isinstance(n, ast.Constant) and isinstance(n.value, str) and 'FP8_MIN_ROWS' in n.value]
        self.assertTrue(texts, 'no cell recipe names FP8_MIN_ROWS any more; re-point this contract')
        stated = [int(v) for text in texts for v in re.findall(r'FP8_MIN_ROWS (\d+) rows', text)]
        self.assertTrue(stated, f'every mention of FP8_MIN_ROWS must state its value, so a drift is visible '
                                f'(prefill_collectives has {actual})')
        for value in stated:
            self.assertEqual(value, actual,
                             f'cells.py states FP8_MIN_ROWS {value} rows; prefill_collectives/__init__.py '
                             f'defines {actual}')


if __name__ == '__main__':
    unittest.main()
