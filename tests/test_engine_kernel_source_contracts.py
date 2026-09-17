"""Source contracts for publication ordering and stated thresholds.

These tests read kernel sources instead of executing them. The served MoE lanes
are CuTe-DSL bodies that only run under a GPU, so a missing ordering step is
invisible to anything that cannot launch a kernel -- and the fleet's CPU lanes
never touch this code at all.

The contract is the one the 2026-09-18 FC1/C=2 audit found missing (#1133): a
body publishes the FC1 packed input pair (packed A and its block scales) with
generic global stores, publishes that work through the resident grid barrier,
then reads the pair back with TMA. `membar.gl` and the barrier epoch order
ordinary global accesses; TMA reads go through the async proxy, so the two need
`fence.proxy.async.global` between them. Without it a consuming CTA may read the
new packed bits with the previous scales -- an operand that is neither the search
winner nor the baseline, with no NaN and no out-of-bounds address needed to see
it. Answer quality, not memory safety, is what fails.

Lanes: `LANES` is every body the audit found doing that store -> resident-grid
barrier -> TMA read sequence -- the static decode lane (v4, and v5 through
inheritance), the stock static lane, the micro lane and the dynamic body. The
prefill *fragment* modules (`moe_dynamic_prefill.py`, `_prefill_m64_bodies.py`,
`_moe_dynamic/gated.py`, `moe_dynamic_gated_sf6.py`, the packet variants) pack
inputs and read `tAgA` too, but carry no resident grid barrier of their own: they
are spliced into a body that has one, so a source-level assertion about the
fragment would be about the wrong file. Add a lane here only after its
publication is shown to be its own.
"""
import ast
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
B12X = ROOT / 'engine/kernels/b12x'
TILED = B12X / 'moe_static_kernel_v5.py'
CELLS = ROOT / 'engine/kernels/cells.py'
PREFILL = ROOT / 'engine/kernels/prefill_collectives/__init__.py'

# module (relative to b12x), class -- every body that publishes the packed pair before its TMA reads.
LANES = (
    ('moe_static_kernel_v4.py', 'MoEStaticKernelV4'),
    ('moe_static_kernel.py', 'MoEStaticKernel'),
    ('moe_micro_kernel.py', 'MoEMicroKernel'),
    ('_moe_dynamic/generic.py', 'MoEDynamicKernel'),
)

# The TMA partitions of the packed input pair the publication produces. Weights
# (`tAgB*`) are static and carry no producer of their own, so they are not here.
PACKED_INPUT = ('tAgA', 'tAgSFA')
# Where a publication writes the pair: the packed bits and their block scales.
PACKED_STORES = ('packed_a_storage', 'scale_storage')

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


def kernel_of(lane):
    """The `kernel` method LANES names, wherever it is defined."""
    where, klass = lane
    return method_named(class_named(module(B12X / where), klass), 'kernel')


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


def packed_store_lines(fn):
    """Lines that write the packed pair: the bit helper, or a store into its storage."""
    lines = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and 'st_global' in ast.unparse(node.func):
            if any(name in ast.unparse(node) for name in PACKED_STORES):
                lines.append(node.lineno)
        elif isinstance(node, ast.Assign):
            if any(any(name in ast.unparse(target) for name in PACKED_STORES)
                   for target in node.targets):
                lines.append(node.lineno)
    return sorted(lines)


def assert_fence_orders_publication(test, fn, where):
    """The fence must sit after the last packed store and before the first packed read."""
    events = publication_events(fn)
    barriers = [e for e in events if e[2] == 'barrier']
    fences = [e for e in events if e[2] == 'fence']
    reads = [e for e in events if e[2] == 'read']
    stores = packed_store_lines(fn)
    test.assertTrue(stores, f'{where}: no generic store into {" or ".join(PACKED_STORES)}; '
                            f're-point this contract at the publication it uses')
    test.assertTrue(barriers, f'{where}: the resident grid barrier that publishes the pair is gone')
    test.assertTrue(reads, f'{where}: no TMA read of {" or ".join(PACKED_INPUT)}; re-point this contract')
    test.assertTrue(fences, f'{where}: no fence_proxy("async.global") -- {PTX_RULE}')
    last_store, first_read = stores[-1], reads[0][0]
    published = max((e for e in barriers if e[0] < first_read), default=None)
    test.assertIsNotNone(published, f'{where}: every barrier sits after the packed read')
    ordered = [e for e in fences
               if last_store < e[0] < first_read and published[0] < e[0]]
    test.assertTrue(ordered,
                    f'{where}: the fence is not between the last packed-input store (line {last_store}) '
                    f'and the first packed read (line {first_read}), after the publication barrier '
                    f'(line {published[0]}); {PTX_RULE}')


class MoEProxyFenceTests(unittest.TestCase):
    def test_every_publishing_lane_fences_its_packed_input(self):
        for lane in LANES:
            with self.subTest(lane=lane[0]):
                kernel = kernel_of(lane)
                self.assertIsNotNone(kernel, f'{lane[1]}.kernel moved; re-point this contract')
                assert_fence_orders_publication(self, kernel, f'{lane[0]}:{lane[1]}.kernel')

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
