"""Exercise the actual grid-barrier control flow under reordered arrivals.

The model distinguishes observing a relaxed counter from acquiring the writes
released through it. It checks the publication contract, not GPU timing.
"""
import ast
import copy
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1] / 'engine/kernels/b12x'
BARRIERS = ('moe_static_kernel_v4.py', 'moe_static_kernel.py',
            'moe_micro_kernel.py', 'moe_direct_micro_kernel.py',
            '_moe_dynamic/generic.py', '_moe_dynamic/gated.py')
PROXY_OWNERS = ('moe_static_kernel_v4.py', 'moe_static_kernel.py',
                'moe_micro_kernel.py', '_moe_dynamic/generic.py',
                '_moe_dynamic/gated.py', 'moe_dynamic_gated_sf6.py',
                'moe_dynamic_prefill.py', '_prefill_m64_bodies.py')


class Events(ast.NodeTransformer):
    def visit_Call(self, node):
        name = ast.unparse(node.func)
        if name in ('threadfence', 'ld_global_acquire_i32',
                    'st_global_release_i32', 'spin_wait_global_eq_i32'):
            name = '_' + name
        if name in ('cute.arch.sync_threads', '_threadfence',
                    '_ld_global_acquire_i32', 'atomic_add_global_i32',
                    'st_global_i32', '_st_global_release_i32',
                    '_spin_wait_global_eq_i32'):
            return ast.copy_location(ast.Yield(ast.Tuple(
                elts=[ast.Constant(name), *node.args], ctx=ast.Load())), node)
        return self.generic_visit(node)


def barrier(path, *, remove_final_acquire=False, token=False):
    tree = ast.parse((ROOT / path).read_text())
    fn = copy.deepcopy(next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                            and n.name in (('_token_publish_fc1_ready',) if token else
                                           ('resident_grid_barrier', '_resident_grid_barrier'))))
    if token:
        waiter = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                      and n.name == '_token_wait_fc1_ready')
        fn.args = ast.parse('def x(self, barrier_count, barrier_epoch, grid_x, is_cta_leader): pass').body[0].args
        fn.body = ast.parse('token_idx = 0\nexpected_epoch = 0\nchunks_per_token = grid_x').body + fn.body + copy.deepcopy(waiter.body)
    fn.decorator_list = []
    fn.returns = None
    for arg in fn.args.args:
        arg.annotation = None
    if remove_final_acquire:
        last = next(n for n in ast.walk(fn) if isinstance(n, ast.If)
                    and ast.unparse(n.test) in ('arrived == grid_x - Int32(1)',
                                               'arrived == chunks_per_token - Int32(1)'))
        last.body = [n for n in last.body if not (
            isinstance(n, ast.Expr) and ast.unparse(n.value) in ('_threadfence()', 'threadfence()'))]
    env = dict(Int32=int, get_ptr_as_int64=lambda tensor, _: tensor)
    exec(compile(ast.fix_missing_locations(Events().visit(
        ast.Module(body=[fn], type_ignores=[]))), str(ROOT / path), 'exec'), env)
    return env[fn.name]


def reordered_arrivals(fn, count=3):
    visible = [{i} for i in range(count)]  # stores before each CTA's entry sync
    observed = [set() for _ in range(count)]
    released = [set() for _ in range(count)]
    sc_before, counter_release, epoch_release = set(), set(), set()
    counter, epoch = 0, 0
    generators = [fn(None, 'count', 'epoch', count, 1) for _ in range(count)]
    pending = [next(g) for g in generators]

    def step(i):
        nonlocal counter, epoch, epoch_release, sc_before, counter_release
        event, *args = pending[i]
        result = None
        if event == '_threadfence':
            visible[i].update(sc_before | observed[i])
            sc_before.update(visible[i])
            released[i] = visible[i].copy()
        elif event == '_ld_global_acquire_i32':
            visible[i].update(epoch_release)
            result = epoch
        elif event == 'atomic_add_global_i32':
            result = counter
            observed[i] = counter_release.copy()
            counter_release.update(released[i])
            counter += args[1]
        elif event == 'st_global_i32':
            counter = args[1]
        elif event == '_st_global_release_i32':
            epoch_release = visible[i].copy()
            epoch = args[1]
        elif event == '_spin_wait_global_eq_i32':
            assert epoch != args[1], 'waiter advanced before epoch publication'
            visible[i].update(epoch_release)
        else:
            assert event == 'cute.arch.sync_threads'
        try:
            pending[i] = generators[i].send(result)
        except StopIteration:
            pending[i] = None

    # CTA 0 fences first but increments last. Its entry fence therefore
    # cannot acquire the later CTAs' writes merely because it arrives last.
    for i in range(count):
        while pending[i][0] != 'atomic_add_global_i32':
            step(i)
    for i in (*range(1, count), 0):
        step(i)
        while pending[i] and pending[i][0] != '_spin_wait_global_eq_i32':
            step(i)
    for i in range(1, count):
        while pending[i]:
            step(i)
    return visible


class GlobalPublicationTests(unittest.TestCase):
    def test_single_token_chunk_publication_has_the_same_contract(self):
        path = 'moe_direct_micro_kernel.py'
        self.assertEqual(reordered_arrivals(barrier(path, token=True)), [{0, 1, 2}] * 3)
        self.assertNotEqual(reordered_arrivals(barrier(path, token=True, remove_final_acquire=True))[0], {0, 1, 2})

    def test_final_arriver_transitively_publishes_every_cta(self):
        for path in BARRIERS:
            for count in (2, 3, 8):
                with self.subTest(path=path, count=count):
                    self.assertEqual(reordered_arrivals(barrier(path), count),
                                     [set(range(count)) for _ in range(count)])

    def test_negative_control_detects_missing_final_acquire(self):
        for path in BARRIERS:
            with self.subTest(path=path):
                seen = reordered_arrivals(barrier(path, remove_final_acquire=True))
                self.assertNotEqual(seen[0], {0, 1, 2})
                self.assertNotEqual(seen[1], {0, 1, 2})

    def test_each_tma_consumer_bridges_after_input_publication(self):
        for path in PROXY_OWNERS:
            tree = ast.parse((ROOT / path).read_text())
            kernel = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                          and n.name == 'kernel')
            fences = [i for i, n in enumerate(kernel.body) if isinstance(n, ast.Expr)
                      and isinstance(n.value, ast.Call)
                      and ast.unparse(n.value.func) == 'cute.arch.fence_proxy'
                      and n.value.args and isinstance(n.value.args[0], ast.Constant)
                      and n.value.args[0].value == 'async.global']
            with self.subTest(path=path):
                self.assertEqual(len(fences), 1)
                before = kernel.body[fences[0] - 1]
                self.assertIsInstance(before, ast.Expr)
                self.assertIn(ast.unparse(before.value.func), (
                    'self._resident_grid_barrier',
                    'self.initialize_route_q0_and_publish'))


if __name__ == '__main__':
    unittest.main()
