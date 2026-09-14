"""Execute pipeline setup and the real C1 publication tail without a GPU."""
import ast
import copy
import random
from types import SimpleNamespace
import unittest

from tests.test_engine_moe_scatter_config import namespace
from tests.test_engine_moe_sf6_staging import CLASS, SOURCE, geometry


KERNEL = next(n for n in CLASS.body if isinstance(n, ast.FunctionDef) and n.name == 'kernel')


def assigned(node, name):
    return isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == name


class SuspendPublication(ast.NodeTransformer):
    def visit_Call(self, node):
        event = {'self.epilog_sync_barrier.arrive_and_wait': 'barrier',
                 'cute.arch.fence_proxy': 'fence',
                 '_st_global_i64': 'stamp'}.get(ast.unparse(node.func))
        if event:
            return ast.copy_location(ast.Yield(ast.Constant(event)), node)
        return self.generic_visit(node)


def publication_tail():
    work = next(n for n in ast.walk(KERNEL) if isinstance(n, ast.While)
                and ast.unparse(n.test) == 'is_valid_tile')
    h_index = next(i for i, n in enumerate(work.body)
                   if isinstance(n, ast.For) and ast.unparse(n.target) == 'h')
    role = work.body[h_index].body[0]
    quant_index = next(i for i, n in enumerate(role.body) if isinstance(n, ast.While)
                       and ast.unparse(n.test) == 'quant_idx < epi_rows * sf_blocks_per_half')
    read_index = next(i for i, n in enumerate(work.body) if assigned(n, 'csA2_p'))
    # Preserve every statement after quantization through the first FC2
    # read, including optional timing stamps and all synchronization.
    body = copy.deepcopy(role.body[quant_index+1:] + work.body[h_index+1:read_index])
    fn = ast.parse('def tail():\n    pass').body[0]
    fn.body = body
    return compile(ast.fix_missing_locations(SuspendPublication().visit(
        ast.Module(body=[fn], type_ignores=[]))), str(SOURCE), 'exec')


class SyncCleanupTests(unittest.TestCase):
    def test_actual_initialization_publishes_used_rings_once(self):
        start = next(i for i, n in enumerate(KERNEL.body) if assigned(n, 'fc1_pipeline'))
        end = next(i for i, n in enumerate(KERNEL.body[start:], start)
                   if isinstance(n, ast.Expr) and ast.unparse(n.value) == 'cute.arch.sync_threads()')
        code = compile(ast.Module(body=copy.deepcopy(KERNEL.body[start:end+1]),
                                  type_ignores=[]), str(SOURCE), 'exec')
        for enabled in (False, True):
            g = geometry(sync_cleanup=enabled)
            events, pointers = [], {}
            for name, count in (('fc1', 2*g.fc1_stages), ('fc2', 2*g.fc2_stages),
                                ('a', g.a_barrier_count)):
                def pointer(name=name, count=count):
                    self.assertGreater(count, 0, 'accessed omitted A barrier storage')
                    return name, count
                pointers[name+'_bars'] = SimpleNamespace(data_ptr=pointer)
            def create(**kw):
                name, count = kw['barrier_storage']
                self.assertEqual(count, 2*kw['num_stages'])
                self.assertEqual(kw['producer_group'], 'producer')
                self.assertEqual(kw['consumer_group'], 'four-warp-consumer')
                self.assertGreater(kw['tx_count'], 0)
                events.append(('init', name, count))
                # Mirrors the inspected installed API contract: create
                # initializes both full/empty barriers even when deferred.
                if not kw.get('defer_sync', False):
                    events.extend(('fence', 'sync'))
                return object()
            env = dict(self=g, prod_group='producer', cons_group='four-warp-consumer',
                fc1_tma_bytes=17936, fc2_tma_bytes=17936, a_tma_bytes=4096,
                storage=SimpleNamespace(**pointers), cta_layout_vmnk=(1, 1, 1, 1),
                pipeline=SimpleNamespace(PipelineTmaAsync=SimpleNamespace(create=create)),
                cutlass=SimpleNamespace(const_expr=bool), cute=SimpleNamespace(arch=SimpleNamespace(
                    mbarrier_init_fence=lambda: events.append('fence'),
                    sync_threads=lambda: events.append('sync'))))
            exec(code, env)
            self.assertEqual(g.cluster_shape_mnk, (1, 1, 1))
            if enabled:
                self.assertEqual(events, [('init', 'fc1', 4), ('init', 'fc2', 4), 'fence', 'sync'])
                self.assertNotIn('a_pipeline', env)
            else:
                self.assertEqual(events, [('init', 'fc1', 4), 'fence', 'sync',
                    ('init', 'fc2', 4), 'fence', 'sync', ('init', 'a', 4), 'fence', 'sync', 'sync'])

    def test_actual_publication_tail_waits_for_late_writes_from_all_warps(self):
        code = publication_tail()
        for enabled in (False, True):
            for stamped in (False, True):
                for seed in range(20):
                    rng = random.Random(seed)
                    g = geometry(sync_cleanup=enabled)
                    self.assertEqual(g.fc1_halves, 1, 'omission requires a single FC1 half')
                    g.stamps = stamped
                    pending, published = {}, {}
                    expected = {(t, k): t*101+k for t in range(128) for k in range(1+t % 9)}
                    def worker(t):
                        # Uneven quantization duration deliberately leaves late
                        # writers in each warp. Publication uses the real tail.
                        for k in range(1+t % 9):
                            yield ('store', (t, k), expected[t, k])
                        env = dict(self=g, cutlass=SimpleNamespace(const_expr=bool),
                                   Int32=int, tidx=t, item_no=0, STAMP_ITEMS=8)
                        exec(code, env)
                        yield from env['tail']()
                        yield 'read'
                    workers = [worker(t) for t in range(128)]
                    ready, blocked, rendezvous = list(range(128)), [], 0
                    while ready or blocked:
                        if not ready:
                            self.assertEqual(len(blocked), 128, 'a thread escaped publication')
                            ready, blocked = blocked, []
                            rendezvous += 1
                        t = rng.choice(ready)
                        try:
                            event = next(workers[t])
                        except StopIteration:
                            ready.remove(t)
                            continue
                        if isinstance(event, tuple):
                            pending[event[1]] = event[2]
                        elif event == 'fence':
                            for key in [key for key in pending if key[0] == t]:
                                published[key] = pending.pop(key)
                        elif event == 'barrier':
                            ready.remove(t)
                            blocked.append(t)
                        elif event == 'stamp':
                            self.assertEqual({**published, **pending}, expected,
                                             'FC1 timestamp preceded a peer write')
                        elif event == 'read':
                            self.assertEqual(published, expected, (enabled, seed, t))
                        else:
                            self.fail(event)
                    self.assertEqual(rendezvous, 1 if enabled and not stamped else 2)

    def test_scope_normalization_and_control_identity(self):
        ns = namespace()
        normalize, key = ns['_static_v2_decode_config'], ns['_static_v2_cache_key']
        for recipe in ('t', 't,r', 't,r,sf6'):
            for rows in (0, 1, 6, 7, 8, 9, 16, 32, 128):
                for compact in (False, True):
                    raw = dict(ns['_parse_glm53_static_v2'](recipe), compact_staging=compact)
                    a = normalize(raw, rows)
                    b = normalize(dict(raw, sync_cleanup=False), rows)
                    enabled = recipe == 't,r,sf6' and 1 <= rows <= 8
                    self.assertEqual(a['sync_cleanup'], enabled)
                    self.assertFalse(b['sync_cleanup'])
                    self.assertEqual(normalize(a, rows), a)
                    self.assertEqual(normalize(b, rows), b)
                    self.assertEqual(key(a, m=rows) != key(b, m=rows), enabled)
                    g = geometry(reform=a['decode_reform'], packed=a.get('reform_sf_pack', False),
                                 compact=compact)
                    self.assertEqual(g.sync_cleanup, enabled)
                    self.assertEqual(g.a_barrier_count, 0 if enabled else 4)


if __name__ == '__main__':
    unittest.main()
