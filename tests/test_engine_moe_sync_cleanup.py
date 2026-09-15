"""Execute pipeline setup and the real C1/C2 publication tail without a GPU."""
import ast
import copy
import random
from types import SimpleNamespace
import unittest

from tests.test_engine_moe_scatter_config import namespace
from tests.test_engine_moe_sf6_staging import CLASS, SOURCE, geometry


KERNEL = next(n for n in CLASS.body if isinstance(n, ast.FunctionDef) and n.name == 'kernel')
# The same M16 reform tile serves C1 rows and the C2 batch tile. C2 adds the
# retained direct scatter (#955) and a third FC2 pipeline slot (#962).
C2 = dict(registers=True, scatter_fp32=True, direct_scatter=True, scatter_reuse=True, fc2_prefetch=True)
TILES = {'C1': {}, 'C2': C2}
TOK, WEIGHT = 1 << 20, 1 << 21


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


class DropBarrier(ast.NodeTransformer):
    def visit_Expr(self, node):
        if ast.unparse(node.value) == 'self.epilog_sync_barrier.arrive_and_wait()':
            return ast.copy_location(ast.Pass(), node)
        return node


class Word(int):
    """A shared i32 load; the kernel bit-casts the weight row to FP32."""
    def bitcast(self, _):
        return float(self)


def publication_tail(*, item_barrier=True):
    work = next(n for n in ast.walk(KERNEL) if isinstance(n, ast.While)
                and ast.unparse(n.test) == 'is_valid_tile')
    h_index = next(i for i, n in enumerate(work.body)
                   if isinstance(n, ast.For) and ast.unparse(n.target) == 'h')
    role = work.body[h_index].body[0]
    quant_index = next(i for i, n in enumerate(role.body) if isinstance(n, ast.While)
                       and ast.unparse(n.test) == 'quant_idx < epi_rows * sf_blocks_per_half')
    read_index = next(i for i, n in enumerate(work.body) if assigned(n, 'csA2_p'))
    # Preserve every statement after quantization through the first A2 read,
    # including optional timing stamps, all synchronization and C2's retained
    # route-metadata loads.
    item = copy.deepcopy(work.body[h_index+1:read_index])
    if not item_barrier:
        item = [DropBarrier().visit(node) for node in item]
    fn = ast.parse('def tail():\n    pass').body[0]
    fn.body = copy.deepcopy(role.body[quant_index+1:]) + item
    return compile(ast.fix_missing_locations(SuspendPublication().visit(
        ast.Module(body=[fn], type_ignores=[]))), str(SOURCE), 'exec')


class SyncCleanupTests(unittest.TestCase):
    def test_actual_initialization_publishes_used_rings_once(self):
        start = next(i for i, n in enumerate(KERNEL.body) if assigned(n, 'fc1_pipeline'))
        end = next(i for i, n in enumerate(KERNEL.body[start:], start)
                   if isinstance(n, ast.Expr) and ast.unparse(n.value) == 'cute.arch.sync_threads()')
        code = compile(ast.Module(body=copy.deepcopy(KERNEL.body[start:end+1]),
                                  type_ignores=[]), str(SOURCE), 'exec')
        for tile, flags in TILES.items():
            for enabled in (False, True):
                g = geometry(sync_cleanup=enabled, **flags)
                self.assertEqual(g.fc2_stages, 3 if tile == 'C2' else 2)
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
                fc1, fc2 = ('init', 'fc1', 2*g.fc1_stages), ('init', 'fc2', 2*g.fc2_stages)
                if enabled:
                    self.assertEqual(events, [fc1, fc2, 'fence', 'sync'], tile)
                    self.assertNotIn('a_pipeline', env)
                else:
                    self.assertEqual(events, [fc1, 'fence', 'sync', fc2, 'fence', 'sync',
                                              ('init', 'a', 4), 'fence', 'sync', 'sync'], tile)

    def publish(self, code, flags, enabled, stamped, seed, *, final_check=True):
        """Run 128 MMA lanes through the real tail under one random schedule.

        Every lane first stores uneven quantization output; on C2 lanes 0..15
        also store their item's token/weight row, and a lane's Phase B loads
        two rows owned by other lanes. Only a lane's own fence publishes its
        stores. Returns the number of all-lane rendezvous.
        """
        rng = random.Random(seed)
        g = geometry(sync_cleanup=enabled, **flags)
        self.assertEqual(g.fc1_halves, 1, 'omission requires a single FC1 half')
        g.stamps = stamped
        g.scatter_row_pairs = (0, 1)  # the CuTe layout check derives this per lane
        valid_rows = rng.randint(1, g.tile_m)
        pending, published = {}, {}
        expected = {(t, k): t*101+k for t in range(128) for k in range(1+t % 9)}
        if g.scatter_reuse:
            expected.update({(t, kind): 7919*t+j for t in range(g.scatter_cache_rows)
                             for j, kind in enumerate(('tok', 'weight'))})

        def load(address):
            row, kind = ((address-TOK)//4, 'tok') if address < WEIGHT else ((address-WEIGHT)//4, 'weight')
            self.assertIn((row, kind), published, 'FC2 loaded unpublished route metadata')
            return Word(published[row, kind])

        def worker(t):
            if g.scatter_reuse and t < g.scatter_cache_rows:
                # Item start: this lane's cache row, published by no earlier fence here.
                yield ('store', (t, 'tok'), expected[t, 'tok'])
                yield ('store', (t, 'weight'), expected[t, 'weight'])
            # Uneven quantization duration deliberately leaves late
            # writers in each warp. Publication uses the real tail.
            for k in range(1+t % 9):
                yield ('store', (t, k), expected[t, k])
            rows, tensors = ((t*3) % 16, (t*3+8) % 16), []
            env = dict(self=g, cutlass=SimpleNamespace(const_expr=bool, range_constexpr=range, Float32=float),
                       Int32=int, tidx=t, item_no=0, STAMP_ITEMS=8, valid_tile_rows=valid_rows,
                       cute=SimpleNamespace(make_rmem_tensor=lambda shape, _: tensors.append([None]*shape[0]) or tensors[-1]),
                       ep_coords=[(rows[0], 0), (rows[0], 1), (rows[1], 0), (rows[1], 1)],
                       scatter_tok_base_addr=TOK, scatter_weight_base_addr=WEIGHT, scatter_N=5,
                       _ld_shared_i32_volatile=load)
            exec(code, env)
            yield from env['tail']()
            if g.scatter_reuse:
                bases, weights = tensors
                for slot, row in enumerate(rows):
                    live = row < valid_rows
                    self.assertEqual(bases[slot], expected[row, 'tok']*5 if live else 0)
                    self.assertEqual(weights[slot], float(expected[row, 'weight']) if live else 0.)
            else:
                self.assertEqual(tensors, [])
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
                if final_check:
                    self.assertEqual(published, expected, (flags, enabled, seed, t))
            else:
                self.fail(event)
        return rendezvous

    def test_actual_publication_tail_waits_for_late_writes_from_all_warps(self):
        code = publication_tail()
        for tile, flags in TILES.items():
            for enabled in (False, True):
                for stamped in (False, True):
                    for seed in range(20):
                        rendezvous = self.publish(code, flags, enabled, stamped, seed)
                        self.assertEqual(rendezvous, 1 if enabled and not stamped else 2,
                                         (tile, enabled, stamped, seed))

    def test_the_final_barrier_is_what_publishes_c2_route_metadata(self):
        # Negative control: without the item-level publication barrier a C2
        # lane loads another lane's route row before that lane has fenced it.
        # The simulation detects it, and the cleaned-up handle relies on
        # exactly the barrier it retains.
        code = publication_tail(item_barrier=False)
        with self.assertRaisesRegex(AssertionError, 'unpublished route metadata'):
            for seed in range(20):
                self.publish(code, C2, True, False, seed, final_check=False)

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

    def test_batch_tile_takes_the_cleanup_with_a_distinct_control(self):
        # `batch` widens decode_reform to the C2 M16 tile (#955); the cleanup
        # follows it there and nowhere else, and OFF keeps its own handle.
        ns = namespace()
        normalize, key = ns['_static_v2_decode_config'], ns['_static_v2_cache_key']
        raw = ns['_parse_glm53_static_v2']('t,r,sf6,batch')
        for rows in range(129):
            a = normalize(raw, rows)
            b = normalize(dict(raw, sync_cleanup=False), rows)
            enabled = 1 <= rows <= 8 or rows == 16
            self.assertEqual(a['sync_cleanup'], enabled, rows)
            self.assertFalse(b['sync_cleanup'])
            self.assertEqual(normalize(a, rows), a)
            self.assertEqual(normalize(b, rows), b)
            self.assertEqual(key(a, m=rows) != key(b, m=rows), enabled, rows)
        c2 = normalize(raw, 16)
        self.assertTrue(all(c2[k] for k in ('c2_direct_scatter', 'c2_scatter_reuse', 'c2_fc2_prefetch')))
        for enabled in (False, True):
            g = geometry(sync_cleanup=enabled, **C2)
            self.assertEqual((g.sync_cleanup, g.a_barrier_count, g.fc2_stages),
                             (enabled, 0 if enabled else 4, 3))


if __name__ == '__main__':
    unittest.main()
