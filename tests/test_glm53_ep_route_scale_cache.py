"""CPU raw-bit oracle for Q0's producer-published selected-scale state."""
import ast
import copy
from pathlib import Path
import random
import struct
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "overlay/modules/glm53_moe/moe_dynamic_ep_local.py"


class FloatBits:
    def __init__(self, bits):
        self.bits = bits & 0xFFFFFFFF

    def number(self):
        return struct.unpack("<f", struct.pack("<I", self.bits))[0]

    def __eq__(self, other):
        return self.number() == other.number()

    def __ne__(self, other):
        return not self == other

    def to(self, dtype):
        return dtype(self)


class ExpertId(int):
    def to(self, dtype):
        return dtype(self)


class Uint32(int):
    def __new__(cls, value):
        return super().__new__(cls, value & 0xFFFFFFFF)

    def bitcast(self, _):
        return FloatBits(self)


def f32(value):
    if isinstance(value, FloatBits):
        return value
    return FloatBits(struct.unpack("<I", struct.pack("<f", value))[0])


def uses_name(node, name):
    return any(isinstance(item, ast.Name) and item.id == name for item in ast.walk(node))


def single_warp_projection(node):
    """Resolve only the new constexpr option, then join its split old guard.

    The dedicated dual-warp tests compare this projection with the pinned
    pre-option method. Arithmetic, route filters and scale oracles stay intact.
    """
    class Project(ast.NodeTransformer):
        def visit_If(self, node):
            spelling = ast.unparse(node.test)
            if spelling in ("cutlass.const_expr(self.q0_dual_warp)",
                            "cutlass.const_expr(not self.q0_dual_warp)"):
                body = node.body if spelling.startswith("cutlass.const_expr(not ") else node.orelse
                return [self.visit(item) for item in body]
            return self.generic_visit(node)

        def visit_Assign(self, node):
            if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id in ("q0_row_idx", "q0_producer_lane")):
                expected = {"q0_row_idx": "warp_idx", "q0_producer_lane": "lane_id"}
                if ast.unparse(node.value) != expected[node.targets[0].id]:
                    raise AssertionError("constant-off Q0 alias changed")
                return None
            return self.generic_visit(node)

        def visit_Name(self, node):
            aliases = {"q0_row_idx": "warp_idx", "q0_producer_lane": "lane_id"}
            return ast.copy_location(ast.Name(aliases.get(node.id, node.id), node.ctx), node)

        def generic_visit(self, node):
            node = super().generic_visit(node)
            for field, value in ast.iter_fields(node):
                if not isinstance(value, list):
                    continue
                joined = []
                for item in value:
                    if (joined and isinstance(item, ast.If) and isinstance(joined[-1], ast.If)
                            and not item.orelse and not joined[-1].orelse
                            and ast.dump(item.test) == ast.dump(joined[-1].test)):
                        previous = joined[-1]
                        body = item.body
                        if (body and previous.body and isinstance(body[0], ast.Assign)
                                and ast.unparse(body[0]) == "route_slot_base = warp_idx * Int32(32)"
                                and ast.dump(body[0]) == ast.dump(previous.body[0])):
                            body = body[1:]  # repeated identical route_slot_base view
                        previous.body.extend(body)
                    else:
                        joined.append(item)
                setattr(node, field, joined)
            return node
    return ast.fix_missing_locations(Project().visit(copy.deepcopy(node)))


class SourceCache:
    """Execute actual lane-0 allocation and each lane's published-state loads.

    Atomic allocation and token-map writes have bounded CPU stand-ins. The
    route filtering, raw-scale load/store, comparison, packed publication and
    consumer decoding are all extracted from the kernel, not reimplemented.
    Consumers use fresh register namespaces; lane 0 registers cannot leak into
    another lane and accidentally conceal missing shared-memory publication.
    """
    ROWS = 0
    SCALES = 1152
    EXPERTS = 2304

    def __init__(self):
        tree = ast.parse(KERNEL.read_text())
        method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                      and node.name == "initialize_route_q0_and_publish")
        method = single_warp_projection(method)
        selected = next(node for node in ast.walk(method) if isinstance(node, ast.If)
                        and ast.unparse(node.test) == "local_topk > Int32(0)")
        token_body = next(node.body for node in ast.walk(method)
                          if isinstance(node, ast.If) and selected in node.body)
        active_guard = next(node.test for node in ast.walk(method)
                            if isinstance(node, ast.If) and node.body is token_body)
        self.active_guard = compile(ast.Expression(active_guard), str(KERNEL), "eval")
        batch_loop = next(node for node in ast.walk(method) if isinstance(node, ast.While)
                          and uses_name(node.test, "produce_active"))
        batch_barriers = [i for i, node in enumerate(batch_loop.body)
                          if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                          and isinstance(node.value.func, ast.Attribute)
                          and node.value.func.attr == "sync_threads"]
        batch_dispatch = next(i for i, node in enumerate(batch_loop.body)
                              if isinstance(node, ast.If) and any(child is selected
                                  for child in ast.walk(node)))
        self.batch_publication_order = (batch_barriers, batch_dispatch)
        producer = next(node for node in token_body if isinstance(node, ast.If)
                        and any(isinstance(child, ast.While)
                                and uses_name(child.test, "topk_slot")
                                for child in node.body))
        producer_index = token_body.index(producer)
        selected_index = token_body.index(selected)
        state_index = next(i for i, node in enumerate(token_body[:selected_index])
                           if isinstance(node, ast.Assign)
                           and uses_name(node, "_ld_shared_i32")
                           and uses_name(node, "route_slot_base"))
        barrier_indices = [i for i, node in enumerate(token_body)
                           if isinstance(node, ast.Expr)
                           and isinstance(node.value, ast.Call)
                           and isinstance(node.value.func, ast.Attribute)
                           and node.value.func.attr == "sync_warp"]
        self.publication_order = (producer_index, barrier_indices, state_index)
        sf_index = next(i for i, node in enumerate(selected.body)
                        if isinstance(node, ast.Assign)
                        and any(isinstance(target, ast.Name) and target.id == "sf_idx"
                                for target in node.targets))
        select = token_body[state_index:selected_index] + [ast.If(
            test=copy.deepcopy(selected.test),
            body=copy.deepcopy(selected.body[:sf_index]), orelse=[])]
        branch = next(node for node in ast.walk(selected) if isinstance(node, ast.If)
                      and ast.unparse(node.test) == "route_scales_equal > Int32(0)")

        def gs_assignment(body):
            return next(node for root in body for node in ast.walk(root)
                        if isinstance(node, ast.Assign)
                        and any(isinstance(target, ast.Name) and target.id == "gs_value"
                                for target in node.targets))

        choose = ast.If(test=copy.deepcopy(branch.test),
                        body=[copy.deepcopy(gs_assignment(branch.body))],
                        orelse=[copy.deepcopy(gs_assignment(branch.orelse))])

        def address_loop(body):
            loop = copy.deepcopy(next(node for node in body if isinstance(node, ast.While)
                                      and uses_name(node.test, "cache_slot")))
            names = {"route_slot", "phys_row", "scale_row_base", "scale_offset"}
            loop.body = [node for node in loop.body if (
                isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id in names
                    or isinstance(target, ast.Subscript) and uses_name(target, "scale_storage")
                    for target in node.targets)) or (
                isinstance(node, ast.AugAssign) and uses_name(node.target, "cache_slot"))]
            # Keep the source's actual loop condition, addresses, output index
            # and increment. Only quantizer arithmetic is omitted here.
            return loop

        addresses = ast.If(test=copy.deepcopy(branch.test),
                           body=[address_loop(branch.body)],
                           orelse=[address_loop(branch.orelse)])

        def compiled(nodes):
            return compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                           str(KERNEL), "exec")

        self.produce = compiled([copy.deepcopy(producer)])
        self.select = compiled(select)
        self.choose = compiled([choose])
        self.addresses = compiled([addresses])
        self.memory = {}
        self.reads = []
        self.writes = []
        self.row_counts = [0] * 72
        self.global_writes = []

        def load(address):
            self.reads.append(address)
            value = self.memory[address]
            return value if value < 0x80000000 else value - 0x100000000

        def save(address, value):
            self.writes.append((address, value & 0xFFFFFFFF))
            self.memory[address] = value & 0xFFFFFFFF

        def atomic(pointer, value):
            name, expert = pointer
            if name != "expert_rows" or not 0 <= expert < 72 or value != 1:
                raise AssertionError("unexpected allocation")
            old = self.row_counts[expert]
            self.row_counts[expert] += value
            return old

        def global_store(pointer, value):
            self.global_writes.append((pointer, value.bits if isinstance(value, FloatBits) else value))

        self.base = dict(Int32=int, Uint32=Uint32, cutlass=SimpleNamespace(Float32=f32),
                         _ld_shared_i32=load, _st_shared_i32=save,
                         get_ptr_as_int64=lambda tensor, index: (tensor, index),
                         atomic_add_global_i32=atomic, st_global_i32=global_store,
                         st_global_f32=global_store,
                         route_phys_rows_addr=self.ROWS, route_scales_addr=self.SCALES,
                         route_expert_ids_addr=self.SCALES, expert_scales_addr=self.EXPERTS,
                         expert_write_rows="expert_rows", token_map="tokens", token_weights="weights",
                         expert_tile_base=[expert * 64 for expert in range(72)],
                         self=SimpleNamespace(tile_shape_mnk=(128, 128, 128)),
                         num_experts=72, num_topk=8, num_k_tiles=64)

    def experts(self, bits):
        for expert, word in enumerate(bits):
            self.memory[self.EXPERTS + expert * 4] = word

    def allocate(self, warp, experts, weights=None, *, lane=0):
        if len(experts) > 8:
            raise ValueError("at most eight source routes")
        weights = list(weights) if weights is not None else [f32(1.0)] * len(experts)
        ids = list(experts) + [72] * (8 - len(experts))
        weights += [f32(0.0)] * (8 - len(weights))
        prefix = [ExpertId(72)] * (warp * 8)
        namespace = dict(self.base, lane_id=lane, token_idx=warp,
                         route_slot_base=warp * 32,
                         topk_ids=prefix + [ExpertId(expert) for expert in ids],
                         topk_weights=[f32(0.0)] * (warp * 8) + weights)
        exec(self.produce, namespace)

    def consume(self, warp, *, lane=0):
        namespace = dict(self.base, lane_id=lane, route_slot_base=warp * 32,
                         first_gs=object(), route_scales_equal=object(),
                         producer_first_gs=object(), producer_scales_equal=object())
        exec(self.select, namespace)
        local_topk = namespace["local_topk"]
        if local_topk == 0:
            return [], None
        words = []
        for slot in range(local_topk):
            namespace.update(route_slot=warp * 32 + slot, cache_slot=slot)
            exec(self.choose, namespace)
            words.append(namespace["gs_value"].bits)
        return words, namespace["route_scales_equal"]

    def state(self, warp):
        return self.memory[self.SCALES + (warp * 32 + 31) * 4]

    def consume_offsets(self, warp, sf_index):
        writes = []

        class Storage:
            def __setitem__(self, index, value):
                if value != 0xA5:
                    raise AssertionError("scale payload changed")
                writes.append(index)

        namespace = dict(self.base, lane_id=sf_index % 32, route_slot_base=warp * 32,
                         sf_idx=sf_index, cache_slot=0, scale_storage=Storage(), scale_byte=0xA5)
        exec(self.select, namespace)
        if namespace["local_topk"]:
            exec(self.addresses, namespace)
        return writes

    def active(self, warp, batch_base, tokens):
        return eval(self.active_guard, dict(Int32=int, warp_idx=warp,
                    producer_batch_tokens=4, token_idx=batch_base + warp, num_tokens=tokens))


def scalar_scale_offset(physical_row, sf_index):
    """Independent original layout: divide into physical tile/row and K fields."""
    physical_tile, tile_row = divmod(physical_row, 128)
    k_tile, inner_k = divmod(sf_index, 4)
    inner_m, outer_m = divmod(tile_row, 32)
    return (physical_tile * 64 * 512 + k_tile * 512
            + outer_m * 16 + inner_m * 4 + inner_k)


def old_quantizer_inputs(selected):
    if not selected:
        return [], None
    equal = all(FloatBits(bits) == FloatBits(selected[0]) for bits in selected[1:])
    return ([selected[0]] * len(selected) if equal else selected), int(equal)


class GuardedWeights:
    """Reading an invalid route's weight fails before any float conversion."""
    def __init__(self, ids, bits):
        self.ids, self.bits, self.reads = ids, bits, []

    def __getitem__(self, index):
        if not 0 <= self.ids[index] < 72:
            raise AssertionError('poisoned remote weight was read')
        self.reads.append(index)
        return FloatBits(self.bits[index])


class RouteWeightReadTests(unittest.TestCase):
    def test_actual_histogram_and_producer_skip_poison_and_preserve_valid_weight_bits(self):
        tree = ast.parse(KERNEL.read_text())
        method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                      and node.name == 'initialize_route_q0_and_publish')
        histogram = next(node for node in ast.walk(method) if isinstance(node, ast.While)
                         and ast.unparse(node.test) == 'hist_idx < total_pairs')
        histogram_code = compile(ast.Module(body=[histogram], type_ignores=[]), str(KERNEL), 'exec')
        # Include both integer boundaries, the E72 sentinel, duplicate local
        # IDs, signed zeros, subnormals, infinities and raw NaN payloads.
        fixtures = (
            ([-1, 72, -(1 << 31), (1 << 31)-1, 73, -2, 72, -1], [0x7FA12345]*8),
            ([0, 0, 71, 71, 1, 1, 2, 2],
             [0, 0x80000000, 0x7FC12345, 0x7FA00001, 1, 0x80000001, 0x7F800000, 0xFF800000]),
            ([72, 71, -1, 71, 0, 72, 0, 1],
             [0x7FA12345, 0x3F800000, 0x7FC12345, 0xBF800000, 0x80000000, 0, 0x7FCABCDE, 0]),
        )
        for ids, bits in fixtures:
            with self.subTest(ids=ids):
                # Reference selection reads only ordinary test-side values,
                # independent of the actual source's control-flow nesting.
                selected = [i for i, expert in enumerate(ids)
                            if 0 <= expert < 72 and FloatBits(bits[i]).number() != 0.0]
                valid = [i for i, expert in enumerate(ids) if 0 <= expert < 72]
                expected_counts = [sum(ids[i] == expert for i in selected) for expert in range(72)]
                counts = [0]*72
                weights = GuardedWeights(ids, bits)
                def increment(address, value):
                    self.assertTrue(0 <= address < 72*4 and address % 4 == 0)
                    self.assertEqual(value, 1)
                    counts[address//4] += value
                # Non-unit cooperative stride proves no route is skipped or
                # visited twice when validity varies across histogram lanes.
                for start in range(3):
                    namespace = dict(Int32=int, cutlass=SimpleNamespace(Float32=f32),
                                     hist_idx=start, flat_stride=3, total_pairs=8, num_experts=72,
                                     topk_ids=[ExpertId(expert) for expert in ids], topk_weights=weights,
                                     route_hist_addr=0, atomic_add_shared_i32=increment)
                    exec(histogram_code, namespace)
                self.assertEqual(sorted(weights.reads), valid)
                self.assertEqual(counts, expected_counts)

                cache = SourceCache()
                scale_bits = [0x3F800000+expert for expert in range(72)]
                cache.experts(scale_bits)
                weights = GuardedWeights(ids, bits)
                namespace = dict(cache.base, lane_id=0, token_idx=0, route_slot_base=0,
                                 topk_ids=[ExpertId(expert) for expert in ids], topk_weights=weights)
                exec(cache.produce, namespace)
                self.assertEqual(weights.reads, valid)
                self.assertEqual(cache.row_counts, expected_counts)
                prior = [0]*72
                expected_writes = []
                for index in selected:
                    expert = ids[index]
                    row = (expert*64 + prior[expert]//128)*128 + prior[expert]%128
                    prior[expert] += 1
                    expected_writes += [(('tokens', row), 0), (('weights', row), bits[index])]
                self.assertEqual(cache.global_writes, expected_writes)
                self.assertEqual(cache.state(0) & 15, len(selected))
                self.assertEqual(cache.consume(0),
                                 old_quantizer_inputs([scale_bits[ids[index]] for index in selected]))


class RouteScaleCacheTests(unittest.TestCase):
    def test_selected_raw_bits_match_previous_quantizer_inputs(self):
        rng = random.Random(478)
        bits = [0, 0x80000000, 0x3F800000, 0xBF800000, 0x7F800000, 0xFF800000,
                0x7FC00001, 0x7FC12345, 0x7FA00001, 1, 0x80000001, 0x7F7FFFFF]
        bits += [rng.getrandbits(32) for _ in range(72 - len(bits))]
        cache = SourceCache()
        cache.experts(bits)
        routes = [[], [0], [1, 0], [0, 1], [6], [8], [6, 6], [8, 8],
                  [6, 2], [2, 6], [2, 8], [8, 2], [2] * 8,
                  [0, 2, 6, 8, 11, 71, 2, 1]]
        for length in range(2, 9):
            routes += [[0, 1] * (length // 2) + [0] * (length % 2),
                       [1, 0] * (length // 2) + [1] * (length % 2),
                       [6] + [2] * (length - 1),
                       [2] * (length - 1) + [6]]
        routes += [[rng.randrange(72) for _ in range(length)]
                   for length in range(9) for _ in range(20)]
        for warp in range(4):
            for selected in routes:
                cache.allocate(warp, selected)
                expected = old_quantizer_inputs([bits[expert] for expert in selected])
                self.assertEqual(cache.consume(warp), expected)
                self.assertEqual(cache.state(warp), len(selected) | ((1 if not selected else expected[1]) << 4))

    def test_each_warp_publishes_only_selected_words_and_packed_count_slot(self):
        cache = SourceCache()
        bits = [0x3F800000 + expert for expert in range(72)]
        canaries = {address: 0xA5A50000 + address for address in range(0, cache.EXPERTS, 4)}
        cache.memory.update(canaries)
        cache.experts(bits)
        for warp in range(4):
            cache.allocate(warp, list(range(warp * 8, warp * 8 + 8)))
        for warp in range(4):
            for lane in range(32):
                self.assertEqual(cache.consume(warp, lane=lane),
                                 (bits[warp * 8:warp * 8 + 8], 0))
            self.assertEqual(cache.state(warp), 8)
        expected_writes = {
            base + (warp * 32 + slot) * 4
            for base in (cache.ROWS, cache.SCALES) for warp in range(4) for slot in range(8)
        } | {cache.ROWS + (warp * 32 + slot) * 4 for warp in range(4) for slot in range(8, 16)
        } | {cache.SCALES + (warp * 32 + 31) * 4 for warp in range(4)}
        self.assertEqual({address for address, _ in cache.writes}, expected_writes)
        self.assertEqual(len(cache.writes), len(expected_writes))
        for address, word in canaries.items():
            if address not in expected_writes:
                self.assertEqual(cache.memory[address], word)
        self.assertEqual([cache.memory[cache.EXPERTS + expert * 4] for expert in range(72)], bits)

    def test_scales_changed_at_same_addresses_replace_previous_batch(self):
        cache = SourceCache()
        transitions = (
            ([0x3F800000] * 72, list(range(8))),
            ([0x7FC00001] * 72, []),
            ([0x7FA00001] * 72, [71]),
            ([0x40000000 + expert for expert in range(72)], list(range(8))),
            ([0x80000000] * 72, [71, 3]),
            ([0x3F800000] * 72, list(range(8))),
        )
        for bits, selected in transitions:
            cache.experts(bits)
            for warp in range(4):
                cache.allocate(warp, selected)
            expected = old_quantizer_inputs([bits[expert] for expert in selected])
            for warp in range(4):
                for lane in range(32):
                    self.assertEqual(cache.consume(warp, lane=lane), expected)
                self.assertEqual(cache.state(warp), len(selected) | ((1 if not selected else expected[1]) << 4))

    def test_no_local_routes_read_no_stale_scale_slots(self):
        cache = SourceCache()
        cache.experts([0x7FC00001] * 72)
        cache.allocate(0, [0] * 8)
        cache.allocate(0, [])
        self.assertEqual(cache.state(0), 16)
        for lane in range(32):
            cache.reads.clear()
            self.assertEqual(cache.consume(0, lane=lane), ([], None))
            self.assertEqual(cache.reads, [cache.SCALES + 31 * 4])

    def test_actual_filter_compacts_zero_through_eight_routes_before_comparison(self):
        cache = SourceCache()
        bits = [0x3F800000 + expert for expert in range(72)]
        bits[71] = 0x7FC01234
        cache.experts(bits)
        # Filling later source slots first checks that the first selected
        # route, rather than topk_slot==0, initializes producer_first_gs.
        order = (7, 3, 5, 1, 6, 2, 4, 0)
        for count in range(9):
            ids = [-1, 72, 3, 4, -1, 72, 5, 6]
            weights = [f32(1), f32(1), f32(0), f32(-0.0)] * 2
            for index, slot in enumerate(order[:count]):
                ids[slot] = 71 if index == 0 else index
                weights[slot] = f32(1)
            chosen = [expert for expert, weight in zip(ids, weights)
                      if 0 <= expert < 72 and weight != f32(0)]
            previous_rows = list(cache.row_counts)
            cache.reads.clear()
            cache.global_writes.clear()
            cache.allocate(0, ids, weights)
            self.assertEqual(cache.reads, [cache.EXPERTS + expert * 4 for expert in chosen])
            expected_global = []
            for slot, expert in enumerate(chosen):
                row = expert * 64 * 128 + previous_rows[expert]
                previous_rows[expert] += 1
                expected_global += [(('tokens', row), 0), (('weights', row), f32(1).bits)]
                self.assertEqual(cache.memory[cache.ROWS + slot * 4], row)
                self.assertEqual(cache.memory[cache.SCALES + slot * 4], bits[expert])
            self.assertEqual(cache.global_writes, expected_global)
            self.assertEqual(cache.row_counts, previous_rows)
            self.assertEqual(cache.consume(0), old_quantizer_inputs([bits[expert] for expert in chosen]))
            self.assertEqual(cache.state(0) & 15, count)

    def test_equal_routes_need_only_existing_state_and_first_scale_reads(self):
        cache = SourceCache()
        cache.experts([0x3F800000] * 72)
        for count in range(1, 9):
            cache.reads.clear()
            cache.writes.clear()
            cache.allocate(0, list(range(count)))
            self.assertEqual(cache.reads, [cache.EXPERTS + expert * 4 for expert in range(count)])
            # Each selected route publishes row, row-only SFA base and raw
            # scale; equality still shares the single count-word store.
            self.assertEqual(len(cache.writes), count * 3 + 1)
            for lane in range(32):
                cache.reads.clear()
                self.assertEqual(cache.consume(0, lane=lane), ([0x3F800000] * count, 1))
                self.assertEqual(cache.reads, [cache.SCALES + 31 * 4, cache.SCALES])

    def test_only_lane_zero_publishes_before_existing_warp_barrier(self):
        cache = SourceCache()
        producer, barriers, consumer = cache.publication_order
        self.assertEqual(len(barriers), 1)
        self.assertLess(producer, barriers[0])
        self.assertLess(barriers[0], consumer)
        batch_barriers, dispatch = cache.batch_publication_order
        self.assertEqual(len(batch_barriers), 1)
        self.assertLess(batch_barriers[0], dispatch)
        cache.experts([0x3F800000] * 72)
        for lane in range(1, 32):
            cache.allocate(0, list(range(8)), lane=lane)
        self.assertEqual(cache.reads, [])
        self.assertEqual(cache.writes, [])
        self.assertEqual(cache.global_writes, [])
        self.assertEqual(cache.row_counts, [0] * 72)


class RouteScaleAddressCacheTests(unittest.TestCase):
    def test_actual_published_bases_and_both_consumers_cover_all_warps_counts_and_sf(self):
        for bits in ([0x3F800000] * 72, [0x3F800000 + expert for expert in range(72)],
                     [0x7FA12345] * 72):
            cache = SourceCache()
            cache.experts(bits)
            for warp in range(4):
                for count in range(9):
                    routes = ([71, 0, 71, 1, 31, 32, 0, 5])[:count]
                    # Unused route metadata is deliberately unreadable. The
                    # actual producer must replace every consumed cache word.
                    for slot in range(32):
                        cache.memory[cache.ROWS + (warp * 32 + slot) * 4] = object()
                    cache.allocate(warp, routes)
                    rows = [cache.memory[cache.ROWS + (warp * 32 + slot) * 4]
                            for slot in range(count)]
                    for slot, row in enumerate(rows):
                        self.assertEqual(cache.memory[cache.ROWS + (warp * 32 + slot + 8) * 4],
                                         scalar_scale_offset(row, 0))
                    for sf_index in range(256):
                        self.assertEqual(cache.consume_offsets(warp, sf_index),
                                         [scalar_scale_offset(row, sf_index) for row in rows])

    def test_unwritten_and_stale_row_bases_cannot_hide_missing_publication(self):
        cache = SourceCache()
        cache.experts([0x3F800000 + expert for expert in range(72)])
        for warp in range(4):
            cache.allocate(warp, list(range(8)))
            old = cache.consume_offsets(warp, 255)
            cache.allocate(warp, [0])
            # The row counter advances at the same shared addresses; a stale
            # base would now produce the old row's scale address.
            current = cache.consume_offsets(warp, 255)
            self.assertNotEqual(current[0], old[0])
            for slot in range(9, 16):
                del cache.memory[cache.ROWS + (warp * 32 + slot) * 4]
            self.assertEqual(cache.consume_offsets(warp, 255), current)
            active_base = cache.ROWS + (warp * 32 + 8) * 4
            del cache.memory[active_base]
            with self.assertRaises(KeyError):
                cache.consume_offsets(warp, 255)
            cache.allocate(warp, [])
            cache.reads.clear()
            self.assertEqual(cache.consume_offsets(warp, 255), [])
            self.assertEqual(cache.reads, [cache.SCALES + (warp * 32 + 31) * 4])

    def test_actual_active_token_guard_excludes_odd_tail_and_idle_warps(self):
        cache = SourceCache()
        for tokens in (4096, 4097, 6912, 8192, 16384):
            for batch in (0, tokens - tokens % 4, tokens + 4):
                for warp in range(9):
                    self.assertEqual(cache.active(warp, batch, tokens),
                                     warp < 4 and batch + warp < tokens)


if __name__ == "__main__":
    unittest.main()
