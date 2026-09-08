"""CPU raw-bit oracle for Q0's shared selected-scale cache."""
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


class Uint32(int):
    def __new__(cls, value):
        return super().__new__(cls, value & 0xFFFFFFFF)

    def bitcast(self, _):
        return FloatBits(self)


def f32(value):
    return FloatBits(struct.unpack("<I", struct.pack("<f", value))[0])


def uses_name(node, name):
    return any(isinstance(item, ast.Name) and item.id == name for item in ast.walk(node))


class SourceCache:
    """Execute the actual store, selection loop and selected quantizer input."""
    def __init__(self):
        tree = ast.parse(KERNEL.read_text())
        method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                      and node.name == "initialize_route_q0_and_publish")
        store = next(node for node in ast.walk(method) if isinstance(node, ast.Expr)
                     and isinstance(node.value, ast.Call)
                     and isinstance(node.value.func, ast.Name)
                     and node.value.func.id == "_st_shared_i32"
                     and uses_name(node.value.args[0], "route_scales_addr"))
        selected = next(node for node in ast.walk(method) if isinstance(node, ast.If)
                        and ast.unparse(node.test) == "local_topk > Int32(0)")
        scale_loop = next(i for i, node in enumerate(selected.body)
                          if isinstance(node, ast.While))
        select = selected.body[:scale_loop + 1]
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

        def compiled(nodes):
            return compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                           str(KERNEL), "exec")

        self.store = compiled([store])
        self.select = compiled(select)
        self.choose = compiled([choose])
        self.memory = {}
        self.reads = []

        def load(address):
            self.reads.append(address)
            value = self.memory[address]
            return value if value < 0x80000000 else value - 0x100000000

        def save(address, value):
            self.memory[address] = value & 0xFFFFFFFF

        self.ns = dict(Int32=int, Uint32=Uint32, cutlass=SimpleNamespace(Float32=f32),
                       _ld_shared_i32=load, _st_shared_i32=save,
                       route_scales_addr=1152, expert_scales_addr=2304)

    def experts(self, bits):
        for expert, word in enumerate(bits):
            self.memory[2304 + expert * 4] = word

    def allocate(self, warp, experts):
        for slot, expert in enumerate(experts):
            self.ns.update(route_slot=warp * 32 + slot, expert_id=expert)
            exec(self.store, self.ns)

    def consume(self, warp, local_topk):
        if not local_topk:
            return [], None
        self.ns.update(route_slot_base=warp * 32, local_topk=local_topk)
        exec(self.select, self.ns)
        words = []
        for slot in range(local_topk):
            self.ns.update(route_slot=warp * 32 + slot, cache_slot=slot)
            exec(self.choose, self.ns)
            words.append(self.ns["gs_value"].bits)
        return words, self.ns["route_scales_equal"]


def old_quantizer_inputs(selected):
    if not selected:
        return [], None
    equal = all(FloatBits(bits) == FloatBits(selected[0]) for bits in selected[1:])
    return ([selected[0]] * len(selected) if equal else selected), int(equal)


class RouteScaleCacheTests(unittest.TestCase):
    def test_selected_raw_bits_match_previous_quantizer_inputs(self):
        rng = random.Random(478)
        bits = [0, 0x80000000, 0x3F800000, 0xBF800000, 0x7F800000, 0xFF800000,
                0x7FC00001, 0x7FC12345, 0x7FA00001, 1, 0x80000001, 0x7F7FFFFF]
        bits += [rng.getrandbits(32) for _ in range(72 - len(bits))]
        cache = SourceCache()
        cache.experts(bits)
        routes = [[], [0], [1, 0], [0, 1], [6], [6, 6], [8, 8], [2] * 8,
                  [0, 2, 6, 8, 11, 71, 2, 1]]
        routes += [[rng.randrange(72) for _ in range(length)]
                   for length in range(9) for _ in range(20)]
        for warp in range(4):
            for selected in routes:
                cache.allocate(warp, selected)
                expected = old_quantizer_inputs([bits[expert] for expert in selected])
                self.assertEqual(cache.consume(warp, len(selected)), expected)

    def test_each_warp_keeps_selected_words_and_count_slot_separate(self):
        cache = SourceCache()
        bits = [0x3F800000 + expert for expert in range(72)]
        cache.experts(bits)
        for warp in range(4):
            # The 31st slot still belongs to the integer local_topk count.
            cache.memory[1152 + (warp * 32 + 31) * 4] = 8
            cache.allocate(warp, list(range(warp * 8, warp * 8 + 8)))
        for warp in range(4):
            self.assertEqual(cache.consume(warp, 8),
                             (bits[warp * 8:warp * 8 + 8], 0))
            self.assertEqual(cache.memory[1152 + (warp * 32 + 31) * 4], 8)
        self.assertEqual([cache.memory[2304 + expert * 4] for expert in range(72)], bits)
        self.assertTrue(all(1152 <= address < 2592 for address in cache.memory))

    def test_scales_changed_at_same_addresses_replace_previous_batch(self):
        cache = SourceCache()
        first = [0x3F800000] * 72
        second = [0x40000000 + expert for expert in range(72)]
        cache.experts(first)
        for warp in range(4):
            cache.allocate(warp, [1, 2, 3, 71])
            self.assertEqual(cache.consume(warp, 4), ([first[1]] * 4, 1))
        cache.experts(second)
        for warp in range(4):
            cache.allocate(warp, [71, 3])
            for _ in range(8):
                self.assertEqual(cache.consume(warp, 2), ([second[71], second[3]], 0))

    def test_no_local_routes_read_no_stale_scale_slots(self):
        cache = SourceCache()
        cache.experts([0x7FC00001] * 72)
        cache.allocate(0, [0] * 8)
        cache.reads.clear()
        self.assertEqual(cache.consume(0, 0), ([], None))
        self.assertEqual(cache.reads, [])


if __name__ == "__main__":
    unittest.main()
