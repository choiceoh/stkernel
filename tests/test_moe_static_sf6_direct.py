"""CPU execution of static SF6 DMA/unpack helpers; no CUDA numerics claim."""
from __future__ import annotations

import ast
import copy
import importlib.util
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "overlay/modules/glm53_moe"
SOURCE = (MODULE / "moe_static_kernel_v4.py").read_text()
TREE = ast.parse(SOURCE)
CLASS = next(node for node in TREE.body if isinstance(node, ast.ClassDef))
spec = importlib.util.spec_from_file_location("static_sf6_pack_oracle", MODULE / "moe_reform_sf_pack.py")
sf6 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sf6
spec.loader.exec_module(sf6)


class BarrierYields(ast.NodeTransformer):
    """Run the production method as 128 coroutines with real phase barriers."""
    def visit_Expr(self, node):
        if (isinstance(node.value, ast.Call)
                and ast.unparse(node.value.func) == "self.sf_expand_barrier.arrive_and_wait"):
            return ast.copy_location(ast.Expr(ast.Yield(ast.Constant(None))), node)
        return self.generic_visit(node)


def method(name, env, *, barriers=False):
    node = copy.deepcopy(next(n for n in CLASS.body if isinstance(n, ast.FunctionDef) and n.name == name))
    node.decorator_list = []
    if barriers:
        node = BarrierYields().visit(node)
    module = ast.Module(body=[ast.ImportFrom(module="__future__",
        names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(MODULE / "moe_static_kernel_v4.py"), "exec"), env)
    return env[name]


def geometry(reform, stages=2):
    env = dict(cutlass=SimpleNamespace(Float32=object()), DenseGemmKernel=object(),
        utils=SimpleNamespace(get_smem_capacity_in_bytes=lambda _: 101376),
        pipeline=SimpleNamespace(NamedBarrier=lambda **kw: SimpleNamespace(**kw)),
        is_gated_activation=lambda _: True)
    for node in TREE.body:
        if isinstance(node, ast.Assign):
            try:
                env[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, AttributeError):
                pass
    obj = SimpleNamespace()
    method("__init__", env)(obj, 16, 4, reform_sf_pack=True,
                             decode_reform=reform, fc2_stages=stages)
    return obj


class SharedMachine:
    def __init__(self, size=16384):
        self.mem = bytearray([0xA5]) * size
        self.gmem = b""
        self.loads, self.stores, self.copies = [], [], []
        env = dict(Int32=int, Int64=int, _ld_shared_i32_volatile=self.load,
                   _st_shared_i32=self.store, _bulk_g2s=self.bulk)
        self.expand = method("_sf_expand_stage", env, barriers=True)
        self.copy_half = method("_sf6_copy_fc2_half", env)

    def load(self, addr):
        assert addr % 4 == 0 and 0 <= addr <= len(self.mem)-4
        self.loads.append(addr)
        return int.from_bytes(self.mem[addr:addr+4], "little", signed=True)

    def store(self, addr, value):
        assert addr % 4 == 0 and 0 <= addr <= len(self.mem)-4
        self.stores.append(addr)
        self.mem[addr:addr+4] = (value & 0xFFFFFFFF).to_bytes(4, "little")

    def bulk(self, dest, source, size, barrier):
        assert dest % 16 == source % 16 == size % 16 == 0
        assert 0 <= source <= len(self.gmem)-size and 0 <= dest <= len(self.mem)-size
        self.mem[dest:dest+size] = self.gmem[source:source+size]
        self.copies.append((dest, source, size, barrier))

    def expand_stage(self, addr, block_bytes, seed=13):
        rng = random.Random(seed)
        workers = [self.expand(SimpleNamespace(), addr, t, block_bytes) for t in range(128)]
        order = list(range(128))
        before_stores = len(self.stores)
        rng.shuffle(order)
        for i in order:
            next(workers[i])
        # No owner can write until every owner's packed bytes are retained.
        assert len(self.stores) == before_stores
        rng.shuffle(order)
        for i in order:
            next(workers[i])
        for worker in workers:
            try:
                next(worker)
            except StopIteration:
                continue
            raise AssertionError("unexpected third unpack barrier")
        assert set(self.stores[before_stores:]) == set(range(addr, addr+block_bytes, 4))


def original_offset(row, col, rows, k, *, shared=False):
    # Independent NVFP4 layout: ((32,4),row128), ((16,4),K64).
    row_stride = 512 if shared else 128*k//16
    k_stride = (rows//128)*512 if shared else 512
    return ((row//128)*row_stride + (col//4)*k_stride
            + (row%32)*16 + ((row//32)%4)*4 + col%4)


class StaticDirectScales(unittest.TestCase):
    def test_ordinary_and_reform_geometry_transaction_contract(self):
        for reform in (False, True):
            for stages in (1, 2, 3):
                g = geometry(reform, stages)
                self.assertEqual(g.sf1_block_bytes, 2048 if reform else 4096)
                self.assertEqual(g.sf1_packed_blocks*1552, 1552 if reform else 3104)
                self.assertEqual(g.sf2_block_bytes, 2048 if reform else 1024)
                self.assertEqual(g.sf2_stage_bytes, 1552 if reform else 784)
                self.assertFalse(hasattr(g, "sf2_extra_bytes"))

    def test_fc1_two_independent_bases_expand_without_neighbor_or_ring_alias(self):
        rng = random.Random(9109)
        for slot in (0, 1, 2, 0):
            machine = SharedMachine()
            base = 256 + slot*4096
            raws = [bytes(b + rng.randrange(64) for _ in range(2048)) for b in (0, 192)]
            for half, raw in enumerate(raws):
                packed = sf6.pack_stage_bytes(raw)
                machine.mem[base+half*2048:base+half*2048+1552] = packed
            before = bytes(machine.mem)
            for half in range(2):
                machine.expand_stage(base+half*2048, 2048, slot+half)
            self.assertEqual(machine.mem[base:base+4096], b"".join(raws))
            self.assertEqual(machine.mem[:base], before[:base])
            self.assertEqual(machine.mem[base+4096:], before[base+4096:])

    def test_legacy_q_4096_in_place_expansion_is_unchanged(self):
        raw = bytes(128+(i*31)%64 for i in range(4096))
        packed = bytearray(3088)
        for i, code in enumerate(raw):
            value = code-128
            packed[i//2] |= (value & 15) << ((i%2)*4)
            packed[2048+i//4] |= (value >> 4) << ((i%4)*2)
        packed[3072] = 128
        machine = SharedMachine()
        machine.mem[512:512+3088] = packed
        before = bytes(machine.mem)
        machine.expand_stage(512, 4096)
        self.assertEqual(machine.mem[512:4608], raw)
        self.assertEqual(machine.mem[:512], before[:512])
        self.assertEqual(machine.mem[4608:], before[4608:])

    def test_fc2_selective_bulk_and_actual_unpack_all_codes_halves_and_ring_slots(self):
        rng = random.Random(784)
        for low in (0, 64, 128, 192, 255):
            raw = bytes(low + rng.randrange(min(64, 256-low)) for _ in range(2048))
            packed = sf6.pack_stage_bytes(raw)
            machine = SharedMachine()
            # Multiple experts/tiles in the real packed stride; address is
            # deliberately nonzero, preserving 16-byte source alignment.
            machine.gmem = packed * 7
            for generation, (slot, half) in enumerate(((0,0), (1,1), (2,0), (0,1))):
                dest, source, barrier = 256+slot*1024, (generation+1)*1552, 0x10+slot
                before = bytes(machine.mem)
                n = len(machine.copies)
                machine.copy_half(SimpleNamespace(), dest, source, barrier, half)
                copies = machine.copies[n:]
                self.assertEqual(len(copies), 5)
                self.assertEqual(sum(row[2] for row in copies), 784)
                self.assertEqual({row[3] for row in copies}, {barrier})
                written = [i for d, _, n, _ in copies for i in range(d, d+n)]
                self.assertEqual(sorted(written), list(range(dest, dest+784)))
                machine.expand_stage(dest, 1024, generation)
                expected = raw[half*512:half*512+512] + raw[1024+half*512:1536+half*512]
                self.assertEqual(machine.mem[dest:dest+1024], expected)
                self.assertEqual(machine.mem[:dest], before[:dest])
                self.assertEqual(machine.mem[dest+1024:], before[dest+1024:])

    def test_source_map_matches_actual_mma_byte_positions_for_both_geometries(self):
        for reform in (False, True):
            for kind, rows, k, rn, kn in (("fc1", 1024, 4096, 128, 256 if reform else 512),
                                          ("fc2", 4096, 512, 256 if reform else 128, 128)):
                nr, nk = rows//rn, k//kn
                for expert, rt, kt in ((0,0,0), (1,1,1), (2,nr-1,nk-1)):
                    seen = set()
                    for row in range(rn):
                        for col in range(kn//16):
                            dest = original_offset(row, col, rn, kn, shared=True)
                            seen.add(dest)
                            expected = expert*rows*k//16 + original_offset(rt*rn+row,
                                kt*(kn//16)+col, rows, k)
                            if reform:
                                actual = sf6.stage_source_offset(rows,k,kind,expert,rt,kt,dest)
                            elif kind == "fc1":
                                actual = sf6.stage_source_offset(rows,k,kind,expert,rt,
                                    kt*2+dest//2048,dest%2048)
                            else:
                                actual = sf6.stage_source_offset(rows,k,kind,expert,rt//2,kt,
                                    (dest//512)*1024+(rt%2)*512+dest%512)
                            self.assertEqual(actual, expected)
                    self.assertEqual(seen, set(range(rn*kn//16)))

    def test_no_original_scale_tensor_is_created_outside_raw_only_branch(self):
        for name in ("moe_static_kernel_v4.py", "moe_static_kernel_v5.py"):
            tree = ast.parse((MODULE/name).read_text())
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
            call = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__call__")
            parents = {child: parent for parent in ast.walk(call) for child in ast.iter_child_nodes(parent)}
            uses = [n for n in ast.walk(call) if isinstance(n, ast.Name)
                    and n.id in ("sfb_w13_ptr", "sfb_down_ptr")]
            self.assertEqual(len(uses), 2)
            for node in uses:
                ancestors = []
                while node in parents:
                    node = parents[node]
                    ancestors.append(node)
                self.assertTrue(any(isinstance(n, ast.If) and
                    ast.unparse(n.test) == "cutlass.const_expr(not self.reform_sf_pack)" for n in ancestors))


if __name__ == "__main__":
    unittest.main()
