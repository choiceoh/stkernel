"""CPU-only byte maps and real producer-method control flow for dynamic SF6.

The production unpack/producer AST runs with integer pointers and fake DMA.
This proves byte identity, no raw-global reconstruction, and stage publication
ordering. It does not claim CuTe compilation, CUDA synchronization or speed.
"""
import ast
import hashlib
import importlib.util
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "overlay/modules/glm53_moe/moe_dynamic_gated_sf6.py"
TEXT = SOURCE.read_text()
TREE = ast.parse(TEXT)
CLASS = next(n for n in TREE.body if isinstance(n, ast.ClassDef))

spec = importlib.util.spec_from_file_location("dynamic_sf6_cpu_pack", ROOT / "overlay/modules/glm53_moe/moe_reform_sf_pack.py")
pack = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pack
spec.loader.exec_module(pack)


def body(name):
    return next(n for n in [*TREE.body, *CLASS.body] if isinstance(n, ast.FunctionDef) and n.name == name)


class StripDSL(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        node.decorator_list = []
        return self.generic_visit(node)

    def visit_Call(self, node):
        # Python's range does not have the DSL-only unroll keyword. This
        # transformation preserves its actual start/stop/step and loop body.
        if isinstance(node.func, ast.Name) and node.func.id == "range":
            node.keywords = [kw for kw in node.keywords if kw.arg != "unroll"]
        return self.generic_visit(node)


def functions(names, namespace=None):
    # Reparse because the transformer must never mutate the audit AST.
    parsed = ast.parse(TEXT)
    nodes = parsed.body + next(n for n in parsed.body if isinstance(n, ast.ClassDef)).body
    chosen = [n for n in nodes if isinstance(n, ast.FunctionDef) and n.name in names]
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *chosen], type_ignores=[])
    module = ast.fix_missing_locations(StripDSL().visit(module))
    values = dict(Int32=int, Int64=int,
                  cutlass=SimpleNamespace(const_expr=bool, range_constexpr=range, Uint8="uint8"),
                  cute=SimpleNamespace(make_rmem_tensor=lambda shape, dtype: [0] * shape[0],
                                       size=lambda value: value if isinstance(value, int) else __import__("math").prod(value)))
    values.update(namespace or {})
    exec(compile(module, str(SOURCE), "exec"), values)
    return values


class ByteMapTests(unittest.TestCase):
    def test_actual_unpack_exact_for_both_physical_halves_and_all_lanes(self):
        rng = random.Random(932)
        for kind in ("fc1", "fc2"):
            for half in (0, 1):
                for base in (0, 73, 192):
                    raw = bytes(base + rng.randrange(64) for _ in range(2048))
                    encoded = pack.pack_stage_bytes(raw)
                    self.assertIsNotNone(encoded)
                    start = 2**35 + 1552 * 713
                    memory, reads, stores = bytearray(1024), [], set()
                    def load(address):
                        offset = address - start
                        self.assertEqual(offset % 4, 0)
                        self.assertTrue(0 <= offset <= 1548)
                        reads.extend(range(offset, offset + 4))
                        return int.from_bytes(encoded[offset:offset + 4], "little")
                    def store(address, value):
                        self.assertEqual(address % 4, 0)
                        self.assertFalse(set(range(address, address + 4)) & stores)
                        stores.update(range(address, address + 4))
                        memory[address:address + 4] = (value & 0xFFFFFFFF).to_bytes(4, "little")
                    ns = functions({"_sf6_expand_dynamic_tile", "dynamic_sf6_byte_index"},
                                   dict(_sf6_ld_global_u32=load, _st_shared_i32=store))
                    for lane in range(32):
                        ns["_sf6_expand_dynamic_tile"](start, 0, half, lane, kind == "fc2")
                    expected = bytes(raw[ns["dynamic_sf6_byte_index"](kind, half, i)] for i in range(1024))
                    self.assertEqual(memory, expected)
                    self.assertEqual(stores, set(range(1024)))
                    # 512 low-plane bytes, 256 high-plane bytes, one u32 base.
                    self.assertEqual(len(set(reads)), 772)
                    self.assertTrue(set(reads) <= set(range(1536)) | set(range(1536, 1540)))

    def test_dynamic_tiles_match_original_raw_layout_through_shared_sf6_format(self):
        ns = functions({"dynamic_sf6_stage_index", "dynamic_sf6_byte_index"})
        for kind, rows, k in (("fc1", 1024, 4096), ("fc1", 256, 512),
                              ("fc2", 4096, 512), ("fc2", 512, 128)):
            nr, nk = pack.stage_shape(rows, k, kind)
            for expert in (0, 1, 287):
                for row_tile in range(rows // 128):
                    for k_tile in range(k // 128):
                        stage, half = ns["dynamic_sf6_stage_index"](kind, rows, k, expert, row_tile, k_tile)
                        packed_expert, tile = divmod(stage, nr * nk)
                        rt, kt = divmod(tile, nk)
                        self.assertEqual(packed_expert, expert)
                        for byte in (0, 3, 31, 127, 511, 512, 515, 767, 1023):
                            original = pack.stage_source_offset(rows, k, kind, expert, rt, kt,
                                ns["dynamic_sf6_byte_index"](kind, half, byte))
                            expected = expert * rows * k // 16 + (row_tile * (k // 128) + k_tile) * 1024 + byte
                            self.assertEqual(original, expected, (kind, expert, row_tile, k_tile, byte))

    def test_bad_geometry_coordinates_and_plane_shapes_fail_closed(self):
        ns = functions({"dynamic_sf6_stage_index", "dynamic_sf6_byte_index", "_check_sf6_shapes"})
        for args in (("bad", 256, 256, 0, 0, 0), ("fc1", 128, 128, 0, 0, 0),
                     ("fc2", 128, 128, 0, 0, 0), ("fc1", 256, 256, -1, 0, 0),
                     ("fc1", 256, 256, 0, 2, 0), ("fc2", 256, 128, 0, 0, 1)):
            with self.assertRaises(ValueError):
                ns["dynamic_sf6_stage_index"](*args)
        for args in (("fc1", 2, 0), ("fc2", 0, 1024), ("fc1", 0, -1)):
            with self.assertRaises(ValueError):
                ns["dynamic_sf6_byte_index"](*args)
        kernel = SimpleNamespace(tile_shape_mnk=(128, 128, 128))
        w13, down = SimpleNamespace(shape=(1024, 4096, 288)), SimpleNamespace(shape=(4096, 512, 288))
        first = SimpleNamespace(shape=(288, 128, 1552), element_type="uint8")
        second = SimpleNamespace(shape=(288, 64, 1552), element_type="uint8")
        ns["_check_sf6_shapes"](kernel, w13, down, first, second)
        first.element_type = "fp8"
        with self.assertRaises(ValueError):
            ns["_check_sf6_shapes"](kernel, w13, down, first, second)


class Tensor:
    def __init__(self, name, pointer=0, shape=(1, 1, 1)):
        self.name, self.iterator, self.shape = name, pointer, shape

    def __getitem__(self, index):
        stage = index[-1] if isinstance(index, tuple) and isinstance(index[-1], int) else 0
        return Tensor(self.name, self.iterator + stage * 1024, self.shape)


class State:
    def __init__(self):
        self.index, self.count = 0, 0

    def reset_count(self):
        self.count = 0

    def advance(self):
        self.index = (self.index + 1) % 3
        self.count += 1


class FakePipeline:
    def __init__(self, events):
        self.events = events

    def producer_acquire(self, state, try_acquire_token=None):
        if try_acquire_token is not True:
            raise AssertionError("SF6 must wait separately, before shared writes")
        self.events.append(("publish", state.index))

    def producer_get_barrier(self, state):
        return state.index

    def producer_commit(self, state):
        self.events.append(("commit", state.index))


class ProducerTests(unittest.TestCase):
    def namespace(self):
        events = []
        kernel = SimpleNamespace(ab_storage_stage=2, _hidden_size=4096,
                                 pass_gate_barrier=SimpleNamespace(wait_unaligned=lambda: events.append(("alias_wait",))))
        cute = SimpleNamespace(arch=SimpleNamespace(thread_idx=lambda: (256, 0, 0),
                sync_warp=lambda: events.append(("sync",))), size=lambda x: x,
                copy=lambda atom, src, dst, **kw: events.append(("dma", atom, kw["tma_bar_ptr"])))
        pipeline = SimpleNamespace(PipelineAsync=SimpleNamespace(producer_acquire=
            lambda pipe, state: events.append(("wait", state.index))))
        ns = functions({"load_fc1_tma_slice", "load_fc2_tma_tile"}, dict(
            cute=cute, pipeline=pipeline, get_ptr_as_int64=lambda tensor, index: tensor.iterator + index,
            shared_ptr_to_u32=lambda pointer: pointer,
            _sf6_expand_dynamic_tile=lambda *args: events.append(("expand", *args))))
        return ns, kernel, events

    def test_fc1_native_n64_halves_replay_correct_physical_n128_scale_blocks(self):
        ns, kernel, events = self.namespace()
        state, up = State(), State()
        pipe = FakePipeline(events)
        base = 2**34
        packed = Tensor("packed", base, (288, 128, 1552))
        smem = tuple(Tensor(name, pointer) for name, pointer in
                     (("a", 10000), ("sfa", 20000), ("gate_b", 30000), ("up_b", 40000),
                      ("gate_sf", 50000), ("up_sf", 60000), ("up_sf_extra", 70000)))
        result = ns["load_fc1_tma_slice"](kernel, 2, 1, 17, 4, 32, state, pipe, up, pipe,
            ("a", "b", "sfa"), (Tensor("a"), Tensor("sfa"), Tensor("b"), packed), smem)
        self.assertIs(result[0], state)
        # Both stock native N64 halves replay the same physical N128 scales.
        expand = [event for event in events if event[0] == "expand"]
        self.assertEqual(len(expand), 128)
        for native_half in range(2):
            for kt in range(32):
                gate, up = expand[(native_half * 32 + kt) * 2:(native_half * 32 + kt) * 2 + 2]
                self.assertEqual(gate[1], base + (17 * 128 + (2 + 4) * 16 + kt // 2) * 1552)
                self.assertEqual(up[1], base + (17 * 128 + 2 * 16 + kt // 2) * 1552)
                self.assertEqual(gate[3:], (kt % 2, 0, False))
                self.assertEqual(up[3:], (kt % 2, 0, False))
                stage = (native_half * 32 + kt) % 3
                self.assertEqual(gate[2], 50000 + stage * 1024)
                self.assertEqual(up[2], 60000 + stage * 1024 if stage < 2 else 70000)
        self.assertEqual(sum(event[0] == "alias_wait" for event in events), 1)
        self.check_publication(events, scale_copies=2, dma_copies=4)

    def test_fc2_stage_mapping_and_alias_publication(self):
        ns, kernel, events = self.namespace()
        state, pipe = State(), FakePipeline(events)
        base = 2**34
        packed = Tensor("packed", base, (288, 64, 1552))
        for output in (0, 1, 14, 15, 30, 31):
            for intermediate in range(4):
                ns["load_fc2_tma_tile"](kernel, intermediate, output, 287, state, pipe,
                    ("b",), (Tensor("b"), packed),
                    (Tensor("b_shared"), Tensor("b_extra"), Tensor("sf_shared", 80000)))
                expanded = next(event for event in reversed(events) if event[0] == "expand")
                self.assertEqual(expanded[1], base + (287 * 64 + (output // 2) * 4 + intermediate) * 1552)
                self.assertEqual(expanded[3:], (output % 2, 0, True))
        self.check_publication(events, scale_copies=1, dma_copies=1)

    def check_publication(self, events, *, scale_copies, dma_copies):
        names = [event[0] for event in events if event[0] != "alias_wait"]
        expected = ["wait"] + ["expand"] * scale_copies + ["sync", "publish"] + ["dma"] * dma_copies + ["commit"]
        self.assertEqual(len(names) % len(expected), 0)
        self.assertEqual(names, expected * (len(names) // len(expected)))


class ForkContractTests(unittest.TestCase):
    def test_no_raw_expert_scale_descriptor_and_extended_abi_are_explicit(self):
        call = body("__call__")
        names = [arg.arg for arg in call.args.args]
        self.assertEqual(names[-5:], ["token_weights", "sfb1_packed", "sfb2_packed", "max_active_clusters", "stream"])
        branch = next(n for n in call.body if isinstance(n, ast.If) and n.orelse)
        raw_names = {n.id for statement in branch.orelse for n in ast.walk(statement) if isinstance(n, ast.Name)}
        self.assertFalse(raw_names & {"sfb_w13_ptr", "sfb_down_ptr", "sfb_w13_tensor", "sfb_down_tensor"})
        kernel_names = {n.id for n in ast.walk(body("kernel")) if isinstance(n, ast.Name)}
        self.assertFalse(kernel_names & {"mSFB_w13", "mSFB_down", "tma_sfb_w13", "tma_sfb_down"})
        methods = {n.name for n in CLASS.body if isinstance(n, ast.FunctionDef)}
        self.assertFalse(methods & {"initialize_route_q0_and_publish", "fc1_gate_up_swiglu_to_sC", "fc1_gate_up_swiglu_to_sC_tail",
                                    "quantize_q1_sC_to_sA_sSFA", "fc2_accumulate_slice", "fc2_accumulate_slice_tail", "scatter_sC_to_gmem"})
        self.assertIn("993783308233288ddfa77293e9dbabdc825ba5bfdcc4dcc41e842a895ec33445", TEXT)
        self.assertIn("@lru_cache(maxsize=1)\ndef stock_contract_matches", TEXT)

    def test_tma_transaction_budgets_exclude_synchronous_scale_bytes(self):
        kernel = body("kernel")
        first = next(i for i, node in enumerate(kernel.body) if isinstance(node, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "fc1_tma_copy_bytes" for t in node.targets))
        stop = next(i for i, node in enumerate(kernel.body) if isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "phase2_tma_copy_bytes" for t in node.targets))
        ns = dict(self=SimpleNamespace(a_dtype="a", b_dtype="b", sf_dtype="sf"),
                  a_smem_one=8192, fc1_b_smem_one=4096, sfa_smem_one=1024,
                  b_smem_one=8192, sequential_branch_compact=False,
                  cute=SimpleNamespace(size_in_bytes=lambda dtype, size: size),
                  cutlass=SimpleNamespace(const_expr=bool))
        exec(compile(ast.Module(body=kernel.body[first:stop+1], type_ignores=[]), str(SOURCE), "exec"), ns)
        self.assertEqual(ns["fc1_tma_copy_bytes"], 17408)
        self.assertEqual(ns["phase2_tma_copy_bytes"], 8192)


if __name__ == "__main__":
    unittest.main()
