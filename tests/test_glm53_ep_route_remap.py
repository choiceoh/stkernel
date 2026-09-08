"""Metadata admission and real Torch oracle for the one-launch EP remap.

GPU callers can use ``legacy_remap`` with CUDA tensors to compare the fused
kernel against the actual serving remap, without duplicating its semantics.
"""
import ast
import importlib.util
import math
import operator
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
REMAP = ROOT / "overlay/modules/glm53_moe/glm53_ep_route_remap.py"
WRAPPER = ROOT / "overlay/modules/glm53_moe/flashinfer_b12x_moe.py"


def extract(path, names, namespace):
    body = [node for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def legacy_remap(topk_ids, topk_weights, **kwargs):
    """Run the actual wrapper implementation with ordinary CPU or CUDA tensors."""
    import torch
    ns = extract(WRAPPER, {"_ep_buf", "remap_b12x_ep_tensors"}, {"torch": torch})
    return ns["remap_b12x_ep_tensors"](topk_ids, topk_weights, **kwargs)


def _i32(value):
    return (int(value) + (1 << 31)) % (1 << 32) - (1 << 31)


def slot_reference(expert, *, expert_map, local_expert_offset):
    """Independent scalar description of the legacy tensor conversion order."""
    if expert_map is not None:
        if not 0 <= expert < len(expert_map) or expert_map[expert] < 0:
            return 72, True
        return _i32(expert_map[expert]), False
    local = _i32(_i32(expert) - local_expert_offset)
    remote = expert < 0 or local < 0 or local >= 72
    return (72 if remote else local), remote


class FakeTensor:
    next_pointer = 1 << 20

    def __init__(self, shape=(4097, 8), dtype="int32", device="cuda:0", contiguous=True):
        self.shape, self.dtype, self.device = shape, dtype, device
        self.is_cuda = device.startswith("cuda")
        self.ndim, self._contiguous = len(shape), contiguous
        self.pointer = FakeTensor.next_pointer
        FakeTensor.next_pointer += 1 << 20

    def is_contiguous(self):
        return self._contiguous

    def numel(self):
        return math.prod(self.shape)

    def data_ptr(self):
        return self.pointer

    def element_size(self):
        return {"int64": 8, "int32": 4, "float32": 4,
                "float16": 2, "bfloat16": 2}[self.dtype]


class ShortPrepareAdmissionTests(unittest.TestCase):
    def fixture(self):
        launch = Mock()
        class Kernel:
            def __getitem__(self, grid):
                self.grid = grid
                return launch
        kernel = Kernel()
        torch = SimpleNamespace(Tensor=FakeTensor, int32="int32", int64="int64",
                                float32="float32", float16="float16", bfloat16="bfloat16")
        ns = extract(REMAP, {"_ep_short_decode_metadata", "ep_short_decode_prepare_supported",
                             "try_prepare_ep_short_decode"}, dict(
            torch=torch, _prepare_ep_short_decode_kernel=kernel))
        args = dict(hidden_states=FakeTensor((6, 4096), "bfloat16"),
                    topk_ids=FakeTensor((6, 8)), topk_weights=FakeTensor((6, 8), "float32"),
                    expert_map=FakeTensor((288,)), num_local_experts=72,
                    local_expert_offset=72, pad_x=FakeTensor((8, 4096), "bfloat16"),
                    pad_ids=FakeTensor((8, 8)), pad_weights=FakeTensor((8, 8), "float32"))
        return ns, kernel, launch, args

    def test_exact_types_and_map_variants_launch_once_with_owned_buffers(self):
        for id_dtype in ("int32", "int64"):
            for weight_dtype in ("float32", "float16", "bfloat16"):
                for map_len in (None, 0, 288):
                    ns, kernel, launch, args = self.fixture()
                    args["topk_ids"].dtype = id_dtype
                    args["topk_weights"].dtype = args["pad_weights"].dtype = weight_dtype
                    args["expert_map"] = None if map_len is None else FakeTensor((map_len,), id_dtype)
                    self.assertTrue(ns["try_prepare_ep_short_decode"](**args))
                    launch.assert_called_once()
                    self.assertEqual(kernel.grid, (8,))
                    sent, constants = launch.call_args
                    self.assertEqual(constants, dict(MAP_LEN=map_len or 0,
                                                     HAS_MAP=map_len is not None, num_warps=4))
                    self.assertIs(sent[0], args["hidden_states"])
                    self.assertIs(sent[4], args["pad_x"])
                    self.assertIs(sent[5], args["pad_ids"])
                    self.assertIs(sent[6], args["pad_weights"])
                    self.assertIs(sent[3], args["expert_map"] if map_len else args["pad_ids"])

    def test_unsupported_metadata_or_partial_alias_never_launches(self):
        changes = (
            {"hidden_states": FakeTensor((7, 4096), "bfloat16")},
            {"hidden_states": FakeTensor((6, 4096), "float16")},
            {"hidden_states": FakeTensor((6, 4096), "bfloat16", contiguous=False)},
            {"hidden_states": FakeTensor((6, 4096), "bfloat16", device="cpu")},
            {"topk_ids": FakeTensor((6, 7))},
            {"topk_weights": FakeTensor((6, 8), "float32", device="cuda:1")},
            {"expert_map": FakeTensor((288, 1))},
            {"expert_map": FakeTensor((288,), "float32")},
            {"expert_map": FakeTensor((288,), contiguous=False)},
            {"expert_map": FakeTensor(((1 << 31),))},
            {"num_local_experts": 73}, {"local_expert_offset": -1},
            {"local_expert_offset": True}, {"local_expert_offset": 1 << 31},
            {"pad_x": FakeTensor((6, 4096), "bfloat16")},
            {"pad_ids": FakeTensor((8, 8), "int64")},
            {"pad_weights": FakeTensor((8, 8), "float16")},
            {"pad_weights": FakeTensor((8, 8), "float32", contiguous=False)},
        )
        for change in changes:
            ns, _, launch, args = self.fixture()
            self.assertFalse(ns["try_prepare_ep_short_decode"](**dict(args, **change)))
            launch.assert_not_called()
        for left, right in (("pad_x", "hidden_states"), ("pad_ids", "topk_ids"),
                            ("pad_weights", "expert_map"), ("pad_weights", "pad_ids")):
            ns, _, launch, args = self.fixture()
            args[left].pointer = args[right].pointer + args[right].element_size()
            self.assertFalse(ns["try_prepare_ep_short_decode"](**args))
            launch.assert_not_called()

    def test_metadata_is_fresh_and_launch_errors_propagate(self):
        ns, _, launch, args = self.fixture()
        inputs = {k: v for k, v in args.items() if not k.startswith("pad_")}
        self.assertTrue(ns["ep_short_decode_prepare_supported"](**inputs))
        launch.assert_not_called()
        self.assertTrue(ns["try_prepare_ep_short_decode"](**args))
        args["topk_ids"].shape = (5, 8)
        launch.reset_mock()
        self.assertFalse(ns["try_prepare_ep_short_decode"](**args))
        launch.assert_not_called()
        args["topk_ids"].shape = (6, 8)
        launch.side_effect = RuntimeError("device launch failed")
        with self.assertRaisesRegex(RuntimeError, "device launch failed"):
            ns["try_prepare_ep_short_decode"](**args)


class Vector:
    """Integer lanes for executing the real Triton preparation source on CPU."""
    def __init__(self, values):
        self.values = list(values)

    def binary(self, other, fn, reverse=False):
        values = other.values if isinstance(other, Vector) else [other] * len(self.values)
        return Vector(fn(b, a) if reverse else fn(a, b)
                      for a, b in zip(self.values, values))

    def to(self, dtype):
        if dtype == "int32":
            return Vector(_i32(v) for v in self.values)
        if dtype in ("uint16", "uint32"):
            mask = (1 << int(dtype[4:])) - 1
            return Vector(int(v) & mask for v in self.values)
        return Vector(self.values)

    def __invert__(self):
        return Vector(not v for v in self.values)


for _name, _op in (("add", operator.add), ("sub", operator.sub), ("mul", operator.mul),
                  ("lt", operator.lt), ("ge", operator.ge),
                  ("and", operator.and_), ("or", operator.or_)):
    setattr(Vector, "__" + _name + "__", lambda self, other, op=_op: self.binary(other, op))
    setattr(Vector, "__r" + _name + "__", lambda self, other, op=_op: self.binary(other, op, True))


class Pointer:
    def __init__(self, values, dtype, offset=0):
        self.values, self.dtype, self.offset = values, SimpleNamespace(element_ty=dtype), offset

    def to(self, dtype):
        return Pointer(self.values, dtype, self.offset)

    def __add__(self, offset):
        return Pointer(self.values, self.dtype.element_ty, self.offset + offset)


class TritonLanes:
    constexpr = object
    float32, uint32, uint16, int32, int64 = "float32", "uint32", "uint16", "int32", "int64"
    pointer_type = staticmethod(lambda kind: kind)
    arange = staticmethod(lambda lo, hi: Vector(range(lo, hi)))

    def program_id(self, _):
        return self.row

    @staticmethod
    def where(condition, left, right):
        if isinstance(condition, Vector):
            ls = left.values if isinstance(left, Vector) else [left] * len(condition.values)
            rs = right.values if isinstance(right, Vector) else [right] * len(condition.values)
            return Vector(l if c else r for c, l, r in zip(condition.values, ls, rs))
        return left if condition else right

    @staticmethod
    def load(pointer, mask=True, other=0):
        offsets = pointer.offset.values
        masks = mask.values if isinstance(mask, Vector) else [mask] * len(offsets)
        result = []
        for index, enabled in zip(offsets, masks):
            if enabled:
                if not 0 <= index < len(pointer.values):
                    raise AssertionError("unmasked out-of-bounds read")
                result.append(pointer.values[index])
            else:
                result.append(other)
        return Vector(result)

    @staticmethod
    def store(pointer, value):
        offsets = pointer.offset.values
        values = value.values if isinstance(value, Vector) else [value] * len(offsets)
        for index, item in zip(offsets, values):
            if not 0 <= index < len(pointer.values):
                raise AssertionError("out-of-bounds write")
            pointer.values[index] = item


class ShortPrepareKernelTests(unittest.TestCase):
    def kernel(self):
        tree = ast.parse(REMAP.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "_prepare_ep_short_decode_kernel")
        node.decorator_list = []
        lanes = TritonLanes()
        ns = {"tl": lanes}
        exec(compile(ast.Module([node], type_ignores=[]), str(REMAP), "exec"), ns)
        return lanes, ns[node.name]

    def test_real_kernel_matches_legacy_padding_bits_and_changed_inputs(self):
        lanes, kernel = self.kernel()
        ids_pattern = [-1, 0, 71, 72, 73, 143, 288, (1 << 32) + 72]
        mapping = [-1] * 288
        mapping[72:76] = [0, 72, 73, (1 << 32) + 7]
        for dtype, nan, negative_zero in (("float32", 0x7FC01234, 0x80000000),
                                         ("float16", 0x7E12, 0x8000),
                                         ("bfloat16", 0x7FC1, 0x8000)):
            for initial_map in (None, [], mapping):
                with self.subTest(dtype=dtype, mapped=initial_map is not None):
                    x = [((i * 31) & 65535) for i in range(6 * 4096)]
                    ids, weights = ids_pattern * 6, [nan, negative_zero, 5, 9] * 12
                    mapped = None if initial_map is None else list(initial_map)
                    px, pi, pw = [77] * (8 * 4096), [77] * 64, [77] * 64
                    for turn in range(2):
                        if turn:
                            x[0], ids[0], weights[0] = 0x7FC1, 72, negative_zero
                            if mapped:
                                mapped[72] = 5
                        saved = (x[:], ids[:], weights[:], None if mapped is None else mapped[:])
                        for row in range(8):
                            lanes.row = row
                            kernel(Pointer(x, "bfloat16"), Pointer(ids, "int64"),
                                   Pointer(weights, dtype), Pointer(mapped or pi, "int64"),
                                   Pointer(px, "bfloat16"), Pointer(pi, "int32"),
                                   Pointer(pw, dtype), 72, MAP_LEN=len(mapped or []),
                                   HAS_MAP=mapped is not None)
                        expected_ids, expected_weights = [], []
                        for row in range(8):
                            source = row if row < 6 else 0
                            for slot in range(8):
                                index = source * 8 + slot
                                expert, remote = slot_reference(ids[index], expert_map=mapped,
                                                                local_expert_offset=72)
                                expected_ids.append(expert)
                                expected_weights.append(weights[index] if row < 6 and not remote else 0)
                        self.assertEqual(px, x + x[:4096] * 2)
                        self.assertEqual(pi, expected_ids)
                        self.assertEqual(pw, expected_weights)
                        self.assertEqual((x, ids, weights, mapped), saved)


class AdmissionTests(unittest.TestCase):
    def namespace(self):
        launcher = Mock()
        # A JITFunction is subscripted with a launch grid.
        class Kernel:
            def __getitem__(self, grid):
                self.grid = grid
                return launcher
        kernel = Kernel()
        torch = SimpleNamespace(Tensor=FakeTensor, int32="int32", int64="int64",
                                float32="float32", float16="float16", bfloat16="bfloat16")
        ns = extract(REMAP, {"_ep_route_remap_metadata", "ep_route_remap_supported",
                             "try_remap_ep_local"}, dict(
            torch=torch, triton=SimpleNamespace(cdiv=lambda n, d: (n+d-1)//d),
            _remap_ep_local_kernel=kernel))
        return ns, kernel, launcher

    def inputs(self, dtype="float32"):
        return dict(topk_ids=FakeTensor(), topk_weights=FakeTensor(dtype=dtype),
                    expert_map=FakeTensor((288,)), num_local_experts=72,
                    local_expert_offset=72, out_ids=FakeTensor(),
                    out_scales=FakeTensor(dtype=dtype))

    def test_one_launch_reuses_scratch_and_preserves_dtype_for_each_arm(self):
        ns, kernel, launch = self.namespace()
        for dtype in ("float32", "float16", "bfloat16"):
            for mapping in (None, FakeTensor((0,)), FakeTensor((288,), dtype="int64")):
                with self.subTest(dtype=dtype, mapping=mapping):
                    launch.reset_mock()
                    args = self.inputs(dtype)
                    args["expert_map"] = mapping
                    self.assertTrue(ns["try_remap_ep_local"](**args))
                    launch.assert_called_once()
                    sent, constants = launch.call_args
                    self.assertIs(sent[0], args["topk_ids"])
                    self.assertIs(sent[1], args["topk_weights"])
                    self.assertIs(sent[3], args["out_ids"])
                    self.assertIs(sent[4], args["out_scales"])
                    self.assertEqual(kernel.grid, (129,))  # 4097*8 tail mask
                    self.assertEqual(constants["HAS_MAP"], mapping is not None)
                    self.assertEqual(constants["MAP_LEN"], mapping.numel() if mapping else 0)
                    if mapping is None or not mapping.numel():
                        self.assertIs(sent[2], args["out_ids"])

    def test_unsupported_metadata_returns_before_any_launch(self):
        ns, _, launch = self.namespace()
        changes = (
            {"num_local_experts": 73}, {"local_expert_offset": -1},
            {"local_expert_offset": True},
            {"topk_ids": FakeTensor((4095, 8))},
            {"topk_ids": FakeTensor((16385, 8))},
            {"topk_ids": FakeTensor((4097, 1))},
            {"topk_ids": FakeTensor(dtype="float32")},
            {"topk_ids": FakeTensor(device="cpu")},
            {"topk_weights": FakeTensor(dtype="float64")},
            {"out_ids": FakeTensor(dtype="int64")},
            {"out_scales": FakeTensor(dtype="float16")},
            {"out_scales": FakeTensor(dtype="float32", contiguous=False)},
            {"out_scales": FakeTensor(dtype="float32", device="cuda:1")},
            {"expert_map": FakeTensor((288,), dtype="float32")},
            {"expert_map": FakeTensor((288, 1))},
            {"expert_map": FakeTensor((288,), contiguous=False)},
            {"expert_map": FakeTensor((288,), device="cuda:1")},
        )
        for change in changes:
            with self.subTest(change=change):
                self.assertFalse(ns["try_remap_ep_local"](**dict(self.inputs(), **change)))
        launch.assert_not_called()

    def test_launch_failure_propagates_instead_of_requesting_fallback(self):
        ns, _, launch = self.namespace()
        launch.side_effect = RuntimeError("launch failed")
        with self.assertRaisesRegex(RuntimeError, "launch failed"):
            ns["try_remap_ep_local"](**self.inputs())

    def test_launch_sizes_are_validated_once_and_recomputed_for_reused_tensors(self):
        ns, kernel, launch = self.namespace()
        args = self.inputs()
        ids, mapping = args["topk_ids"], args["expert_map"]
        # Admission already has the dense shape; avoid additional Tensor size
        # queries to prepare the launch, or reading the map size a second time.
        ids.numel = Mock(side_effect=AssertionError("redundant ids size query"))
        mapping.numel = Mock(wraps=mapping.numel)
        for rows, map_len in ((4097, 288), (8192, 144), (4096, 0)):
            with self.subTest(rows=rows, map_len=map_len):
                for name in ("topk_ids", "topk_weights", "out_ids", "out_scales"):
                    args[name].shape = (rows, 8)
                mapping.shape = (map_len,)
                mapping.numel.reset_mock()
                launch.reset_mock()
                self.assertTrue(ns["try_remap_ep_local"](**args))
                mapping.numel.assert_called_once_with()
                launch.assert_called_once()
                sent, constants = launch.call_args
                self.assertEqual(sent[5], rows * 8)
                self.assertEqual(kernel.grid, ((rows * 8 + 255) // 256,))
                self.assertEqual(constants["MAP_LEN"], map_len)
                self.assertTrue(constants["HAS_MAP"])
                self.assertIs(sent[2], mapping if map_len else args["out_ids"])
        ids.numel.assert_not_called()

    def test_reused_tensor_metadata_is_revalidated_after_every_launch(self):
        ns, _, launch = self.namespace()
        args = self.inputs()
        self.assertTrue(ns["try_remap_ep_local"](**args))
        for name, attribute, invalid in (
            ("topk_ids", "shape", (4095, 8)),
            ("topk_ids", "_contiguous", False),
            ("out_ids", "device", "cuda:1"),
            ("topk_weights", "dtype", "float16"),
            ("expert_map", "device", "cuda:1"),
            ("expert_map", "shape", ((1 << 31),)),
        ):
            with self.subTest(tensor=name, attribute=attribute):
                tensor = args[name]
                original = getattr(tensor, attribute)
                setattr(tensor, attribute, invalid)
                launch.reset_mock()
                self.assertFalse(ns["try_remap_ep_local"](**args))
                launch.assert_not_called()
                setattr(tensor, attribute, original)
                self.assertTrue(ns["try_remap_ep_local"](**args))
                launch.assert_called_once()

    def test_public_support_check_is_boolean_and_never_launches(self):
        ns, _, launch = self.namespace()
        args = self.inputs()
        self.assertIs(ns["ep_route_remap_supported"](**args), True)
        args["out_ids"].device = "cuda:1"
        self.assertIs(ns["ep_route_remap_supported"](**args), False)
        launch.assert_not_called()


class WrapperScratchTests(unittest.TestCase):
    class Buffer:
        def __init__(self, rows):
            self.rows = rows
            self.slices = []

        def size(self, axis):
            return self.rows if axis == 0 else 8

        def __getitem__(self, key):
            view = SimpleNamespace(parent=self, key=key)
            self.slices.append(view)
            return view

    def fixture(self):
        tree = ast.parse(WRAPPER.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == "FlashInferB12xExperts")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                      and node.name == "_remap_ep_tensors")
        legacy = Mock(return_value=("legacy ids", "legacy weights"))
        namespace = {"remap_b12x_ep_tensors": legacy}
        future = ast.parse("from __future__ import annotations").body[0]
        exec(compile(ast.Module(body=[future, method], type_ignores=[]),
                     str(WRAPPER), "exec"), namespace)
        buffers = {name: self.Buffer(8192) for name in (
            "_ep_ids", "_ep_scales", "_ep_long", "_ep_mapped",
            "_ep_remote", "_ep_tmp_a", "_ep_tmp_b")}
        wrapper = SimpleNamespace(num_local_experts=72, local_expert_offset=72, **buffers)
        module = ModuleType("test_remap")
        module.try_remap_ep_local = Mock(return_value=True)
        return namespace["_remap_ep_tensors"], wrapper, module, legacy

    def test_full_candidate_reuses_tensor_objects_without_any_scratch_views(self):
        method, wrapper, module, legacy = self.fixture()
        name = "flashinfer.fused_moe.cute_dsl.blackwell_sm12x.glm53_ep_route_remap"
        with patch.dict(sys.modules, {name: module}):
            result = method(wrapper, self.Buffer(8192), object(), object(),
                            fuse_local_prefill=True)
        self.assertIs(result[0], wrapper._ep_ids)
        self.assertIs(result[1], wrapper._ep_scales)
        for value in vars(wrapper).values():
            if isinstance(value, self.Buffer):
                self.assertEqual(value.slices, [])
        legacy.assert_not_called()

    def test_partial_candidate_uses_current_storage_and_exact_row_views(self):
        method, wrapper, module, _ = self.fixture()
        name = "flashinfer.fused_moe.cute_dsl.blackwell_sm12x.glm53_ep_route_remap"
        with patch.dict(sys.modules, {name: module}):
            for rows in (4096, 6912):
                # Replacement models a dtype/device scratch rebuild. No view
                # from the earlier storage may be reused by the wrapper.
                wrapper._ep_ids = self.Buffer(8192)
                wrapper._ep_scales = self.Buffer(8192)
                result = method(wrapper, self.Buffer(rows), object(), object(),
                                fuse_local_prefill=True)
                for actual, parent in zip(result, (wrapper._ep_ids, wrapper._ep_scales)):
                    self.assertIs(actual.parent, parent)
                    self.assertEqual(actual.key, slice(None, rows))

    def test_declined_candidate_and_disabled_path_keep_legacy_scratch(self):
        method, wrapper, module, legacy = self.fixture()
        module.try_remap_ep_local.return_value = False
        name = "flashinfer.fused_moe.cute_dsl.blackwell_sm12x.glm53_ep_route_remap"
        with patch.dict(sys.modules, {name: module}):
            result = method(wrapper, self.Buffer(8192), object(), object(),
                            fuse_local_prefill=True)
            self.assertEqual(result, legacy.return_value)
            self.assertIs(legacy.call_args.kwargs["out_ids"], wrapper._ep_ids)
            self.assertIs(legacy.call_args.kwargs["out_scales"], wrapper._ep_scales)
            for name in ("long_idx", "mapped", "remote", "tmp_a", "tmp_b"):
                self.assertEqual(legacy.call_args.kwargs[name].key, slice(None, 8192))
            module.try_remap_ep_local.reset_mock()
            method(wrapper, self.Buffer(8192), object(), object())
            module.try_remap_ep_local.assert_not_called()
            self.assertIs(legacy.call_args.kwargs["out_ids"].parent, wrapper._ep_ids)
            self.assertIs(legacy.call_args.kwargs["out_scales"].parent, wrapper._ep_scales)


@unittest.skipUnless(importlib.util.find_spec("torch"), "pinned CPU image supplies Torch")
class ReferenceTests(unittest.TestCase):
    def test_actual_tensor_remap_matches_scalar_conversion_and_range_oracle(self):
        import torch
        pattern = [-1, -(1 << 40), 0, 71, 72, 73, 143, 144, 287, 288,
                   (1 << 32)+72, (1 << 32)+73, 72, 72, 74, 75]
        mapping = [-1]*288
        mapping[72:76] = [0, 72, 73, (1 << 32)+7]
        maps = (None, [], mapping)
        for offset in (0, 72, 216):
            for values in maps:
                with self.subTest(offset=offset, mapped=values is not None):
                    ids = torch.tensor(pattern, dtype=torch.int64).reshape(2, 8)
                    weights = torch.arange(16, dtype=torch.float32).reshape(2, 8)
                    mapped = None if values is None else torch.tensor(values, dtype=torch.int64)
                    out_ids, out_weights = legacy_remap(
                        ids, weights, expert_map=mapped, num_local_experts=72,
                        local_expert_offset=offset)
                    expected = [slot_reference(x, expert_map=values, local_expert_offset=offset)
                                for x in pattern]
                    self.assertEqual(out_ids.reshape(-1).tolist(), [x[0] for x in expected])
                    self.assertEqual(out_weights.reshape(-1).tolist(),
                                     [0. if x[1] else float(i) for i, x in enumerate(expected)])

    def test_actual_remap_preserves_local_bits_and_zeros_remote_nonfinite_weights(self):
        import torch
        for dtype, bits_dtype, nan, negative_zero in (
            (torch.float32, torch.int32, 0x7FC01234, -(1 << 31)),
            (torch.float16, torch.int16, 0x7E12, -(1 << 15)),
            (torch.bfloat16, torch.int16, 0x7FC1, -(1 << 15)),
        ):
            with self.subTest(dtype=dtype):
                bits = torch.tensor([nan, negative_zero, nan, negative_zero,
                                     nan, negative_zero, nan, negative_zero], dtype=bits_dtype)
                weights = bits.view(dtype).reshape(1, 8)
                ids = torch.tensor([[72, 72, -1, 288, 72, 72, 71, 144]], dtype=torch.int32)
                mapping = torch.full((288,), -1, dtype=torch.int32)
                mapping[72] = 0
                _, remapped = legacy_remap(ids, weights, expert_map=mapping,
                                          num_local_experts=72, local_expert_offset=72)
                self.assertEqual(remapped.view(bits_dtype).reshape(-1).tolist(),
                                 [nan, negative_zero, 0, 0, nan, negative_zero, 0, 0])
                _, empty = legacy_remap(ids, weights, expert_map=mapping[:0],
                                       num_local_experts=72, local_expert_offset=72)
                self.assertEqual(empty.view(bits_dtype).reshape(-1).tolist(), [0]*8)


if __name__ == "__main__":
    unittest.main()
