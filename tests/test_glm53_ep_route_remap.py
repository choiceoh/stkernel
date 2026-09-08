"""Metadata admission and real Torch oracle for the one-launch EP remap.

GPU callers can use ``legacy_remap`` with CUDA tensors to compare the fused
kernel against the actual serving remap, without duplicating its semantics.
"""
import ast
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

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
    def __init__(self, shape=(4097, 8), dtype="int32", device="cuda:0", contiguous=True):
        self.shape, self.dtype, self.device = shape, dtype, device
        self.is_cuda = device.startswith("cuda")
        self.ndim, self._contiguous = len(shape), contiguous

    def is_contiguous(self):
        return self._contiguous

    def numel(self):
        return math.prod(self.shape)


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
        ns = extract(REMAP, {"ep_route_remap_supported", "try_remap_ep_local"}, dict(
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
