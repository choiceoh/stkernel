#!/usr/bin/env python3
"""Execute production MLA routing with CPU mocks; no CUDA or vLLM needed.

Run: python3 tests/test_megakernel_wiring_regressions.py
"""
from __future__ import annotations

import ast
import math
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch


MODULE = Path(__file__).resolve().parents[1] / "overlay/modules/glm53_model"


def load_functions(filename, names, namespace, class_name=None):
    """Compile complete source functions, preserving their control flow."""
    tree = ast.parse((MODULE / filename).read_text())
    body = tree.body
    if class_name is not None:
        body = next(n for n in body if isinstance(n, ast.ClassDef)
                    and n.name == class_name).body
    functions = [n for n in body if isinstance(n, ast.FunctionDef)
                 and n.name in names]
    assert {n.name for n in functions} == set(names)
    code = ast.Module(body=functions, type_ignores=[])
    exec(compile(code, str(MODULE / filename), "exec"), namespace)


class Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value

    def all(self):
        return self


class Tensor:
    """Constant-valued tensor sufficient for the routing arithmetic."""
    def __init__(self, value=1.0, shape=(16, 1), dtype="bf16"):
        self.value, self.shape, self.dtype = value, shape, dtype
        self.device = "cuda"

    def is_contiguous(self):
        return True

    def contiguous(self):
        return self

    def clone(self):
        return Tensor(self.value, self.shape, self.dtype)

    def to(self, dtype):
        return Tensor(self.value, self.shape, dtype)

    def view(self, dtype):
        return self.to(dtype)

    def float(self):
        return self

    def norm(self):
        return Scalar(abs(self.value))

    def max(self):
        return Scalar(self.value)

    def __sub__(self, other):
        return Tensor(self.value - other.value, self.shape)

    def size(self, dim):
        return self.shape[dim]

    def split(self, sizes, dim=-1):
        return tuple(Tensor(self.value, self.shape[:-1] + (n,)) for n in sizes)

    def unsqueeze(self, dim):
        return Tensor(self.value, self.shape[:dim] + (1,) + self.shape[dim:])

    def reshape(self, *shape):
        if -1 in shape:
            inferred = math.prod(self.shape) // -math.prod(shape)
            shape = tuple(inferred if n == -1 else n for n in shape)
        return Tensor(self.value, shape)


def fake_torch(capturing):
    return types.SimpleNamespace(
        Tensor=Tensor, int32="int32", uint8="uint8", bfloat16="bf16",
        cuda=types.SimpleNamespace(
            is_current_stream_capturing=lambda: capturing[0],
            Event=lambda **kw: types.SimpleNamespace(
                record=lambda: None, elapsed_time=lambda other: 0.01),
            synchronize=Mock()),
        isfinite=lambda t: Scalar(math.isfinite(t.value)),
        empty=lambda shape, **kw: Tensor(shape=shape),
    )


class MLAFailureTests(unittest.TestCase):
    def setUp(self):
        self.capturing = [False]
        self.module = types.SimpleNamespace(
            MLA_D=512, MLA_H=16, _ARMED={"mla": True},
            maybe_arm=Mock(), mla_decode=Mock(return_value=Tensor(1.0)))
        self.reference = Mock(return_value=Tensor(1.0))
        self.ns = {
            "math": math, "torch": fake_torch(self.capturing), "logger": Mock(),
            "_MK_MLA_MAX_T": 1 << 20, "_MK_MLA_SHADOW_MAX_T": 4096,
            "_MK_MLA_SHADOW_MIN_ROWS": 16, "_MK_MLA_SHADOW_MIN_NORM": 1e-3,
            "_MK_MLA_SHADOW": {"checked": False, "failure": None},
            "_mk_mla_mod": lambda: self.module,
            "_sm90_wrapper_run": self.reference,
        }
        load_functions("flashinfer_mla_sparse_sm90.py",
                       {"_mk_mla_check_failure", "_mk_mla_route", "_mk_mla_run"},
                       self.ns)
        self.impl = types.SimpleNamespace(
            qk_rope_head_dim=0, kv_lora_rank=512, num_heads=16,
            use_fp8_kv_cache=True, head_size=512, scale=1.0)
        self.args = (self.impl, Tensor(), Tensor(), Tensor(),
                     Tensor(dtype="int32"), Tensor(32, dtype="int32"),
                     types.SimpleNamespace(_k_scale_float=1.0))

    def run_mla(self):
        return self.ns["_mk_mla_run"](*self.args)

    def test_mismatch_after_capture_poisoned_worker_cannot_fall_back(self):
        self.capturing[0] = True
        self.run_mla()
        self.reference.assert_not_called()
        self.assertTrue(self.ns["_MK_MLA_SHADOW"]["captured"])
        self.capturing[0] = False
        self.module.mla_decode.return_value = Tensor(1.5)
        with self.assertRaisesRegex(RuntimeError, "worker invalid.*restart") as error:
            self.run_mla()
        self.assertFalse(self.module._ARMED["mla"])
        launches = self.module.mla_decode.call_count
        # Even an ineligible shape, a disabled module, or a direct launch
        # attempt must not bypass the sticky failure after it was observed.
        self.ns["_mk_mla_mod"] = lambda: None
        for capturing in (False, True):
            self.capturing[0] = capturing
            with self.assertRaises(RuntimeError) as retry:
                self.ns["_mk_mla_route"](self.impl, 0)
            self.assertEqual(str(retry.exception), str(error.exception))
            with self.assertRaises(RuntimeError):
                self.run_mla()
        self.assertEqual(self.module.mla_decode.call_count, launches)

    def test_mismatch_before_capture_also_requires_restart(self):
        self.module.mla_decode.return_value = Tensor(1.5)
        with self.assertRaisesRegex(RuntimeError, "SHADOW FAIL"):
            self.run_mla()
        self.assertFalse(self.module._ARMED["mla"])

    def test_nonfinite_output_or_reference_never_passes(self):
        for output, reference in ((float("nan"), 1.0), (1.0, float("nan")),
                                  (1.0, float("inf")), (float("nan"), 0.0)):
            with self.subTest(output=output, reference=reference):
                self.setUp()
                self.module.mla_decode.return_value = Tensor(output)
                self.reference.return_value = Tensor(reference)
                with self.assertRaisesRegex(RuntimeError, "finite=False"):
                    self.run_mla()

    def test_empty_cache_defers_judgement_then_valid_output_passes(self):
        self.module.mla_decode.return_value = Tensor(0.0)
        self.reference.return_value = Tensor(0.0)
        self.run_mla()
        self.assertFalse(self.ns["_MK_MLA_SHADOW"]["checked"])
        self.module.mla_decode.return_value = Tensor(1.01)
        self.reference.return_value = Tensor(1.0)
        self.assertIs(self.run_mla(), self.module.mla_decode.return_value)
        self.assertTrue(self.ns["_MK_MLA_SHADOW"]["checked"])
        self.assertTrue(self.ns["_mk_mla_route"](self.impl, 16))


if __name__ == "__main__":
    unittest.main()
