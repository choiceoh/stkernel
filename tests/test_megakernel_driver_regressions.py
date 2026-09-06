"""CPU regressions for the real megakernel boot gates and lane dispatch."""
import importlib.util
import logging
import math
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / "overlay/modules/glm53_megakernel/glm53_megakernel.py"


def driver():
    spec = importlib.util.spec_from_file_location("mk_driver_regression", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.logger = logging.getLogger("mk_regression")
    module.logger.addHandler(logging.NullHandler())
    module.logger.propagate = False
    return module


class Vector:
    def __init__(self, *values):
        self.values = values

    def float(self):
        return self

    def __sub__(self, other):
        return Vector(*(a - b for a, b in zip(self.values, other.values)))

    def norm(self):
        return math.sqrt(sum(v * v for v in self.values))


class Tensor:
    def __init__(self, *shape):
        self.shape = shape

    def __getitem__(self, key):
        return Tensor(key.stop, *self.shape[1:])

    def __mul__(self, _):
        return self

    def to(self, *_):
        return self

    def view(self, *_):
        return self

    def float(self):
        return self


def torch_stub():
    return types.SimpleNamespace(
        manual_seed=lambda *_: None,
        randn=lambda *shape, **_: Tensor(*shape),
        randint=lambda *args, **_: Tensor(*args[2]),
        full=lambda shape, *args, **_: Tensor(*shape),
        bfloat16=object(), float8_e4m3fn=object(), int32=object(), uint8=object(),
        cuda=types.SimpleNamespace(synchronize=lambda: None),
        equal=lambda a, b: a == b,
        tensor=lambda values, **_: tuple(values),
    )


class Extension:
    """The binding's probe surface since 34차 §8: one knob, the forced split."""

    def __init__(self, state=None):
        self.state = list(state or [0])  # 0 = the rule's split
        self.calls = []

    def probe_state(self):
        return self.state.copy()

    def restore_probe_state(self, state):
        self.state = list(state)

    def set_gemm2(self, ksr):
        if ksr >= 0:
            self.state[0] = ksr

    def gemm2_plan(self, m, n, k):
        ksr = self.state[0] if self.state[0] > 0 else 1
        return [ksr, 32 * ksr, 2]

    def run(self, x, pack, n):
        self.calls.append((x.shape[0], n, self.state[0]))
        return (x.shape[0], n)  # equal mathematical output, distinct dispatch


class DriverRegressions(unittest.TestCase):
    def test_nonfinite_errors_fail_closed(self):
        mk = driver()
        with patch.dict(sys.modules, {"torch": torch_stub()}):
            for bad in (math.nan, math.inf, -math.inf):
                for a, b in ((Vector(bad), Vector(1)), (Vector(1), Vector(bad))):
                    self.assertEqual(mk._rel_err(a, b), math.inf)
            self.assertEqual(mk._rel_err(Vector(0), Vector(0)), 0)
            self.assertEqual(mk._rel_err(Vector(3, 4), Vector(0, 0)), 5)
            self.assertEqual(mk._rel_err(Vector(2), Vector(1)), 1)
            # Finite operands can still overflow the reference norm. A
            # finite numerator divided by inf must not turn into a pass.
            self.assertEqual(mk._rel_err(Vector(1e154), Vector(1.9e154)), math.inf)

    def test_mla_rejects_nan_in_every_shape_position(self):
        mk = driver()
        mk.mla_decode = mk.mla_decode_ref = lambda *a: Tensor()
        with patch.dict(sys.modules, {"torch": torch_stub()}):
            for position in range(6):
                for bad in (math.nan, math.inf, 0.021):
                    errors = iter([0.001] * position + [bad] + [0.001] * (5 - position))
                    mk._rel_err = lambda *a: next(errors)
                    self.assertFalse(mk._selftest_mla(), (position, bad))
            mk._rel_err = lambda *a: 0.001
            self.assertTrue(mk._selftest_mla())

    def test_exact_gate_runs_row_classes_and_forced_split_then_restores(self):
        mk = driver()
        ext = Extension()
        mk._EXT, mk._gemm_call = ext, ext.run
        original = ext.probe_state()
        def fixture(*args, shape=None):
            n, k, m = shape
            return Tensor(m, k), object(), object(), Tensor(m, n)
        mk.exact_fixture = fixture
        mk._exact_gate = lambda *a: (0.0, 0)
        mk._rel_err = lambda *a: 0.0
        with patch.dict(sys.modules, {"torch": torch_stub()}):
            self.assertEqual(mk._selftest_gemm_exact(), 0)
        self.assertEqual({m for m, n, ksr in ext.calls if n == 1024}, {8, 16, 32})
        self.assertIn((8, 2048, 3), ext.calls)  # the forced odd split
        self.assertEqual(ext.probe_state(), original)

    def test_restore_after_launch_failure_or_refused_split(self):
        mk = driver()
        for refuse in (False, True):
            ext = Extension()
            if refuse:
                ext.set_gemm2 = lambda ksr: None  # the extension ignores the force
            original = ext.probe_state()
            mk._EXT = ext
            def launch(*args):
                if not refuse:
                    raise RuntimeError("injected launch failure")
                return ext.run(*args)
            mk._gemm_call = launch
            mk.exact_fixture = lambda *a, shape=None: (Tensor(shape[2], shape[1]), object(), object(), Tensor(shape[2], shape[0]))
            mk._exact_gate = lambda *a: (0.0, 0)
            mk._rel_err = lambda *a: 0.0
            with patch.dict(sys.modules, {"torch": torch_stub()}), self.assertRaises(RuntimeError):
                mk._selftest_gemm_exact()
            self.assertEqual(ext.probe_state(), original)

    def test_exact_gate_rejects_nan_and_restores_state(self):
        mk = driver()
        ext = Extension()
        original = ext.probe_state()
        mk._EXT, mk._gemm_call = ext, ext.run
        mk.exact_fixture = lambda **_: (Tensor(32, 4096), object(), object(), Tensor(32, 1024))
        mk._rel_err = lambda *a: 0.0
        errors = iter([(math.nan, 0)])  # the first row class is NaN
        mk._exact_gate = lambda *a: next(errors)
        with patch.dict(sys.modules, {"torch": torch_stub()}), self.assertRaises(RuntimeError):
            mk._selftest_gemm_exact()
        self.assertEqual(ext.probe_state(), original)

if __name__ == "__main__":
    unittest.main()
