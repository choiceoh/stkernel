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
    def __init__(self, state=None):
        self.state = list(state or [5, 1, 32, 24, 1, 3, 1])
        self.calls = []
        self.refuse_local = False

    def probe_state(self):
        return self.state.copy()

    def restore_probe_state(self, state):
        self.state = list(state)

    def set_probe(self, *values):
        self.state[:4] = values

    def set_gemm2(self, *values):
        for i, value in enumerate(values, 4):
            if value >= 0:
                self.state[i] = value

    def gemm2_plan(self, m, n, k):
        return [self.state[4], self.state[5], 32, 2, self.state[6]]

    def gemm_plan(self, m, n, k, bg):
        local = int(self.state[1] == 2 and not self.refuse_local)
        return [96, self.state[0], 32, local, 32 if local else 96, 0]

    def run(self, x, pack, n):
        lane = "v2" if self.state[4] else "local" if self.state[1] == 2 else "global"
        self.calls.append((lane, x.shape[0], n, self.state[0]))
        return (x.shape[0], n)  # equal mathematical output, distinct dispatch


class DriverRegressions(unittest.TestCase):
    def test_short_kda_case_keeps_previous_step_state_and_weight_packs(self):
        mk = driver()
        fixture = mk._KdaFixture.__new__(mk._KdaFixture)
        fixture.T, fixture.acc = 8, 3
        fixture.x, fixture.sidx = Tensor(8, 4096), Tensor(1, 8)
        fixture.conv_st, fixture.rec_st = object(), object()
        fixture._mk_cache, fixture._stock_cache = (object(), object()), object()
        with patch.dict(sys.modules, {"torch": torch_stub()}):
            case = fixture.for_query(1, 8)
            for nq, acc in ((0, 1), (9, 1), (1, 0), (1, 9)):
                with self.assertRaises(ValueError):
                    fixture.for_query(nq, acc)
        self.assertEqual((case.T, case.acc, case.x.shape), (1, 8, (1, 4096)))
        self.assertEqual((case.cu, case.nacc), ((0, 1), (8,)))
        self.assertIs(case.sidx, fixture.sidx)
        self.assertIs(case.conv_st, fixture.conv_st)
        self.assertIs(case.rec_st, fixture.rec_st)
        self.assertIs(case._mk_cache[0], fixture._mk_cache[0])
        self.assertIs(case._stock_cache, fixture._stock_cache)
        self.assertEqual(case._mk_cache[1].num_actual_tokens, 1)
        self.assertEqual(fixture.T, 8)

    def test_kda_errors_cover_every_written_recurrent_position(self):
        mk = driver()
        fixture = mk._KdaFixture.__new__(mk._KdaFixture)
        fixture.T = 4
        class State:
            def __getitem__(self, key):
                return (key.start, key.stop)
        got = dict(out="output", conv_state=State(), rec_state=State())
        mk._rel_err = lambda a, b: a
        errors = fixture.errors(got, got)
        self.assertEqual(errors, dict(out="output", conv_state=(1, 2), rec_state=(1, 5)))

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

    def test_v1_dispatch_and_exact_state_restore(self):
        mk = driver()
        ext = Extension()
        mk._EXT, mk._gemm_call = ext, ext.run
        original = ext.probe_state()
        mk.run_both_kernels(Tensor(8, 4096), object(), 2048, ksr=3)
        self.assertEqual(ext.calls, [("global", 8, 2048, 3), ("local", 8, 2048, 3)])
        self.assertEqual(ext.probe_state(), original)

    def test_restore_after_launch_failure_or_wrong_plan(self):
        mk = driver()
        for refuse_local in (False, True):
            ext = Extension()
            ext.refuse_local = refuse_local
            original = ext.probe_state()
            mk._EXT = ext
            def launch(*args):
                if not refuse_local:
                    raise RuntimeError("injected launch failure")
                return ext.run(*args)
            mk._gemm_call = launch
            with self.assertRaises(RuntimeError):
                mk.run_both_kernels(Tensor(8, 4096), object(), 1024)
            self.assertEqual(ext.probe_state(), original)

    def test_exact_gate_runs_v2_row_classes_and_odd_v1_split(self):
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
        self.assertEqual({m for lane, m, n, ksr in ext.calls if lane == "v2"}, {8, 16, 32})
        self.assertIn(("local", 8, 2048, 3), ext.calls)
        self.assertEqual(ext.probe_state(), original)

    def test_exact_gate_rejects_v2_nan_and_restores_state(self):
        mk = driver()
        ext = Extension()
        original = ext.probe_state()
        mk._EXT, mk._gemm_call = ext, ext.run
        mk.exact_fixture = lambda **_: (Tensor(32, 4096), object(), object(), Tensor(32, 1024))
        errors = iter([(0.0, 0), (0.0, 0), (math.nan, 0)])
        mk._exact_gate = lambda *a: next(errors)
        with patch.dict(sys.modules, {"torch": torch_stub()}), self.assertRaises(RuntimeError):
            mk._selftest_gemm_exact()
        self.assertEqual(ext.probe_state(), original)


if __name__ == "__main__":
    unittest.main()
