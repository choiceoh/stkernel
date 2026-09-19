"""The Qwen3.8 dense sweep probe (engine/QWEN38_CARRY.md C2): its cases are the lanes Qwen38Net.prepare_dense builds for the
served facts (probes/qwen38_config.json) and the widths the wizard asks, built with prepare_dense's arguments; its two arms
are DenseLinear's two branches on either side of the switch; its gate, crossover and plan arithmetic is right. CPU only:
prepare_dense runs over meta tensors with recording lanes, and where a branch is followed the native W4 kernel and the
FP8 lane are stand-ins."""
import contextlib
import importlib.util
from types import SimpleNamespace
from pathlib import Path
import unittest
from unittest.mock import patch

torch = None
if importlib.util.find_spec("torch") is not None:
    import torch


@contextlib.contextmanager
def recording_lanes():
    """engine/kernels/dense's three lane classes replaced by recorders of what they were built from."""
    from engine.kernels import dense

    class Recorder:
        label = None

        def __init__(self, weight, **options):
            self.shape, self.options = tuple(weight.shape), options

    stand_ins = {label: type(label, (Recorder,), {"label": label})
                 for label in ("DenseLinear", "PaddedDenseLinear", "FP8Linear")}
    with contextlib.ExitStack() as stack:
        for label, cls in stand_ins.items():
            stack.enter_context(patch.object(dense, label, cls))
        yield


@unittest.skipUnless(torch is not None, "requires torch")
class Cases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from probes import engine_qwen38_dense
        from probes.engine_qwen38_cells import facts
        cls.probe, cls.F = engine_qwen38_dense, facts()

    def prepare_dense(self):
        """{key: (class, weight shape, options)}: the real Qwen38Net.prepare_dense, no store, over meta tensors of every
        served spec (the MTP head's included, as the fleet serves it)."""
        from engine.profiles.qwen38 import specs
        from engine.profiles.qwen38.net import Qwen38Net
        net = object.__new__(Qwen38Net)
        net.p = {s.name: torch.empty(s.shape, dtype=s.dtype, device="meta") for s in specs.all_specs(self.F, mtp=True)}
        net.hc_fp8 = False
        with recording_lanes():
            net.prepare_dense(None)
        return {key: (lane.label, lane.shape, lane.options) for key, lane in net.dense.items()}

    def test_the_cases_are_the_lanes_prepare_dense_builds(self):
        built = self.prepare_dense()
        self.assertEqual(built.pop("head")[0], "FP8Linear")          # FP8 only: no W4A8/FP8 switch to measure
        expected = {}
        for key, (label, shape, _) in built.items():
            expected.setdefault((label, *shape), []).append(key)
        cases = self.probe.qwen38_cases(self.F)
        self.assertEqual({(c.lane, c.rows, c.cols): list(c.keys) for c in cases}, expected)
        self.assertEqual(len(cases), len(expected))
        # every layer's mixer in/out and shared expert gate_up/down; the MTP head's are BF16 (no lane) by default
        self.assertEqual(sum(len(c.keys) for c in cases), 4 * self.F.layers)
        self.assertTrue(all(c.model == "qwen38" for c in cases))

    def test_the_probe_builds_each_lane_with_prepare_dense_s_arguments(self):
        built = self.prepare_dense()
        for case in self.probe.qwen38_cases(self.F) + self.probe.dsv41_cases():
            with self.subTest(case=case.label), recording_lanes():
                lane = self.probe.build_lane(case, torch.empty(case.rows, case.cols, dtype=torch.bfloat16, device="meta"))
                self.assertEqual(lane.label, case.lane)
                if case.keys:
                    self.assertEqual((lane.label, lane.shape, lane.options), built[case.keys[0]])
                else:           # V4.1 has no net: the padded lane as prepare_dense builds one, without a name
                    self.assertEqual(lane.options, dict(prefill=True, store=None, name=None, smooth=None))

    def test_the_widths_are_the_ones_the_wizard_asks(self):
        from engine.kernels import cells
        from engine.kernels.dense import padded_columns
        from engine.profiles.dsv41 import shapes as dsv41
        from tests.test_engine_kernel_shape import DSV41_TEXT_CONFIG
        qwen = self.probe.qwen38_cases(self.F)
        for shape, cases in ((self.F.kernel_shape(), qwen), (dsv41.kernel_shape(DSV41_TEXT_CONFIG), self.probe.dsv41_cases())):
            widths = {width for _, width in cells._dense_widths(shape, shape.moe)[0]}
            with self.subTest(hidden=shape.hidden):
                self.assertEqual({c.cols for c in cases if c.lane == "PaddedDenseLinear"},
                                 {w for w in widths if w % cells.DENSE_ALIGN})
        self.assertIn(self.F.hidden, {c.cols for c in qwen})
        self.assertEqual([(c.model, c.rows, c.cols, padded_columns(c.cols)) for c in self.probe.dsv41_cases()],
                         [("dsv41", 5120, 576, 640)])
        self.assertEqual({(c.cols, padded_columns(c.cols)) for c in qwen if c.lane == "PaddedDenseLinear"}, {(160, 256)})

    def test_the_captured_rows_are_the_decode_graphs(self):
        from engine.profiles.qwen38.fleet import MAX_SEQS
        self.assertEqual(self.probe.captured_rows(self.F), MAX_SEQS * (self.F.spec_k + 1))
        self.assertEqual([m for m in self.probe.ROWS if m <= self.probe.captured_rows(self.F)], [1, 2, 4, 8])

    def test_every_case_has_a_distinct_label(self):
        cases = self.probe.qwen38_cases(self.F) + self.probe.dsv41_cases()
        self.assertEqual(len({c.label for c in cases}), len(cases))
        self.assertIn("qwen38:gdn.out_proj+attn.o_proj:2560x1536", {c.label for c in cases})


@unittest.skipUnless(torch is not None, "requires torch")
class Arms(unittest.TestCase):
    """probe.arms are DenseLinear.__call__'s branches on either side of its switch, with PaddedDenseLinear's widened input.
    The native W4 kernel and the FP8 lane are stand-ins computing different products of the widened input, so a call
    that took the other branch, or an unwidened input, cannot match."""

    def layer(self, cls, rows, cols, pad):
        from engine.kernels import dense
        generator = torch.Generator().manual_seed(2560)
        w4 = torch.randn(rows, cols + pad, generator=generator)
        fp8 = torch.randn(rows, cols + pad, generator=generator)

        class FP8:
            cublas = None

            def __call__(self, x, rows_ok=None, *, out=None, decode=False, normalization=None):
                return (x.float() @ fp8.T).bfloat16()

        def run_gemm(x, data, scale, out, n, *rest):
            out.copy_((x.float() @ w4.T).bfloat16())

        layer = object.__new__(cls)
        layer.rows, layer.cols = rows, cols + pad
        if cls is dense.PaddedDenseLinear:
            layer.input_cols, layer.pad = cols, pad
        layer.packs = (dense.W4Pack(torch.zeros(1, dtype=torch.uint8), torch.zeros(1, dtype=torch.int8),
                                    torch.ones(rows), rows, cols + pad),)
        layer.fp8, layer.decode_fp8, layer.decode_precision, layer.calibrated = FP8(), None, "w4", False
        layer.workspace, layer.decode_input_rows, layer.observer, layer.executed = None, (), None, 0
        layer.bound_input_executed, layer.producer_pack_executed = set(), set()
        stub = SimpleNamespace(run_gemm=run_gemm, run_gemm_wide_input=run_gemm)
        return layer, patch.object(dense, "extension", lambda: stub)

    def test_the_arms_are_the_dispatch_branches(self):
        from engine.kernels import dense
        from probes import engine_qwen38_dense as probe
        for cls, cols, pad in ((dense.DenseLinear, 256, 0), (dense.PaddedDenseLinear, 160, 96)):
            layer, stubbed = self.layer(cls, 64, cols, pad)
            self.assertEqual(probe.served_form(layer), [])
            with self.subTest(lane=cls.__name__), stubbed:
                arms = probe.arms(layer)
                for rows in (1, probe.W4A8_ROWS, probe.W4A8_ROWS + 1, 48):
                    x = torch.randn(rows, cols).bfloat16()
                    served = layer(x)
                    if rows <= probe.W4A8_ROWS:
                        self.assertTrue(torch.equal(served, arms["w4a8"](x)), rows)
                        self.assertFalse(torch.equal(served, arms["fp8"](x)), rows)
                    else:
                        self.assertTrue(torch.equal(served, arms["fp8"](x)), rows)
                        with self.assertRaisesRegex(ValueError, "1..32 BF16 rows"):
                            arms["w4a8"](x)

    def test_a_layer_off_the_served_form_is_refused(self):
        from engine.kernels import dense
        from probes import engine_qwen38_dense as probe
        layer, _ = self.layer(dense.DenseLinear, 64, 256, 0)
        layer.decode_input_rows = (8,)
        layer.fp8.cublas = object()
        self.assertEqual(len(probe.served_form(layer)), 2)


class Arithmetic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch is None:
            raise unittest.SkipTest("the probe imports torch")
        from probes import engine_qwen38_dense
        cls.probe = engine_qwen38_dense

    def test_the_crossover_is_where_fp8_stays_no_slower(self):
        crossover = self.probe.crossover
        w4a8 = {1: 10., 2: 10., 4: 10., 8: 10., 16: 10., 32: 10.}
        self.assertEqual(crossover(w4a8, {1: 30., 2: 20., 4: 12., 8: 9., 16: 8., 32: 5.}), 8)
        self.assertEqual(crossover(w4a8, {1: 30., 2: 9., 4: 12., 8: 11., 16: 8., 32: 5.}), 16)    # not the first win
        self.assertIsNone(crossover(w4a8, {1: 9., 2: 9., 4: 9., 8: 9., 16: 9., 32: 11.}))
        self.assertEqual(crossover(w4a8, {m: 1. for m in w4a8}), 1)
        self.assertEqual(crossover(w4a8, {m: 10. for m in w4a8}), 1)                              # a tie is no slower
        # only the row counts both arms ran count: FP8's rows above the switch do not move it
        self.assertIsNone(crossover(w4a8, {1: 30., 2: 30., 4: 30., 8: 30., 16: 30., 32: 30., 48: 1., 4096: 1.}))

    def test_the_bands_are_the_dense_test_s_by_lane(self):
        band = self.probe.band
        self.assertEqual([band("w4a8", m) for m in (1, 32)], [.16, .16])
        self.assertEqual([band("fp8", m) for m in (1, 32, 33, 512, 1023, 1024, 4096)], [.05, .05, .05, .05, .05, .16, .16])

    def test_the_gate_fails_every_way_it_names(self):
        failures = self.probe.gate_failures
        good = dict(rows=8, dispatch="w4a8", dispatch_exact=True, relative_error={"w4a8": .1, "fp8": .02},
                    band={"w4a8": .16, "fp8": .05}, twin_error=.001, padded_exact=True)
        self.assertEqual(failures(good), [])
        aligned = {k: v for k, v in good.items() if k != "padded_exact"}          # an aligned lane has no padded check
        self.assertEqual(failures(aligned), [])
        for change in (dict(dispatch_exact=False), dict(relative_error={"w4a8": .16, "fp8": .02}),
                       dict(relative_error={"w4a8": float("nan"), "fp8": .02}), dict(twin_error=.006),
                       dict(twin_error=float("nan")), dict(padded_exact=False)):
            with self.subTest(change=change):
                self.assertEqual(len(failures({**good, **change})), 1)
        fp8_only = dict(rows=4096, dispatch="fp8", dispatch_exact=True, relative_error={"fp8": .03}, band={"fp8": .16})
        self.assertEqual(failures(fp8_only), [])
        self.assertEqual(len(failures({**fp8_only, "relative_error": {"fp8": .2}})), 1)

    def test_an_unserved_arm_off_its_band_is_recorded_not_raised(self):
        """The first lane run (measurements/qwen38_lane_20260917): FP8 at 4 rows of 320x2560 was 22% off while W4A8
        served those rows; the gate stopped the whole run on it."""
        failures, broken = self.probe.gate_failures, self.probe.broken_arms
        row = dict(rows=4, dispatch="w4a8", dispatch_exact=True, relative_error={"w4a8": .085, "fp8": .22},
                   band={"w4a8": .16, "fp8": .05}, twin_error=0.0)
        self.assertEqual((failures(row), broken(row)), ([], ["fp8"]))
        self.assertEqual(broken({**row, "relative_error": {"w4a8": .085, "fp8": float("nan")}}), ["fp8"])
        self.assertEqual(broken({**row, "relative_error": {"w4a8": .085, "fp8": .04}}), [])
        served = dict(rows=64, dispatch="fp8", dispatch_exact=True, relative_error={"fp8": .22}, band={"fp8": .05})
        self.assertEqual((len(failures(served)), broken(served)), (1, []))

    def test_a_broken_cell_is_neither_captured_nor_timed(self):
        source = (Path(__file__).resolve().parents[1] / "probes" / "engine_qwen38_dense.py").read_text()
        self.assertIn("if (name == 'w4a8' and m > W4A8_ROWS) or (m, name) in broken:", source)
        self.assertIn("if (m, name) in broken[i]:\n            continue", source)

    def test_relative_error_and_summary(self):
        want = torch.tensor([[3., 4.]])
        self.assertEqual(self.probe.relative_error(want.bfloat16(), want), 0.)
        self.assertAlmostEqual(self.probe.relative_error(torch.tensor([[3., 4.5]]), want), .1)
        row = self.probe.summary([5., 1., 3.], [2., 2., 8.], 4)
        self.assertEqual((row["cold_us"], row["warm_us"], row["cold_min_us"], row["warm_min_us"]), (3., 2., 1., 2.))
        self.assertEqual((row["cold_us_per_row"], row["samples"]), (.75, 3))

    def test_the_plan_times_w4a8_where_it_is_admitted_and_captures_decode_rows(self):
        cases = [object(), object()]
        cells = self.probe.plan(cases, self.probe.ROWS, 8)
        for i in range(2):
            self.assertEqual([m for j, m, arm, _ in cells if j == i and arm == "w4a8"], [1, 2, 4, 8, 16, 32])
            self.assertEqual([m for j, m, arm, _ in cells if j == i and arm == "fp8"], list(self.probe.ROWS))
        self.assertEqual({m for _, m, _, mode in cells if mode == "captured"}, {1, 2, 4, 8})
        self.assertEqual(self.probe.ROWS, (1, 2, 4, 8, 16, 32, 48, 64, 128, 512, 4096))
        self.assertEqual(self.probe.projection("mtp.L0.attn.in_proj"), "attn.in_proj")
        self.assertEqual(self.probe.projection("L12.moe.sh_down"), "moe.sh_down")


class LaneRoutingTests(unittest.TestCase):
    def test_kernel_check_routes_the_lane_to_the_probe(self):
        from pathlib import Path
        text = (Path(__file__).resolve().parents[1] / "probes" / "engine_kernel_check.py").read_text()
        self.assertIn("args.lanes == 'qwen38_dense'", text)
        self.assertIn("from probes.engine_qwen38_dense import run as qwen38_dense", text)


if __name__ == "__main__":
    unittest.main()
