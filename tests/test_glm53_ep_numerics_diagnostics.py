"""CPU-only failure diagnostics; fake tensors never import an accelerator."""
import copy
import math
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "probes"))
import glm53_ep_local_check as probe
import glm53_ep_numerics_diagnostics as diagnostics


def word(value):
    return struct.unpack("<I", struct.pack("<f", value))[0] >> 16


def value(bits):
    return struct.unpack("<f", struct.pack("<I", (bits & 65535) << 16))[0]


class Tensor:
    """Small list-backed adapter with an observable host-copy boundary."""
    def __init__(self, data, dtype="float", name="", copies=None):
        self.data, self.dtype, self.name = data, dtype, name
        self.copies = copies if copies is not None else []

    @property
    def shape(self):
        if not isinstance(self.data, list):
            return ()
        return (len(self.data),) + (Tensor(self.data[0]).shape if self.data else ())

    def __getitem__(self, index):
        return Tensor(self.data[index], self.dtype, f"{self.name}[{index}]", self.copies)

    def __float__(self):
        return float(self.data)

    def __int__(self):
        return int(self.data)

    def __rmul__(self, other):
        return Tensor([diagnostics._f32(other * x) for x in self.data])

    def __gt__(self, other):
        return Tensor([a > b for a, b in zip(self.data, other.data)])

    def __or__(self, other):
        return Tensor([a or b for a, b in zip(self.data, other.data)])

    def sum(self):
        return sum(self.data)

    def max(self):
        return max(self.data)

    def detach(self):
        return self

    def cpu(self):
        self.copies.append((self.name, self.shape))
        return self

    def tolist(self):
        return copy.deepcopy(self.data)

    def nonzero(self, *, as_tuple):
        assert as_tuple is False
        return Tensor([[i] for i, v in enumerate(self.data) if v])

    def flatten(self):
        return Tensor([x for row in self.data for x in row])

    def view(self, dtype):
        assert self.dtype == "bf16" and dtype == "int16"
        return Tensor([((x + 32768) % 65536) - 32768 for x in self.data])

    def float(self):
        assert self.dtype == "bf16"
        return Tensor([value(x) for x in self.data])

    def norm(self):
        return math.sqrt(sum(x*x for x in self.data))

    def abs(self):
        return Tensor([abs(x) for x in self.data])

    def amax(self):
        return max(self.data)


def fake_torch():
    return SimpleNamespace(
        bfloat16="bf16", int16="int16",
        maximum=lambda a, b: Tensor([max(x, y) for x, y in zip(a.data, b.data)]),
        full_like=lambda a, fill: Tensor([fill]*len(a.data)))


def row_record(row_id=0):
    return dict(
        row_id=row_id,
        metrics=dict(relative_l2=.0118, relative_peak=.0407,
                     stock_relative_l2=.004, stock_relative_peak=.01,
                     l2_limit=.02, peak_limit=.04, b1_l2_norm=4., b1_absmax=1.),
        routes=[dict(slot=i, global_expert_id=i+3, local_expert_id=i+3,
                     weight=.125, scales=dict(fc1_input=.9, fc1_alpha=.9,
                                             fc2_input=.9, fc2_alpha=1.1)) for i in range(8)],
        raw_bf16={name: [word(1.)]*12 for name in ("B1", "B2", "B3", "C1", "X")})


class RecordTests(unittest.TestCase):
    def test_bounds_selection_and_original_row_metrics(self):
        def records():
            for i in range(8):
                row = row_record(i*2)
                row["raw_bf16"]["C1"] = [word(float(j+1)) for j in range(12)]
                yield row
            raise AssertionError("must not consume a ninth row")
        got = diagnostics.build_failure_record(records(), total_bad_rows=11)
        self.assertEqual(got["captured_bad_rows"], 8)
        self.assertTrue(got["truncated"])
        self.assertEqual([r["row_id"] for r in got["rows"]], list(range(0, 16, 2)))
        for row in got["rows"]:
            self.assertEqual(row["metrics"], row_record()["metrics"])
            self.assertEqual(row["violations"], {"l2": False, "peak": True})
            self.assertEqual([c["column"] for c in row["worst_columns"]], list(range(11, 3, -1)))
            self.assertNotIn("raw_bf16", row)

    def test_raw_words_signed_zero_and_stable_ties(self):
        row = row_record()
        for i, name in enumerate(("B1", "B2", "B3", "C1", "X")):
            row["raw_bf16"][name][0] = [0, -32768, -16512, 16256, -16256][i]
        got = diagnostics.build_failure_record([row], total_bad_rows=1)
        columns = got["rows"][0]["worst_columns"]
        self.assertEqual([c["column"] for c in columns], list(range(8)))
        self.assertEqual(columns[0]["absolute_delta_f32"], 1.)
        self.assertEqual(columns[0]["raw_bf16_u16"],
                         dict(B1=0, B2=32768, B3=49024, C1=16256, X=49280))
        self.assertIn("not a causal attribution", got["input_column_scope"])

    def test_rejects_nonfailed_or_malformed_rows(self):
        mutations = [lambda r: r["metrics"].update(peak_limit=.0407),
                     lambda r: r["metrics"].update(b1_absmax=float("nan")),
                     lambda r: r["raw_bf16"]["B3"].pop(),
                     lambda r: r["routes"].pop()]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                row = row_record(); mutate(row)
                with self.assertRaises(ValueError):
                    diagnostics.build_failure_record([row], total_bad_rows=1)
        with self.assertRaises(ValueError):
            diagnostics.build_failure_record([row_record()], total_bad_rows=0)

    def test_capture_copies_only_selected_rows_and_preserves_route_scale_mapping(self):
        copies = []
        outputs = {name: Tensor([[word(1.+i/8)]*12 for _ in range(12)], "bf16", name, copies)
                   for i, name in enumerate(("B1", "B2", "B3", "C1", "X"))}
        ids = [0, 1, 2, -1, 999, 0, 2, 1]
        with patch.dict(sys.modules, torch=fake_torch()):
            got = diagnostics.capture_failure(
                outputs["C1"], outputs["B1"], outputs["B2"], outputs["B3"],
                bad=Tensor([False]+[True]*11), error=Tensor([.01]*12), peak=Tensor([.05]*12),
                noise=Tensor([.002]*12), peak_noise=Tensor([.005]*12),
                l2_limits=Tensor([.02]*12), peak_limits=Tensor([.04]*12), total_bad_rows=11,
                route_ids=Tensor([ids]*12), route_weights=Tensor([[.125]*8]*12),
                expert_map=Tensor([0, -1, 1]), scales={"fc1_input": Tensor([.8, 1.2])},
                inputs=outputs["X"])
        self.assertEqual([r["row_id"] for r in got["rows"]], list(range(1, 9)))
        self.assertEqual(copies, [(f"{name}[{i}]", (12,)) for i in range(1, 9)
                                 for name in ("B1", "B2", "B3", "C1", "X")])
        row = got["rows"][0]
        self.assertEqual(row["metrics"]["b1_l2_norm"], math.sqrt(12))
        self.assertEqual(row["metrics"]["b1_absmax"], 1.)
        self.assertEqual([r["local_expert_id"] for r in row["routes"]], [0, -1, 1, -1, -1, 0, 1, -1])
        self.assertEqual([r["scales"]["fc1_input"] for r in row["routes"]],
                         [.8, None, 1.2, None, None, .8, 1.2, None])
        self.assertEqual(row["worst_columns"][0]["raw_bf16_u16"]["B3"], word(1.25))


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.context = dict(result=dict(phase="changed-candidate", verdict="RUNNING"),
                            third=object(), route_ids=object(), route_weights=object(),
                            expert_map=object(), scales=object(), inputs=object())
        self.outputs = [object(), object(), object()]

    def compare(self, *, fail, capture=None):
        errors = (Tensor([.01, .01]), Tensor([.045 if fail else .039, .05]))
        noise = (Tensor([.002, .009]), Tensor([.01, .018]))
        with (patch.dict(sys.modules, torch=fake_torch()),
              patch.object(probe, "check_control"),
              patch.object(probe, "row_errors", side_effect=[errors, noise]),
              patch.object(probe, "capture_failure", capture or self.capture)):
            return probe.compare(*self.outputs, failure_context=self.context)

    def test_success_has_no_capture_or_result_mutation(self):
        with patch.object(probe, "capture_failure") as self.capture:
            got = self.compare(fail=False)
            self.capture.assert_not_called()
        self.assertEqual(got["bad_rows"], 0)
        self.assertEqual(self.context["result"], dict(phase="changed-candidate", verdict="RUNNING"))

    def test_failure_uses_per_row_limits_and_never_replaces_first_failure(self):
        with patch.object(probe, "capture_failure", return_value={"captured": 1}) as self.capture:
            with self.assertRaisesRegex(AssertionError, "CANDIDATE_NUMERICS_FAIL") as thrown:
                self.compare(fail=True)
            args, kwargs = self.capture.call_args
            self.assertEqual(args, (*self.outputs, self.context["third"]))
            self.assertEqual(kwargs["bad"].data, [True, False])
            self.assertEqual(kwargs["l2_limits"].data, [.02, diagnostics._f32(3*.009)])
            self.assertEqual(kwargs["peak_limits"].data, [.04, diagnostics._f32(3*.018)])
            self.assertEqual(kwargs["total_bad_rows"], 1)
            first = copy.deepcopy(self.context["result"])
            self.context["result"]["phase"] = "later-candidate"
            with self.assertRaisesRegex(AssertionError, "CANDIDATE_NUMERICS_FAIL"):
                self.compare(fail=True)
            self.capture.assert_called_once()
        self.assertEqual(self.context["result"]["verdict"], "FAIL")
        self.assertEqual(self.context["result"]["candidate_first_failure"], first["candidate_first_failure"])
        self.assertEqual(self.context["result"]["candidate_failure_diagnostics"], {"captured": 1})
        self.assertEqual(thrown.exception.args[0]["bad_rows"], 1)

    def test_capture_failure_cannot_replace_original_assertion(self):
        with patch.object(probe, "capture_failure", side_effect=RuntimeError("copy failed")) as self.capture:
            with self.assertRaisesRegex(AssertionError, "CANDIDATE_NUMERICS_FAIL"):
                self.compare(fail=True)
        self.assertEqual(self.context["result"]["verdict"], "FAIL")
        self.assertIn("copy failed", self.context["result"]["candidate_failure_diagnostics_error"])
        self.assertEqual(self.context["result"]["candidate_first_failure"]["phase"], "changed-candidate")

    def test_stock_failure_is_not_reclassified_or_captured(self):
        with (patch.dict(sys.modules, torch=fake_torch()),
              patch.object(probe, "check_control", side_effect=AssertionError("UNSTABLE_STOCK_CONTROL")),
              patch.object(probe, "capture_failure") as capture,
              self.assertRaisesRegex(AssertionError, "UNSTABLE_STOCK_CONTROL")):
            probe.compare(*self.outputs, failure_context=self.context)
        capture.assert_not_called()
        self.assertNotIn("candidate_first_failure", self.context["result"])


if __name__ == "__main__":
    unittest.main()
