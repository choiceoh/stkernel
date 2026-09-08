"""Exercise real wrapper control flow/staging on CPU; not GPU numerics."""
import ast
import copy
from pathlib import Path
import sys
import tempfile
from types import MethodType, ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "overlay/modules/glm53_moe/flashinfer_b12x_moe.py"
DISPATCH = "flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch"
sys.path.insert(0, str(ROOT / "bench"))
import proof


class Tensor:
    """Small row-view storage model for copy/broadcast/lifetime assertions."""
    def __init__(self, rows, dtype="bf16", device="cuda:0"):
        self.rows, self.dtype, self.device = rows, dtype, device

    def size(self, dim):
        return len(self.rows) if dim == 0 else len(self.rows[0])

    @property
    def shape(self):
        return self.size(0), self.size(1)

    def numel(self):
        return self.size(0) * self.size(1)

    def __getitem__(self, index):
        return Tensor(self.rows[index], self.dtype, self.device)

    def copy_(self, other):
        assert len(self.rows) == len(other.rows)
        for dst, src in zip(self.rows, other.rows):
            dst[:] = src
        return self

    def zero_(self):
        for row in self.rows:
            row[:] = [0] * len(row)
        return self

    def expand(self, rows, columns):
        assert len(self.rows) == 1 and columns == -1
        return Tensor(self.rows * rows, self.dtype, self.device)


class Harness:
    def __init__(self):
        self.calls, self.fallbacks, self.logs, self.allocations = [], [], [], []
        self.fail_allocate = False
        names = {"b12x_ep_zero_weight_micro_chunks", "b12x_ep_micro_tail",
                 "_apply_ep_zero_weight_micro", "_ep_tail_padded_micro",
                 "_ep_tail_buffers", "apply"}
        tree = ast.parse(SOURCE.read_text())
        constants = [n for n in tree.body if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and
                             t.id.startswith("B12X_EP_ZERO_WEIGHT_MICRO_")
                             for t in n.targets)]
        functions = [copy.deepcopy(n) for n in ast.walk(tree)
                     if isinstance(n, ast.FunctionDef) and n.name in names]
        assert {n.name for n in functions} == names
        ns = dict(torch=SimpleNamespace(zeros=self.zeros),
                  logger=SimpleNamespace(info_once=self.log, warning_once=self.log),
                  _EP_LOCAL_PREFILL_ENABLED=True,
                  ep_local_prefill_eligible=lambda **kw: False,
                  b12x_ep_stock_topk_micro_chunks=lambda *a, **kw: (),
                  b12x_ep_should_compact=lambda *a, **kw: False)
        module = ast.Module(body=[ast.ImportFrom("__future__", [ast.alias("annotations")], 0),
                                  *constants, *functions], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), ns)
        self.ns = ns
        self.owner = SimpleNamespace(
            _ep_zero_weight_micro=True, _kernel_num_experts=72,
            _ep_zero_weight_workspace=object(), _activation_str="swigluoai_uninterleave",
            _swiglu_alpha=1., _swiglu_beta=0., _swiglu_limit=10.,
            _use_ep=True, _ep_no_dummy=True, _ep_stock_topk_micro=False,
            _ep_compact_enabled=False, hidden_dim=3, intermediate_size_per_partition=2048,
            num_local_experts=72, w1_scale=object(), w2_scale=object(),
            w1_sf_mma=object(), w2_sf_mma=object(), _fc2_input_scale=object(),
            g1_alphas=Tensor([[1] for _ in range(72)]),
            g2_alphas=Tensor([[1] for _ in range(72)]),
            _ensure_ep_scratch=lambda *args: None,
            _remap_ep_tensors=lambda ids, weights, *args, **kw: (ids, weights),
            _apply_ep_fixed=self.fixed,
        )
        ns["torch"].cuda = SimpleNamespace(is_current_stream_capturing=lambda: True)
        ns["torch"].int32 = "i32"
        for name in names - {"b12x_ep_zero_weight_micro_chunks", "b12x_ep_micro_tail"}:
            setattr(self.owner, name, MethodType(ns[name], self.owner))
        self.dispatch = ModuleType(DISPATCH)
        self.dispatch.launch_sm120_moe = self.launch

    def zeros(self, shape, *, dtype, device):
        if self.fail_allocate:
            raise RuntimeError("test allocation failure")
        value = Tensor([[0] * shape[1] for _ in range(shape[0])], dtype, device)
        self.allocations.append(value)
        return value

    def log(self, message, *args):
        self.logs.append(message % args)

    @staticmethod
    def write_rows(output, x, ids, weights):
        for dst, row, experts, scales in zip(output.rows, x.rows, ids.rows, weights.rows):
            assert all(expert < 72 or weight == 0 for expert, weight in zip(experts, scales))
            total = sum(weight for expert, weight in zip(experts, scales) if expert < 72)
            dst[:] = [value * total for value in row]

    def launch(self, **kw):
        assert kw["a"].size(0) == 8 and kw["top_k"] == 8
        assert kw["num_experts"] == kw["num_local_experts"] == 72
        assert kw["_workspace"] is self.owner._ep_zero_weight_workspace
        self.calls.append({key: copy.deepcopy(kw[key].rows)
                           for key in ("a", "topk_ids", "topk_weights")})
        self.write_rows(kw["scatter_output"], kw["a"], kw["topk_ids"], kw["topk_weights"])

    def fixed(self, output, x, w1, w2, ids, weights):
        self.fallbacks.append(x.size(0))
        self.write_rows(output, x, ids, weights)
        return output

    def run(self, tokens, *, all_remote=False, shift=0):
        x = Tensor([[shift + r + 1, 2, -3] for r in range(tokens)])
        ids = Tensor([[72] * 8 if all_remote else [1, 3] + [72] * 6
                      for _ in range(tokens)], "i32")
        weights = Tensor([[0] * 8 if all_remote else [.25, .5] + [0] * 6
                          for _ in range(tokens)], "fp32")
        out = Tensor([[999] * 3 for _ in range(tokens)])
        with patch.dict(sys.modules, {DISPATCH: self.dispatch}):
            self.owner.apply(out, x, Tensor([[0]] * 72), object(), weights, ids,
                             None, 288, None, None, None, None, None, None, None)
        return x, ids, weights, out


class ShortDecodeTests(unittest.TestCase):
    def test_actual_apply_reaches_single_padded_call_for_all_short_batches(self):
        for tokens in range(1, 8):
            with self.subTest(tokens=tokens):
                h = Harness()
                x, ids, weights, out = h.run(tokens)
                self.assertEqual(len(h.calls), 1)
                self.assertEqual(h.fallbacks, [])
                self.assertEqual(out.rows, [[v * .75 for v in row] for row in x.rows])
                call = h.calls[0]
                self.assertEqual(call["a"][tokens:], [x.rows[0]] * (8 - tokens))
                self.assertEqual(call["topk_ids"][tokens:], [ids.rows[0]] * (8 - tokens))
                self.assertEqual(call["topk_weights"][tokens:], [[0] * 8] * (8 - tokens))

    def test_spec5_graph_shapes_reuse_staging_and_preserve_every_real_row(self):
        h = Harness()
        for tokens, calls in ((24, 3), (18, 3), (12, 2), (6, 1)):
            before = len(h.calls)
            x, _, _, out = h.run(tokens, shift=tokens * 10)
            self.assertEqual(len(h.calls) - before, calls)
            self.assertEqual(out.rows, [[v * .75 for v in row] for row in x.rows])
        self.assertEqual(len(h.allocations), 4)
        self.assertEqual(h.fallbacks, [])

    def test_all_remote_rows_clear_previous_staging_output(self):
        h = Harness()
        h.run(6)
        _, _, _, out = h.run(6, all_remote=True)
        self.assertEqual(out.rows, [[0] * 3] * 6)
        self.assertEqual(len(h.allocations), 4)

    def test_padding_allocation_failure_keeps_disjoint_fixed_fallback(self):
        for tokens, full_calls, remainder in ((6, 0, 6), (18, 2, 2)):
            h = Harness()
            h.fail_allocate = True
            x, _, _, out = h.run(tokens)
            self.assertEqual(len(h.calls), full_calls)
            self.assertEqual(h.fallbacks, [remainder])
            self.assertEqual(out.rows, [[v * .75 for v in row] for row in x.rows])
            self.assertFalse(any("b12x EP zero-weight micro:" in line for line in h.logs))

    def test_disabled_path_and_invalid_tail_geometry_retain_fallback(self):
        h = Harness()
        h.owner._ep_zero_weight_micro = False
        h.run(6)
        self.assertEqual(h.calls, [])
        self.assertEqual(h.fallbacks, [6])
        tail = h.ns["b12x_ep_micro_tail"]
        for tokens, topk, experts, enabled in ((0, 8, 72, True), (81, 8, 72, True),
                                              (6, 7, 72, True), (6, 8, 71, True),
                                              (6, 8, 72, False)):
            self.assertIsNone(tail(tokens, topk, experts, enabled=enabled))

    def test_proof_requires_completed_coherent_call_counts(self):
        knob = "VLLM_B12X_EP_ZERO_WEIGHT_MICRO"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "boot.log"
            good = "b12x EP zero-weight micro: 6 tokens -> 1 top-k=8 calls (8 tokens / 64 routed pairs each; padded tail=1)"
            for text, expected in ((good, True), ("armed", False),
                                   (good.replace("-> 1", "-> 6"), False),
                                   (good.replace("tail=1", "tail=0"), False),
                                   (good.replace("6 tokens", "0 tokens"), False),
                                   (good.replace("; padded tail=1", ""), False)):
                path.write_text(text)
                self.assertIs(proof.check([knob], str(path))["proof"][knob], expected)


if __name__ == "__main__":
    unittest.main()
