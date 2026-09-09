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
PREPARE = "flashinfer.fused_moe.cute_dsl.blackwell_sm12x.glm53_ep_route_remap"
sys.path.insert(0, str(ROOT / "bench"))
import proof


class Tensor:
    """Small row-view storage model for copy/broadcast/lifetime assertions."""
    next_pointer = 0x100000

    def __init__(self, rows, dtype="bf16", device="cuda:0", operations=None):
        self.rows, self.dtype, self.device = rows, dtype, device
        self.operations = operations
        self.pointer = Tensor.next_pointer
        Tensor.next_pointer += 0x100000
        self.contiguous = True

    def size(self, dim):
        return len(self.rows) if dim == 0 else len(self.rows[0])

    @property
    def shape(self):
        return self.size(0), self.size(1)

    def numel(self):
        return self.size(0) * self.size(1)

    def element_size(self):
        return {"bf16": 2, "fp16": 2, "fp32": 4, "i32": 4, "i64": 8}.get(self.dtype, 1)

    def data_ptr(self):
        return self.pointer

    def is_contiguous(self):
        return self.contiguous

    def __getitem__(self, index):
        view = Tensor(self.rows[index], self.dtype, self.device, self.operations)
        start, _, step = index.indices(self.size(0))
        view.pointer = self.pointer + start * self.size(1) * self.element_size()
        view.contiguous = self.contiguous and step == 1
        return view

    def copy_(self, other):
        if self.operations is not None:
            self.operations.append("copy")
        assert len(self.rows) == len(other.rows)
        for dst, src in zip(self.rows, other.rows):
            dst[:] = src
        return self

    def zero_(self):
        if self.operations is not None:
            self.operations.append("zero")
        for row in self.rows:
            row[:] = [0] * len(row)
        return self

    def expand(self, rows, columns):
        assert len(self.rows) == 1 and columns == -1
        return Tensor(self.rows * rows, self.dtype, self.device, self.operations)


class Harness:
    def __init__(self):
        self.calls, self.fallbacks, self.logs, self.allocations = [], [], [], []
        self.launch_kwargs = []
        self.remaps, self.operations, self.prepares = [], [], []
        self.fail_allocate = False
        names = {"b12x_ep_zero_weight_micro_chunks", "b12x_ep_micro_tail",
                 "_apply_ep_zero_weight_micro", "_ep_tail_padded_micro",
                 "_ep_tail_buffers", "_try_apply_ep_fused_short_decode",
                 "_ensure_ep_scratch", "apply"}
        tree = ast.parse(SOURCE.read_text())
        constants = [n for n in tree.body if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and
                             t.id.startswith("B12X_EP_ZERO_WEIGHT_MICRO_")
                             for t in n.targets)]
        functions = [copy.deepcopy(n) for n in ast.walk(tree)
                     if isinstance(n, ast.FunctionDef) and n.name in names]
        assert {n.name for n in functions} == names
        ns = dict(torch=SimpleNamespace(zeros=self.zeros, empty=self.empty),
                  logger=SimpleNamespace(info_once=self.log, warning_once=self.log),
                  _EP_LOCAL_PREFILL_ENABLED=True,
                  ep_local_prefill_eligible=lambda **kw: False,
                  b12x_ep_stock_topk_micro_chunks=lambda *a, **kw: (),
                  b12x_ep_should_compact=lambda *a, **kw: False)
        module = ast.Module(body=[ast.ImportFrom("__future__", [ast.alias("annotations")], 0),
                                  *constants, *functions], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), ns)
        self.ns = ns
        self.owner = SimpleNamespace(_sf6_weight_views=None,
            _ep_zero_weight_micro=True, _kernel_num_experts=72,
            _ep_zero_weight_workspace=SimpleNamespace(
                ep_micro_scatter_fp32=Tensor([[0] * 4096 for _ in range(8)], "fp32")),
            _ep_fixed_workspace=SimpleNamespace(
                ep_micro_scatter_fp32=Tensor([[0] * 4096 for _ in range(8)], "fp32")),
            _activation_str="swigluoai_uninterleave",
            _swiglu_alpha=1., _swiglu_beta=0., _swiglu_limit=10.,
            _use_ep=True, _ep_no_dummy=True, _ep_stock_topk_micro=False,
            _ep_compact_enabled=False, hidden_dim=3, intermediate_size_per_partition=2048,
            num_local_experts=72, local_expert_offset=0,
            w1_scale=object(), w2_scale=object(),
            w1_sf_mma=object(), w2_sf_mma=object(), _fc2_input_scale=object(),
            g1_alphas=Tensor([[1] for _ in range(72)]),
            g2_alphas=Tensor([[1] for _ in range(72)]),
            _ep_ids=None, _ep_scales=None, _ep_mapped=None, _ep_fill_ids=None,
            max_num_tokens=80, topk=8,
            _remap_ep_tensors=self.remap,
            _apply_ep_fixed=self.fixed,
        )
        ns["torch"].cuda = SimpleNamespace(is_current_stream_capturing=lambda: True)
        ns["torch"].int32 = "i32"
        ns["torch"].int64 = "i64"
        ns["torch"].bool = "bool"
        ns["torch"].bfloat16 = "bf16"
        for name in names - {"b12x_ep_zero_weight_micro_chunks", "b12x_ep_micro_tail"}:
            setattr(self.owner, name, MethodType(ns[name], self.owner))
        self.dispatch = ModuleType(DISPATCH)
        self.dispatch.launch_sm120_moe = self.launch
        self.prepare = ModuleType(PREPARE)
        self.prepare.ep_short_decode_prepare_supported = lambda *args, **kw: False
        self.prepare.try_prepare_ep_short_decode = self.prepare_rows

    def zeros(self, shape, *, dtype, device):
        if self.fail_allocate:
            raise RuntimeError("test allocation failure")
        value = Tensor([[0] * shape[1] for _ in range(shape[0])], dtype, device, self.operations)
        self.allocations.append(value)
        return value

    @staticmethod
    def empty(shape, *, dtype, device):
        # Execute the actual scratch allocator's dtype/shape choices without
        # counting those independent buffers as the four tail allocations.
        return Tensor([[0] * shape[1] for _ in range(shape[0])], dtype, device)

    def log(self, message, *args):
        self.logs.append(message % args)

    def remap(self, ids, weights, *args, **kwargs):
        self.remaps.append(ids.dtype)
        # The actual _ensure_ep_scratch, called by actual apply, chooses these
        # dtypes. Do not assume FP32 or return the raw router tensor unchanged.
        out_ids = self.owner._ep_ids[:ids.size(0)]
        out_scales = self.owner._ep_scales[:weights.size(0)]
        out_ids.copy_(ids)
        out_scales.copy_(weights)
        return out_ids, out_scales

    def prepare_rows(self, x, ids, weights, **kw):
        # Admission/kernel semantics are tested in test_glm53_ep_route_remap;
        # this stub supplies their output to the actual wrapper continuation.
        self.prepares.append((x, ids, weights))
        for row in range(8):
            source = row if row < 6 else 0
            kw["pad_x"].rows[row][:] = x.rows[source]
            kw["pad_ids"].rows[row][:] = ids.rows[source]
            kw["pad_weights"].rows[row][:] = weights.rows[source] if row < 6 else [0] * 8
        return True

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
        self.launch_kwargs.append(kw)
        if "_ep_short_output" in kw:
            plane = kw["_workspace"].ep_micro_scatter_fp32
            self.write_rows(plane, kw["a"], kw["topk_ids"], kw["topk_weights"])
            return kw["_ep_short_output"].copy_(plane[:6])
        self.write_rows(kw["scatter_output"], kw["a"], kw["topk_ids"], kw["topk_weights"])
        return kw["scatter_output"]

    def fixed(self, output, x, w1, w2, ids, weights):
        self.fallbacks.append(x.size(0))
        self.write_rows(output, x, ids, weights)
        return output

    def run(self, tokens, *, all_remote=False, shift=0, id_dtype="i32",
            weights_dtype="fp32", output=None):
        width = self.owner.hidden_dim
        x = Tensor([[shift + r + 1, 2, -3] + [0] * (width - 3) for r in range(tokens)])
        ids = Tensor([[72] * 8 if all_remote else [1, 3] + [72] * 6
                      for _ in range(tokens)], id_dtype)
        weights = Tensor([[0] * 8 if all_remote else [.25, .5] + [0] * 6
                          for _ in range(tokens)], weights_dtype)
        out = output if output is not None else Tensor(
            [[999] * width for _ in range(tokens)], operations=self.operations)
        with patch.dict(sys.modules, {DISPATCH: self.dispatch, PREPARE: self.prepare}):
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


class FusedShortDecodeTests(unittest.TestCase):
    def harness(self):
        h = Harness()
        h.owner.hidden_dim = 4096
        h.prepare.ep_short_decode_prepare_supported = lambda *args, **kw: True
        return h

    def test_exact_six_rows_bypass_torch_remap_and_all_staging_copies(self):
        h = self.harness()
        x, ids, weights, out = h.run(6, id_dtype="i64", weights_dtype="bf16")
        self.assertEqual(h.remaps, [])
        self.assertEqual(len(h.prepares), 1)
        self.assertEqual(len(h.calls), 1)
        self.assertEqual(h.operations, ["copy"])  # one direct final FP32 -> BF16 cast
        self.assertIs(h.launch_kwargs[-1]["_ep_short_output"], out)
        self.assertEqual(h.owner._ep_tail_out.rows, [[0] * 4096] * 8)
        self.assertEqual(h.owner._ep_tail_ids.dtype, "i32")
        self.assertEqual(h.owner._ep_tail_w.dtype, "bf16")
        self.assertEqual(ids.dtype, "i64")
        self.assertEqual(weights.dtype, "bf16")
        self.assertIs(h.prepares[0][1], ids)
        self.assertIs(h.prepares[0][2], weights)
        self.assertEqual(out.rows, [[v * .75 for v in row] for row in x.rows])
        self.assertEqual(h.calls[0]["a"][6:], [x.rows[0]] * 2)
        self.assertEqual(h.calls[0]["topk_ids"][6:], [ids.rows[0]] * 2)
        self.assertEqual(h.calls[0]["topk_weights"][6:], [[0] * 8] * 2)
        self.assertIn("[ep-short-prepare fused=1]", h.logs[-1])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "boot.log"
            path.write_text(h.logs[-1])
            self.assertTrue(proof.check(["VLLM_B12X_EP_ZERO_WEIGHT_MICRO"], str(path))[
                "proof"]["VLLM_B12X_EP_ZERO_WEIGHT_MICRO"])

    def test_same_weight_dtype_preserves_staging_in_both_capture_orders(self):
        names = ("_ep_tail_x", "_ep_tail_ids", "_ep_tail_w", "_ep_tail_out")
        for weights_dtype in ("bf16", "fp16", "fp32"):
            for order in ((18, 6, 12, 6), (6, 18, 12, 6)):
                with self.subTest(weights_dtype=weights_dtype, order=order):
                    h = self.harness()
                    planes = (h.owner._ep_zero_weight_workspace.ep_micro_scatter_fp32,
                              h.owner._ep_fixed_workspace.ep_micro_scatter_fp32)
                    buffers = None
                    for index, tokens in enumerate(order):
                        h.operations.clear()
                        x, ids, weights, out = h.run(
                            tokens, all_remote=index == 2, shift=123 * index,
                            id_dtype="i64", weights_dtype=weights_dtype)
                        current = tuple(getattr(h.owner, name) for name in names)
                        self.assertIs(h.owner._ep_zero_weight_workspace.ep_micro_scatter_fp32,
                                      planes[0])
                        self.assertIs(h.owner._ep_fixed_workspace.ep_micro_scatter_fp32,
                                      planes[1])
                        if buffers is None:
                            buffers = current
                        for before, after in zip(buffers, current):
                            self.assertIs(before, after)
                        self.assertEqual(current[1].dtype, "i32")
                        self.assertEqual(current[2].dtype, weights_dtype)
                        self.assertEqual((ids.dtype, weights.dtype), ("i64", weights_dtype))
                        if tokens == 6:
                            self.assertIs(h.prepares[-1][1], ids)
                            self.assertIs(h.prepares[-1][2], weights)
                            self.assertEqual(h.operations, ["copy"])
                            self.assertIs(h.launch_kwargs[-1]["_ep_short_output"], out)
                        else:
                            self.assertEqual(h.owner._ep_scales.dtype, weights_dtype)
                        scale = 0 if index == 2 else .75
                        self.assertEqual(out.rows, [[v * scale for v in row] for row in x.rows])
                    self.assertEqual(len(h.allocations), 4)
                    self.assertEqual(len(h.remaps), 2)  # only T18 and T12

    def test_direct_output_declines_each_staging_or_plane_alias_and_bad_layout(self):
        for name in ("_ep_tail_x", "_ep_tail_ids", "_ep_tail_w", "_ep_tail_out",
                     "plane", "noncontiguous", "dtype"):
            with self.subTest(name=name):
                h = self.harness()
                h.run(6)
                out = Tensor([[999] * 4096 for _ in range(6)], operations=h.operations)
                if name == "noncontiguous":
                    out.contiguous = False
                elif name == "dtype":
                    out.dtype = "fp32"
                else:
                    buf = (h.owner._ep_zero_weight_workspace.ep_micro_scatter_fp32
                           if name == "plane" else getattr(h.owner, name))
                    out.pointer = buf.data_ptr() + buf.element_size()
                h.operations.clear()
                x, _, _, actual = h.run(6, output=out)
                self.assertIs(actual, out)
                self.assertNotIn("_ep_short_output", h.launch_kwargs[-1])
                self.assertEqual(h.operations, ["copy"])
                self.assertEqual(out.rows, [[v * .75 for v in row] for row in x.rows])

    def test_direct_output_failure_does_not_publish_completion_or_copy_again(self):
        h = self.harness()
        h.dispatch.launch_sm120_moe = lambda **kw: kw["scatter_output"]
        with self.assertRaisesRegex(RuntimeError, "direct T6 output was not published"):
            h.run(6)
        self.assertEqual(h.operations, [])
        self.assertFalse(any("[ep-short-prepare fused=1]" in line for line in h.logs))

    def test_unsupported_or_prelaunch_decline_reuses_original_remap_and_padding(self):
        for decline_at in ("admission", "preparation"):
            h = self.harness()
            if decline_at == "admission":
                h.prepare.ep_short_decode_prepare_supported = lambda *args, **kw: False
            else:
                h.prepare.try_prepare_ep_short_decode = lambda *args, **kw: False
            x, _, _, out = h.run(6, id_dtype="i64")
            self.assertEqual(len(h.remaps), 1)
            self.assertNotIn("_ep_short_output", h.launch_kwargs[-1])
            self.assertEqual(h.operations.count("copy"), 6)
            self.assertEqual(h.operations.count("zero"), 1)
            self.assertEqual(len(h.allocations), 4)
            self.assertEqual(out.rows, [[v * .75 for v in row] for row in x.rows])
            self.assertIn("[ep-short-prepare fused=0]", h.logs[-1])

    def test_allocation_failure_falls_back_but_launch_errors_never_do(self):
        h = self.harness()
        h.fail_allocate = True
        h.run(6)
        self.assertEqual(len(h.remaps), 1)
        self.assertEqual(h.prepares, [])
        self.assertEqual(h.calls, [])
        self.assertEqual(h.fallbacks, [6])
        for fail_at in ("prepare", "compute"):
            h = self.harness()
            def fail(*args, **kwargs):
                raise RuntimeError("device operation failed")
            if fail_at == "prepare":
                h.prepare.try_prepare_ep_short_decode = fail
            else:
                h.dispatch.launch_sm120_moe = fail
            with self.assertRaisesRegex(RuntimeError, "device operation failed"):
                h.run(6)
            self.assertEqual(h.remaps, [])
            self.assertEqual(h.fallbacks, [])
            self.assertFalse(any("[ep-short-prepare fused=1]" in line for line in h.logs))

    def test_other_token_counts_geometry_or_disabled_flag_never_prepare(self):
        for tokens, width, intermediate, enabled in (
            (5, 4096, 2048, True), (7, 4096, 2048, True), (8, 4096, 2048, True),
            (6, 4095, 2048, True), (6, 4096, 1024, True), (6, 4096, 2048, False),
        ):
            h = self.harness()
            h.owner.hidden_dim, h.owner.intermediate_size_per_partition = width, intermediate
            h.owner._ep_zero_weight_micro = enabled
            h.run(tokens)
            self.assertEqual(h.prepares, [])
            self.assertEqual(len(h.remaps), 1)


if __name__ == "__main__":
    unittest.main()
