"""GLM precision ports: selection before BF16 rounding, real-input calibration, and W8A16 head arithmetic."""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest import mock

torch = None
if importlib.util.find_spec("torch"):
    import torch


@unittest.skipUnless(torch is not None, "requires torch")
class PrecisionPortTests(unittest.TestCase):
    def test_router_preserves_a_boundary_that_bf16_collapses_and_shared_gate_rounding(self):
        from engine.profiles.qwen38.net import Qwen38Net
        from engine.profiles.qwen38.lanes import route_softmax_topk
        x = torch.tensor([[1., 1.]], dtype=torch.bfloat16)
        gates = torch.tensor([[1., 0.], [1., 2. ** -9], [0.25, 0.5]], dtype=torch.bfloat16)
        seen = {}
        def experts(x, ids, weights, **kw):
            seen["ids"] = ids
            return x
        net = NS(F=NS(experts=2, topk_experts=1), p={"L0.moe.gates": gates},
                 lanes=NS(route=route_softmax_topk, route_local=None), _experts={"L0.": experts})
        _, shared = Qwen38Net._routed(net, "L0.", x, compact=True)
        self.assertEqual((x @ gates[:2].T).tolist(), [[1., 1.]])
        self.assertEqual(seen["ids"].tolist(), [[1]])
        torch.testing.assert_close(shared, torch.sigmoid((x @ gates[2:].T).float()), rtol=0, atol=0)

    def test_router_storage_is_admitted_once_for_target_and_mtp(self):
        from engine.profiles.qwen38.net import Qwen38Net
        from engine.base.arena import Arena
        net = object.__new__(Qwen38Net)
        net.F, net.layers, net.mtp = NS(experts=8, hidden=16), [0, 2], True
        net.p = {p + "moe.gates": torch.randn(9, 16).bfloat16() for p in ("L0.", "L2.", "mtp.L0.")}
        net._router_weights = {}
        arena = Arena(net.router_nbytes(), device="cpu")
        net.prepare_routers(arena)
        self.assertEqual(arena.used, net.router_nbytes())
        for prefix, w in net._router_weights.items():
            self.assertEqual(w.untyped_storage().data_ptr(), arena.buf.untyped_storage().data_ptr())
            torch.testing.assert_close(w, net.p[prefix + "moe.gates"][:8].float(), rtol=0, atol=0)
        with self.assertRaisesRegex(RuntimeError, "already prepared"):
            net.prepare_routers(arena)

    def test_full_profile_calibration_uses_padded_inputs_and_omits_speculative_mtp(self):
        from engine.profiles.qwen38 import calibration as qcal, specs
        from engine.profiles.qwen38.net import Qwen38Net
        from engine.kernels.dense.calibration import Calibration, BUDGET_BYTES
        from engine.kernels.dense.store import PackStore
        from probes.engine_qwen38_cells import facts
        net = NS(dense_names=Qwen38Net.dense_names)
        with tempfile.TemporaryDirectory() as root:
            store = PackStore(root, 0, "qwen", require_identity=True)
            entries, used, deferred = qcal.plan(net, specs.all_specs(facts(), mtp=True), store)
            self.assertTrue(entries)
            self.assertFalse(deferred)
            self.assertLess(used, BUDGET_BYTES)
            self.assertEqual(used, sum(Calibration.nbytes(m) for _, _, m in entries))
            self.assertTrue(all(not key.startswith("mtp.") for key, _, _ in entries))
            self.assertEqual(sum(key == "head" for key, _, _ in entries), 1)
            downs = [m for key, _, m in entries if key.endswith("sh_down")]
            self.assertEqual(len(downs), 48)
            self.assertTrue(all(m[0].width == 256 for m in downs))
            limited, size, left = qcal.plan(net, specs.all_specs(facts()), store, budget=100_000)
            self.assertLessEqual(size, 100_000)
            self.assertTrue(left)

    def test_collect_file_reload_and_gptq_improves_held_out_correlated_rows(self):
        from engine.kernels.dense.store import PackStore
        from engine.kernels.dense.packing import fp8_rtn
        from engine.profiles.qwen38 import calibration as qcal
        from engine.profiles.qwen38.net import HEAD_NAME
        from engine.base.params import Spec
        gen = torch.Generator().manual_seed(5)
        w = (torch.randn(128, 256, generator=gen) * .05).bfloat16()
        mix = torch.randn(256, 256, generator=gen) * .4 + torch.eye(256)
        x = (torch.randn(4096, 256, generator=gen) @ mix).bfloat16()
        held = (torch.randn(512, 256, generator=gen) @ mix).bfloat16().float()
        net = NS(p={"head": w}, dense_names=lambda keys: {}, dense={})
        with tempfile.TemporaryDirectory() as root:
            store = PackStore(root, 0, "qwen-test", require_identity=True)
            entries, _, _ = qcal.plan(net, [Spec("head", w.shape, w.dtype)], store)
            c = qcal.attach(net, entries, None, max_decode_rows=64)
            net.head_observer(torch.full_like(x, float("nan")), None)  # warmup
            self.assertEqual(c.progress(), 0)
            c.arm()
            net.head_observer(x[:64], None)                           # unmasked decode/ghost rows
            self.assertEqual(c.progress(), 0)
            net.head_observer(x, None)
            torch.testing.assert_close(c.H[HEAD_NAME], x.float().T @ x.float(), rtol=0, atol=0)
            model = qcal.Lifecycle()
            model.composition = NS(net=NS(comm=NS(rank=0)))
            model.calibration, model.calibration_root, model.calibration_weights_id = c, root, "qwen-test"
            with mock.patch.object(c, "complete", return_value=True):
                model.housekeeping(256)                              # auto-save uses the same stamp as manual save
            self.assertEqual(len(c.filed), 1)
            blob = torch.load(c.filed[0], weights_only=True)
            self.assertEqual((blob["ntok"], blob["weights_id"]), (4096, "qwen-test"))
            fresh = PackStore(root, 0, "qwen-test", require_identity=True)
            self.assertFalse(fresh.missing_calibration(HEAD_NAME, 256))
            q, scales = fresh.pack_fp8(w, HEAD_NAME)
            qr, sr = fp8_rtn(w)
            deq = lambda q, s: q.float() * s.repeat_interleave(128, 0).repeat_interleave(128, 1)
            error = lambda q, s: ((held @ (deq(q, s) - w.float()).T) ** 2).mean().item()
            self.assertLess(error(q, scales), .65 * error(qr, sr))
            foreign = PackStore(root, 0, "another-checkpoint", require_identity=True)
            self.assertFalse(foreign.calibrated(HEAD_NAME))
            self.assertIsNone(foreign.pack_fp8(w, HEAD_NAME))
            blob.pop("weights_id")
            torch.save(blob, c.filed[0])
            self.assertFalse(PackStore(root, 0, "qwen-test", require_identity=True).calibrated(HEAD_NAME))
            self.assertTrue(PackStore(root, 0, "legacy").calibrated(HEAD_NAME))

    def test_identity_changes_with_export_file_or_arithmetic(self):
        from engine.profiles.qwen38.calibration import identity
        with tempfile.TemporaryDirectory() as root:
            file = Path(root) / "rank0.safetensors"
            file.write_bytes(b"one")
            get = lambda hc=False: identity({"source_revision": "123"}, [file], {"hidden": 2560}, hc_fp8=hc)
            first = get()
            self.assertEqual(first, get())
            self.assertNotEqual(first, get(True))
            file.write_bytes(b"different")
            self.assertNotEqual(first, get())


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), "requires CUDA")
class GpuPrecisionPortTests(unittest.TestCase):
    def test_served_prefill_and_decode_routes_agree_including_ties(self):
        # The cell suite also admits GLM kernels. Qwen's EP admission is process-global and must be bound before
        # its lane table is constructed, exactly as fleet.main does. Use a fresh process instead of resetting it.
        if os.environ.get("ST_QWEN_ROUTING_TEST") != "1":
            root = str(Path(__file__).resolve().parents[1])
            result = subprocess.run([sys.executable, "-m", "unittest", self.id()], capture_output=True, text=True,
                                    cwd=root, env={**os.environ, "ST_QWEN_ROUTING_TEST": "1",
                                                   "PYTHONPATH": root + os.pathsep + os.environ.get("PYTHONPATH", "")},
                                    timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            return
        from engine.base.kernel_shape import bind
        from engine.profiles.qwen38.lanes import served
        from engine.profiles.qwen38.net import Qwen38Net
        from probes.engine_qwen38_cells import facts
        F, seen = facts(), []
        bind(F.kernel_shape())
        def experts(x, ids, weights, **kw):
            seen.append((ids, weights))
            return torch.zeros_like(x)
        gates = torch.randn(F.experts + 1, F.hidden, device="cuda").bfloat16()
        x = torch.randn(4, F.hidden, device="cuda").bfloat16()
        x[0].zero_()                                            # exact ties must take the same expert set/order
        net = NS(F=F, lanes=served(), p={"L0.moe.gates": gates,
                 "L0.moe.w13": torch.empty(F.experts // 4, F.moe_inter * 2, 1, device="meta")},
                 _router_weights={"L0.": gates[:F.experts].float()}, _experts={"L0.": experts}, first_expert=0)
        _, eager_gate = Qwen38Net._routed(net, "L0.", x, compact=True)
        _, captured_gate = Qwen38Net._routed(net, "L0.", x, compact=False)
        (global_ids, global_weights), (local_ids, local_weights) = seen
        keep = global_ids < F.experts // 4
        self.assertEqual(global_ids[0].tolist(), list(range(F.topk_experts)))
        torch.testing.assert_close(local_ids[keep], global_ids[keep], rtol=0, atol=0)
        torch.testing.assert_close(local_weights, global_weights * keep, rtol=0, atol=0)
        torch.testing.assert_close(eager_gate, captured_gate, rtol=0, atol=0)

    def test_ieee_router_at_qwen_shape_and_graph_replay(self):
        from engine.kernels.router_fp32 import router_logits
        from engine.kernels.moe_route import softmax_topk
        from probes.engine_qwen38_cells import facts
        F = facts()
        gen = torch.Generator().manual_seed(74)
        w = (torch.randn(F.experts, F.hidden, generator=gen) * .03).bfloat16().float().cuda()
        x = torch.randn(4, F.hidden, generator=gen).bfloat16().cuda()
        # Force an exact near tie in the first row: the selector must see the bits a BF16 logit loses.
        x[0].zero_(); x[0, :2] = 1
        w[:, :2] = -1; w[0, :2] = torch.tensor([1., 0.], device="cuda")
        w[1, :2] = torch.tensor([1., 2. ** -9], device="cuda")
        y = router_logits(x, w)
        ids, _ = softmax_topk(y, F.topk_experts)
        self.assertEqual(ids[0, 0].item(), 1)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = router_logits(x, w)
            picks, weights = softmax_topk(captured, F.topk_experts)
        for shift in (0., .5, -1.):
            x[1:].fill_(shift)
            graph.replay()
            ref = x.double() @ w.double().T
            self.assertEqual(captured.dtype, torch.float32)
            torch.testing.assert_close(captured.double(), ref, atol=2e-5, rtol=2e-5)
            self.assertEqual(picks[0, 0].item(), 1)
            self.assertTrue(torch.isfinite(weights).all())
            reference_weights = ref.softmax(-1).gather(1, picks.long())
            reference_weights /= reference_weights.sum(-1, keepdim=True)
            torch.testing.assert_close(weights.double(), reference_weights, rtol=2e-6, atol=0)

    def test_head_w8a16_removes_activation_error_at_served_shape(self):
        from engine.kernels.dense import FP8Linear
        from probes.engine_qwen38_cells import facts
        F = facts()
        gen = torch.Generator().manual_seed(39)
        w = (torch.randn(F.vocab_local, F.hidden, generator=gen) * .02).bfloat16().cuda()
        before = FP8Linear(w, decode_rows=True)
        after = FP8Linear(w, quantized=before.weight, decode_rows="w8a16")
        q, scales = after.weight
        # Same quantized weight on both sides: only activation quantization/accumulation is judged here.
        exact = q.float() * scales.repeat_interleave(128, 0).repeat_interleave(128, 1)
        reports = []
        for rows in (1, 4, 16):
            x = torch.randn(rows, F.hidden, generator=gen).bfloat16().cuda()
            old, new = before(x).float(), after(x).float()
            ref = (x.double() @ exact.double().T)[:, :F.vocab_local].float()
            old_rmse = ((old - ref) ** 2).mean().sqrt().item()
            new_rmse = ((new - ref) ** 2).mean().sqrt().item()
            self.assertLess(new_rmse, old_rmse * .2)
            torch.testing.assert_close(new, ref, atol=.008, rtol=.004)
            reports.append(dict(rows=rows, w8a8_rmse=old_rmse, w8a16_rmse=new_rmse))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = after(x)
        x.mul_(.5)
        graph.replay()
        torch.testing.assert_close(out, after(x), rtol=0, atol=0)
        print(json.dumps(dict(qwen_head_same_weight=reports)), flush=True)

    def test_native_prefill_collection_at_hidden_and_padded_width(self):
        from engine.profiles.qwen38 import calibration as qcal
        from engine.profiles.qwen38.net import HEAD_NAME
        from engine.kernels.dense.store import Need
        from engine.base.arena import Arena
        from engine.kernels.dense.calibration import Calibration
        keys = [("head", HEAD_NAME, [Need(HEAD_NAME, 0, 2560)]),
                ("L0.moe.sh_down", "shared-down", [Need("shared-down", 0, 256)])]
        net = NS(p={"head": torch.empty(0, device="cuda")}, dense={"L0.moe.sh_down": NS()})
        arena = Arena(sum(Calibration.nbytes(m) for _, _, m in keys))
        c = qcal.attach(net, keys, arena, max_decode_rows=64)
        x = torch.randn(256, 2560, device="cuda").bfloat16()
        down = torch.nn.functional.pad(x[:, :160], (0, 96))
        observe = lambda: (net.head_observer(x, None), net.dense["L0.moe.sh_down"].observer(down, None))
        observe()
        self.assertEqual(c.progress(), 0)
        c.arm(); observe()
        self.assertEqual(c.progress(), 256)
        for name, value in ((HEAD_NAME, x), ("shared-down", down)):
            ref = value.double().T @ value.double()
            torch.testing.assert_close(c.H[name].double(), ref, atol=1e-4, rtol=1e-5)
            torch.testing.assert_close(c.amax[name], value.float().abs().amax(0), atol=0, rtol=0)
