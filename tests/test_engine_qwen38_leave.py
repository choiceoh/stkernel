"""A leave as its TP sum's programmatic dependent, prefetching the site's down projection (engine/QWEN38_CARRY.md H4):
where the kernel waits, what a launch asks for in each mode, the lanes' three modes, the net's prefetch target and the
knob down to the launcher. On a device the bytes are probes/engine_qwen38_leave's (the single-GPU lane): the three modes
behind a stand-in sum that lands late, against the ordinary launch.

    docker exec -w <repo> stk-test python3 -m unittest tests.test_engine_qwen38_leave
    docker exec -e TRITON_INTERPRET=1 -w <repo> stk-test python3 -m unittest tests.test_engine_qwen38_leave
"""
import importlib.util
import inspect
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
torch = None
if importlib.util.find_spec("torch") is not None:
    import torch
READY = torch is not None and importlib.util.find_spec("triton") is not None
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
HC, HIDDEN, RANK, EPS = 4, 2560, 320, 1e-6


def operands(rows, hidden=HIDDEN, seed=0):
    gen = torch.Generator().manual_seed(seed)
    width = HC * hidden
    return ((torch.randn(rows, width, generator=gen)).bfloat16(), torch.randn(rows, hidden, generator=gen).bfloat16(),
            (torch.rand(rows, HC, generator=gen) * 2).bfloat16(), (torch.randn(width, generator=gen) * 0.1).bfloat16())


@unittest.skipUnless(READY, "torch and triton required")
class KernelTests(unittest.TestCase):
    def test_only_the_norm_weight_and_the_prefetch_come_before_the_wait(self):
        from engine.kernels import gated_residual as hcr
        source = inspect.getsource(hcr._leave_norm.fn)
        wait = source.index("tl.extra.cuda.gdc_wait()")
        self.assertLess(source.index("w = tl.load(W + off"), wait)
        self.assertLess(source.index("_prefetch_l2(NEXT, SECTORS"), wait)
        for load in ("tl.load(H + r * sH", "tl.load(OUT + r * sO", "tl.load(INJ + r * sI"):
            self.assertGreater(source.index(load), wait, load)
        self.assertEqual(source.count("tl.load("), 4)                         # the weight, the streams, the sum, the gate
        prefetch = inspect.getsource(hcr._prefetch_l2.fn)
        self.assertIn('"prefetch.global.L2 [$1]; // $0"', prefetch)
        self.assertIn("NEXT + sector * 16", prefetch)                        # a sector is 32 bytes, 16 BF16

    def test_the_budget_counts_whole_sectors_of_a_decode_steps_weight(self):
        from engine.kernels import gated_residual as hcr
        weight = mock.Mock(dtype=torch.bfloat16, device=torch.device("cuda"), is_contiguous=lambda: True,
                           numel=lambda: (RANK + HC) * HC * HIDDEN, element_size=lambda: 2)
        self.assertEqual(hcr._prefetch_sectors(None, 4), 0)
        self.assertEqual(hcr._prefetch_sectors(weight, 4), (RANK + HC) * HC * HIDDEN * 2 // 32)
        self.assertEqual(hcr._prefetch_sectors(weight, hcr.DECODE_ROWS + 1), 0)   # the cuBLAS mixer re-reads its tiles
        with mock.patch.object(hcr, "_PREFETCH_BYTES_OVERRIDE", 4 << 20):
            self.assertEqual(hcr._prefetch_sectors(weight, 4), (4 << 20) // 32)
        with mock.patch.object(hcr, "PREFETCH_BYTES", 1 << 20):
            self.assertEqual(hcr._prefetch_sectors(weight, 1), (1 << 20) // 32)
            with mock.patch.object(hcr, "_PREFETCH_BYTES_OVERRIDE", 0):
                self.assertEqual(hcr._prefetch_sectors(weight, 1), 0)
        with self.assertRaises(ValueError):
            hcr._prefetch_sectors(mock.Mock(dtype=torch.float32, device=torch.device("cuda"),
                                            is_contiguous=lambda: True), 4)


@unittest.skipUnless(READY, "torch and triton required")
class LaunchTests(unittest.TestCase):
    """What `leave_norm` asks of the launch in each mode, the kernel replaced by a recorder and the CPU tensors made to
    look like a GPU's -- or, `compiled` False, like the interpreter's (is_cuda stood in, the device still the CPU)."""

    def launch(self, rows=4, compiled=True, norm=True, **kwargs):
        from engine.kernels import gated_residual as hcr
        h, out, inject, w = operands(rows, hidden=64)
        calls = []

        class Recorder:
            def __getitem__(self, grid):
                return lambda *args, **kw: calls.append((grid, args, kw))

        with mock.patch.object(hcr, "_leave_norm", Recorder()), \
                mock.patch.object(torch.Tensor, "is_cuda", property(lambda tensor: True)):
            if compiled:
                with mock.patch.object(torch.Tensor, "device", property(lambda tensor: torch.device("cuda"))):
                    self.result = hcr.leave_norm(h, out, inject, w, EPS, HC, **kwargs) if norm else \
                        hcr.leave(h, out, inject, HC, **kwargs)
            else:
                self.result = hcr.leave_norm(h, out, inject, w, EPS, HC, **kwargs)
        self.assertEqual(len(calls), 1)
        grid, args, kw = calls[0]
        self.assertEqual(grid, (rows, HC))
        return args, kw

    def test_the_ordinary_launch_by_default(self):
        args, kw = self.launch()
        self.assertEqual((kw["PDL"], kw["PREFETCH"], kw["launch_pdl"], args[11]), (False, False, False, 0))

    def test_pdl_is_a_dependent_launch_without_a_prefetch(self):
        args, kw = self.launch(pdl=True)
        self.assertEqual((kw["PDL"], kw["PREFETCH"], kw["launch_pdl"], args[11]), (True, False, True, 0))
        args, kw = self.launch(norm=False, pdl=True)                          # the leave before an injection feature
        self.assertEqual((kw["PDL"], kw["NORM"], kw["launch_pdl"]), (True, False, True))

    def test_a_prefetch_names_the_weight_and_its_sectors(self):
        weight = torch.randn(RANK + HC, HC * 64).bfloat16()
        args, kw = self.launch(pdl=True, prefetch=weight)
        self.assertEqual((kw["PDL"], kw["PREFETCH"], kw["launch_pdl"]), (True, True, True))
        self.assertIs(args[5], weight)                                       # NEXT
        self.assertEqual(args[11], weight.numel() * 2 // 32)                 # SECTORS

    def test_no_prefetch_without_the_wait_or_past_decode_rows(self):
        from engine.kernels import gated_residual as hcr
        weight = torch.randn(RANK + HC, HC * 64).bfloat16()
        args, kw = self.launch(prefetch=weight)                              # launched after its sum: nothing to fill
        self.assertEqual((kw["PDL"], kw["PREFETCH"], args[11]), (False, False, 0))
        args, kw = self.launch(rows=hcr.DECODE_ROWS + 1, pdl=True, prefetch=weight)
        self.assertEqual((kw["PDL"], kw["PREFETCH"], args[11]), (True, False, 0))

    def test_the_interpreter_launches_the_ordinary_kernel(self):
        weight = torch.randn(RANK + HC, HC * 64).bfloat16()
        args, kw = self.launch(compiled=False, pdl=True, prefetch=weight)
        self.assertEqual((kw["PDL"], kw["PREFETCH"], kw["launch_pdl"], args[11]), (False, False, False, 0))

    def test_pdl_is_declared(self):
        with self.assertRaises(ValueError):
            self.launch(pdl=1)


@unittest.skipUnless(READY and INTERPRET, "TRITON_INTERPRET=1 with torch and triton")
class InterpretedTests(unittest.TestCase):
    def test_every_mode_leaves_the_ordinary_bytes(self):
        """Under the interpreter every mode is the ordinary kernel; the modes' own bytes on a GB10 are the probe's."""
        from engine.kernels import gated_residual as hcr
        from tests.test_engine_qwen38_kernels import served_kernels
        h, out, inject, w = operands(3, hidden=16)
        weight = torch.randn(RANK + HC, HC * 16).bfloat16()
        with served_kernels():
            base = hcr.leave_norm(h.clone(), out, inject, w, EPS, HC)
            for kwargs in (dict(pdl=True), dict(pdl=True, prefetch=weight)):
                got = hcr.leave_norm(h.clone(), out, inject, w, EPS, HC, **kwargs)
                with self.subTest(**{k: type(v).__name__ for k, v in kwargs.items()}):
                    self.assertTrue(all(torch.equal(a, b) for a, b in zip(got, base)))


@unittest.skipUnless(READY, "torch and triton required")
class LanesTests(unittest.TestCase):
    def test_three_modes_and_prefetch_the_default(self):
        from engine.profiles.qwen38 import lanes
        self.assertEqual(lanes.LEAVES, ("off", "pdl", "prefetch"))
        self.assertEqual(lanes.LEAVE, "prefetch")
        with self.assertRaises(ValueError):
            lanes.served(leave="early")

    def test_the_served_leaves_pass_their_mode(self):
        """The served table's leaves, read from lanes.served: the mode decides pdl, and only `prefetch` hands the
        weight on (the table's reference leaves take and ignore it)."""
        source = inspect.getsource(__import__("engine.profiles.qwen38.lanes", fromlist=["served"]).served)
        self.assertIn('pdl = leave != "off"', source)
        self.assertIn("return hcr.leave(h, out, inject, hc, pdl=pdl)", source)
        self.assertIn('pdl=pdl, prefetch=prefetch if leave == "prefetch" else None)', source)
        self.assertIn("bound = [hcr.norm_streams, hc_leave, hc_leave_norm, hcr.mix,", source)
        self.assertIn("leave=leave)", source)
        from engine.profiles.qwen38 import lanes
        table = lanes.reference()
        h, out, inject, w = operands(2, hidden=16)
        got = table.hc_leave_norm(h.clone(), out, inject, w, EPS, HC, prefetch=torch.ones(3))
        want = table.hc_leave_norm(h.clone(), out, inject, w, EPS, HC)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(got, want)))


@unittest.skipUnless(READY, "torch and triton required")
class NetTests(unittest.TestCase):
    def net(self, hc_fp8=False):
        from engine.profiles.qwen38.net import Qwen38Net
        calls = []
        p = {"L0.hc.mlp.norm": "norm", "L0.hc.mlp.down_inject": "down_inject", "L0.hc.mlp.up": "up"}
        lanes = SimpleNamespace(hc_leave_norm=lambda *a, **k: calls.append(k) or ("h", "normed"))
        net = SimpleNamespace(F=SimpleNamespace(rms_eps=EPS, hc=HC), p=p, lanes=lanes,
                              _hc_projections={"L0.hc.mlp.": object()} if hc_fp8 else {},
                              _mix=lambda prefix, normed, down, inject: ("x", "injection"))
        net._mixer_weight = lambda prefix, down: Qwen38Net._mixer_weight(net, prefix, down)
        return net, calls, Qwen38Net

    def test_a_site_prefetches_its_own_down_projection(self):
        net, calls, Qwen38Net = self.net()
        self.assertEqual(Qwen38Net._site(net, "L0.hc.mlp.", "h", "out", "inject"), ("x", "injection", "h"))
        self.assertEqual(calls, [{"prefetch": "down_inject"}])

    def test_an_fp8_mixer_reads_another_weight(self):
        net, calls, Qwen38Net = self.net(hc_fp8=True)
        Qwen38Net._site(net, "L0.hc.mlp.", "h", "out", "inject")
        self.assertEqual(calls, [{"prefetch": None}])

    def test_the_closing_mixers_prefetch_their_down_projection(self):
        source = (ROOT / "engine/profiles/qwen38/net.py").read_text(encoding="utf-8")
        self.assertIn('prefetch=self._mixer_weight("close.", "down"))', source)
        self.assertIn('prefetch=self._mixer_weight("mtp.close.", "down"))', source)


class KnobTests(unittest.TestCase):
    def test_the_knob_reaches_the_lanes_from_the_launcher(self):
        fleet = (ROOT / "engine/profiles/qwen38/fleet.py").read_text(encoding="utf-8")
        self.assertIn('ap.add_argument("--leave", choices=lane_tables.LEAVES, default=lane_tables.LEAVE,', fleet)
        self.assertIn("lanes = lane_tables.served(leave=a.leave)", fleet)
        self.assertIn('print("  leave: "', fleet)                               # a boot's log says which
        launcher = (ROOT / "launchers/start-st-qwen38.sh").read_text(encoding="utf-8")
        self.assertIn('case "${ST_LEAVE:-prefetch}" in', launcher)
        self.assertIn('off|pdl) LEAVE_ARG="--leave $ST_LEAVE" ;;', launcher)   # off: the rollback
        self.assertIn("$DRAFTER_ARG $LEAVE_ARG $HC_ARG", launcher)

    def test_the_lane_probe_is_admitted(self):
        check = (ROOT / "probes/engine_kernel_check.py").read_text(encoding="utf-8")
        self.assertIn("args.lanes == 'qwen38_leave'", check)
        probe = (ROOT / "probes/engine_qwen38_leave.py").read_text(encoding="utf-8")
        self.assertIn('("off", "off", None), ("pdl", "pdl", None)', probe)
        self.assertIn("launch_pdl=True", probe)                              # the stand-in is a dependent like the sum


if __name__ == "__main__":
    unittest.main()
