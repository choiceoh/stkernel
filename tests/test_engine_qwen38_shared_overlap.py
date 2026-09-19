"""Qwen3.8's shared expert forked beside the routed experts (engine/QWEN38_CARRY.md M5, GLM-5.3's #789): which steps
fork, that a fork launches what the unforked layer launches and joins before the gated sum, the declared flag and its
knob down to the launcher -- and, on a device, the forked layer's bytes against the unforked one's, eagerly and as a
captured graph replayed over new inputs.

    docker exec -w <repo> stk-test python3 -m unittest tests.test_engine_qwen38_shared_overlap
"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
torch = None
if importlib.util.find_spec("torch") is not None:
    import torch
CUDA = torch is not None and torch.cuda.is_available()
EXPERTS, TOPK, HIDDEN, INTER, SPEC_K = 8, 2, 64, 32, 3


def stand_in(device, shared_overlap, calls=None):
    """A net of one MoE layer on plain torch lanes: what Qwen38Net._moe reads, nothing else."""
    from engine.base.comm import Comm
    gen = torch.Generator().manual_seed(5)
    rand = lambda *shape: (torch.randn(*shape, generator=gen) * 0.3).to(device=device, dtype=torch.bfloat16)
    weights = {"L0.moe.gates": rand(EXPERTS + 1, HIDDEN), "L0.moe.sh_gate_up": rand(2 * INTER, HIDDEN),
               "L0.moe.sh_down": rand(HIDDEN, INTER), "experts_up": rand(EXPERTS, INTER, HIDDEN),
               "experts_down": rand(EXPERTS, HIDDEN, INTER)}
    calls = [] if calls is None else calls

    def route(scores, k):
        calls.append("route")
        top = scores.float().softmax(-1).topk(k, dim=-1)
        return top.indices.to(torch.int32), (top.values / top.values.sum(-1, keepdim=True)).to(scores.dtype)

    def experts(x, ids, w, compact=False, local=False):
        calls.append("experts")
        up = torch.einsum("nh,nkih->nki", x.float(), weights["experts_up"][ids.long()].float())
        out = torch.einsum("nki,nkhi->nkh", torch.nn.functional.silu(up), weights["experts_down"][ids.long()].float())
        return (out * w.float()[..., None]).sum(1).to(x.dtype)

    def swiglu(y, pad_to=None):
        calls.append("swiglu")
        gate, up = y.float().chunk(2, dim=-1)
        return (torch.nn.functional.silu(gate) * up).to(y.dtype)

    def linear(x, name):
        calls.append(name.rsplit(".", 1)[1])
        return torch.nn.functional.linear(x, weights[name])

    lanes = SimpleNamespace(rows_linear=None, route=route, route_local=None, swiglu=swiglu,
                            moe_finish=lambda routed, shared, gate: (routed.float() + shared.float() * gate).to(routed.dtype))
    return SimpleNamespace(F=SimpleNamespace(experts=EXPERTS, topk_experts=TOPK, spec_k=SPEC_K), p=weights, lanes=lanes,
                           comm=Comm(), linear=linear, _experts={"L0.": experts}, dense={}, first_expert=0,
                           shared_overlap=shared_overlap, _overlap=None), calls


@unittest.skipUnless(torch is not None, "requires torch")
class RuleTests(unittest.TestCase):
    def test_which_steps_fork(self):
        from engine.profiles.qwen38.net import Qwen38Net
        cpu, fake = torch.zeros(4, 8), SimpleNamespace(is_cuda=True, shape=(4, 8))
        wide = SimpleNamespace(is_cuda=True, shape=(SPEC_K + 2, 8))
        for wanted, x, compact, want in ((False, fake, False, False), (True, fake, False, True), (True, fake, True, False),
                                         (True, cpu, False, False), (True, wide, False, False), ("all", wide, False, True),
                                         ("all", wide, True, False)):
            net = SimpleNamespace(shared_overlap=wanted, F=SimpleNamespace(spec_k=SPEC_K))
            with self.subTest(wanted=wanted, rows=x.shape[0], compact=compact):
                self.assertIs(Qwen38Net._forks(net, x, compact), want)
        self.assertFalse(Qwen38Net._forks(SimpleNamespace(F=SimpleNamespace(spec_k=SPEC_K)), fake, False))   # an older stand-in

    def test_the_flag_is_declared(self):
        from engine.profiles.qwen38.net import Qwen38Net
        import inspect
        parameter = inspect.signature(Qwen38Net.__init__).parameters["shared_overlap"]
        self.assertIs(parameter.default, False)
        source = inspect.getsource(Qwen38Net.__init__)
        self.assertIn('if shared_overlap not in (False, True, "all") or type(shared_overlap) not in (bool, str):', source)

    def test_the_unforked_layer_is_the_layer_it_was(self):
        from engine.profiles.qwen38.net import Qwen38Net
        net, calls = stand_in("cpu", True)                                  # a CPU step never forks, whatever was asked
        x = torch.randn(3, HIDDEN, generator=torch.Generator().manual_seed(1)).to(torch.bfloat16)
        out = Qwen38Net._moe(net, "L0.", x, compact=False)
        self.assertEqual(calls, ["route", "experts", "sh_gate_up", "swiglu", "sh_down"])
        routed, gate = Qwen38Net._routed(net, "L0.", x, compact=False)
        self.assertEqual((tuple(gate.shape), gate.dtype), ((3, 1), torch.float32))
        want = net.lanes.moe_finish(routed, Qwen38Net._shared(net, "L0.", x), gate)
        self.assertTrue(torch.equal(out, want))
        self.assertIsNone(net._overlap)

    def test_the_knob_reaches_the_net_from_the_launcher(self):
        fleet = (ROOT / "engine/profiles/qwen38/fleet.py").read_text(encoding="utf-8")
        self.assertIn('ap.add_argument("--shared-overlap", choices=("off", "one", "all"), default="one",', fleet)
        self.assertIn('"  shared expert: "', fleet)                          # a boot's log says which
        self.assertIn('shared_overlap={"off": False, "one": True, "all": "all"}[a.shared_overlap],', fleet)
        self.assertIn("shared_overlap=shared_overlap)", fleet)
        launcher = (ROOT / "launchers/start-st-qwen38.sh").read_text(encoding="utf-8")
        self.assertIn('case "${ST_SHARED_OVERLAP:-one}" in', launcher)
        self.assertIn('off|all) OVERLAP_ARG="--shared-overlap $ST_SHARED_OVERLAP" ;;', launcher)     # off: the rollback
        self.assertIn("$EXPERTS_ARG $OVERLAP_ARG $ONESHOT_ARG", launcher)

    def test_the_step_probe_builds_the_arms(self):
        probe = (ROOT / "probes/engine_qwen38_step.py").read_text(encoding="utf-8")
        self.assertIn('OVERLAP_ARMS = ("served", "overlap-one", "overlap-all")', probe)
        self.assertIn('shared_overlap={"overlap-one": True, "overlap-all": "all"}.get(arm, False))', probe)
        check = (ROOT / "probes/engine_kernel_check.py").read_text(encoding="utf-8")
        self.assertIn("args.lanes == 'qwen38_step_overlap'", check)
        self.assertIn("layer_sets=LAYER_SETS[:1], arms=OVERLAP_ARMS", check)


@unittest.skipUnless(CUDA, "requires CUDA")
class ForkTests(unittest.TestCase):
    def layers(self, mode):
        from engine.profiles.qwen38.net import Qwen38Net
        plain, _ = stand_in("cuda", False)
        forked, calls = stand_in("cuda", mode)
        return Qwen38Net, plain, forked, calls

    def test_a_forked_step_is_the_unforked_step_s_bytes(self):
        gen = torch.Generator().manual_seed(2)
        for mode, rows, forks in ((True, SPEC_K + 1, True), (True, SPEC_K + 2, False), ("all", 9, True)):
            Net, plain, forked, calls = self.layers(mode)
            x = torch.randn(rows, HIDDEN, generator=gen).to(device="cuda", dtype=torch.bfloat16)
            with self.subTest(mode=mode, rows=rows):
                want = Net._moe(plain, "L0.", x, compact=False)
                got = Net._moe(forked, "L0.", x, compact=False)
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(got, want))
                self.assertEqual(forked._overlap is not None and forked._overlap.executed, forks)
                # the fork launches the shared expert first, on its own stream, then the routed branch on the step's
                self.assertEqual(calls, ["sh_gate_up", "swiglu", "sh_down", "route", "experts"] if forks else
                                 ["route", "experts", "sh_gate_up", "swiglu", "sh_down"])
                self.assertTrue(bool(got.float().abs().sum() > 0))
            eager = Net._moe(forked, "L0.", x, compact=True)                 # an eager step never forks
            self.assertTrue(torch.equal(eager, want))

    def test_a_captured_fork_replays_over_new_inputs(self):
        Net, plain, forked, _ = self.layers(True)
        gen = torch.Generator().manual_seed(3)
        x = torch.randn(SPEC_K + 1, HIDDEN, generator=gen).to(device="cuda", dtype=torch.bfloat16)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                Net._moe(forked, "L0.", x, compact=False)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out = Net._moe(forked, "L0.", x, compact=False)
        for trial in range(4):
            x.copy_(torch.randn(SPEC_K + 1, HIDDEN, generator=gen).to(device="cuda", dtype=torch.bfloat16))
            graph.replay()
            torch.cuda.synchronize()
            with self.subTest(trial=trial):
                self.assertTrue(torch.equal(out, Net._moe(plain, "L0.", x, compact=False)))
        graph.reset()


if __name__ == "__main__":
    unittest.main()
