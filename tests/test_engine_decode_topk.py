"""engine/kernels/decode_topk: the shared-memory plan, the block width, the gate, and the tie rule.

The kernel itself needs a GPU (`probes/engine_decode_select_rows.py`, arms `fused`/`fused_rows`
against the same build's `control`). What is checkable here is everything the host decides --
a plan that never overruns the device budget, a gate that declines instead of raising, and the
selection contract the kernel encodes: strictly greater wins, and an exact tie goes to the
lower pool id, which is the set torch.topk(sorted=False) returns.
"""
import ast
import importlib.util
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "engine/kernels/decode_topk.cu"


def module():
    """Import the wrapper without importing the engine package (no torch extension is built)."""
    spec = importlib.util.spec_from_file_location("st_decode_topk",
                                                  ROOT / "engine/kernels/decode_topk.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def defines():
    text = SOURCE.read_text(encoding="utf-8")
    return {name: int(value)
            for name, value in re.findall(r"^#define (ST_[A-Z_]+) (\d+)$", text, re.M)}


class SourceContractTests(unittest.TestCase):
    """The wrapper's constants are the kernel's; a drift here is a silent out-of-bounds."""

    def test_the_wrapper_and_the_kernel_agree_on_k_radix_and_the_block_cap(self):
        d, m = defines(), module()
        self.assertEqual(d["ST_K"], m.SELECT_K)
        self.assertEqual(d["ST_RADIX"], m.ST_RADIX)
        self.assertEqual(d["ST_THREADS"], m.MAX_THREADS)

    def test_the_binding_takes_the_stash_the_bin_cache_and_the_block_width(self):
        text = SOURCE.read_text(encoding="utf-8")
        signature = re.search(r"void run\((.*?)\) \{", text, re.S).group(1)
        for name in ("scores", "ke", "out", "stash_slots", "bin_bytes", "threads"):
            self.assertIn(name, signature)
        call = ast.parse((ROOT / "engine/kernels/decode_topk.py").read_text(encoding="utf-8"))
        run = [n for n in ast.walk(call)
               if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "run"]
        self.assertEqual([len(n.args) for n in run], [6], "run takes six arguments")

    def test_the_kernel_reserves_no_more_static_shared_memory_than_the_wrapper_holds_back(self):
        d, m = defines(), module()
        static = (d["ST_RADIX"] + 1) * 4          # hist
        static += d["ST_REPLICAS"] * d["ST_RADIX"] * 4  # hist_rep
        static += d["ST_K"] * 4                   # selected
        static += 64                              # counter, threshold, num_input, last_remain, prefix
        self.assertLessEqual(static, m.STATIC_SMEM)

    def test_launch_bytes_cover_both_key_id_rings_and_fit_the_planned_budget(self):
        # Exercise the actual C++ launch expression, independently of the Python
        # planner. The regression budgeted 16 bytes/slot but launched with 8,
        # leaving the second key/id ring outside dynamic shared memory.
        source, m = SOURCE.read_text(encoding="utf-8"), module()
        expression = re.search(r"const size_t smem = (.*?);", source).group(1)
        expression = expression.replace("sizeof(int)", "4").replace("(size_t)", "")
        for budget in (48 * 1024, 88 * 1024, 92 * 1024, 200 * 1024):
            for columns in (1024, 4096, 50688, 1 << 20):
                bins, stash = m.plan(columns, budget)
                allocated = eval(expression, {"__builtins__": {}}, dict(bin_bytes=bins, stash_slots=stash))
                last_id_end = bins + (3 * stash + stash) * 4
                self.assertGreaterEqual(allocated, last_id_end, (budget, columns))
                self.assertLessEqual(allocated, budget, (budget, columns))


class ControlArmTests(unittest.TestCase):
    """`_select_rows(native=False)` must really be the Torch path.

    A same-build control that quietly runs the candidate makes every number the probe
    reports a comparison of the kernel with itself. This caught exactly that: the switch
    was added to the signature while the body still called the kernel directly.
    """

    def test_the_control_switch_is_the_only_gate_on_the_native_selection(self):
        source = (ROOT / "engine/profiles/glm53/net.py").read_text(encoding="utf-8")
        select_rows = next(n for n in ast.walk(ast.parse(source))
                           if isinstance(n, ast.FunctionDef) and n.name == "_select_rows")
        self.assertIn("native", [a.arg for a in select_rows.args.kwonlyargs])
        # every mention of the kernel sits on the `native` side of a conditional on `native`
        gated = {id(node) for exp in ast.walk(select_rows) if isinstance(exp, ast.IfExp)
                 and isinstance(exp.test, ast.Name) and exp.test.id == "native"
                 for node in ast.walk(exp.body) if isinstance(node, ast.Name)}
        mentions = [n for n in ast.walk(select_rows)
                    if isinstance(n, ast.Name) and n.id == "select_native"]
        self.assertTrue(mentions, "the selection must reach the kernel at all")
        for node in mentions:
            self.assertIn(id(node), gated, "native=False must not reach the kernel")

    def test_the_probe_runs_the_torch_control_and_the_fused_candidate(self):
        arms = (ROOT / "probes/engine_decode_select_rows.py").read_text(encoding="utf-8")
        table = arms.split("def arms(")[1].split("def capture(")[0]
        self.assertIn("'control'", table)
        self.assertIn("native=False", table)
        self.assertIn("'fused'", table)


class PlanTests(unittest.TestCase):
    CONTEXTS = (512, 1024, 2048, 4096, 8192, 16384, 32768, 50688)   # n_cand = context // kpool

    def test_the_plan_never_overruns_the_budget_and_the_cache_covers_the_row(self):
        m = module()
        for budget in (48 * 1024, 88 * 1024, 92 * 1024, 200 * 1024):
            for n in self.CONTEXTS + (1 << 20,):
                bins, stash = m.plan(n, budget)
                self.assertLessEqual(bins + 16 * stash, budget, (budget, n))
                self.assertLessEqual(stash, m.MAX_STASH)
                if bins:
                    self.assertGreaterEqual(bins, n, "a kept cache must cover every column")
                    self.assertEqual(bins % 4, 0, "the sift reads the cache four bins to a word")
                    self.assertGreaterEqual(stash, m.MIN_STASH)

    def test_a_row_too_wide_for_a_usable_stash_drops_the_cache_rather_than_the_answer(self):
        m = module()
        bins, stash = m.plan(1 << 20, 88 * 1024)
        self.assertEqual(bins, 0, "the kernel reads the row twice instead")
        self.assertGreaterEqual(stash, m.SELECT_K)

    def test_a_budget_too_small_for_even_the_stash_makes_select_decline(self):
        m = module()
        _, stash = m.plan(4096, 4 * 1024)
        self.assertLess(stash, m.SELECT_K, "select() returns None below this, it does not launch")

    def test_the_block_is_a_quad_a_thread_between_the_radix_and_the_cap(self):
        m = module()
        for n in self.CONTEXTS + (600, 1 << 20):
            threads = m.block_threads(n)
            self.assertEqual(threads & (threads - 1), 0, n)
            self.assertGreaterEqual(threads, m.ST_RADIX)
            self.assertLessEqual(threads, m.MAX_THREADS)
        # the measured ladder (sm_120, nine shapes against 128/256/512/1024)
        self.assertEqual([m.block_threads(n) for n in (1024, 2048, 4096, 32768)],
                         [256, 512, 1024, 1024])


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "requires PyTorch")
class GateTests(unittest.TestCase):
    """Every shape it does not admit returns None so the caller keeps Torch. None of them raise."""

    def test_a_shape_it_does_not_admit_declines_instead_of_raising(self):
        import torch
        m = module()
        rows, n = 8, 4096
        good = torch.zeros(rows, n)
        ke = torch.full((rows,), n, dtype=torch.int32)
        for label, logits, horizon, k in (
                ("cpu logits", good, ke, m.SELECT_K),                 # a reference lane
                ("k is not 512", good, ke, 128),
                ("not fp32", torch.zeros(rows, n, dtype=torch.bfloat16), ke, m.SELECT_K),
                ("three dims", torch.zeros(2, rows, n), ke, m.SELECT_K),
                ("strided columns", torch.zeros(rows, n, 2)[..., 0], ke, m.SELECT_K),
                ("horizon not int32", good, ke.float(), m.SELECT_K),
                ("horizon per column", good, torch.zeros(n, dtype=torch.int32), m.SELECT_K),
                ("empty", torch.zeros(0, n), torch.zeros(0, dtype=torch.int32), m.SELECT_K)):
            self.assertIsNone(m.select(logits, horizon, k), label)

    def test_the_kernel_key_orders_a_row_the_way_the_docstring_says(self):
        """key = (ordered float bits, ~index): strictly greater wins, an exact tie takes the
        LOWER pool id. This is the rule the .cu encodes; that CUDA's torch.topk returns the
        same set is a device fact, checked below where there is a device."""
        import torch
        torch.manual_seed(0)

        def ordered(x):                       # st_ordered_key, in Python
            if x == 0.0:
                x = 0.0                       # the kernel folds -0.0 onto +0.0 the same way
            bits = int(torch.tensor(x, dtype=torch.float32).view(torch.int32)) & 0xFFFFFFFF
            return (~bits & 0xFFFFFFFF) if bits & 0x80000000 else (bits | 0x80000000)

        row = torch.tensor([3.0, -0.0, 0.0, 3.0, float("nan"), -1.0, 3.0])
        by_key = sorted(range(row.numel()),
                        key=lambda i: (-ordered(float(row[i])), i))
        self.assertEqual(by_key[:4], [4, 0, 3, 6], "NaN is the largest key; then ties by id")
        self.assertEqual(ordered(0.0), ordered(-0.0), "signed zeros are one value")

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None
                         and __import__("torch").cuda.is_available(), "requires CUDA")
    def test_native_selection_refines_more_than_512_visible_pools_on_replay(self):
        # Boot's zero-context warmup only exercises the all-visible-pools-win
        # shortcut. A 2681-token request exposes 670 pools and enters refinement.
        import torch
        m = module()
        torch.manual_seed(1014)
        logits = torch.randn(8, 1024, device="cuda")
        ke = torch.full((8,), 670, dtype=torch.int32, device="cuda")
        m.select(logits, ke, 512)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            winners = m.select(logits, ke, 512)
        for horizon in (513, 670, 1024):
            for tied in (False, True):
                logits.copy_(torch.randn_like(logits) if not tied else torch.ones_like(logits))
                ke.fill_(horizon)
                graph.replay()
                torch.cuda.synchronize()
                expected = (torch.arange(512, device="cuda").expand(8, -1) if tied else
                            torch.topk(logits[:, :horizon], 512, dim=-1).indices)
                self.assertTrue(torch.equal(winners.sort(-1).values.long(), expected.sort(-1).values))

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None
                         and __import__("torch").cuda.is_available(), "requires CUDA")
    def test_on_cuda_that_rule_is_the_set_torch_topk_returns(self):
        """The reason the swap is exact. CPU torch.topk breaks ties differently, which is why
        this is asserted on a device and why the reference lane keeps torch either way."""
        import torch
        torch.manual_seed(0)
        k = 64
        for trial in range(20):
            n = 200 + trial * 37
            row = torch.randn(n, device="cuda")
            row[torch.randperm(n, device="cuda")[:40]] = row.topk(max(1, k - 20)).values[-1]
            lower_index = sorted(range(n), key=lambda i: (-row[i].item(), i))[:k]
            self.assertEqual(set(torch.topk(row, k, sorted=False).indices.tolist()),
                             set(lower_index), n)
        flat = torch.full((300,), 1.25, device="cuda")
        self.assertEqual(set(torch.topk(flat, k, sorted=False).indices.tolist()), set(range(k)))


if __name__ == "__main__":
    unittest.main()
