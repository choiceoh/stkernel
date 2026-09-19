"""probes/engine_qwen38_input_reuse.py and the lane's wiring (engine/QWEN38_CARRY.md S2), on the CPU: the shapes are
Qwen3.8's projections at TP=4, the rows its verify steps', the weights a graph cycles through clear the L2, the
kernel's admission names the same pairs, and the lane, the queue and the cells' GPU cases reach the probe and the test.

    docker exec -w <repo> stk-test python3 -m unittest tests.test_probe_qwen38_input_reuse
"""
import importlib.util
import os
from pathlib import Path
import re
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
TORCH = importlib.util.find_spec("torch") is not None


def probe():
    import probes.engine_qwen38_input_reuse as module
    return module


@unittest.skipUnless(TORCH, "requires torch")
class TableTests(unittest.TestCase):
    def test_the_probe_imports_without_a_gpu_or_the_kernel_package(self):
        code = "import sys, probes.engine_qwen38_input_reuse as p; print(callable(p.run), 'engine.kernels' in sys.modules)"
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                                env={**os.environ, "PYTHONPATH": str(ROOT)})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.split(), ["True", "False"])

    def test_the_shapes_are_qwen38_s_projections(self):
        from probes.engine_qwen38_cells import facts
        p, F = probe(), facts()
        qsa_in = F.heads_local * 2 * F.head_dim + 2 * F.kv_heads_local * F.head_dim + F.idx_heads * F.idx_dim + F.idx_dim
        gdn_in = F.qkv_local + F.v_heads_local * F.v_dim + 2 * F.v_heads_local
        self.assertEqual(p.SHAPES, {"gdn_in": (gdn_in, F.hidden), "qsa_in": (qsa_in, F.hidden),
                                    "o_proj": (F.hidden, F.heads_local * F.head_dim)})
        self.assertEqual(F.heads_local * F.head_dim, F.v_heads_local * F.v_dim)      # both output projections: 1536
        self.assertEqual(p.SHAPES, {"gdn_in": (4120, 2560), "qsa_in": (4224, 2560), "o_proj": (2560, 1536)})
        self.assertEqual(p.ROWS, (2, 4, 6, 8))                                      # C x (K+1) at K=1, C = 1..4
        self.assertEqual(p.MODES, (0, 2))
        for n, k in p.SHAPES.values():
            count = p.layers_for(n, k)
            self.assertGreaterEqual(count * (n * k // 2), 40 * 2 ** 20)            # the weights clear the L2

    def test_the_kernel_admits_the_same_pairs(self):
        source = (ROOT / "engine" / "kernels" / "dense" / "kernels.cu").read_text(encoding="utf-8")
        body = source.split("bool mk_qwen38_input_shape(int m, int n, int k) {")[1].split("}")[0]
        self.assertIn("mk_gemm_input_mode() == 2 && m >= 2 && m <= 8", body)
        pairs = {(int(n), int(k)) for k, ns in re.findall(r"k == (\d+) && \(([^)]*)\)", body)
                 for n in re.findall(r"n == (\d+)", ns)}
        pairs |= {(int(n), int(k)) for k, n in re.findall(r"k == (\d+) && n == (\d+)", body)}
        self.assertEqual(pairs, set(probe().SHAPES.values()))
        self.assertIn("|| mk_qwen38_input_shape(m, n, k)", source)
        self.assertIn("const bool cta3=enabled && (n==4096 || n==6144);", source)

    def test_the_lane_the_queue_and_the_cells_reach_it(self):
        check = (ROOT / "probes" / "engine_kernel_check.py").read_text()
        self.assertIn("args.lanes == 'qwen38_input_reuse'", check)
        self.assertIn("from probes.engine_qwen38_input_reuse import run as qwen38_input_reuse", check)
        self.assertIn("'probes/engine_qwen38_input_reuse.py'", (ROOT / "bench" / "fleet_onepass.py").read_text())
        from probes.engine_qwen38_cells import GLUE_CASES
        self.assertIn("tests.test_engine_qwen38_input_reuse.InputReuseTests", GLUE_CASES)


if __name__ == "__main__":
    unittest.main()
