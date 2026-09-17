"""The shared expert's activation writes its padded consumer's input in its own launch (engine/QWEN38_CARRY.md S1).

Qwen3.8's shared expert (per rank: [320, 2560] gate_up, [2560, 160] down) runs its down projection through
dense.PaddedDenseLinear, which zero-extends the 160 activation columns to 256 with `F.pad` -- two launches (the zero
buffer and the copy) on each of the 49 MoE layers of a step, after the SwiGLU launch that wrote the 160 columns.
`common.swiglu(pad_to=256)` writes the zeros in that launch and PaddedDenseLinear takes an input already at its padded
width. The bytes the dense lane receives are the ones it built before; `pad_to=None` keeps every other caller's launch
(GLM-5.3's drafter) writing exactly what it wrote.

    docker exec -e TRITON_INTERPRET=1 -w <repo> stk-test python3 -m unittest tests.test_engine_swiglu_pad
"""
import importlib.util
import os
from pathlib import Path
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
torch = None
if importlib.util.find_spec("torch") is not None:
    import torch
TRITON = importlib.util.find_spec("triton") is not None
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
RUNS = torch is not None and TRITON and (INTERPRET or torch.cuda.is_available())
DEVICE = "cpu" if INTERPRET else "cuda"


@unittest.skipUnless(RUNS, "requires Triton with CUDA, or TRITON_INTERPRET=1")
class SwigluPadTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(973)
        if INTERPRET:
            patch = mock.patch.object(torch.Tensor, "is_cuda", property(lambda tensor: True))
            patch.start()
            self.addCleanup(patch.stop)

    def fused(self, rows, inter=160, dtype=torch.float32):
        """FP32 on the CPU: Triton's interpreter does not round BF16 as a GPU does, so the structure (the padded
        columns, the masks, the rows) is held in FP32 there and the BF16 bytes on a GPU."""
        return (torch.randn(rows, 2 * inter, device=DEVICE) * 2).to(dtype)

    def test_the_padded_launch_is_the_launch_then_the_pad(self):
        from engine.kernels.common.swiglu import swiglu
        for rows in (1, 2, 33):
            with self.subTest(rows=rows):
                fused = self.fused(rows)
                padded = swiglu(fused, pad_to=256)
                self.assertEqual(tuple(padded.shape), (rows, 256))
                self.assertTrue(torch.equal(padded, torch.nn.functional.pad(swiglu(fused), (0, 96))))

    def test_without_pad_to_the_launch_writes_what_the_old_launch_wrote(self):
        """The kernel as it was before `pad_to` (kept here verbatim), on the same inputs: GLM-5.3's drafter calls it
        without a pad, and nothing it wrote may move. (Against the torch pair it is bit-identical only after BF16
        rounding on a GPU -- the GPU case below.)"""
        import triton
        import triton.language as tl
        from engine.kernels.common.swiglu import swiglu

        @triton.jit
        def old(X, OUT, sX, sO, width, BLOCK: tl.constexpr):
            row = tl.program_id(0)
            c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
            live = c < width
            gate = tl.load(X + row * sX + c, live, other=0.0).to(tl.float32)
            up = tl.load(X + row * sX + width + c, live, other=0.0)
            gate = (gate * tl.sigmoid(gate)).to(up.dtype)
            tl.store(OUT + row * sO + c, gate * up, live)

        for inter in (160, 1536):
            with self.subTest(inter=inter):
                fused = self.fused(4, inter=inter)
                expected = torch.empty(4, inter, device=DEVICE, dtype=fused.dtype)
                block = 1024 if inter >= 1024 else triton.next_power_of_2(inter)
                old[(4, triton.cdiv(inter, block))](fused, expected, fused.stride(0), expected.stride(0), inter, BLOCK=block)
                self.assertTrue(torch.equal(swiglu(fused), expected))

    @unittest.skipIf(INTERPRET or torch is None or not torch.cuda.is_available(), "the GPU's BF16 rounding")
    def test_on_a_gpu_the_bf16_launch_is_the_torch_pair_padded(self):
        from engine.kernels.common.swiglu import swiglu
        fused = self.fused(8, dtype=torch.bfloat16)
        gate, up = fused.chunk(2, -1)
        self.assertTrue(torch.equal(swiglu(fused), torch.nn.functional.silu(gate) * up))
        self.assertTrue(torch.equal(swiglu(fused, pad_to=256),
                                    torch.nn.functional.pad(torch.nn.functional.silu(gate) * up, (0, 96))))

    def test_it_refuses_a_narrower_output(self):
        from engine.kernels.common.swiglu import swiglu
        with self.assertRaises(ValueError):
            swiglu(self.fused(2), pad_to=128)


@unittest.skipUnless(torch is not None, "requires torch")
class PaddedInputTests(unittest.TestCase):
    def test_a_padded_lane_passes_an_already_padded_input_through_and_pads_a_narrow_one(self):
        from engine.kernels import dense
        seen = []
        lane = object.__new__(dense.PaddedDenseLinear)
        lane.input_cols, lane.pad = 160, 96
        narrow = torch.randn(3, 160).to(torch.bfloat16)
        wide = torch.nn.functional.pad(narrow, (0, 96))
        with mock.patch.object(dense.DenseLinear, "__call__", lambda self, x, rows_ok=None, observe=True: seen.append(x)):
            lane(narrow)
            lane(wide)
        self.assertTrue(torch.equal(seen[0], seen[1]))
        self.assertIs(seen[1], wide)                                   # no second pad: the producer's buffer itself
        with self.assertRaises(ValueError):
            lane(torch.randn(3, 200).to(torch.bfloat16))

    def test_the_layer_asks_the_activation_for_the_padded_width(self):
        source = (ROOT / "engine/profiles/qwen38/net.py").read_text()
        self.assertIn('lanes.swiglu(self.linear(x, n + "sh_gate_up"), pad_to=pad_to)', source)

    def test_the_reference_lane_pads_the_same(self):
        from engine.profiles.qwen38 import lanes
        fused = torch.randn(2, 320)
        ref = lanes.reference().swiglu
        self.assertTrue(torch.equal(ref(fused, pad_to=256), torch.nn.functional.pad(ref(fused), (0, 96))))


if __name__ == "__main__":
    unittest.main()
