"""The kernel glue (engine/kernels/cells.GLUE): exact adapters that put other cells on the compiled kernels.

On the CPU every adapter is held to its oracle with the kernel's torch twin injected, and its refusals and argument
contracts are pinned before any kernel launches. The KDA glue runs its real Triton kernels against the oracle on a GPU
or on the CPU under TRITON_INTERPRET=1 (Triton's interpreter executes the same kernels). The native cases -- the MLA
megakernel, the V4.1 mHC seam, the packed dense lane -- need CUDA and the ST image; they are the judgment the wizard's
table still marks as pending.

    docker exec -e TRITON_INTERPRET=1 -w <repo> stk-test python3 -m unittest tests.test_engine_kernel_glue
"""
import builtins
import contextlib
import importlib.util
import math
import os
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from engine.base import kernel_shape as ks
from engine.base.kernel_shape import MEASURED, Attention, LinearAttention
from tests.image_kernels import PRESENT, REASON

ROOT = Path(__file__).resolve().parents[1]
torch = None
if importlib.util.find_spec("torch") is not None:
    import torch
TRITON = importlib.util.find_spec("triton") is not None
GPU = torch is not None and torch.cuda.is_available() and PRESENT
GPU_REASON = "requires CUDA; " + REASON
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
KDA_DEVICE = "cpu" if INTERPRET else "cuda"
KDA_RUNS = torch is not None and TRITON and (INTERPRET or (torch.cuda.is_available() and PRESENT))

if TRITON:
    import triton
    import triton.language as tl

    # the interpreter patches `tl` inside a kernel but not a builtin bound under another name (kda/op.py: exp = tl.exp);
    # these stand in for those names under TRITON_INTERPRET=1 only (see `kda_kernels`)
    @triton.jit
    def _interpreted_exp(x):
        return tl.exp(x)

    @triton.jit
    def _interpreted_exp2(x):
        return tl.exp2(x)

    @triton.jit
    def _interpreted_log(x):
        return tl.log(x)


@contextlib.contextmanager
def kda_kernels():
    """The KDA kernels as they run here. On a GPU: untouched. Under TRITON_INTERPRET=1, four test-only accommodations
    for what the interpreter does differently from a compiled kernel -- none changes a kernel's arithmetic:

    - a `tl` builtin bound under another name (kda/op.py: `exp = tl.exp`) is not patched by the interpreter, so those
      names get @triton.jit wrappers that call the builtin;
    - a kernel loop variable is a Python int where the compiled one is a tl.int32 (chunk_delta_h.py: `i_t.to(...)`),
      so that module's `range` yields int32 tensors;
    - autotuning benchmarks through a device driver, so every config times equal and the first one runs;
    - the wrappers' CUDA-only argument checks see these CPU tensors as the interpreter's device."""
    if not INTERPRET:
        yield
        return
    import numpy as np
    from triton.runtime.autotuner import Autotuner
    from triton.runtime.interpreter import TensorHandle
    from engine.kernels.kda import chunk_delta_h, cumsum, fused_recurrent, kda, l2norm, solve_tril

    def int32_range(*bounds):
        ints = [int(b.handle.data.item()) if hasattr(b, "handle") else int(b) for b in bounds]
        for i in builtins.range(*ints):
            yield tl.core.tensor(TensorHandle(np.array([i], dtype=np.int32), tl.int32), tl.int32)

    wrappers = {"exp": _interpreted_exp, "exp2": _interpreted_exp2, "log": _interpreted_log}
    with contextlib.ExitStack() as stack:
        for module in (chunk_delta_h, cumsum, fused_recurrent, kda, l2norm, solve_tril):
            for name, wrapper in wrappers.items():
                if hasattr(module, name):
                    stack.enter_context(patch.object(module, name, wrapper))
        stack.enter_context(patch.object(chunk_delta_h, "range", int32_range, create=True))
        stack.enter_context(patch.object(Autotuner, "_bench", lambda self, *args, config, **meta: [0.0, 0.0, 0.0]))
        stack.enter_context(patch.object(torch.Tensor, "is_cuda", property(lambda tensor: True)))
        yield

GQA_QWEN = Attention("gqa", heads=6, head_dim=256, kv_heads=1, sink=False)      # Qwen3.8 per rank, the sink read as absent
HEAD_DECAY = replace(MEASURED, attention=GQA_QWEN,
                     linear=LinearAttention(heads=4, v_heads=12, k_dim=128, v_dim=128, conv=4, decay="head"))
SPLIT_SINKHORN = replace(MEASURED, hc_variant="split_sinkhorn")                  # the V4.1 seam is compiled at 4096 too


def bench():
    """engine/kernels/dense/mhc_reference.py: the V4.1 seam's torch form (stdlib at import, torch inside)."""
    from engine.kernels.dense import mhc_reference
    import importlib
    importlib.reload(mhc_reference)
    return mhc_reference


def twin(q, ckv, slots, lens, sm_scale, ckv_scale, out=None):
    """The MLA kernel's torch twin (engine/kernels/mla.mla_decode_ref), held to the compiled cell's call contract."""
    from engine.kernels.mla import MLA_D, MLA_H, mla_decode_ref
    assert tuple(q.shape[1:]) == (MLA_H, MLA_D) and q.is_contiguous(), tuple(q.shape)
    assert out is not None and out.is_contiguous() and out.shape == q.shape and out.dtype == q.dtype
    out.copy_(mla_decode_ref(q, ckv, slots, lens, sm_scale, ckv_scale))
    return out


def selection(t, positions, width, device="cpu"):
    slots = torch.randint(0, positions, (t, width), dtype=torch.int32, device=device)
    lens = torch.randint(1, width + 1, (t,), dtype=torch.int32, device=device)
    lens[0] = width
    return slots, lens


@unittest.skipUnless(torch is not None, "requires torch")
class MlaGlueTests(unittest.TestCase):
    """engine/kernels/mla/glue.py: heads grouped, the latent zero-extended, a GQA key and value packed side by side."""

    def setUp(self):
        torch.manual_seed(20260913)
        ks.reset()
        self.addCleanup(ks.reset)

    def test_grouped_heads_over_a_padded_latent_are_the_attention(self):
        from engine.kernels.mla import glue
        from engine.modules.sparse_attention import mla_sparse_mqa
        for heads, latent in ((16, 512), (32, 512), (20, 512), (6, 512), (16, 256), (37, 128)):
            with self.subTest(heads=heads, latent=latent):
                t, positions, width = 5, 300, 24
                q = torch.randn(t, heads, latent)
                rows = torch.randn(positions, latent)
                slots, lens = selection(t, positions, width)
                launches = []
                got = glue.grouped(q, glue.pad_rows(rows), slots, lens, latent ** -0.5, 1.0,
                                   attend=lambda *a, **kw: launches.append(1) or twin(*a, **kw))
                self.assertEqual((tuple(got.shape), got.is_contiguous()), ((t, heads, latent), True))
                self.assertEqual(len(launches), math.ceil(heads / 16))                 # one launch per group of 16
                torch.testing.assert_close(got, mla_sparse_mqa(q, rows, slots, lens, latent ** -0.5), rtol=1e-5, atol=1e-5)

    def test_a_packed_key_and_value_are_grouped_query_attention(self):
        from engine.kernels.mla import glue
        from engine.modules.sparse_attention import gqa_sparse
        for heads, kv_heads, dim, k_gain, v_gain in ((6, 1, 256, 1.0, 1.0), (8, 2, 128, 2.0, 0.25),
                                                     (24, 3, 64, 0.5, 4.0), (32, 1, 256, 1.0, 8.0)):
            with self.subTest(heads=heads, kv_heads=kv_heads, dim=dim, gains=(k_gain, v_gain)):
                t, positions, width = 4, 200, 17
                q = torch.randn(t, heads, dim)
                k, v = torch.randn(positions, kv_heads, dim), torch.randn(positions, kv_heads, dim)
                slots, lens = selection(t, positions, width)
                rows = glue.pack_kv(k, v, k_gain=k_gain, v_gain=v_gain)
                self.assertEqual(tuple(rows.shape), (positions, kv_heads, 512))
                got = glue.gqa(q, rows.reshape(positions * kv_heads, 512), slots, lens, dim ** -0.5, 1.0,
                               kv_heads=kv_heads, k_gain=k_gain, v_gain=v_gain, attend=twin)
                self.assertEqual((tuple(got.shape), got.is_contiguous()), ((t, heads, dim), True))
                torch.testing.assert_close(got, gqa_sparse(q, k, v, slots, lens, dim ** -0.5), rtol=1e-5, atol=1e-5)

    def test_the_latent_precision_is_the_only_change(self):
        """With the rows in the latent's e4m3, the adapter computes GQA over exactly the K and V those bytes hold."""
        from engine.kernels.mla import glue
        from engine.modules.sparse_attention import gqa_sparse
        t, positions, width, dim, k_gain, v_gain = 3, 64, 9, 256, 4.0, 2.0
        q = torch.randn(t, 6, dim, dtype=torch.bfloat16)
        k, v = torch.randn(positions, 1, dim), torch.randn(positions, 1, dim)
        rows = glue.pack_kv(k, v, k_gain=k_gain, v_gain=v_gain).reshape(positions, 512).to(torch.float8_e4m3fn)
        held = rows.float().view(positions, 1, 512)
        slots, lens = selection(t, positions, width)
        got = glue.gqa(q, rows, slots, lens, dim ** -0.5, 1.0, kv_heads=1, k_gain=k_gain, v_gain=v_gain, attend=twin)
        self.assertEqual(got.dtype, torch.bfloat16)
        ref = gqa_sparse(q.float(), held[..., :dim] / k_gain, held[..., dim:2 * dim] / v_gain, slots, lens, dim ** -0.5)
        torch.testing.assert_close(got.float(), ref, rtol=2 ** -8 + 1e-4, atol=1e-4)      # the output's BF16 half-ulp only
        self.assertTrue(torch.equal(held[..., 2 * dim:], torch.zeros(positions, 1, 512 - 2 * dim)))

    def test_the_adapters_refuse_what_they_cannot_carry(self):
        from engine.kernels.mla import glue
        slots, lens = torch.zeros(2, 3, dtype=torch.int32), torch.ones(2, dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "side by side"):
            glue.gqa(torch.zeros(2, 4, 300), torch.zeros(10, 512), slots, lens, 1.0, 1.0, kv_heads=1, attend=twin)
        with self.assertRaisesRegex(ValueError, "not a multiple"):
            glue.gqa(torch.zeros(2, 5, 64), torch.zeros(10, 512), slots, lens, 1.0, 1.0, kv_heads=2, attend=twin)
        with self.assertRaisesRegex(ValueError, "power of two"):
            glue.pack_kv(torch.zeros(3, 1, 8), torch.zeros(3, 1, 8), k_gain=3.0)
        with self.assertRaisesRegex(ValueError, "power of two"):
            glue.gqa(torch.zeros(2, 4, 64), torch.zeros(10, 512), slots, lens, 1.0, 1.0, kv_heads=1, v_gain=0.0, attend=twin)
        with self.assertRaisesRegex(ValueError, "must match"):
            glue.pack_kv(torch.zeros(3, 1, 8), torch.zeros(3, 1, 16))
        with self.assertRaisesRegex(ValueError, "does not fit"):
            glue.grouped(torch.zeros(1, 16, 1024), torch.zeros(4, 512), slots[:1], lens[:1], 1.0, 1.0, attend=twin)
        with self.assertRaisesRegex(ValueError, "does not fit"):
            glue.pad_rows(torch.zeros(2, 600))
        from engine.kernels import mla
        if not mla._ARMED["mla"]:
            with self.assertRaisesRegex(RuntimeError, "not armed"):
                glue.grouped(torch.zeros(1, 16, 512), torch.zeros(4, 512), slots[:1], lens[:1], 1.0, 1.0)

    def test_arm_admits_the_bound_attention_by_the_glue_rule(self):
        from engine.kernels import mla
        from engine.kernels.mla import glue
        ks.bind(HEAD_DECAY)
        checks = []
        with patch.object(mla, "maybe_arm", side_effect=lambda check=None: checks.append(check) or check()):
            glue.arm()
        self.assertEqual(checks, [glue.check])
        with self.assertRaisesRegex(RuntimeError, "compiled for the 16 heads"):
            mla._check_cell()                                                     # the lane itself still refuses the cell
        ks.reset()
        ks.bind(replace(MEASURED, attention=replace(GQA_QWEN, sink=True)))
        with patch.object(mla, "maybe_arm", side_effect=lambda check=None: check()):
            with self.assertRaisesRegex(RuntimeError, "no sink term"):
                glue.arm()
        # maybe_arm runs the admission check before it builds anything
        source = (ROOT / "engine/kernels/mla/__init__.py").read_text().split("def maybe_arm(", 1)[1].split("\ndef ", 1)[0]
        self.assertLess(source.index("(_check_cell if check is None else check)()"), source.index("_build()"))


@unittest.skipUnless(torch is not None, "requires torch")
class MhcV41GlueTests(unittest.TestCase):
    """engine/kernels/dense/mhc.MHCV41: the megakernel's V4.1 seam, decode and prefill in pieces."""

    def setUp(self):
        torch.manual_seed(41)
        ks.reset()
        self.addCleanup(ks.reset)

    class Recorder:
        def __init__(self):
            self.calls = []

        def run_mhc_v41(self, ptrs, scalars, ints, hidden):
            self.calls.append((list(ptrs), list(scalars), list(ints), hidden))

    def inputs(self, tokens, hidden=4096):
        g = lambda *shape, scale=1.0, dtype=torch.float32: (torch.randn(*shape) * scale).to(dtype)
        return dict(x=g(tokens, hidden, scale=.1, dtype=torch.bfloat16), res=g(tokens, 4, hidden, scale=.1, dtype=torch.bfloat16),
                    post=g(tokens, 4, scale=.5), comb=g(tokens, 4, 4, scale=.25), scale=torch.tensor([.8, 1.1, .7]),
                    base=g(24, scale=.2), norm=g(hidden, scale=.5, dtype=torch.bfloat16), pre=torch.rand(tokens, 4))

    def test_the_wrapper_holds_its_weights_and_the_kernel_scratch(self):
        from engine.kernels.dense import mhc
        ks.bind(SPLIT_SINKHORN)
        layer = mhc.MHCV41({"L0.attn": torch.zeros(24, 4 * 4096)}, ext=self.Recorder())
        self.assertEqual((layer.hidden, layer.hc, layer.nout), (4096, 4, 24))
        self.assertEqual([(t.numel(), t.dtype) for t in layer.workspace],
                         [(16 * 128 * 24, torch.float32), (16 * 128, torch.float32), (16 * 128, torch.float32),
                          (128 * 4, torch.float32), (128 * 4096, torch.bfloat16), (8, torch.int32)])
        with self.assertRaisesRegex(ValueError, "FP32"):
            mhc.MHCV41({"L0.attn": torch.zeros(24, 4 * 4096, dtype=torch.bfloat16)}, ext=self.Recorder())
        ks.reset()
        ks.bind(MEASURED)
        with self.assertRaisesRegex(ValueError, "mixes by mhc"):
            mhc.MHCV41({"L0.attn": torch.zeros(24, 4 * 4096)}, ext=self.Recorder())

    def test_the_launch_passes_the_twenty_pointer_contract(self):
        from engine.kernels.dense import mhc
        ks.bind(SPLIT_SINKHORN)
        ext = self.Recorder()
        fn = torch.randn(24, 4 * 4096)
        layer = mhc.MHCV41({"L3.ffn": fn}, ext=ext)
        a = self.inputs(5)
        with patch.object(mhc.MHCV41, "_check", lambda *args: None):                 # CPU tensors: the pointers only
            outs = layer("L3.ffn", a["x"], a["res"], a["post"], a["comb"], a["scale"], a["base"], a["norm"], a["pre"],
                         1e-6, 1e-6, 1e-6, 2.0, 1e-6, 20)
        (ptrs, scalars, ints, hidden), = ext.calls
        residual, post, comb, layer_input, pre = outs
        expected = [a["x"], a["res"], a["post"], a["comb"], fn, a["scale"], a["base"], a["norm"], residual, post, comb,
                    layer_input, *layer.workspace, a["pre"], pre]
        self.assertEqual(ptrs, [t.data_ptr() for t in expected])
        self.assertEqual((scalars, ints, hidden), ([1e-6, 1e-6, 1e-6, 2.0, 1e-6], [5, 20], 4096))
        self.assertEqual([(tuple(t.shape), t.dtype) for t in outs],
                         [((5, 4, 4096), torch.bfloat16), ((5, 4), torch.float32), ((5, 16), torch.float32),
                          ((5, 4096), torch.bfloat16), ((5, 4), torch.float32)])
        self.assertEqual(layer.executed, {"L3.ffn"})
        with self.assertRaisesRegex(ValueError, "prefill"):
            layer("L3.ffn", *[torch.zeros(129, *t.shape[1:], dtype=t.dtype) if t.shape[0] == 5 else t
                              for t in (a["x"], a["res"], a["post"], a["comb"], a["scale"], a["base"], a["norm"], a["pre"])],
                  1e-6, 1e-6, 1e-6, 2.0, 1e-6, 20)
        with self.assertRaisesRegex(ValueError, "contiguous CUDA"):                    # the real check refuses CPU tensors
            layer("L3.ffn", a["x"], a["res"], a["post"], a["comb"], a["scale"], a["base"], a["norm"], a["pre"],
                  1e-6, 1e-6, 1e-6, 2.0, 1e-6, 20)
        self.assertEqual(len(ext.calls), 1)                                            # and nothing launched

    def test_prefill_in_pieces_is_the_seam_token_by_token(self):
        from engine.kernels.dense import mhc
        seam = bench()
        ks.bind(SPLIT_SINKHORN)
        launches = []

        class Twin(mhc.MHCV41):
            def _launch(self, key, rows, coefficients, outs, scalars, sinkhorn):
                x, res, post, comb, pre = rows
                rms_eps, pre_eps, sinkhorn_eps, post_mult, norm_eps = scalars
                got = seam.v41_component_reference(x, res, post, comb, self.weights[key], *coefficients, pre,
                                                   rms_eps=rms_eps, norm_eps=norm_eps, pre_eps=pre_eps,
                                                   sinkhorn_eps=sinkhorn_eps, post_mult=post_mult,
                                                   sinkhorn_iters=sinkhorn)
                for out, value in zip(outs, got):
                    out.copy_(value.reshape(out.shape))
                launches.append(x.shape[0])

        fn = torch.randn(24, 4 * 4096) * .02
        layer = Twin({"L0": fn}, ext=self.Recorder())
        a = self.inputs(300)
        params = (1e-6, 1e-6, 1e-6, 2.0, 1e-6, 20)
        outs = layer.prefill("L0", a["x"], a["res"], a["post"], a["comb"], a["scale"], a["base"], a["norm"], a["pre"], *params)
        self.assertEqual(launches, [128, 128, 44])
        for i in (0, 127, 128, 255, 256, 299):
            one = seam.v41_component_reference(a["x"][i:i + 1], a["res"][i:i + 1], a["post"][i:i + 1], a["comb"][i:i + 1],
                                               fn, a["scale"], a["base"], a["norm"], a["pre"][i:i + 1],
                                               rms_eps=1e-6, norm_eps=1e-6, pre_eps=1e-6, sinkhorn_eps=1e-6,
                                               post_mult=2.0, sinkhorn_iters=20)
            for name, out, value in zip(("residual", "post", "comb", "layer input", "pre"), outs, one):
                with self.subTest(token=i, output=name):
                    torch.testing.assert_close(out[i:i + 1].float(), value.reshape(out[i:i + 1].shape).float(),
                                               rtol=1e-4, atol=1e-4)


@unittest.skipUnless(torch is not None and TRITON, "the KDA modules import Triton")
class KdaDecayGlueTests(unittest.TestCase):
    """engine/kernels/kda/ring.recurrent_decay_ring(_rows) and kda/chunk_decay.chunk_kda_with_decay: the fused lanes'
    launches on a decay computed outside the kernel. Their refusals run before any launch."""

    def setUp(self):
        ks.reset()
        self.addCleanup(ks.reset)

    def test_the_ring_forms_are_checked_by_the_bound_decay(self):
        from engine.kernels.kda import ring
        ks.bind(HEAD_DECAY)
        with self.assertRaisesRegex(ValueError, "recurrent_decay_ring"):
            ring._check_cell(False)                                            # GDN cannot take the fused gate
        ring._check_cell(True)
        ks.reset()
        ks.bind(replace(MEASURED, linear=None))
        for decay in (False, True):
            with self.subTest(decay=decay), self.assertRaisesRegex(ValueError, "no linear attention"):
                ring._check_cell(decay)

    def test_the_decay_entries_take_the_decay_and_nothing_else(self):
        from engine.kernels import linear_decay
        from engine.kernels.kda import ring
        ks.bind(HEAD_DECAY)
        t, h, hv, kd = 3, 4, 12, 128
        q, k = torch.zeros(1, t, h, kd), torch.zeros(1, t, h, kd)
        v, beta = torch.zeros(1, t, hv, kd), torch.zeros(1, t, hv)
        states = torch.zeros(2, 4, hv, kd, kd)
        with self.assertRaisesRegex(ValueError, "no gate parameters"):
            ring._recurrent(q, k, v, torch.zeros(1, t, hv), beta, torch.zeros(h), None, states, 0, 0, None, decay=True)
        with self.assertRaisesRegex(ValueError, "per head"):
            ring.recurrent_decay_ring(q, k, v, torch.zeros(1, t, hv, kd, 1), beta, states, 0, 0)
        widened = []
        real = linear_decay.per_channel
        with patch.object(linear_decay, "per_channel", side_effect=lambda g, d: widened.append(d) or real(g, d)):
            with self.assertRaisesRegex(ValueError, "CUDA"):                      # CPU tensors stop at the launch contract
                ring.recurrent_decay_ring(q, k, v, torch.zeros(1, t, hv), beta, states, 0, 0)
        self.assertEqual(widened, [kd])                                            # the per-head decay, read per channel
        with self.assertRaisesRegex(ValueError, "slot and one CUDA context"):
            ring.recurrent_decay_ring_rows(q, k, v, torch.zeros(1, t, hv), beta, states, 0, 0)

    def test_the_chunk_entry_refuses_before_any_launch(self):
        from engine.kernels.kda.chunk_decay import chunk_kda_with_decay
        t, h, hv, kd = 70, 4, 12, 128
        q, k, v = torch.zeros(1, t, h, kd), torch.zeros(1, t, h, kd), torch.zeros(1, t, hv, kd)
        decay, beta = torch.zeros(1, t, hv), torch.zeros(1, t, hv)
        with self.assertRaisesRegex(ValueError, "not a multiple"):
            chunk_kda_with_decay(q, k, torch.zeros(1, t, 5, kd), torch.zeros(1, t, 5), torch.zeros(1, t, 5))
        with self.assertRaisesRegex(ValueError, "per head"):
            chunk_kda_with_decay(q, k, v, torch.zeros(1, t, h), beta)
        with self.assertRaisesRegex(ValueError, "beta"):
            chunk_kda_with_decay(q, k, v, decay, torch.zeros(1, t, h))
        with self.assertRaisesRegex(ValueError, "CUDA contiguous"):
            chunk_kda_with_decay(q, k, v, decay, beta, out=torch.zeros(1, t, hv, kd))

    def test_the_adaptation_is_the_recurrence_the_kernels_read(self):
        """Summing a per-head decay then widening it is the sum of the widened decay, and repeating the key heads is the
        recurrent kernel's grouping (i_h = i_hv // (HV // H))."""
        from engine.kernels.kda.kda import RCP_LN2
        g = -torch.rand(1, 64, 12, dtype=torch.float64)
        torch.testing.assert_close(g.cumsum(1).unsqueeze(-1).expand(1, 64, 12, 128),
                                   g.unsqueeze(-1).expand(1, 64, 12, 128).cumsum(1), rtol=0, atol=0)
        self.assertAlmostEqual(RCP_LN2, 1 / math.log(2), places=15)
        self.assertTrue(torch.equal(torch.arange(4).repeat_interleave(3), torch.arange(12) // 3))


@unittest.skipUnless(KDA_RUNS, "requires CUDA and the ST image, or TRITON_INTERPRET=1 with Triton")
class KdaDecayKernelTests(unittest.TestCase):
    """The KDA decay glue's real Triton kernels against modules/linear_attention and against the fused lanes: GDN's
    per-head decay, value heads a multiple of the key heads."""

    def setUp(self):
        torch.manual_seed(129613)
        ks.reset()
        self.addCleanup(ks.reset)
        # the interpreter is slow: small widths there, Qwen3.8's per-rank cell on a GPU
        self.h, self.hv, self.kd = (2, 4, 16) if INTERPRET else (4, 12, 128)
        ks.bind(replace(HEAD_DECAY, linear=LinearAttention(heads=self.h, v_heads=self.hv, k_dim=self.kd, v_dim=self.kd,
                                                          conv=4, decay="head")))

    @staticmethod
    def rel(a, b):
        return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30))

    def step(self, t, dtype):
        g = lambda *shape: torch.randn(*shape, device=KDA_DEVICE, dtype=dtype)
        decay = -torch.nn.functional.softplus(torch.randn(1, t, self.hv, device=KDA_DEVICE)) * .5
        return g(1, t, self.h, self.kd), g(1, t, self.h, self.kd), g(1, t, self.hv, self.kd), decay, g(1, t, self.hv)

    def oracle(self, q, k, v, decay, beta, state=None):
        from engine.modules.linear_attention import gated_delta_rule
        group = self.hv // self.h
        return gated_delta_rule(q.repeat_interleave(group, 2), k.repeat_interleave(group, 2), v, decay, beta, state,
                                scale=self.kd ** -0.5, qk_l2norm=True)

    def test_the_ring_decay_entry_is_the_recurrence_and_the_functional_lane(self):
        from engine.kernels.kda import fused_recurrent_kda
        from engine.kernels.kda.ring import recurrent_decay_ring
        from engine.kernels.linear_decay import per_channel
        dtype = torch.float32 if INTERPRET else torch.bfloat16
        # a ring holds at least a step's tokens (the launcher refuses T > R): 8 cells for the GPU's 7-token step, which
        # the first lane run (measurements/qwen38_lane_20260917) met as a refusal with 6
        cells_ = 8
        for t in ((1, 3) if INTERPRET else (1, 3, 7)):
            q, k, v, decay, beta = self.step(t, dtype)
            ring = torch.randn(3, cells_, self.hv, self.kd, self.kd, device=KDA_DEVICE) * .1
            for slot, context in ((0, 0), (1, 1), (2, cells_ + 2)):
                with self.subTest(tokens=t, slot=slot, context=context):
                    initial = (ring[slot, (context - 1) % cells_][None].clone() if context
                               else torch.zeros(1, self.hv, self.kd, self.kd, device=KDA_DEVICE))
                    expected = ring.clone()
                    wide = ring.clone()
                    with kda_kernels():
                        # GDN sigmoids in the projection's dtype before the FP32 recurrence. The generic KDA
                        # functional entry takes those beta values, without applying a second sigmoid.
                        out_ref, states = fused_recurrent_kda(q, k, v, per_channel(decay, self.kd), beta.sigmoid(),
                                                              scale=self.kd ** -0.5, initial_state=initial,
                                                              inplace_final_state=False, use_qk_l2norm_in_kernel=True,
                                                              sigmoid_beta=False, compute_gate=False, state_layout="kv")
                        out = recurrent_decay_ring(q, k, v, decay, beta, ring, slot, context)
                        out_wide = recurrent_decay_ring(q, k, v, per_channel(decay, self.kd).contiguous(), beta, wide,
                                                        slot, context)
                    for i, state in enumerate(states):
                        expected[slot, (context + i) % cells_].copy_(state)
                    if INTERPRET:
                        # FP32 torch.sigmoid and the interpreter's numpy exp differ in their last bit; GPU BF16
                        # rounds both to the model's beta. Keep the storage equality check exact there.
                        torch.testing.assert_close(out, out_ref, rtol=2e-6, atol=2e-7)
                        torch.testing.assert_close(ring, expected, rtol=2e-6, atol=2e-7)
                    else:
                        self.assertTrue(torch.equal(out, out_ref))             # the functional lane, byte for byte
                        self.assertTrue(torch.equal(ring, expected))          # its states, written in the ring
                    self.assertTrue(torch.equal(out_wide, out) and torch.equal(wide, ring))   # stride 0 == contiguous
                    oracle, final = self.oracle(q, k, v, decay, torch.sigmoid(beta), initial)
                    self.assertLessEqual(self.rel(out, oracle), 1e-5 if INTERPRET else 1e-2)
                    self.assertLessEqual(self.rel(ring[slot, (context + t - 1) % cells_], final[0]),
                                         1e-5 if INTERPRET else 1e-2)

    def test_the_decay_entry_with_kda_s_gate_is_the_fused_entry(self):
        """Hand the decay entry the per-channel decay KDA's fused gate computes and it is the fused ring entry."""
        from engine.kernels.kda.ring import recurrent_decay_ring, recurrent_kda_ring
        from engine.modules.linear_attention import kda_gate
        h, kd, t = self.h, self.kd, 3
        ks.reset()
        ks.bind(replace(MEASURED, linear=LinearAttention(heads=h, v_heads=h, k_dim=kd, v_dim=kd, conv=4)))
        g = lambda *shape: torch.randn(*shape, device=KDA_DEVICE)
        q, k, v, raw, beta = g(1, t, h, kd), g(1, t, h, kd), g(1, t, h, kd), g(1, t, h, kd), g(1, t, h)
        a_log, bias = g(h) * .2, g(h * kd) * .1
        fused, direct = (torch.randn(2, 4, h, kd, kd, device=KDA_DEVICE) * .1 for _ in range(2))
        direct.copy_(fused)
        decay = kda_gate(raw, a_log, bias, -5.0, safe_gate=True)
        with kda_kernels():
            out_fused = recurrent_kda_ring(q, k, v, raw, beta, a_log, bias, fused, 1, 3, -5.0)
            out = recurrent_decay_ring(q, k, v, decay, beta, direct, 1, 3)
        self.assertLessEqual(self.rel(out, out_fused), 1e-5)
        self.assertLessEqual(self.rel(direct, fused), 1e-5)

    def test_the_rows_fold_is_the_one_row_launches(self):
        from engine.kernels.kda.ring import recurrent_decay_ring, recurrent_decay_ring_rows
        rows, t = 2, 2
        q, k, v, decay, beta = self.step(rows * t, torch.float32 if INTERPRET else torch.bfloat16)
        ring = torch.randn(3, 6, self.hv, self.kd, self.kd, device=KDA_DEVICE) * .1
        one = ring.clone()
        slots = torch.tensor([2, 0], device=KDA_DEVICE, dtype=torch.int32)
        contexts = torch.tensor([5, 0], device=KDA_DEVICE, dtype=torch.int32)
        with kda_kernels():
            folded = recurrent_decay_ring_rows(q, k, v, decay, beta, ring, slots, contexts)
            parts = [recurrent_decay_ring(*(x[:, i * t:(i + 1) * t] for x in (q, k, v, decay, beta)), one,
                                          int(slots[i]), int(contexts[i])) for i in range(rows)]
        self.assertTrue(torch.equal(folded, torch.cat(parts, 1)))
        self.assertTrue(torch.equal(ring, one))

    def test_the_chunk_decay_entry_is_the_recurrence(self):
        from engine.kernels.kda.chunk_decay import chunk_kda_with_decay
        from engine.kernels.kda.index import single_sequence_bounds
        t = 130 if INTERPRET else 300                                   # three 64-token chunks either way
        dtype = torch.float32 if INTERPRET else torch.bfloat16
        q, k, v, decay, _ = self.step(t, dtype)
        decay = decay * .2
        beta = torch.sigmoid(torch.randn(1, t, self.hv, device=KDA_DEVICE))
        bounds = single_sequence_bounds(t, q.device)
        # without `out` the pipeline writes its output over v's storage (as the fused entry does): hand it a copy
        run = lambda d, **kw: chunk_kda_with_decay(q, k, v.clone(), d, beta, use_qk_l2norm_in_kernel=True, cu_seqlens=bounds,
                                                   out=None if INTERPRET else torch.empty_like(v), **kw)
        with kda_kernels():
            o, state, marks = run(decay, output_final_state=True, states_at=[1, 2])
            wide, _ = run(decay.unsqueeze(-1).expand(1, t, self.hv, self.kd))
        tolerance = 1e-5 if INTERPRET else 1e-2
        oracle, final = self.oracle(q, k, v, decay, beta)
        self.assertLessEqual(self.rel(o, oracle), tolerance)
        self.assertLessEqual(self.rel(state.transpose(-1, -2), final), tolerance)
        for n, chunk in enumerate((1, 2)):
            _, at = self.oracle(*(x[:, :64 * chunk] for x in (q, k, v, decay, beta)))
            self.assertLessEqual(self.rel(marks[n].transpose(-1, -2), at[0]), tolerance)
        self.assertLessEqual(self.rel(wide, o), 1e-5 if INTERPRET else 1e-3)       # per channel: the same decay


@unittest.skipUnless(torch is not None, "requires torch")
class DenseGlueTests(unittest.TestCase):
    """engine/kernels/dense.PaddedDenseLinear: zero columns in, zero-extended input, the same product."""

    def fake_lane(self):
        from engine.kernels import dense
        seen = {}

        def init(layer, weight, *, prefill=True, hessians=None, store=None, name=None, smooth=None,
                 decode_precision="w4", fp8_decode_rows=False):
            layer.rows, layer.cols, layer.weight, layer.observer = *weight.shape, weight, None
            seen["smooth"] = smooth
            seen["decode"] = (decode_precision, fp8_decode_rows)     # the padded layer hands its parent both (#1226)

        def call(layer, x, rows_ok=None, *, observe=True):
            if x.shape[-1] != layer.cols:
                raise ValueError("dense input does not match its bound weight")
            seen["x"] = x
            return torch.nn.functional.linear(x.float(), layer.weight.float())        # FP32: the padding, not a rounding

        return seen, (patch.object(dense.DenseLinear, "__init__", init), patch.object(dense.DenseLinear, "__call__", call))

    def test_padded_columns_are_the_same_product(self):
        from engine.kernels import dense
        seen, patches = self.fake_lane()
        torch.manual_seed(576)
        weight = torch.randn(64, 576, dtype=torch.bfloat16)
        with patches[0], patches[1]:
            layer = dense.PaddedDenseLinear(weight, smooth=torch.full((576,), 2.0))
            self.assertEqual((layer.cols, layer.input_cols, layer.pad), (640, 576, 64))
            self.assertTrue(torch.equal(layer.weight[:, :576], weight))
            self.assertFalse(layer.weight[:, 576:].any())
            self.assertTrue(torch.equal(seen["smooth"], torch.cat([torch.full((576,), 2.0), torch.ones(64)])))
            x = torch.randn(3, 5, 576, dtype=torch.bfloat16)
            out = layer(x)
            self.assertEqual(tuple(seen["x"].shape), (3, 5, 640))
            self.assertFalse(seen["x"][..., 576:].any())
            torch.testing.assert_close(out, torch.nn.functional.linear(x.float(), weight.float()), rtol=1e-5, atol=1e-4)
            # an input already at the padded width (common.swiglu's pad_to writes one) passes through unpadded again
            wide = torch.zeros(2, 640, dtype=torch.bfloat16)
            layer(wide)
            self.assertIs(seen["x"], wide)
            with self.assertRaisesRegex(ValueError, "does not match"):
                layer(torch.zeros(2, 600, dtype=torch.bfloat16))
            self.assertIsNone(layer.packet_projector())
            self.assertIsNone(layer.slot_writer(4))
            aligned = dense.PaddedDenseLinear(torch.randn(8, 512, dtype=torch.bfloat16))
            self.assertEqual((aligned.pad, aligned.cols), (0, 512))
            self.assertEqual((dense.padded_columns(576), dense.padded_columns(512), dense.padded_columns(20480)), (640, 512, 20480))
            with self.assertRaisesRegex(ValueError, "widest K"):
                dense.padded_columns(20500)
            with self.assertRaisesRegex(ValueError, r"\[N, K\]"):
                dense.PaddedDenseLinear(torch.zeros(4, 4, 576, dtype=torch.bfloat16))
            with self.assertRaisesRegex(ValueError, "Hessians"):
                dense.PaddedDenseLinear(weight, hessians=torch.eye(576))
            with self.assertRaisesRegex(ValueError, "widest K"):
                dense.PaddedDenseLinear(torch.zeros(4, 20500, dtype=torch.bfloat16))


@unittest.skipUnless(GPU, GPU_REASON)
class GlueOnTheGpuTests(unittest.TestCase):
    """The adapters over the armed kernels, against the same oracles: the judgment the table marks pending."""

    def setUp(self):
        torch.manual_seed(129613)
        ks.reset()
        self.addCleanup(ks.reset)

    @staticmethod
    def rel(a, b):
        return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30))

    def test_mla_glue_on_the_megakernel(self):
        from engine.kernels.mla import glue
        from engine.modules.sparse_attention import gqa_sparse, mla_sparse_mqa
        ks.bind(HEAD_DECAY)
        glue.arm()
        t, positions, width, dim = 40, 4096, 512, 256
        q = torch.randn(t, 6, dim, device="cuda", dtype=torch.bfloat16) * .3
        k = torch.randn(positions, 1, dim, device="cuda") * .5
        v = torch.randn(positions, 1, dim, device="cuda") * .5
        rows = glue.pack_kv(k, v, k_gain=2.0).reshape(positions, 512).to(torch.float8_e4m3fn)
        held = rows.float().view(positions, 1, 512)
        slots, lens = selection(t, positions, width, "cuda")
        got = glue.gqa(q, rows.view(torch.uint8), slots, lens, dim ** -0.5, 1.0, kv_heads=1, k_gain=2.0)
        ref = gqa_sparse(q, held[..., :dim] / 2.0, held[..., dim:2 * dim], slots, lens, dim ** -0.5)
        self.assertLessEqual(self.rel(got, ref), 2e-2)
        q = torch.randn(t, 20, 256, device="cuda", dtype=torch.bfloat16) * .3
        latent = (torch.randn(positions, 256, device="cuda") * .5).to(torch.float8_e4m3fn)
        got = glue.grouped(q, glue.pad_rows(latent.float()).to(torch.float8_e4m3fn).view(torch.uint8), slots, lens,
                           256 ** -0.5, 1.0)
        self.assertLessEqual(self.rel(got, mla_sparse_mqa(q, latent, slots, lens, 256 ** -0.5)), 2e-2)

    def test_mhc_v41_on_the_megakernel(self):
        from engine.kernels.dense import mhc
        seam = bench()
        ks.bind(replace(SPLIT_SINKHORN, hidden=5120, comm=replace(MEASURED.comm, hidden=5120),
                        moe=replace(MEASURED.moe, hidden=5120)))
        fn = torch.randn(24, 4 * 5120, device="cuda") * .02
        layer = mhc.MHCV41({"L0": fn})
        params = dict(rms_eps=1e-6, norm_eps=1e-6, pre_eps=1e-6, sinkhorn_eps=1e-6, post_mult=2.0, sinkhorn_iters=20)
        order = (params["rms_eps"], params["pre_eps"], params["sinkhorn_eps"], params["post_mult"], params["norm_eps"], 20)
        for tokens in (1, 5, 128):
            x, res, post, comb, _, scale, base, norm = seam.fixture(tokens, 5120, device="cuda", seed=tokens)
            pre = torch.rand(tokens, 4, device="cuda")
            got = layer("L0", x, res, post, comb, scale, base, norm, pre, *order)
            ref = seam.v41_component_reference(x, res, post, comb, fn, scale, base, norm, pre, **params)
            rows = seam.compare_outputs([g.reshape(r.shape) for g, r in zip(got, ref)], ref)
            self.assertTrue(all(r["passed"] for r in rows), rows)
        g = lambda *s, scale=1.0, dtype=torch.float32: (torch.randn(*s, device="cuda") * scale).to(dtype)
        n = 300
        x, res = g(n, 5120, scale=.1, dtype=torch.bfloat16), g(n, 4, 5120, scale=.1, dtype=torch.bfloat16)
        post, comb, pre = g(n, 4, scale=.5), g(n, 4, 4, scale=.25), torch.rand(n, 4, device="cuda")
        scale, base, norm = torch.tensor([.8, 1.1, .7], device="cuda"), g(24, scale=.2), g(5120, scale=.5, dtype=torch.bfloat16)
        outs = layer.prefill("L0", x, res, post, comb, scale, base, norm, pre, *order)
        for lo, hi in mhc.pieces(n):
            ref = seam.v41_component_reference(x[lo:hi], res[lo:hi], post[lo:hi], comb[lo:hi], fn, scale, base, norm,
                                               pre[lo:hi], **params)
            rows = seam.compare_outputs([o[lo:hi].reshape(r.shape) for o, r in zip(outs, ref)], ref)
            self.assertTrue(all(r["passed"] for r in rows), (lo, rows))

    def test_padded_dense_is_the_lane_over_padded_bytes(self):
        from engine.kernels.dense import DenseLinear, PaddedDenseLinear
        weight = torch.randn(512, 576, device="cuda", dtype=torch.bfloat16)
        glued = PaddedDenseLinear(weight)
        manual = DenseLinear(torch.nn.functional.pad(weight, (0, 64)))
        for rows in (1, 7, 32, 100):
            x = torch.randn(rows, 576, device="cuda", dtype=torch.bfloat16)
            self.assertTrue(torch.equal(glued(x), manual(torch.nn.functional.pad(x, (0, 64)))), rows)


if __name__ == "__main__":
    unittest.main()
