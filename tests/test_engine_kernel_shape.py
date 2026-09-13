"""The kernel shape descriptor (engine/base/kernel_shape): the GLM profile derives the measured
cell from its checkpoint, binding is once per process, and the kernel wrappers read the bound
shape instead of a model's literals. CPU contracts, no accelerator."""
import ast
import importlib.util
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from engine.base import kernel_shape as ks
from engine.base.kernel_shape import (MEASURED, Attention, Comm, Device, Drafter, Indexer, KernelShape,
                                      LinearAttention, MoE)

ROOT = Path(__file__).resolve().parents[1]
DISPATCH = ROOT / "engine/kernels/b12x/moe_dispatch.py"
torch = None
if importlib.util.find_spec("torch") is not None:
    import torch

# The served checkpoint's text config, read off srv2 on 2026-09-13
# (/home/choiceoh/models/st-glm53-nvidia-tp4-9391/config.json): every key facts.architecture() checks.
GLM53_TEXT_CONFIG = {
    "model_type": "glm5_next_text", "hidden_size": 4096, "num_hidden_layers": 45, "first_k_dense_replace": 3,
    "vocab_size": 154880, "rms_norm_eps": 1e-05, "linear_lower_bound": -5.0,
    "num_attention_heads": 64, "num_key_value_heads": 64, "head_dim": 0, "qk_nope_head_dim": 256,
    "qk_rope_head_dim": 0, "v_head_dim": 256, "q_lora_rank": 1536, "kv_lora_rank": 512, "mla_use_nope": True,
    "rope_parameters": None, "index_n_heads": 32, "index_head_dim": 128, "index_topk": 2048, "index_kpool": 4,
    "index_kpool_compress": True, "index_kpool_always_select_tail": True, "indexer_rope_interleave": True,
    "max_position_embeddings": 1048576, "n_routed_experts": 288, "num_experts_per_tok": 8,
    "moe_intermediate_size": 2048, "intermediate_size": 12288, "routed_scaling_factor": 2.5, "swiglu_limit": 10.0,
    "hc_mult": 4, "hc_eps": 1e-06, "hc_sinkhorn_iters": 20, "mhc": True, "topk_method": "noaux_tc",
    "scoring_func": "sigmoid", "norm_topk_prob": True, "n_group": 1, "topk_group": 1, "moe_router_dtype": "float32",
    "n_shared_experts": 1, "hidden_act": "silu", "tie_word_embeddings": False,
    "linear_attn_config": {
        "full_attn_layers": [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43], "gate_lower_bound": -5.0, "head_dim": 128,
        "kda_layers": [i for i in range(45) if i % 4 != 3], "num_heads": 64, "short_conv_kernel_size": 4},
}

# Qwen3.8-Flash-Next as the profile documents it (plan.py, shapes.py, the b12x tile comment): the
# checkpoint is not on the fleet yet (2026-09-13), so these are the documented widths, not a config read.
QWEN38_TEXT_CONFIG = {
    "hidden_size": 2560, "hc_count": 4, "num_attention_heads": 24, "num_key_value_heads": 2, "head_dim": 256,
    "linear_num_key_heads": 16, "linear_num_value_heads": 48, "linear_key_head_dim": 128, "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4, "indexer_kv_heads": 1, "indexer_head_dim": 128, "indexer_compress_ratio": 4,
    "indexer_budget": 2048, "num_experts": 512, "moe_intermediate_size": 512, "num_experts_per_tok": 10,
    "hidden_act": "silu", "shared_expert_intermediate_size": 2048,
}


def glm_shape():
    from engine.profiles.glm53 import facts
    return facts.architecture(GLM53_TEXT_CONFIG).kernel_shape()


def qwen_shape():
    from engine.profiles.qwen38 import shapes
    return shapes.kernel_shape(QWEN38_TEXT_CONFIG)


def load_dispatch(names, cell):
    """The dispatcher's gate functions from their source, with the admitted cell injected."""
    tree = ast.parse(DISPATCH.read_text())
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in selected} == set(names), "dispatch gate renamed"
    ns = {"_admitted_moe": lambda: cell}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(DISPATCH), "exec"), ns)
    return ns


class DescriptorTests(unittest.TestCase):
    def setUp(self):
        ks.reset()
        self.addCleanup(ks.reset)

    def test_the_glm_checkpoint_derives_the_measured_cell(self):
        """The proof that the served path did not move: GLM's derivation IS the cell the kernels
        were compiled for, field by field (the drafter is bound later, when it loads)."""
        self.assertEqual(glm_shape(), replace(MEASURED, drafter=None))
        self.assertEqual(MEASURED.describe().split(" | ")[0], "hidden 4096 hc 4 tp 4")
        self.assertEqual(MEASURED.device, Device(capability=(12, 1), sms=48))

    def test_unbound_means_the_measured_cell(self):
        self.assertFalse(ks.is_bound())
        self.assertIs(ks.bound(), MEASURED)
        self.assertEqual(ks.drafter(), MEASURED.drafter)

    def test_binding_is_once_per_process(self):
        glm = glm_shape()
        self.assertIs(ks.bind(glm), glm)
        self.assertTrue(ks.is_bound())
        ks.bind(replace(glm))                                   # equal again: a no-op
        with self.assertRaisesRegex(RuntimeError, "already bound"):
            ks.bind(qwen_shape())
        with self.assertRaises(TypeError):
            ks.bind("glm")
        d = Drafter(head_dim=128, kv_heads=8, layers=5, window=2048)
        ks.bind_drafter(d)
        ks.bind_drafter(Drafter(128, 8, 5, 2048))
        with self.assertRaisesRegex(RuntimeError, "drafter"):
            ks.bind_drafter(Drafter(64, 8, 5, 2048))
        self.assertEqual(ks.drafter(), d)
        ks.reset()
        self.assertFalse(ks.is_bound())

    def test_impossible_shapes_are_refused_at_construction(self):
        with self.assertRaises(ValueError):
            Comm(world=4, hidden=0)
        with self.assertRaises(ValueError):
            LinearAttention(heads=2, v_heads=3, k_dim=128, v_dim=128, conv=4)
        with self.assertRaises(ValueError):
            LinearAttention(heads=2, v_heads=2, k_dim=96, v_dim=128, conv=4)
        with self.assertRaises(ValueError):
            LinearAttention(heads=2, v_heads=2, k_dim=128, v_dim=128, conv=4, decay="token")
        with self.assertRaises(ValueError):
            Attention(kind="mqa", heads=1, head_dim=128)
        with self.assertRaises(ValueError):
            Indexer(heads=1, head_dim=128, pool=4, topk=2050)
        with self.assertRaises(ValueError):
            replace(MEASURED.moe, dynamic_tile_m=48)
        with self.assertRaises(ValueError):
            replace(MEASURED, comm=Comm(world=4, hidden=2560))
        with self.assertRaises(ValueError):
            replace(MEASURED, tp=2)
        with self.assertRaises(ValueError):
            replace(MEASURED, moe=replace(MEASURED.moe, inter_local=1024))
        with self.assertRaises(ValueError):
            Device(capability=(12,), sms=48)

    def test_qwen38_declares_a_different_cell_through_the_same_descriptor(self):
        q = qwen_shape()
        self.assertEqual((q.hidden, q.hc, q.tp, q.comm), (2560, 4, 4, Comm(4, 2560)))
        self.assertEqual(q.attention, Attention("gqa", heads=6, head_dim=256, kv_heads=1))
        self.assertEqual(q.linear, LinearAttention(heads=4, v_heads=12, k_dim=128, v_dim=128, conv=4, decay="head"))
        self.assertEqual(q.indexer, Indexer(heads=1, head_dim=128, pool=4, topk=2048))
        self.assertEqual(q.moe, MoE(experts=512, experts_local=128, hidden=2560, inter=512, inter_local=512, topk=10,
                                    quant="nvfp4", activation="silu", swiglu_limit=None, dense_inter_local=512))
        self.assertEqual((q.spec_k, q.drafter), (1, None))
        self.assertNotEqual(q, MEASURED)
        self.assertIn("decay/head", q.describe())
        ks.bind(q)
        self.assertIs(ks.bound(), q)

    def test_the_tiny_test_facts_still_derive_a_valid_shape(self):
        from tests.test_engine_glm53 import tiny_facts
        shape = tiny_facts().kernel_shape()
        self.assertEqual((shape.hidden, shape.linear.k_dim, shape.moe.inter_local), (128, 8, 16))


class DispatchGateTests(unittest.TestCase):
    """The b12x dispatcher's exact-shape gates, from source, with the cell injected."""
    GLM = dict(num_experts=288, num_local_experts=288, hidden_size=4096, intermediate_size=512, num_topk=8,
               quant_mode="nvfp4", activation="swigluoai_uninterleave", swiglu_limit=10.0)
    Q0 = dict(enabled=True, E=288, m=4096, k=4096, n=512, num_topk=8, tile_m=128, quant_mode="nvfp4", tiled=True,
              reform_sf_pack=True, activation="swigluoai_uninterleave", swiglu_alpha=1., swiglu_beta=0.,
              swiglu_limit=10., share_input_across_experts=False)

    def test_the_admission_gate_is_the_bound_cell(self):
        gate = load_dispatch({"_is_admitted_tp_geometry"}, MEASURED.moe)["_is_admitted_tp_geometry"]
        self.assertTrue(gate(**self.GLM))
        self.assertTrue(gate(**dict(self.GLM, intermediate_size=2048)))     # the model's spelling of the width
        for change in (dict(num_experts=287), dict(num_local_experts=72), dict(hidden_size=2560),
                       dict(intermediate_size=1024), dict(num_topk=10), dict(quant_mode="mxfp4"),
                       dict(activation="silu"), dict(swiglu_limit=None)):
            with self.subTest(change=change):
                self.assertFalse(gate(**dict(self.GLM, **change)))
        gate = load_dispatch({"_is_admitted_tp_geometry"}, qwen_shape().moe)["_is_admitted_tp_geometry"]
        self.assertFalse(gate(**self.GLM))
        self.assertTrue(gate(num_experts=512, num_local_experts=128, hidden_size=2560, intermediate_size=512,
                             num_topk=10, quant_mode="nvfp4", activation="silu", swiglu_limit=None))

    def test_q0_and_scatter_gates_take_the_cell(self):
        ns = load_dispatch({"_tp_sf6_q0_eligible", "_glm_tp_scatter_shape", "_glm_tp_scatter_fp32"}, MEASURED.moe)
        q0, shape, fp32 = ns["_tp_sf6_q0_eligible"], ns["_glm_tp_scatter_shape"], ns["_glm_tp_scatter_fp32"]
        qwen = qwen_shape().moe
        self.assertTrue(q0(**self.Q0))                                   # the admitted cell by default
        self.assertTrue(q0(**self.Q0, cell=MEASURED.moe))
        self.assertFalse(q0(**self.Q0, cell=qwen))
        # E at the launch is the weights' expert count -- this rank's 128 of Qwen's 512 (EP), all 288 of GLM's (TP)
        self.assertTrue(q0(**dict(self.Q0, E=128, k=2560, num_topk=10, activation="silu", swiglu_limit=None), cell=qwen))
        self.assertFalse(q0(**dict(self.Q0, E=512, k=2560, num_topk=10, activation="silu", swiglu_limit=None), cell=qwen))
        self.assertFalse(q0(**dict(self.Q0, tile_m=64), cell=qwen))      # the Q0 kernel's own tile stays 128
        self.assertTrue(shape(288, 288, 4096, 512, 8))
        self.assertTrue(shape(1, 1, 4096, 3072, 1))                      # the dense/shared MLP through the E=1 lane
        self.assertFalse(shape(1, 1, 4096, 512, 1))
        self.assertTrue(shape(1, 1, 2560, 512, 1, qwen))
        self.assertTrue(shape(128, 128, 2560, 512, 10, qwen))
        self.assertFalse(shape(512, 512, 2560, 512, 10, qwen))
        self.assertFalse(shape(288, 288, 4096, 512, 8, qwen))
        args = dict(state_E=288, weight_E=288, k=4096, n=512, num_topk=8, quant_mode="nvfp4",
                    activation="swigluoai_uninterleave", swiglu_alpha=1., swiglu_beta=0., swiglu_limit=10.)
        self.assertTrue(fp32(**args))
        self.assertFalse(fp32(**dict(args, swiglu_limit=9.)))
        self.assertFalse(fp32(**args, cell=qwen))


@unittest.skipUnless(torch is not None, "requires PyTorch")
class WrapperTests(unittest.TestCase):
    """The wrappers that used to spell GLM's numbers read the bound shape."""

    def setUp(self):
        ks.reset()
        self.addCleanup(ks.reset)

    def test_mk_mhc_geometry_follows_the_shape_and_refuses_uncompiled_widths(self):
        from engine.kernels.dense import mhc
        self.assertEqual(mhc.geometry(MEASURED), (4096, 4, 24, 16))
        self.assertEqual(mhc.geometry(), (4096, 4, 24, 16))
        self.assertEqual(mhc.workspace_sizes(4096, 4, 24, 16),
                         [(16 * 128 * 24, torch.float32), (16 * 128, torch.float32), (16 * 128, torch.float32),
                          (128 * 4, torch.float32), (128 * 4096, torch.bfloat16), (8, torch.int32)])
        v41 = replace(MEASURED, hidden=5120, comm=Comm(4, 5120), moe=replace(MEASURED.moe, hidden=5120))
        self.assertEqual(mhc.geometry(v41), (5120, 4, 24, 20))
        with self.assertRaisesRegex(ValueError, "2560"):
            mhc.geometry(qwen_shape())
        ks.bind(qwen_shape())
        with self.assertRaisesRegex(ValueError, "hidden 2560"):
            mhc.geometry()

    def test_oneshot_takes_its_row_width_from_the_shape(self):
        from engine.kernels import oneshot
        self.assertEqual((oneshot._cell().world, oneshot._cell().hidden), (4, 4096))
        self.assertEqual((oneshot.MAX_ELEMENTS, oneshot.CONSUMER_MAX_ELEMENTS), (64 * 4096, 8 * 4096))
        ks.bind(qwen_shape())
        self.assertEqual(oneshot._cell().hidden, 2560)
        ks.reset()
        ks.bind(replace(MEASURED, tp=2, comm=Comm(2, 4096), moe=replace(MEASURED.moe, inter_local=1024)))
        with self.assertRaisesRegex(ValueError, "compiled for 4 ranks"):
            oneshot._cell()

    @unittest.skipUnless(importlib.util.find_spec("triton") is not None, "the packet kernels import Triton")
    def test_prefill_collectives_take_their_width_and_world_from_the_shape(self):
        from engine.kernels.prefill_collectives import PrefillCollectives
        comm = SimpleNamespace(world_size=4, group=None)
        tensor = lambda rows, width: SimpleNamespace(ndim=2, shape=(rows, width), dtype=torch.bfloat16,
                                                     is_cuda=True, is_contiguous=lambda: True)
        with patch("torch.cuda.is_current_stream_capturing", return_value=False):
            pc = PrefillCollectives(comm)
            self.assertEqual((pc.world, pc.hidden), (4, 4096))
            pc.check(tensor(64, 4096))
            with self.assertRaisesRegex(ValueError, "4096"):
                pc.check(tensor(64, 2560))
            with self.assertRaisesRegex(ValueError, "TP4"):
                PrefillCollectives(SimpleNamespace(world_size=2))
            ks.bind(qwen_shape())
            pc = PrefillCollectives(comm)
            self.assertEqual(pc.hidden, 2560)
            pc.check(tensor(64, 2560))
            with self.assertRaisesRegex(ValueError, "2560"):
                pc.check(tensor(64, 4096))
            with self.assertRaisesRegex(ValueError, "blocks"):
                pc.check(tensor(33, 2560))

    def test_a_per_head_decay_reaches_the_kernels_as_a_stride_zero_channel_axis(self):
        from engine.kernels.linear_decay import is_per_channel, per_channel
        g = torch.zeros(1, 3, 2)
        wide = per_channel(g, 8)
        self.assertEqual(tuple(wide.shape), (1, 3, 2, 8))
        self.assertEqual(wide.stride(-1), 0)
        self.assertFalse(wide.is_contiguous())
        self.assertTrue(is_per_channel(wide, 8))
        self.assertFalse(is_per_channel(g, 8))
        with self.assertRaises(ValueError):
            per_channel(wide, 8)

    def test_the_boot_binds_the_checkpoint_shape_before_the_lanes(self):
        source = (ROOT / "engine/profiles/glm53/boot.py").read_text()
        fleet = source.split("def fleet(a) -> int:", 1)[1]
        self.assertLess(fleet.index("kernel_shape.bind(facts.load(a.ckpt_meta).kernel_shape())"),
                        fleet.index("comm.prepare_oneshot()"))
        local = source.split("def local(a) -> int:", 1)[1].split("def ", 1)[0]
        self.assertLess(local.index("kernel_shape.bind(facts.load(a.ckpt_meta).kernel_shape())"),
                        local.index("lane_tables.served()"))
        self.assertIn("kernel_shape.bind_drafter(kernel_shape.Drafter(head_dim=D.head_dim", source)


if __name__ == "__main__":
    unittest.main()
