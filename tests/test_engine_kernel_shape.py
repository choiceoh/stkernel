"""The kernel shape descriptor (engine/base/kernel_shape): the GLM profile derives the measured
cell from its checkpoint, binding is once per process, and the kernel wrappers read the bound
shape instead of a model's literals. CPU contracts, no accelerator."""
import ast
import contextlib
import importlib.util
import io
import json
import re
import tempfile
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

# Qwen3.8-Flash-Next's text config as the checkpoint on srv2 states it (/home/choiceoh/models/qwen38-flash-next-nvfp4/config.json,
# read 2026-09-13): the fields the shape derivation reads.
QWEN38_TEXT_CONFIG = {
    "hidden_size": 2560, "hc_count": 4, "num_attention_heads": 24, "num_key_value_heads": 2, "head_dim": 256,
    "linear_num_key_heads": 16, "linear_num_value_heads": 48, "linear_key_head_dim": 128, "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4, "indexer_n_heads": 4, "indexer_kv_heads": 1, "indexer_head_dim": 128, "indexer_compress_ratio": 4,
    "indexer_budget": 2048, "num_experts": 512, "moe_intermediate_size": 640, "num_experts_per_tok": 10,
    "hidden_act": "silu", "shared_expert_intermediate_size": 640,
}


def glm_shape():
    from engine.profiles.glm53 import facts
    return facts.architecture(GLM53_TEXT_CONFIG).kernel_shape()


def qwen_shape():
    from engine.profiles.qwen38 import shapes
    return shapes.kernel_shape(QWEN38_TEXT_CONFIG)


# DeepSeek-V4.1-Flash, read off srv4 on 2026-09-13 (/home/choiceoh/models/DeepSeek-V4.1-Flash/config.json):
# the second real checkpoint on the fleet, outside the engine's scope (D5) -- the wizard's second model.
DSV41_TEXT_CONFIG = {
    "model_type": "deepseek_v41_text", "hidden_size": 5120, "num_hidden_layers": 40, "num_attention_heads": 64,
    "num_key_value_heads": 1, "head_dim": 512, "q_lora_rank": 1280, "qk_rope_head_dim": 64,
    "index_n_heads": 32, "index_head_dim": 128, "index_topk": 512,
    "compress_ratios": [0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0], "candidate_block_size": 8,
    "n_routed_experts": 384, "num_experts_per_tok": 6, "moe_intermediate_size": 2304, "intermediate_size": None,
    "n_shared_experts": 1, "hidden_act": "silu", "swiglu_limit": 10.0, "hc_mult": 4, "hc_sinkhorn_iters": 20,
    "hc_eps": 1e-06, "dspark_block_size": 5, "num_nextn_predict_layers": 3, "sliding_window": 128,
}


def dsv41_shape():
    from engine.profiles.dsv41 import shapes
    return shapes.kernel_shape(DSV41_TEXT_CONFIG)


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
        """Model geometry is unchanged; operator-selected draft width is recorded separately."""
        from engine.profiles.glm53.facts import SPEC_K
        self.assertEqual(SPEC_K, 7)
        self.assertEqual(glm_shape(), replace(MEASURED, drafter=None, spec_k=SPEC_K))
        self.assertEqual(MEASURED.describe().split(" | ")[0], "hidden 4096 hc 4 mhc tp 4")
        self.assertEqual(MEASURED.describe().split(" | ")[1], "mla 16x512 no sink")
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
            Attention(kind="mqa", heads=1, head_dim=128, sink=False)
        with self.assertRaises(ValueError):
            Indexer(heads=1, head_dim=128, pool=4, topk=2050, compress="kpool")
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
        # the operation variants: part of the math, validated, and never defaulted
        with self.assertRaises(ValueError):
            Attention(kind="mla", heads=16, head_dim=512, sink="yes")
        with self.assertRaises(ValueError):
            Indexer(heads=32, head_dim=128, pool=4, topk=2048, compress="lsh")
        with self.assertRaises(ValueError):
            replace(MEASURED, hc_variant="hyper")
        with self.assertRaises(TypeError):
            Attention("mla", 16, 512)
        with self.assertRaises(TypeError):
            Indexer(32, 128, 4, 2048)

    def test_qwen38_declares_a_different_cell_through_the_same_descriptor(self):
        q = qwen_shape()
        self.assertEqual((q.hidden, q.hc, q.tp, q.comm), (2560, 4, 4, Comm(4, 2560)))
        # no sink (a sigmoid output gate instead) and the gated residual form, both read off the pinned HF file
        self.assertEqual(q.attention, Attention("gqa", heads=6, head_dim=256, kv_heads=1, sink=False))
        self.assertEqual(q.linear, LinearAttention(heads=4, v_heads=12, k_dim=128, v_dim=128, conv=4, decay="head"))
        self.assertEqual(q.indexer, Indexer(heads=4, head_dim=128, pool=4, topk=2048, compress="qsa"))   # 4 index heads
        self.assertEqual(q.hc_variant, "gated_residual")
        self.assertEqual(q.moe, MoE(experts=512, experts_local=128, hidden=2560, inter=640, inter_local=640, topk=10,
                                    quant="nvfp4", activation="silu", swiglu_limit=None, dense_inter_local=160))
        # three drafts a step: the one MTP layer chained (facts.SPEC_K, the 09-18 fleet pair) -- the served count,
        # not the config's layer count, and the wizard's derivation takes the same one
        self.assertEqual((q.spec_k, q.drafter), (3, None))
        self.assertNotEqual(q, MEASURED)
        self.assertIn("decay/head", q.describe())
        ks.bind(q)
        self.assertIs(ks.bound(), q)

    def test_dsv41_declares_no_linear_attention_and_its_own_expert_encoding(self):
        from engine.kernels import cells
        s = dsv41_shape()
        self.assertEqual((s.hidden, s.hc, s.tp, s.spec_k, s.linear, s.drafter), (5120, 4, 4, 3, None, None))
        self.assertEqual(s.attention, Attention("mla", heads=16, head_dim=512, kv_heads=1, sink=True, window=128))
        self.assertEqual(s.indexer, Indexer(heads=32, head_dim=128, pool=2, topk=512, compress="ced"))
        self.assertEqual(s.hc_variant, "split_sinkhorn")
        self.assertEqual(s.moe, MoE(experts=384, experts_local=96, hidden=5120, inter=2304, inter_local=2304, topk=6,
                                    quant="mxfp4-a8", activation="silu", swiglu_limit=10.0,
                                    dense_inter_local=576))                     # the shared expert, 2304 / 4
        self.assertEqual(ks.from_dict(json.loads(json.dumps(ks.to_dict(s)))), s)      # None survives the record
        self.assertIn("linear none", s.describe())
        verdicts = {v.lane: v for v in cells.admission(s)}
        status = {lane: v.status for lane, v in verdicts.items()}
        admitted = ("device", "universal")
        unmeasured = ("oneshot", "prefill_collectives")                             # math-free transports, measured at 4096
        refused = ("mla", "indexer", "mhc_decode", "mhc_prefill", "moe", "dense")   # different math, or 576 unaligned columns
        self.assertEqual({k: status[k] for k in admitted}, dict.fromkeys(admitted, cells.ADMITTED))
        self.assertEqual({k: status[k] for k in unmeasured}, dict.fromkeys(unmeasured, cells.UNMEASURED))
        self.assertEqual({k: status[k] for k in refused}, dict.fromkeys(refused, cells.REFUSED))
        self.assertFalse({"kda_recurrent", "kda_ring", "kda_chunk", "draft"} & set(status))
        self.assertIn("sink term", verdicts["mla"].why)                             # 16 x 512 matches, the softmax does not
        self.assertEqual(verdicts["mla"].recipe.kind, "kernel")
        self.assertIn("ced", verdicts["indexer"].why)
        self.assertIn("split_sinkhorn", verdicts["mhc_decode"].why)
        self.assertIn("run_mhc_v41", verdicts["mhc_decode"].recipe.how)            # the glue, its GPU probe pending
        self.assertIn("dsv41_mhc_20260910", verdicts["mhc_decode"].recipe.how)
        self.assertEqual((verdicts["mhc_decode"].recipe.kind, verdicts["mhc_prefill"].recipe.kind), ("wire", "wire"))
        self.assertEqual(verdicts["dense"].recipe.kind, "wire")                     # PaddedDenseLinear: 576 -> 640
        self.assertIn("every 2 rows", verdicts["prefill_collectives"].why)
        self.assertEqual(verdicts["oneshot"].recipe.kind, "measure")
        ks.bind(s)
        self.assertIs(ks.bound(), s)

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
        self.assertTrue(gate(num_experts=512, num_local_experts=128, hidden_size=2560, intermediate_size=640,
                             num_topk=10, quant_mode="nvfp4", activation="silu", swiglu_limit=None))

    def test_q0_and_scatter_gates_take_the_cell(self):
        ns = load_dispatch({"_tp_sf6_q0_eligible", "_glm_tp_scatter_shape", "_glm_tp_scatter_fp32"}, MEASURED.moe)
        q0, shape, fp32 = ns["_tp_sf6_q0_eligible"], ns["_glm_tp_scatter_shape"], ns["_glm_tp_scatter_fp32"]
        qwen = qwen_shape().moe
        self.assertTrue(q0(**self.Q0))                                   # the admitted cell by default
        self.assertTrue(q0(**self.Q0, cell=MEASURED.moe))
        self.assertFalse(q0(**self.Q0, cell=qwen))
        # E at the launch is the weights' expert count -- this rank's 128 of Qwen's 512 (EP), all 288 of GLM's (TP)
        self.assertTrue(q0(**dict(self.Q0, E=128, k=2560, n=640, num_topk=10, activation="silu", swiglu_limit=None), cell=qwen))
        self.assertFalse(q0(**dict(self.Q0, E=512, k=2560, n=640, num_topk=10, activation="silu", swiglu_limit=None), cell=qwen))
        self.assertFalse(q0(**dict(self.Q0, tile_m=64), cell=qwen))      # the Q0 kernel's own tile stays 128
        self.assertTrue(shape(288, 288, 4096, 512, 8))
        self.assertTrue(shape(1, 1, 4096, 3072, 1))                      # the dense/shared MLP through the E=1 lane
        self.assertFalse(shape(1, 1, 4096, 512, 1))
        self.assertTrue(shape(1, 1, 2560, 160, 1, qwen))                    # the shared expert, 640 over four ranks
        self.assertTrue(shape(128, 128, 2560, 640, 10, qwen))
        self.assertFalse(shape(512, 512, 2560, 640, 10, qwen))
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
        with self.assertRaisesRegex(ValueError, "mixes by gated_residual"):
            mhc.geometry(qwen_shape())                                  # Qwen3.8's form is not the segment's
        wide = replace(MEASURED, hidden=2560, comm=Comm(4, 2560), moe=replace(MEASURED.moe, hidden=2560))
        with self.assertRaisesRegex(ValueError, "2560"):
            mhc.geometry(wide)
        ks.bind(wide)
        with self.assertRaisesRegex(ValueError, "hidden 2560"):
            mhc.geometry()

    def test_oneshot_takes_its_row_width_from_the_shape(self):
        from engine.kernels import oneshot
        self.assertEqual((oneshot._cell().world, oneshot._cell().hidden), (4, 4096))
        self.assertEqual((oneshot.MAX_ELEMENTS, oneshot.CONSUMER_MAX_ELEMENTS), (64 * 4096, 16 * 4096))
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
        self.assertLess(fleet.index("kernel_shape.bind_recorded(a.ranks, Path(a.ckpt_meta) / \"config.json\""),
                        fleet.index("comm.prepare_oneshot("))
        local = source.split("def local(a) -> int:", 1)[1].split("def ", 1)[0]
        self.assertLess(local.index("kernel_shape.bind_recorded(a.ranks, Path(a.ckpt_meta) / \"config.json\""),
                        local.index("lane_tables.served()"))
        self.assertIn("kernel_shape.bind_drafter(kernel_shape.Drafter(head_dim=D.head_dim", source)
        for tool in ("preshard.py", "preshard_modelopt.py"):
            self.assertIn("kernel_shape.write_record(", (ROOT / "engine/profiles/glm53" / tool).read_text(), tool)


# A config.json the GLM loader accepts offline: the text config plus the Red Hat NVFP4 encoding facts.load() checks.
GLM53_CONFIG_FILE = dict(GLM53_TEXT_CONFIG, quantization_config={"config_groups": {"group_0": {
    "format": "nvfp4-pack-quantized", "weights": {"group_size": 16}, "input_activations": {"group_size": 16},
    "targets": ["re:.*\\.layers\\.(?:[3-9]|[1-3][0-9]|4[0-4])\\.mlp\\.experts\\..*(gate|up|down)_proj$"]}}})


class CellTests(unittest.TestCase):
    """engine/kernels/cells: the compiled cells the wrappers refuse against, judged before a boot."""

    def test_the_measured_cell_is_admitted_on_every_lane(self):
        from engine.kernels import cells
        verdicts = {v.lane: v for v in cells.admission(MEASURED)}
        self.assertEqual({v.status for v in verdicts.values()}, {cells.ADMITTED}, verdicts)
        self.assertIn("draft", verdicts)
        self.assertIn(" admitted, 0 unmeasured, 0 refused", cells.table(list(verdicts.values())))

    def test_qwen38_gets_its_table_before_any_boot(self):
        from engine.kernels import cells
        status = {v.lane: v.status for v in cells.admission(qwen_shape())}
        refused = ("mla", "indexer", "mhc_decode", "mhc_prefill", "kda_chunk", "dense")   # dense: 160 columns
        admitted = ("device", "universal")
        unmeasured = ("oneshot", "prefill_collectives", "kda_recurrent", "kda_ring", "moe")   # ring: GDN gate in-kernel
        self.assertEqual({k: status[k] for k in refused}, dict.fromkeys(refused, cells.REFUSED))
        self.assertEqual({k: status[k] for k in admitted}, dict.fromkeys(admitted, cells.ADMITTED))
        self.assertEqual({k: status[k] for k in unmeasured}, dict.fromkeys(unmeasured, cells.UNMEASURED))
        self.assertEqual(len(status), 13)
        self.assertNotIn("draft", status)                                  # no drafter declared
        kinds = {v.lane: v.recipe.kind for v in cells.admission(qwen_shape()) if v.recipe}
        self.assertEqual((kinds["mhc_decode"], kinds["mhc_prefill"], kinds["mla"]), ("wire", "wire", "wire"))
        self.assertNotIn("establish", kinds.values())                      # every variant is read off the reference
        pinned = {v.lane: v for v in cells.admission(ks.pin(qwen_shape(), "moe.dynamic_tile_m", 32))}
        self.assertEqual(pinned["moe"].status, cells.UNMEASURED)          # a pin is not a measurement
        mx = {v.lane: v for v in cells.admission(replace(MEASURED, moe=replace(MEASURED.moe, quant="mxfp4")))}
        self.assertEqual(mx["moe"].status, cells.REFUSED)

    def test_the_wrappers_refuse_against_the_cells(self):
        from engine.kernels import cells, mla, oneshot
        from engine.kernels.dense import mhc
        self.assertEqual((mla.MLA_H, mla.MLA_D, mla.MLA_SINK), (cells.MLA_HEADS, cells.MLA_LATENT, cells.MLA_SINK))
        self.assertEqual(mhc.MHC_VARIANT, cells.MHC_VARIANT)
        ks.reset()
        self.addCleanup(ks.reset)
        ks.bind(replace(MEASURED, attention=replace(MEASURED.attention, sink=True)))
        with self.assertRaisesRegex(RuntimeError, "sink"):
            mla._check_cell()
        with self.assertRaisesRegex(ValueError, "split_sinkhorn"):
            mhc.geometry(replace(MEASURED, hc_variant="split_sinkhorn"))
        ks.reset()
        # the TileLang mixes check the variant before any work, in both entries
        tree = ast.parse((ROOT / "engine/kernels/mhc/__init__.py").read_text())
        for name in ("mhc_pre_tilelang", "mhc_post_tilelang"):
            fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
            body = [b for b in fn.body if not (isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant))]
            self.assertEqual(ast.unparse(body[0]), "_check_variant()", name)
        self.assertEqual((mhc.COMPILED_HIDDEN, mhc.COMPILED_HC, mhc.MHC_MAX_TOK, mhc.HCHUNK),
                         (cells.MHC_HIDDEN, cells.MHC_HC, cells.MHC_MAX_TOK, cells.MHC_HCHUNK))
        self.assertEqual((oneshot.COMPILED_WORLD, oneshot.MAX_ELEMENTS, oneshot.CONSUMER_MAX_ELEMENTS),
                         (cells.ONESHOT_WORLD, cells.ONESHOT_MAX_ELEMENTS, cells.ONESHOT_CONSUMER_MAX_ELEMENTS))
        if importlib.util.find_spec("triton") is not None:
            from engine.kernels import kpool, prefill_collectives
            self.assertEqual((kpool.INDEX_HEAD_DIM, kpool.INDEXER_KEY_COMPRESS), (cells.INDEXER_HEAD_DIM, cells.INDEXER_KEY_COMPRESS))
            ks.bind(replace(MEASURED, indexer=replace(MEASURED.indexer, compress="ced")))
            with self.assertRaisesRegex(ValueError, "ced"):
                kpool._indexer_cell()
            ks.reset()
            self.assertEqual(prefill_collectives.BLOCK, cells.PREFILL_BLOCK)
        # the adapters refuse by the glue rules, before any kernel is touched
        from engine.kernels.mla import glue
        with self.assertRaisesRegex(RuntimeError, "no sink term"):
            glue.check(replace(MEASURED.attention, sink=True))
        glue.check(Attention("gqa", heads=6, head_dim=256, kv_heads=1, sink=False))
        ks.bind(qwen_shape())
        glue.check()                                                           # Qwen3.8: no sink, 256 + 256 fit the latent
        ks.reset()
        ks.bind(replace(qwen_shape(), attention=replace(qwen_shape().attention, sink=None)))
        with self.assertRaisesRegex(RuntimeError, "not established"):
            glue.check()
        ks.reset()
        self.assertEqual(mhc.geometry_v41(dsv41_shape()), (5120, 4, 24, 20))
        with self.assertRaisesRegex(ValueError, "mixes by mhc"):
            mhc.geometry_v41(MEASURED)
        with self.assertRaisesRegex(ValueError, "hidden 2560"):
            mhc.geometry_v41(RecipeTests.SHAPES["hc_split2560"])
        self.assertEqual(mhc.pieces(300), [(0, 128), (128, 256), (256, 300)])
        self.assertEqual(mhc.pieces(128), [(0, 128)])
        if importlib.util.find_spec("torch") is not None:
            from engine.kernels import dense
            self.assertEqual(dense.KMAX, cells.DENSE_KMAX)
        source = (ROOT / "engine/kernels/decode_projection.py").read_text()   # #816's candidates read the shape
        self.assertNotIn("2048", source)
        self.assertNotIn("(128, 4096)", source)


class RecordTests(unittest.TestCase):
    """The record the wizard writes when a model is taken in, and what a boot does with it."""

    def setUp(self):
        ks.reset()
        self.addCleanup(ks.reset)

    def test_a_record_binds_while_it_still_describes_the_config(self):
        from engine.kernels import cells
        with tempfile.TemporaryDirectory() as d:
            ranks, cfg = Path(d) / "ranks", Path(d) / "config.json"
            ranks.mkdir()
            cfg.write_text(json.dumps(GLM53_CONFIG_FILE))
            shape = glm_shape()
            path = ks.write_record(ranks, shape, profile="glm53", config_sha256=ks.config_sha256(cfg),
                                   admission=cells.admission(shape))
            self.assertEqual(path, ranks / ks.RECORD)
            record = ks.read_record(ranks)
            self.assertEqual((record["profile"], record["version"]), ("glm53", ks.RECORD_VERSION))
            self.assertEqual(ks.from_dict(record["shape"]), shape)
            self.assertEqual(record["admission"][0]["lane"], "device")
            derived = []
            bound_shape, source = ks.bind_recorded(ranks, cfg, lambda: derived.append(1) or shape)
            self.assertEqual((source, derived, ks.bound()), ("record", [], shape))
            ks.reset()
            cfg.write_text(json.dumps(dict(GLM53_CONFIG_FILE, hidden_size=2560)))     # the checkpoint moved
            with self.assertRaisesRegex(RuntimeError, "rerun"):
                ks.bind_recorded(ranks, cfg, lambda: shape)
            self.assertFalse(ks.is_bound())
            _, source = ks.bind_recorded(Path(d) / "no-ranks", cfg, lambda: shape)   # no record: derived, as before
            self.assertEqual((source, ks.bound()), ("derived", shape))
            (ranks / ks.RECORD).write_text(json.dumps({"version": 0}))
            with self.assertRaisesRegex(RuntimeError, "version"):
                ks.read_record(ranks)

    def test_pins_are_part_of_the_recorded_shape(self):
        pinned = ks.pin(glm_shape(), "moe.dynamic_tile_m", 32)
        self.assertEqual(pinned.moe.dynamic_tile_m, 32)
        self.assertEqual(ks.from_dict(json.loads(json.dumps(ks.to_dict(pinned)))), pinned)
        self.assertEqual(ks.from_dict(json.loads(json.dumps(ks.to_dict(MEASURED)))), MEASURED)   # drafter, capability
        for key in ("moe", "moe.tile", "nope.x", "drafter.head_dim"):          # the derived shape has no drafter
            with self.subTest(key=key), self.assertRaises(ValueError):
                ks.pin(glm_shape(), key, 1)
        with self.assertRaises(ValueError):
            ks.pin(glm_shape(), "moe.dynamic_tile_m", 48)

    def test_the_wizard_judges_a_checkpoint_and_records_it(self):
        with tempfile.TemporaryDirectory() as d:
            ckpt, ranks, qckpt = Path(d) / "ckpt", Path(d) / "ranks", Path(d) / "qwen"
            ckpt.mkdir()
            (ckpt / "config.json").write_text(json.dumps(GLM53_CONFIG_FILE))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = ks.main(["wizard", "--profile", "glm53", "--ckpt", str(ckpt), "--ranks", str(ranks),
                                "--write", "--pin", "moe.dynamic_tile_m=64"])
            self.assertEqual(code, 0)
            text = out.getvalue()
            self.assertIn("recorded ->", text)
            self.assertIn("tile pinned at 64", text)
            self.assertIn("0 refused", text)
            self.assertIn("work: none", text)
            record = ks.read_record(ranks)
            self.assertEqual(ks.from_dict(record["shape"]), ks.pin(glm_shape(), "moe.dynamic_tile_m", 64))
            self.assertEqual(record["config_sha256"], ks.config_sha256(ckpt / "config.json"))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(ks.main(["show", "--ranks", str(ranks)]), 0)
            self.assertIn("glm53 recorded", out.getvalue())
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(ks.main(["show", "--ranks", str(Path(d) / "none")]), 1)
            qckpt.mkdir()
            (qckpt / "config.json").write_text(json.dumps({"text_config": QWEN38_TEXT_CONFIG}))
            result = ks.wizard("qwen38", qckpt)
            self.assertEqual(result["shape"], qwen_shape())
            self.assertIsNone(result["path"])
            self.assertIn("refused", {v.status for v in result["admission"]})
            dckpt = Path(d) / "dsv41"
            dckpt.mkdir()
            (dckpt / "config.json").write_text(json.dumps({"text_config": DSV41_TEXT_CONFIG}))
            result = ks.wizard("dsv41", dckpt, ranks=Path(d) / "dranks", write=True)
            self.assertEqual(result["shape"], dsv41_shape())
            self.assertEqual(ks.from_dict(ks.read_record(Path(d) / "dranks")["shape"]), dsv41_shape())
            self.assertEqual({v.lane for v in result["admission"] if v.status == "refused"},
                             {"mla", "indexer", "mhc_decode", "mhc_prefill", "moe", "dense"})
            with self.assertRaises(ValueError):
                ks.wizard("glm53", ckpt, pins=["moe.dynamic_tile_m"])
            with self.assertRaises(ValueError):
                ks.derive_for("dsv4", ckpt)                                    # not a profile this engine knows


class RecipeTests(unittest.TestCase):
    """Every verdict that asks for work says what to do, where, what judges it and what done is -- in the order to do
    it -- and names files that exist, so the table is a work list an agent can start from."""

    SHAPES = {
        "glm": MEASURED, "qwen38": None, "dsv41": None,
        "mla32": replace(MEASURED, attention=Attention("mla", heads=32, head_dim=512, sink=False)),
        "mla_latent256": replace(MEASURED, attention=Attention("mla", heads=16, head_dim=256, sink=False)),
        "mla_latent1024": replace(MEASURED, attention=Attention("mla", heads=16, head_dim=1024, sink=False)),
        "gqa": replace(MEASURED, attention=Attention("gqa", heads=8, head_dim=128, kv_heads=2, sink=False)),
        "gqa_wide": replace(MEASURED, attention=Attention("gqa", heads=8, head_dim=512, kv_heads=2, sink=False)),
        "dense_wide": replace(MEASURED, moe=replace(MEASURED.moe, dense_inter_local=20500)),
        "no_dense": replace(MEASURED, moe=replace(MEASURED.moe, dense_inter_local=0)),
        "hc_split2560": replace(MEASURED, hidden=2560, comm=Comm(4, 2560), moe=replace(MEASURED.moe, hidden=2560),
                                hc_variant="split_sinkhorn"),
        "sink": replace(MEASURED, attention=replace(MEASURED.attention, sink=True)),
        "sink_unknown": replace(MEASURED, attention=replace(MEASURED.attention, sink=None)),
        "ced": replace(MEASURED, indexer=replace(MEASURED.indexer, compress="ced")),
        "hc_split": replace(MEASURED, hc_variant="split_sinkhorn"),
        "hc_unknown": replace(MEASURED, hc_variant=None),
        "hidden2560": replace(MEASURED, hidden=2560, comm=Comm(4, 2560), moe=replace(MEASURED.moe, hidden=2560)),
        "hidden5120": replace(MEASURED, hidden=5120, comm=Comm(4, 5120), moe=replace(MEASURED.moe, hidden=5120)),
        "hc8": replace(MEASURED, hc=8),
        "hidden4000": replace(MEASURED, hidden=4000, comm=Comm(4, 4000), moe=replace(MEASURED.moe, hidden=4000)),
        "tp2": replace(MEASURED, tp=2, comm=Comm(2, 4096), moe=replace(MEASURED.moe, inter_local=1024)),
        "index64": replace(MEASURED, indexer=Indexer(heads=32, head_dim=64, pool=4, topk=2048, compress="kpool")),
        "draft64": replace(MEASURED, drafter=Drafter(head_dim=64, kv_heads=8, layers=5, window=2048)),
        "kda32": replace(MEASURED, linear=LinearAttention(heads=32, v_heads=32, k_dim=128, v_dim=128, conv=4)),
        # a card no lane has ever been built for: still refused outright
        "device": replace(MEASURED, device=Device(capability=(9, 0), sms=132)),
        # one that is built and self-tested but never measured here (engine/kernels/arch.py):
        # it runs by declaration, so it is `unmeasured` and can never be `admitted`
        "device_check": replace(MEASURED, device=Device(capability=(12, 0), sms=20)),
        "mxfp4": replace(MEASURED, moe=replace(MEASURED.moe, quant="mxfp4")),
    }

    def shapes(self):
        for name, shape in self.SHAPES.items():
            yield name, {"qwen38": qwen_shape, "dsv41": dsv41_shape}.get(name, lambda: shape)()

    def test_a_recipe_rides_on_every_verdict_that_is_not_admitted(self):
        from engine.kernels import cells
        seen = set()
        for name, shape in self.shapes():
            for v in cells.admission(shape):
                with self.subTest(shape=name, lane=v.lane):
                    self.assertEqual(v.recipe is None, v.status == cells.ADMITTED)
                    if v.recipe:
                        seen.add((v.lane, v.status, v.recipe.kind))
        # every refusal and every measurement the table can produce is exercised by the shapes above
        self.assertTrue({("mla", "refused", "kernel"), ("mla", "refused", "wire"), ("mla", "refused", "instance"),
                         ("mhc_decode", "refused", "instance"), ("mhc_decode", "refused", "rewrite"),
                         ("mhc_decode", "unmeasured", "measure"), ("oneshot", "refused", "rewrite"),
                         ("oneshot", "unmeasured", "measure"), ("prefill_collectives", "unmeasured", "measure"),
                         ("dense", "refused", "wire"), ("dense", "refused", "rewrite"), ("dense", "unmeasured", "measure"),
                         ("indexer", "refused", "kernel"), ("draft", "unmeasured", "measure"),
                         ("kda_recurrent", "unmeasured", "measure"),
                         ("kda_ring", "unmeasured", "measure"), ("kda_chunk", "refused", "wire"),
                         ("kda_chunk", "unmeasured", "measure"), ("moe", "refused", "convert"),
                         ("moe", "unmeasured", "measure"), ("device", "refused", "rewrite"),
                         ("device", "unmeasured", "measure"),
                         ("mla", "refused", "establish"), ("mhc_decode", "refused", "establish"),
                         ("mhc_prefill", "refused", "establish"), ("mhc_decode", "refused", "wire"),
                         ("mhc_prefill", "refused", "wire"), ("mhc_prefill", "refused", "instance")} <= seen, seen)

    def test_the_plan_is_cheapest_first_with_refusals_first_at_equal_cost(self):
        from engine.kernels import cells
        self.assertEqual(cells.plan(cells.admission(MEASURED)), [])
        self.assertIn("work: none", cells.work_table(cells.admission(MEASURED)))
        qwen = cells.plan(cells.admission(qwen_shape()))
        self.assertEqual([(v.lane, v.status, v.recipe.kind, v.recipe.cost) for v in qwen], [
            ("dense", "refused", "wire", "hours"), ("kda_chunk", "refused", "wire", "hours"),
            ("mhc_decode", "refused", "wire", "hours"), ("mhc_prefill", "refused", "wire", "hours"),
            ("mla", "refused", "wire", "hours"), ("kda_recurrent", "unmeasured", "measure", "hours"),
            ("kda_ring", "unmeasured", "measure", "hours"), ("moe", "unmeasured", "measure", "hours"),
            ("oneshot", "unmeasured", "measure", "hours"), ("prefill_collectives", "unmeasured", "measure", "hours"),
            ("indexer", "refused", "kernel", "days")])
        self.assertEqual([v.lane for v in cells.plan(cells.admission(dsv41_shape()))],
                         ["dense", "mhc_decode", "mhc_prefill", "oneshot", "prefill_collectives", "indexer", "mla", "moe"])
        text = cells.work_table(cells.admission(qwen_shape()))
        self.assertIn("work (11)", text)
        self.assertIn("1. dense [refused] wire, hours", text)
        for field in ("how:", "where:", "judge:", "done:"):
            self.assertEqual(text.count(field), 11, field)

    def test_the_recipes_name_files_that_exist(self):
        import re
        from engine.kernels import cells
        pattern = re.compile(r"(?:engine|probes|tests|bench|measurements|overlay)/[A-Za-z0-9_./<>-]+")
        checked = set()
        for _, shape in self.shapes():
            for v in cells.admission(shape):
                texts = (v.recipe.where, v.recipe.how, v.recipe.judge, v.recipe.done) if v.recipe else ()
                texts += (v.serve.kernel, v.serve.note) if v.serve else ()
                for text in texts:
                    for token in pattern.findall(text):
                        token = token.rstrip(".,;:)").replace("<profile>", "glm53")
                        with self.subTest(lane=v.lane, token=token):
                            self.assertTrue(self.resolves(token), token)
                        checked.add(token)
        self.assertIn("engine/kernels/dense/mhc_reference.py", checked)
        self.assertIn("measurements/dsv41_mhc_20260910", checked)
        self.assertIn("engine/kernels/dense/mhc.MHCV41.prefill", checked)
        self.assertFalse(self.resolves("engine/kernels/dense/mhc.MHCV42"))      # a named attribute must be in its file
        self.assertFalse(self.resolves("engine/kernels/nowhere.py"))

    @staticmethod
    def resolves(token: str) -> bool:
        """A path that exists, or a module path followed by attributes (`dense/mhc.MHCV41.prefill`) whose file names
        every one of them."""
        if (ROOT / token).exists() or (ROOT / (token + ".py")).exists():
            return True
        head, attributes = token, []
        while "." in head.rpartition("/")[2]:
            head, _, attribute = head.rpartition(".")
            attributes.insert(0, attribute)
            for path in (ROOT / head, ROOT / (head + ".py")):
                if path.is_file():
                    source = path.read_text()
                    return all(re.search(rf"\b{re.escape(name)}\b", source) for name in attributes)
                if path.is_dir():
                    return (path / "__init__.py").is_file() and all(
                        re.search(rf"\b{re.escape(name)}\b", (path / "__init__.py").read_text()) for name in attributes)
        return False

    def test_a_recipe_names_its_kind_cost_and_every_field(self):
        from engine.kernels import cells
        good = dict(kind="measure", where="w", how="h", judge="j", done="d", cost="hours")
        self.assertEqual(cells.Recipe(**good).cost, "hours")
        for change in (dict(kind="guess"), dict(cost="weeks"), dict(where=""), dict(judge=""), dict(done="")):
            with self.subTest(change=change), self.assertRaises(ValueError):
                cells.Recipe(**dict(good, **change))

    def test_the_record_and_the_json_carry_the_work(self):
        from engine.kernels import cells
        with tempfile.TemporaryDirectory() as d:
            qckpt, ranks = Path(d) / "qwen", Path(d) / "ranks"
            qckpt.mkdir()
            (qckpt / "config.json").write_text(json.dumps({"text_config": QWEN38_TEXT_CONFIG}))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(ks.main(["wizard", "--profile", "qwen38", "--ckpt", str(qckpt), "--ranks", str(ranks),
                                          "--write", "--json"]), 0)
            doc = json.loads(out.getvalue())
            expected = [v.lane for v in cells.plan(cells.admission(qwen_shape()))]
            self.assertEqual(doc["plan"], expected)
            self.assertEqual(doc["counts"], {"admitted": 2, "unmeasured": 5, "refused": 6})   # K1: the GDN ring is unmeasured
            self.assertEqual(doc["serving"], {"specialized": 7, "glue": 3, "generic": 2, "none": 0, "glue_judged": 0,
                                              "generic_judged": 1})         # #986: qsa and gated_residual serve three lanes
            self.assertEqual(ks.from_dict(doc["shape"]), qwen_shape())
            self.assertEqual(doc["record"], str(ranks / ks.RECORD))
            by_lane = {entry["lane"]: entry for entry in doc["admission"]}
            self.assertEqual(set(by_lane["mhc_decode"]["recipe"]), {"kind", "where", "how", "judge", "done", "cost"})
            self.assertIsNone(by_lane["device"]["recipe"])
            self.assertIsNone(by_lane["device"]["serve"])
            self.assertEqual((by_lane["mla"]["serve"]["tier"], by_lane["mla"]["serve"]["judged"]), ("generic", False))   # #986: the ported qsa op
            record = ks.read_record(ranks)
            self.assertEqual(record["plan"], expected)
            self.assertEqual([cells.from_dict(v) for v in record["admission"]], cells.admission(qwen_shape()))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(ks.main(["show", "--ranks", str(ranks), "--json"]), 0)
            shown = json.loads(out.getvalue())
            self.assertEqual((shown["plan"], shown["counts"], shown["serving"]), (expected, doc["counts"], doc["serving"]))
            self.assertTrue(shown["recorded_table_current"])
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(ks.main(["show", "--ranks", str(ranks)]), 0)
            self.assertIn("work (11)", out.getvalue())
            self.assertNotIn("older cells", out.getvalue())
            # a record written before recipes and the serving column: `show` judges its shape against today's cells, so
            # the serving counts and lines are there, and it says the recorded table is stale
            old = dict(record, admission=[{k: v for k, v in entry.items() if k not in ("recipe", "serve")}
                                          for entry in record["admission"]])
            old.pop("plan")
            (ranks / ks.RECORD).write_text(json.dumps(old))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(ks.main(["show", "--ranks", str(ranks), "--json"]), 0)
            shown = json.loads(out.getvalue())
            self.assertEqual((shown["serving"], shown["plan"]), (doc["serving"], expected))
            self.assertFalse(shown["recorded_table_current"])
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(ks.main(["show", "--ranks", str(ranks)]), 0)
            self.assertIn("older cells", out.getvalue())
            self.assertIn("serving: 7 specialized, 3 glue", out.getvalue())


class ServeTests(unittest.TestCase):
    """What serves each layer: the lane's own kernel, that kernel through an exact adapter, a shape-generic fast kernel
    for the same math, or nothing fast -- and never an oracle."""

    def verdicts(self, shape):
        from engine.kernels import cells
        return {v.lane: v for v in cells.admission(shape)}

    def test_every_layer_names_what_serves_it_and_an_oracle_never_does(self):
        from engine.kernels import cells
        for name, shape in RecipeTests().shapes():
            for lane, v in self.verdicts(shape).items():
                with self.subTest(shape=name, lane=lane):
                    if lane == "device":
                        self.assertIsNone(v.serve)                      # a precondition, not a layer
                        continue
                    self.assertIsNotNone(v.serve)
                    tier = v.serve.tier
                    if lane == "universal":
                        self.assertEqual(tier, cells.GENERIC)
                    elif v.status == cells.REFUSED:
                        self.assertIn(tier, (cells.SPECIALIZED, cells.GLUE, cells.GENERIC, cells.NONE))
                        if tier == cells.SPECIALIZED:
                            # a kernel written for the shape serves a lane the compiled cell refuses (Qwen3.8's qsa
                            # indexer and gated residual, #986): it lives in engine/kernels and no GPU has judged it yet
                            self.assertTrue(v.serve.kernel.startswith("engine/kernels/"), v.serve.kernel)
                            self.assertFalse(v.serve.judged)
                    else:
                        self.assertIn(tier, (cells.SPECIALIZED, cells.GLUE))
                    if tier == cells.GLUE:
                        self.assertFalse(v.serve.judged)                # no adapter has run on a GPU yet
                        self.assertIn("glue" if lane == "mla" else "engine/kernels", v.serve.kernel)
                    self.assertNotIn("engine/modules", v.serve.kernel)
                    self.assertFalse(v.serve.kernel.startswith("modules/"))

    def test_glm_is_served_by_its_own_lanes(self):
        from engine.kernels import cells
        verdicts = self.verdicts(MEASURED)
        self.assertEqual({lane: (v.serve.tier, v.serve.judged) for lane, v in verdicts.items() if v.serve},
                         {**{lane: (cells.SPECIALIZED, True) for lane in verdicts if lane not in ("device", "universal")},
                          "universal": (cells.GENERIC, True)})

    def test_the_second_models_name_their_fastest_kernels(self):
        from engine.kernels import cells
        S, L, G, N = cells.SPECIALIZED, cells.GLUE, cells.GENERIC, cells.NONE
        dsv41 = {lane: (v.serve.tier, v.serve.judged) for lane, v in self.verdicts(dsv41_shape()).items() if v.serve}
        self.assertEqual(dsv41, {"mla": (G, False), "indexer": (N, False), "mhc_decode": (L, False), "mhc_prefill": (L, False),
                                 "oneshot": (S, True), "prefill_collectives": (S, False), "dense": (L, False),
                                 "moe": (G, False), "universal": (G, True)})
        d = self.verdicts(dsv41_shape())
        self.assertIn("trtllm_batch_decode_sparse_mla_dsv4", d["mla"].serve.kernel)   # the sink-capable decode kernel
        self.assertIn("MHCV41", d["mhc_decode"].serve.kernel)
        self.assertIn("MHCV41.prefill", d["mhc_prefill"].serve.kernel)
        self.assertIn("PaddedDenseLinear", d["dense"].serve.kernel)
        self.assertIn("576 -> 640", d["dense"].serve.note)
        self.assertIn("MXFP4", d["moe"].serve.kernel)
        self.assertIn("torch", d["indexer"].serve.note)                               # the CED compressor has no fast kernel
        self.assertEqual(cells.serving(list(d.values())), {"specialized": 2, "glue": 3, "generic": 3, "none": 1,
                                                            "glue_judged": 0, "generic_judged": 1})
        qwen = {lane: (v.serve.tier, v.serve.judged) for lane, v in self.verdicts(qwen_shape()).items() if v.serve}
        self.assertEqual(qwen, {"mla": (G, False), "indexer": (S, False), "mhc_decode": (S, False), "mhc_prefill": (S, False),
                                "oneshot": (S, True), "prefill_collectives": (S, False), "dense": (L, False),
                                "kda_recurrent": (L, False), "kda_ring": (S, False), "kda_chunk": (L, False),
                                "moe": (S, False), "universal": (G, True)})
        q = self.verdicts(qwen_shape())
        self.assertIn("qsa_sparse_paged_attention in engine/kernels/qsa.py", q["mla"].serve.kernel)   # the ported BF16-KV kernel
        self.assertIn("glue.gqa", q["mla"].serve.note)                                    # 256 + 256 fill the latent: the e4m3 alternative
        self.assertIn("engine/kernels/qsa", q["indexer"].serve.kernel)
        self.assertIn("engine/kernels/gated_residual", q["mhc_decode"].serve.kernel)       # five launches a site
        self.assertIn("gated_residual", q["mhc_decode"].serve.note)
        self.assertIn("PaddedDenseLinear", q["dense"].serve.kernel)                       # the shared expert's 160 columns
        self.assertIn("recurrent_gdn_ring", q["kda_ring"].serve.kernel)                    # GDN's gate in the ring kernel
        self.assertIn("chunk_kda_with_decay", q["kda_chunk"].serve.kernel)
        self.assertEqual(cells.serving(list(q.values())), {"specialized": 7, "glue": 3, "generic": 2, "none": 0,
                                                            "glue_judged": 0, "generic_judged": 1})
        unread = self.verdicts(replace(qwen_shape(), attention=replace(qwen_shape().attention, sink=None)))
        self.assertEqual(unread["mla"].serve.tier, N)                                     # an unread sink serves nothing

    def test_the_tier_follows_the_refusal(self):
        from engine.kernels import cells
        L, G, N = cells.GLUE, cells.GENERIC, cells.NONE
        attention = lambda *args, **kw: replace(MEASURED, attention=Attention(*args, **kw))
        qsa = Indexer(heads=1, head_dim=128, pool=4, topk=2048, compress="qsa")
        cases = {
            "grouped heads": (attention("mla", heads=32, head_dim=512, sink=False), "mla", L, "glue.grouped", "world 1"),
            "odd heads": (attention("mla", heads=20, head_dim=512, sink=False), "mla", L, "glue.grouped", "zero-query"),
            "narrow latent": (attention("mla", heads=16, head_dim=256, sink=False), "mla", L, "zero-extended", "512/256"),
            "wide latent": (attention("mla", heads=16, head_dim=1024, sink=False), "mla", G,
                            "BatchDecodeMlaWithPagedKVCacheWrapper", ""),
            "unknown sink": (attention("mla", heads=16, head_dim=512, sink=None), "mla", N, "", "establish"),
            "gqa": (attention("gqa", heads=8, head_dim=128, kv_heads=2, sink=False), "mla", L, "glue.gqa", "e4m3"),
            # an unestablished sink serves nothing, whatever would fit: the adapter refuses it (cells.mla_glue_refusal)
            "gqa unknown sink": (attention("gqa", heads=8, head_dim=128, kv_heads=2, sink=None), "mla", N, "",
                                 "with no sink, engine/kernels/mla/glue.gqa"),
            "wide gqa unknown, qsa": (replace(attention("gqa", heads=8, head_dim=512, kv_heads=2, sink=None), indexer=qsa),
                                      "mla", N, "", "with no sink, qsa_sparse_paged_attention"),
            "wide gqa": (attention("gqa", heads=8, head_dim=512, kv_heads=2, sink=False), "mla", G,
                         "BatchDecodeWithPagedKVCacheWrapper", ""),
            "wide gqa, qsa": (replace(attention("gqa", heads=8, head_dim=512, kv_heads=2, sink=False), indexer=qsa),
                              "mla", G, "qsa_sparse_paged_attention", "gqa_sparse"),
            # a sink is math the sink-free kernels cannot compute, whatever else would fit (Codex on #847)
            "gqa sink, qsa": (replace(attention("gqa", heads=8, head_dim=128, kv_heads=2, sink=True), indexer=qsa),
                              "mla", N, "", "sinks"),
            "sink 16x512": (attention("mla", heads=16, head_dim=512, sink=True), "mla", G,
                            "trtllm_batch_decode_sparse_mla_dsv4", ""),
            "sink 16x256": (attention("mla", heads=16, head_dim=256, sink=True), "mla", N, "", "head size 512"),
            "sink 192 heads": (attention("mla", heads=192, head_dim=512, sink=True), "mla", N, "", "at most 128"),
            "tp2": (replace(MEASURED, tp=2, comm=Comm(2, 4096), moe=replace(MEASURED.moe, inter_local=1024)), "oneshot",
                    G, "NCCL", ""),
            "hidden4000": (replace(MEASURED, hidden=4000, comm=Comm(4, 4000), moe=replace(MEASURED.moe, hidden=4000)),
                           "dense", L, "PaddedDenseLinear", "4000 -> 4096"),
            "dense past KMAX": (RecipeTests.SHAPES["dense_wide"], "dense", G, "cuBLAS", ""),
            "hidden2560": (replace(MEASURED, hidden=2560, comm=Comm(4, 2560), moe=replace(MEASURED.moe, hidden=2560)),
                           "mhc_decode", G, "TileLang", ""),
            "split-sinkhorn at 2560": (RecipeTests.SHAPES["hc_split2560"], "mhc_decode", N, "", "compiled for hidden"),
            "index64": (replace(MEASURED, indexer=Indexer(heads=32, head_dim=64, pool=4, topk=2048, compress="kpool")),
                        "indexer", N, "", ""),
        }
        for name, (shape, lane, tier, kernel, note) in cases.items():
            with self.subTest(case=name):
                serve = self.verdicts(shape)[lane].serve
                self.assertEqual(serve.tier, tier)
                self.assertIn(kernel, serve.kernel)
                self.assertIn(note, serve.note)
        judged = {name: self.verdicts(shape)[lane].serve.judged for name, (shape, lane, *_) in cases.items()}
        self.assertEqual({n for n, j in judged.items() if j}, {"tp2", "dense past KMAX", "hidden2560"})

    def test_a_model_without_a_dense_mlp_is_judged_on_its_projections(self):
        from engine.kernels import cells
        # zero is the descriptor's "no dense or shared MLP" (engine/profiles/qwen38/shapes.py): the lane still packs the
        # projections at K = hidden, so that width alone is judged -- not a nonexistent MLP width (Codex on #847)
        dense = self.verdicts(RecipeTests.SHAPES["no_dense"])["dense"]
        self.assertEqual((dense.status, dense.serve.tier), (cells.ADMITTED, cells.SPECIALIZED))
        self.assertNotIn("dense intermediate", dense.why)
        unaligned = self.verdicts(replace(RecipeTests.SHAPES["no_dense"], hidden=4000, comm=Comm(4, 4000),
                                          moe=replace(RecipeTests.SHAPES["no_dense"].moe, hidden=4000)))["dense"]
        self.assertEqual((unaligned.status, unaligned.serve.tier, unaligned.recipe.kind), (cells.REFUSED, cells.GLUE, "wire"))
        self.assertIn("projections alone", unaligned.why)
        self.assertIn("4000 -> 4096", unaligned.serve.note)
        self.assertIn("dense intermediate 3072", self.verdicts(RecipeTests.SHAPES["hidden4000"])["dense"].why)

    def test_the_v41_seam_gets_its_own_instance_recipe(self):
        """A split-sinkhorn model at a width the seam is not compiled for asks for a V4.1 instance, judged by the V4.1
        reference -- not for the MK segment's mhc instance."""
        from engine.kernels import cells
        for lane in ("mhc_decode", "mhc_prefill"):
            with self.subTest(lane=lane):
                v = self.verdicts(RecipeTests.SHAPES["hc_split2560"])[lane]
                self.assertEqual((v.status, v.serve.tier, v.recipe.kind, v.recipe.cost), (cells.REFUSED, cells.NONE, "instance", "hours"))
                self.assertIn("mk_mhc_v41_launch", v.recipe.where)
                self.assertIn("v41_component_reference", v.recipe.judge)
                self.assertIn("MHCV41", v.recipe.done)
                self.assertNotIn("mhc_pre/mhc_post", v.recipe.judge)
        hc8 = self.verdicts(replace(MEASURED, hc=8, hc_variant="split_sinkhorn"))["mhc_decode"]
        self.assertEqual((hc8.recipe.kind, hc8.recipe.cost), ("rewrite", "days"))
        self.assertIn("V4.1 seam included", hc8.recipe.how)
        mhc = self.verdicts(RecipeTests.SHAPES["hidden2560"])["mhc_decode"]        # the mhc form keeps its own names
        self.assertIn("mk_mhc_launch", mhc.recipe.where)
        self.assertIn("mhc_pre/mhc_post", mhc.recipe.judge)

    def test_the_glue_rules_are_the_adapters_rules(self):
        from engine.kernels import cells
        self.assertIsNone(cells.mla_glue_refusal(Attention("mla", heads=20, head_dim=256, sink=False)))
        self.assertIsNone(cells.mla_glue_refusal(Attention("gqa", heads=6, head_dim=256, kv_heads=1, sink=False)))
        for a, why in ((Attention("gqa", heads=6, head_dim=512, kv_heads=1, sink=False), "side by side"),
                       (Attention("mla", heads=16, head_dim=1024, sink=False), "does not fit"),
                       (Attention("mla", heads=16, head_dim=512, sink=True), "no sink term"),
                       (Attention("gqa", heads=6, head_dim=256, kv_heads=1, sink=None), "not established")):
            with self.subTest(attention=a):
                self.assertIn(why, cells.mla_glue_refusal(a))
        self.assertIsNone(cells.mhc_v41_refusal(dsv41_shape()))
        self.assertIsNone(cells.mhc_v41_refusal(replace(MEASURED, hc_variant="split_sinkhorn")))   # 4096 is compiled too
        self.assertIn("mixes by mhc", cells.mhc_v41_refusal(MEASURED))
        self.assertIn("hidden 2560", cells.mhc_v41_refusal(RecipeTests.SHAPES["hc_split2560"]))
        self.assertIsNone(cells.dense_glue_refusal(576))
        self.assertIsNone(cells.dense_glue_refusal(20480))
        self.assertIn("20608", cells.dense_glue_refusal(20500))
        self.assertIsNotNone(cells.dense_glue_refusal(0))

    def test_a_serve_is_validated(self):
        from engine.kernels import cells
        cells.Serve(cells.GENERIC, "flashinfer", False, "")
        cells.Serve(cells.GLUE, "engine/kernels/mla/glue.gqa", False, "")
        for bad in ((cells.NONE, "x", False, ""), (cells.NONE, "", True, ""), ("best", "x", False, ""),
                    (cells.GENERIC, "", False, ""), (cells.GENERIC, "engine/modules/moe.py", True, ""),
                    (cells.GLUE, "", False, ""), (cells.GLUE, "modules/sparse_attention.gqa_sparse", False, ""),
                    (cells.SPECIALIZED, "x", 1, "")):
            with self.subTest(serve=bad), self.assertRaises(ValueError):
                cells.Serve(*bad)
        self.assertEqual(cells.TIERS, (cells.SPECIALIZED, cells.GLUE, cells.GENERIC, cells.NONE))   # fastest first
        text = cells.table(cells.admission(dsv41_shape()))
        self.assertIn("serving: 2 specialized, 3 glue (0 judged), 3 generic (1 judged), 1 with nothing fast", text)
        self.assertIn("nothing fast -- the CED compressor", text)
        self.assertIn("glue, unjudged: engine/kernels/dense/mhc.MHCV41", text)


if __name__ == "__main__":
    unittest.main()
