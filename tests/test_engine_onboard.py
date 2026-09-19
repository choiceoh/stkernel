"""The front door (engine/base/onboard, tools/onboard.py): a checkpoint no profile was written for is READ, and what
its config does not settle comes back as a blank rather than a guess.

CHARTER D5 names no model, so attaching one has to be a path and not an edit: a profile package declares itself and
the wizard discovers it (`kernel_shape.profiles`/`claims`), and a config nobody claims still gets a shape when it
settles one -- with the lane table and the work list that follow. These cases pin three things:

  * what the door READS, including the axes a config NAMES (`index_kpool_compress` -> kpool, a `kda_layers` key ->
    the per-channel decay, `mhc: true` -> the mixer);
  * what it refuses to guess, and that an operator may STATE such a field but never overrule a key;
  * that stating exactly those fields makes the generic derivation EQUAL to each profile's own, field for field --
    so the list of what a profile knows beyond its config is short, written down, and true of the three checkpoints
    this repo serves today.

CPU only, no checkpoint.
"""
import ast
import importlib.util
import json
import unittest
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DSV41_CONFIG = ROOT / "measurements/dsv41_mhc_20260910/reference-config.json"

# a routed transformer the ordinary way: no hyper-connection, no sparse indexer, no linear attention
PLAIN_MOE = {"model_type": "some_moe_text", "hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32,
             "num_key_value_heads": 8, "head_dim": 128, "num_experts": 128, "num_experts_per_tok": 8,
             "moe_intermediate_size": 768, "intermediate_size": 12288, "hidden_act": "silu"}


def pinned(name: str, **seed) -> dict:
    """A *_CONFIG fixture out of tests/test_engine_kernel_shape.py without importing torch. `seed` supplies the
    names a fixture is built from (GLM53_CONFIG_FILE is GLM53_TEXT_CONFIG plus its encoding)."""
    tree = ast.parse((ROOT / "tests/test_engine_kernel_shape.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == name:
            ns = dict(seed)
            exec(compile(ast.Module(body=[node], type_ignores=[]), "fixture", "exec"), ns)
            return ns[name]
    raise AssertionError(f"{name} is not in tests/test_engine_kernel_shape.py")


def glm53_config() -> dict:
    """GLM-5.3's config.json as the loader reads it: the text config plus the NVFP4 encoding (facts.load checks it)."""
    return pinned("GLM53_CONFIG_FILE", GLM53_TEXT_CONFIG=pinned("GLM53_TEXT_CONFIG"))


def qwen38_config() -> dict:
    """Qwen3.8's text config plus the quant method its profile refuses to load without (facts.load: "this profile
    serves NVIDIA's ModelOpt NVFP4 checkpoint")."""
    return dict(pinned("QWEN38_TEXT_CONFIG"), quantization_config={"quant_method": "modelopt"})


def dsv41_config() -> dict:
    """DSv4.1's pinned reference config, flattened the way tools/onboard.py flattens a checkpoint's."""
    cfg = json.loads(DSV41_CONFIG.read_text())
    return dict(cfg["text_config"], quantization_config=cfg["quantization_config"])


def cli():
    spec = importlib.util.spec_from_file_location("onboard_cli", ROOT / "tools/onboard.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReadingTests(unittest.TestCase):
    def read(self, cfg, **kw):
        from engine.base.onboard import read_config
        return read_config(cfg, **kw)

    def test_a_config_that_settles_everything_gets_a_shape(self):
        r = self.read(PLAIN_MOE, placement="ep")
        self.assertEqual((r.blanks, r.complete), ((), True))
        s = r.shape
        self.assertEqual((s.hidden, s.hc, s.tp, s.hc_variant), (4096, 1, 4, None))
        self.assertEqual((s.attention.kind, s.attention.heads, s.attention.head_dim, s.attention.kv_heads), ("gqa", 8, 128, 2))
        self.assertEqual((s.linear, s.indexer, s.spec_k), (None, None, 1))
        self.assertEqual((s.moe.experts, s.moe.experts_local, s.moe.inter, s.moe.inter_local), (128, 32, 768, 768))
        self.assertEqual((s.moe.topk, s.moe.quant, s.moe.activation), (8, "bf16", "silu"))
        self.assertEqual(s.moe.dense_inter_local, 12288 // 4)

    def test_every_value_carries_the_key_it_was_read_from(self):
        r = self.read(PLAIN_MOE, placement="ep")
        self.assertEqual(r.sources["hidden"], "hidden_size")
        self.assertEqual(r.sources["attention.kind"], "num_key_value_heads")
        self.assertEqual(r.sources["moe.experts"], "num_experts")
        self.assertEqual(r.sources["tp"], "hardware (CHARTER D5)")            # not the model's
        self.assertIn("no MTP key", r.sources["spec_k"])
        self.assertIn("no indexer key", r.sources["indexer"])

    def test_the_read_table_is_the_value_beside_the_key(self):
        """A reading that cannot build a shape still says what it DID read -- the operator needs it to see which
        blank is left, and tools/onboard.py prints this table."""
        r = self.read(PLAIN_MOE)                                              # no placement: one blank, no shape
        table = dict((f, (v, s)) for f, v, s in r.read())
        self.assertIsNone(r.shape)
        self.assertEqual(table["hidden"], (4096, "hidden_size"))
        self.assertEqual(table["attention.heads"], (8, "num_attention_heads"))  # per rank, as the shape holds it
        self.assertEqual(table["moe.dense_inter_local"], (12288 // 4, "intermediate_size"))
        self.assertEqual(r.values["tp"], 4)

    def test_the_sink_is_carried_as_not_established_not_as_false(self):
        """False would be a claim about the model's softmax. None is the shape's own 'not established', and the lane
        refuses it by name -- which is the outcome we want from a config that never mentions it."""
        r = self.read(PLAIN_MOE, placement="ep")
        self.assertIsNone(r.shape.attention.sink)
        self.assertEqual([b.field for b in r.unsettled], ["attention.sink"])
        from engine.kernels import cells
        mla = [v for v in cells.admission(r.shape) if v.lane == "mla"][0]
        self.assertEqual(mla.status, "refused")

    def test_placement_is_the_operators_and_a_routed_model_without_it_is_a_blank(self):
        ep, tp = self.read(PLAIN_MOE, placement="ep").shape.moe, self.read(PLAIN_MOE, placement="tp").shape.moe
        self.assertEqual((ep.experts_local, ep.inter_local), (32, 768))        # whole experts a rank
        self.assertEqual((tp.experts_local, tp.inter_local), (128, 192))       # every expert's intermediate sliced
        blanks = self.read(PLAIN_MOE).blanks
        self.assertEqual([b.field for b in blanks], ["moe.experts_local"])
        self.assertIn("operator", blanks[0].why)
        with self.assertRaises(ValueError):
            self.read(PLAIN_MOE, placement="everywhere")

    def test_a_state_fills_a_blank_and_is_marked_as_the_operators(self):
        quantized = dict(PLAIN_MOE, quantization_config={"quant_method": "modelopt"})
        self.assertEqual([b.field for b in self.read(quantized, placement="ep").blanks], ["moe.quant"])
        r = self.read(quantized, placement="ep", states={"attention.sink": True, "moe.quant": "nvfp4"})
        self.assertEqual((r.blanks, r.unsettled), ((), ()))
        self.assertEqual((r.shape.attention.sink, r.shape.moe.quant), (True, "nvfp4"))
        self.assertIn("operator", r.sources["attention.sink"])

    def test_a_config_that_declares_no_quantization_has_settled_it(self):
        """Silence is an answer here: no `quantization_config` means the weights are the checkpoint's own dtype, so
        naming a cell would be claiming something the file contradicts."""
        self.assertEqual(self.read(PLAIN_MOE, placement="ep").shape.moe.quant, "bf16")
        with self.assertRaises(ValueError):
            self.read(PLAIN_MOE, placement="ep", states={"moe.quant": "nvfp4"})

    def test_a_state_may_not_overrule_a_key_the_config_states(self):
        """The door is not a knob (D11): every argument it takes fills something the checkpoint left unsaid. A state
        for a field the config settles is a contradiction, and it raises rather than serving the operator's word."""
        with self.assertRaises(ValueError) as e:
            self.read(PLAIN_MOE, placement="ep", states={"attention.kind": "mla"})
        self.assertIn("num_key_value_heads", str(e.exception))
        with self.assertRaises(ValueError):                                    # `mhc: true` names the mixer
            self.read(glm53_config(), placement="tp", states={"hc_variant": "split_sinkhorn"})

    def test_a_state_for_a_part_this_model_has_not_got_is_refused(self):
        """A plain routed transformer has no indexer, no linear attention and one residual stream: a state for one
        of those would sit in the reading doing nothing, which is how a reading starts meaning less than it says."""
        for field, value in (("indexer.compress", "qsa"), ("linear.decay", "head"), ("hc_variant", "mhc")):
            with self.subTest(field=field), self.assertRaises(ValueError) as e:
                self.read(PLAIN_MOE, placement="ep", states={field: value})
            self.assertIn("nothing for", str(e.exception))

    def test_a_field_no_operator_states_and_a_value_no_field_takes_are_refused(self):
        for states in ({"moe.experts": 8}, {"indexer.compress": "kmeans"}, {"linear.decay": "layer"},
                       {"moe.quant": 4}, {"attention.sink": "yes"}):
            with self.subTest(states=states), self.assertRaises(ValueError):
                self.read(PLAIN_MOE, placement="ep", states=states)

    def test_a_config_without_the_width_says_so_and_stops(self):
        r = self.read({"model_type": "x"})
        self.assertEqual(([b.field for b in r.blanks], r.shape), (["hidden"], None))

    def test_numbers_that_contradict_the_descriptor_come_back_as_a_blank_not_a_traceback(self):
        """The door reads checkpoints nobody wrote a profile for: one whose own numbers do not make a shape says so
        in the same voice as every other blank."""
        r = self.read(dict(PLAIN_MOE, num_attention_heads=30), placement="ep")  # 30/4 heads over 8/4 kv heads
        self.assertEqual([b.field for b in r.blanks], ["shape"])
        self.assertIn("kv_heads", r.blanks[0].why)

    def test_the_streams_and_their_mixer_are_two_different_facts(self):
        r = self.read(dict(PLAIN_MOE, hc_mult=4), placement="ep")
        self.assertEqual((r.shape.hc, r.shape.hc_variant), (4, None))          # four streams, mixer not established
        self.assertIn("hc_variant", [b.field for b in r.unsettled])
        from engine.kernels import cells
        lanes = {v.lane: v.status for v in cells.admission(r.shape)}
        self.assertEqual(lanes.get("mhc_decode"), "refused")

    def test_one_stream_has_no_mhc_lane_at_all(self):
        """hc 1 is a plain residual: no streams to mix, so the mHC lanes do not apply -- the rule the indexer and KDA
        lanes already follow. The measured cell keeps all fourteen of its lanes."""
        from engine.base.kernel_shape import MEASURED
        from engine.kernels import cells
        r = self.read(PLAIN_MOE, placement="ep")
        self.assertEqual([v.lane for v in cells.admission(r.shape) if v.lane.startswith("mhc")], [])
        self.assertEqual(len(cells.admission(MEASURED)), 14)

    def test_judging_an_incomplete_reading_gives_no_table(self):
        """A lane verdict on a guessed shape is worth less than no verdict."""
        from engine.base.onboard import judge
        out = judge(self.read(PLAIN_MOE))                                      # no placement
        self.assertEqual((out["admission"], out["plan"]), ([], []))


class DenseModelTests(unittest.TestCase):
    """A checkpoint with no routed experts -- the commonest kind of LLM there is.

    The descriptor has no "no MoE" (`KernelShape.moe` is not optional), and for a while that meant the door answered
    "no shape" for a model it can describe perfectly well. It does not need to: the one MLP every token passes is
    what this engine serves through the **E=1 cell**, which b12x's own gate admits beside the routed tuple
    (`engine/kernels/b12x/moe_dispatch._glm_tp_scatter_shape`). So that is what the door reads it as -- and since
    that MLP is TP-sharded like every dense and shared MLP here, there is no expert placement to ask for.

    The other half is the one that matters more: a config that DOES name experts, in a spelling this door cannot
    read, must come back as a blank. Reading it as dense would serve every token through a single MLP.
    """
    LLAMA = {"model_type": "dense_text", "hidden_size": 8192, "num_hidden_layers": 80, "num_attention_heads": 64,
             "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 28672, "hidden_act": "silu"}

    def read(self, cfg, **kw):
        from engine.base.onboard import read_config
        return read_config(cfg, **kw)

    def test_a_model_without_routed_experts_gets_a_shape_and_asks_for_no_placement(self):
        r = self.read(self.LLAMA)                                              # no placement, and none is wanted
        self.assertEqual((r.blanks, r.complete), ((), True))
        m = r.shape.moe
        self.assertEqual((m.experts, m.experts_local, m.topk), (1, 1, 1))
        self.assertEqual((m.inter, m.inter_local, m.dense_inter_local), (28672, 7168, 7168))
        self.assertEqual((m.quant, m.activation), ("bf16", "silu"))
        self.assertIn("E=1", r.sources["moe"])
        self.assertNotIn("moe.placement", r.sources)                           # nothing to place

    def test_the_cell_it_reads_is_the_one_the_dispatcher_admits(self):
        """Not an assertion about what the E=1 cell ought to be: the gate that decides it is compiled out of
        b12x's source and asked."""
        from tests.test_engine_kernel_shape import load_dispatch
        cell = self.read(self.LLAMA).shape.moe
        gate = load_dispatch({"_glm_tp_scatter_shape"}, cell)["_glm_tp_scatter_shape"]
        self.assertTrue(gate(1, 1, cell.hidden, cell.dense_inter_local, 1))    # the MLP, as this engine serves it
        self.assertFalse(gate(1, 1, cell.hidden, cell.dense_inter_local * 2, 1))

    def test_the_lane_table_answers_for_it(self):
        """What the door is for: the work a dense checkpoint would take, and no lane that does not apply."""
        from engine.kernels import cells
        lanes = {v.lane: v.status for v in cells.admission(self.read(self.LLAMA).shape)}
        self.assertEqual(lanes["dense"], "unmeasured")                         # judged at hidden 8192 / I 7168
        self.assertEqual(lanes["universal"], "admitted")
        self.assertEqual(lanes["moe"], "refused")                              # bf16 weights against the b12x cell
        for absent in ("kda_ring", "kda_chunk", "indexer", "mhc_decode", "mhc_prefill"):
            self.assertNotIn(absent, lanes)

    def test_a_spelling_this_door_cannot_read_is_a_blank_not_a_dense_model(self):
        mixture = dict(self.LLAMA, num_local_experts=8, num_experts_per_tok=2)  # Mixtral counts experts this way
        blanks = {b.field: b.why for b in self.read(mixture).blanks}
        self.assertEqual(list(blanks), ["moe"])
        self.assertIn("num_local_experts", blanks["moe"])                      # the key it saw
        self.assertIn("`moe_intermediate_size`", blanks["moe"])                # the one it wanted
        self.assertIn("never a dense model", blanks["moe"])

    def test_a_config_with_no_mlp_at_all_says_that(self):
        r = self.read({k: v for k, v in self.LLAMA.items() if k != "intermediate_size"})
        self.assertEqual([b.field for b in r.blanks], ["moe"])
        self.assertIn("no MLP", r.blanks[0].why)

    def test_a_width_the_four_ranks_cannot_split_is_a_blank(self):
        r = self.read(dict(self.LLAMA, intermediate_size=28673))
        self.assertEqual([b.field for b in r.blanks], ["shape"])
        self.assertIn("inter_local", r.blanks[0].why)


class ServedModelsTests(unittest.TestCase):
    """The three checkpoints this repo serves today all have profiles -- so the generic door must READ what their
    configs name and must NOT pretend to settle the rest. Each blank names exactly what its profile supplies."""

    def reading(self, cfg, placement):
        from engine.base.onboard import read_config
        return read_config(cfg, placement=placement)

    def blanks(self, cfg, placement="ep"):
        return {b.field: b.why for b in self.reading(cfg, placement).blanks}

    def test_glm53_names_its_compression_its_decay_and_its_mixer(self):
        """Three axes other configs leave open are written into this one's keys, and reading a named key is not a
        guess: `index_kpool_compress`, a `kda_layers` key (the per-key-channel decay, LinearAttention's axis) and
        `mhc: true`."""
        r = self.reading(glm53_config(), "tp")
        self.assertEqual(r.values["indexer.compress"], "kpool")
        self.assertEqual(r.sources["indexer.compress"], "index_kpool_compress")
        self.assertEqual(r.values["linear.decay"], "channel")
        self.assertIn("kda", r.sources["linear.decay"])
        self.assertEqual((r.values["hc_variant"], r.sources["hc_variant"]), ("mhc", "mhc"))
        self.assertEqual(sorted(self.blanks(glm53_config(), "tp")), ["moe.activation", "moe.quant"])

    def test_qwen38_needs_its_indexer_compression_its_decay_and_its_expert_format(self):
        blanks = self.blanks(qwen38_config())
        self.assertEqual(sorted(blanks), ["indexer.compress", "linear.decay", "moe.quant"])
        self.assertIn("kpool, ced, qsa", blanks["indexer.compress"])
        self.assertIn("KDA", blanks["linear.decay"])
        self.assertIn("modelopt", blanks["moe.quant"])                         # what the config actually says

    def test_dsv41_needs_its_attention_kind_indexer_gate_and_expert_format(self):
        blanks = self.blanks(dsv41_config())
        self.assertEqual(sorted(blanks), ["attention.kind", "indexer.compress", "moe.activation", "moe.quant"])
        self.assertIn("kv_lora_rank", blanks["attention.kind"])                # one KV head and no latent key
        self.assertIn("fp8", blanks["moe.quant"])                              # what the config actually says

    def test_the_clamped_gate_is_a_blank_because_two_checkpoints_write_it_the_same_and_differ(self):
        """GLM-5.3 and DSv4.1 both declare `hidden_act` silu with `swiglu_limit` 10.0, and are served with different
        gates (swigluoai_uninterleave, silu). The MoE cell is compared by name, so the door refuses to pick."""
        for cfg, placement in ((glm53_config(), "tp"), (dsv41_config(), "ep")):
            with self.subTest(model=cfg["model_type"]):
                why = self.blanks(cfg, placement)["moe.activation"]
                self.assertIn("swiglu_limit", why)
        self.assertNotIn("moe.activation", self.blanks(qwen38_config()))       # no clamp: `hidden_act` is the answer

    def test_the_indexer_pool_is_read_from_the_one_ratio_above_one(self):
        """DSv4.1 states a compress ratio per layer (0, 2, 1) instead of one pool key. One pooling group above 1 is
        the indexer's; a config naming several would leave the pool blank instead of picking."""
        from engine.base.onboard import read_config
        states = dict(AGREEMENT[2][4])
        r = read_config(dsv41_config(), placement="ep", states=states)
        self.assertEqual(r.shape.indexer.pool, 2)
        self.assertIn("compress_ratios", r.sources["indexer"])
        several = dict(dsv41_config(), compress_ratios=[0, 2, 4])
        blanks = read_config(several, placement="ep", states=states).blanks
        self.assertEqual([b.field for b in blanks], ["indexer.pool"])
        self.assertIn("compress_ratios", blanks[0].why)

    def test_a_profile_claims_its_model_type_and_the_wizard_discovers_it(self):
        from engine.base import kernel_shape as ks
        self.assertEqual(set(ks.profiles()), {"glm53", "qwen38", "dsv41"})
        self.assertTrue(all(m.startswith("engine.profiles.") for m in ks.profiles().values()))
        self.assertEqual(ks.claims("glm5_next_text"), "glm53")
        self.assertEqual(ks.claims("qwen4_exp_text"), "qwen38")
        self.assertEqual(ks.claims("deepseek_v41_text"), "dsv41")
        self.assertIsNone(ks.claims("some_moe_text"))
        self.assertIsNone(ks.claims(None))

    def test_the_claimed_model_type_is_the_one_the_profile_asserts(self):
        """A profile's claim and the assert in its own facts must be the same string, or a checkpoint would be routed
        to a derivation that then refuses it."""
        import importlib
        for name, asserted in (("glm53", "glm5_next_text"), ("qwen38", "qwen4_exp_text")):
            source = (ROOT / f"engine/profiles/{name}/facts.py").read_text()
            self.assertIn(f'== "{asserted}"', source, name)
            self.assertEqual(importlib.import_module(f"engine.profiles.{name}").MODEL_TYPES, (asserted,))


def glm53_shape():
    from engine.profiles.glm53 import facts
    return facts.architecture(pinned("GLM53_TEXT_CONFIG")).kernel_shape()


def qwen38_shape():
    from engine.profiles.qwen38 import shapes
    return shapes.kernel_shape(pinned("QWEN38_TEXT_CONFIG"))


def dsv41_shape():
    from engine.profiles.dsv41 import shapes
    return shapes.kernel_shape(json.loads(DSV41_CONFIG.read_text())["text_config"])


#: What each profile knows that its config does not state -- the whole list, per model, and every value is read off
#: that model's reference implementation (the comment beside it in the profile). `exceptions` are fields the config
#: cannot state at all because they are not the model's: the drafter is the operator's choice, not the checkpoint's.
AGREEMENT = (
    ("glm53", glm53_config, "tp", glm53_shape,
     {"moe.quant": "nvfp4",                          # quantization_config: nvfp4-pack-quantized, group 16
      "moe.activation": "swigluoai_uninterleave",    # facts.kernel_shape: the clamped OpenAI gate, uninterleaved
      "attention.sink": False},                      # mla_sparse_mqa: "No sink"
     {"spec_k": "DFlash2 with 7 draft slots is the operator's (facts.SPEC_K, 2026-09-13); this config declares no "
                "MTP head at all, and the one at layer 45 is not served"}),
    ("qwen38", qwen38_config, "ep", qwen38_shape,
     {"moe.quant": "nvfp4",                          # ModelOpt NVFP4, group 16
      "indexer.compress": "qsa",                     # nvidia/qsa.py
      "linear.decay": "head",                        # GatedDeltaNet: one decay a head
      "attention.sink": False,                       # "QSA does not support ALiBi or attention sinks"
      "hc_variant": "gated_residual"},               # Qwen4ExpTextGatedResidual
     {"spec_k": "the served draft count is the operator's (facts.SPEC_K 3: the one MTP layer chained, the 09-18 "
                "fleet pair); this config declares one MTP layer and the door reads one token a step"}),
    ("dsv41", dsv41_config, "ep", dsv41_shape,
     {"moe.quant": "mxfp4-a8",                       # FP4 e2m1 in groups of 32 with E8M0 scales, FP8 activations
      "moe.activation": "silu",
      "indexer.compress": "ced",                     # the Compressor of inference/model.py
      "attention.kind": "mla",                       # one KV head over a compressed latent
      "attention.sink": True,                        # sparse_attn: attn_sink
      "hc_variant": "split_sinkhorn"},               # hc_split_sinkhorn
     {}),
)


class AgreementTests(unittest.TestCase):
    """The generic door and each profile's own derivation, field for field.

    This is the load-bearing case of the front door: if the two disagree about a checkpoint the fleet serves, one of
    them is wrong about a model that boots. Each model's `states` above is the complete list of what its profile
    knows beyond its config -- three, five and six facts -- and nothing else may differ."""

    def test_the_door_reads_every_profiles_shape_once_its_reference_is_stated(self):
        from engine.base.onboard import read_config
        for name, config, placement, profile, states, exceptions in AGREEMENT:
            with self.subTest(model=name):
                r = read_config(config(), placement=placement, states=states)
                self.assertEqual([b.field for b in r.blanks], [], name)
                door, own = asdict(r.shape), asdict(profile())
                for field in exceptions:
                    self.assertNotEqual(door.pop(field), own.pop(field), f"{name}.{field} no longer differs")
                self.assertEqual(door, own, name)

    def test_the_stated_fields_are_exactly_the_blanks(self):
        """Neither more nor fewer: a state the door did not ask for would be overruling a key (and raises), and a
        blank left unstated would block the shape. `unsettled` counts -- the shape carries those, the lane refuses
        them by name, and a profile that states one is telling the lane to run."""
        from engine.base.onboard import read_config
        for name, config, placement, _, states, _ in AGREEMENT:
            with self.subTest(model=name):
                bare = read_config(config(), placement=placement)
                self.assertEqual(sorted(states), sorted(b.field for b in bare.blanks + bare.unsettled), name)

    def test_the_door_and_the_profile_are_judged_the_same_by_the_cells(self):
        """The shapes are equal, so the lane table must be too -- this is what an operator actually reads."""
        from engine.base.onboard import read_config
        from engine.kernels import cells
        for name, config, placement, profile, states, _ in AGREEMENT:
            with self.subTest(model=name):
                shape = read_config(config(), placement=placement, states=states).shape
                self.assertEqual(cells.to_dicts(cells.admission(shape)), cells.to_dicts(cells.admission(profile())))


class CliTests(unittest.TestCase):
    def test_the_command_prints_what_it_read_a_shape_a_table_and_the_work(self):
        module = cli()
        result = module.run(PLAIN_MOE, placement="ep", ckpt=None)
        text = module.render(result)
        self.assertEqual(result["door"], "generic")
        self.assertIn("read (per rank", text)
        self.assertIn("[hidden_size]", text)
        self.assertIn("lanes:", text)
        self.assertIn("work, cheapest first:", text)
        self.assertIn("moe", text)
        payload = json.loads(module.as_json(result))
        self.assertEqual(payload["door"], "generic")
        self.assertEqual(payload["plan"], [v.lane for v in result["plan"]])
        self.assertTrue(payload["shape"])
        self.assertEqual(payload["read"][0], {"field": "tp", "value": 4, "source": "hardware (CHARTER D5)"})

    def test_a_config_that_does_not_settle_prints_its_blanks_and_exits_two(self):
        module = cli()
        result = module.run(PLAIN_MOE, placement=None, ckpt=None)              # no placement: one blank
        text = module.render(result)
        self.assertIn("no shape -- the config does not settle", text)
        self.assertIn("moe.experts_local", text)
        self.assertIn("engine/profiles/", text)                                # what fills a blank
        self.assertIn("--state field=value", text)
        self.assertIn("[hidden_size]", text)                                   # and what it did read

    def test_states_come_off_the_command_line_by_field_name(self):
        module = cli()
        self.assertEqual(module.parse_states(["moe.quant=mxfp4-a8", "attention.sink=true"]),
                         {"moe.quant": "mxfp4-a8", "attention.sink": True})
        for bad in (["moe.quant"], ["attention.sink=maybe"]):
            with self.subTest(arg=bad), self.assertRaises(SystemExit):
                module.parse_states(bad)

    def test_the_whole_command_reads_the_pinned_dsv41_config(self):
        """End to end on a real checkpoint's config: the door that claims no model reproduces the profile's own lane
        table, from the file plus the six facts its reference states (engine/DSV41_CARRY_20260919.md)."""
        module = cli()
        states = dict(AGREEMENT[2][4])
        cfg = module.text_config(DSV41_CONFIG)
        result = module.run(cfg, placement="ep", ckpt=None, states=states)
        self.assertEqual(result["shape"], dsv41_shape())
        table = {v.lane: v.status for v in result["admission"]}
        self.assertEqual(table["moe"], "refused")                              # mxfp4-a8 against the b12x cell
        self.assertEqual(table["universal"], "admitted")
        self.assertIn("work, cheapest first:", module.render(result))

    def test_it_reads_a_nested_text_config_and_the_outer_quantization(self):
        module = cli()
        cfg = module.text_config(DSV41_CONFIG)
        self.assertEqual(cfg["model_type"], "deepseek_v41_text")               # the nested one, not the wrapper
        self.assertEqual(cfg["quantization_config"]["quant_method"], "fp8")    # lifted from the outer config

    def test_the_door_names_the_profile_when_one_claims_the_checkpoint(self):
        module = cli()
        result = module.run({"model_type": "glm5_next_text", "hidden_size": 4096}, placement="ep", ckpt=None)
        self.assertIn("profile glm53", result["door"])                         # without --ckpt it still says who owns it


if __name__ == "__main__":
    unittest.main()
