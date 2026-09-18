"""engine/profiles/qwen38/preshard over a synthetic checkpoint in NVIDIA's Hub export layout (nvidia/Qwen3.8-Flash-Next-
NVFP4 @ fc694b54): the MIXED_PRECISION quantization config naming every quantised module, per-expert NVFP4 four-tensor
experts, the MTP head's per-expert FP8 block-scaled experts, the PLE table as e4m3 shards under one scalar scale --
cut into four TEP=4 rank files and four PLE table files (the operator's decision of 2026-09-18: the table on the SSD).

What is checked: facts.load reads the export's encoding off its config (and still reads the older NVFP4-only config);
every rank file holds every spec of layout v3 and no table part; the MTP experts are the FP8 ones dequantised (x times
its scale, held on srv2 against the BF16 copy) and NVFP4-encoded like the target's; each table file is its rank's
shards' bytes exactly, its sidecar and hash agree, and ple_table.PLETable opens and gathers it; SHA256SUMS, the
manifest and the shape record are written; a second run into the same directory is refused."""
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

torch = None
if importlib.util.find_spec("torch") is not None:
    import torch
    import numpy as np
    from tests.test_engine_qwen38_weights import write_safetensors

P = "model.language_model."
TINY = {                                                        # every key facts.architecture reads, at widths that split four ways
    # hidden is a multiple of 128: the NVFP4 scale swizzle pads rows to 128 and the layout declares unpadded sizes
    "model_type": "qwen4_exp_text", "hidden_size": 128, "vocab_size": 256, "rms_norm_eps": 1e-6,
    "max_position_embeddings": 4096, "eos_token_id": 2,
    "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    "full_attention_interval": 4, "hc_count": 4, "hc_lowrank": 8,
    "linear_num_key_heads": 4, "linear_num_value_heads": 8, "linear_key_head_dim": 16, "linear_value_head_dim": 16,
    "linear_conv_kernel_dim": 4,
    "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 32,
    "rope_parameters": {"rope_theta": 10000.0, "partial_rotary_factor": 0.25, "mrope_interleaved": True,
                        "mrope_section": [2, 1, 1]},
    "indexer_n_heads": 4, "indexer_head_dim": 16, "indexer_budget": 64, "indexer_compress_ratio": 4, "indexer_kv_heads": 1,
    "num_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 128, "shared_expert_intermediate_size": 128,
    "hidden_act": "silu", "output_gate_type": "sigmoid", "norm_topk_prob": True,
    "ple_layer_ids": [2], "ngram_size": 3, "heads_per_ngram": 8, "ple_embed_dim": 128, "ple_conv_kernel_size": 4,
    "ngram_vocab_size_base": 64, "split_ngram_parts": 8, "make_ngram_vocab_size_divisible_by": 128, "seed": 1234,
    "mtp_num_hidden_layers": 1, "mtp_use_dedicated_embeddings": False, "tie_word_embeddings": False,
}
MTP_BLOCK = 16


def nvidia_config(text: dict = TINY, *, mtp_block: int = MTP_BLOCK) -> dict:
    """config.json as the Hub export writes it: the multimodal wrapper's text_config and the mixed-precision
    quantization config naming every quantised module."""
    nvfp4 = {"num_bits": 4, "type": "float", "group_size": 16, "dynamic": False}
    quantised = {f"{P}layers.{L}.mlp.experts": {"quant_algo": "NVFP4", "group_size": 16} for L in range(len(text["layer_types"]))}
    for i in text["ple_layer_ids"]:
        quantised[f"{P}layers.{i - 1}.ple.ple_embedding.ngram_embedding"] = {"quant_algo": "FP8"}
    quantised["mtp.layers.0.mlp.experts"] = {"quant_algo": "FP8_PB_WO", "group_size": mtp_block}     # config.json's name for it
    return {"architectures": ["Qwen4ExpForConditionalGeneration"], "model_type": "qwen4_exp", "text_config": dict(text),
            "quantization_config": {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION",
                                    "producer": {"name": "modelopt", "version": "test"},
                                    "config_groups": {"group_0": {"weights": nvfp4, "input_activations": nvfp4,
                                                                  "targets": sorted(k for k in quantised if k.endswith("experts"))}},
                                    "ignore": ["lm_head", f"{P}embed_tokens", f"{P}hyper_connection_mixer*"],
                                    "quantized_layers": quantised}}


def old_config(text: dict = TINY) -> dict:
    """The older copy's config: NVFP4 only, the rest named by the glob patterns it ignores."""
    nvfp4 = {"num_bits": 4, "type": "float", "group_size": 16, "dynamic": False}
    return {"text_config": dict(text),
            "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4",
                                    "config_groups": {"group_0": {"weights": nvfp4, "input_activations": nvfp4}},
                                    "ignore": ["model.embed_tokens", "mtp.*", "model.mtp.*", "*.self_attn.*", "*.linear_attn.*",
                                               "*.mlp.gate*", "*.mlp.shared_expert.*", "*hyper_connection*", "*.ple.*",
                                               "lm_head"]}}


def facts_of(config: dict):
    from engine.profiles.qwen38 import facts
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "config.json").write_text(json.dumps(config))
        return facts.load(d)


def nvidia_checkpoint(root: Path, seed: int = 0) -> dict:
    """Every tensor the layout reads, in the export's encodings, across three shard files under an index."""
    from engine.modules.ngram_embedding import NGramHash
    F = facts_of(nvidia_config())
    t = TINY
    H, hc, V = t["hidden_size"], t["hc_count"], t["vocab_size"]
    g = torch.Generator().manual_seed(seed)
    r = lambda *shape, scale=0.1: torch.randn(*shape, generator=g) * scale
    bf = lambda *shape, scale=0.1: r(*shape, scale=scale).to(torch.bfloat16)
    u8 = lambda *shape: torch.randint(0, 256, shape, generator=g, dtype=torch.int32).to(torch.uint8)
    e4 = lambda *shape, lo=0.5, hi=4.0: (torch.rand(*shape, generator=g) * (hi - lo) + lo).to(torch.float8_e4m3fn)
    dense, experts, fp8 = {}, {}, {}
    dense[f"{P}embed_tokens.weight"], dense["lm_head.weight"] = bf(V, H, scale=0.5), bf(V, H)

    def hyper(base, inject):
        dense[f"{base}.hc_norm.weight"] = bf(hc * H, scale=0.3)
        dense[f"{base}.input_mix_weight_down.weight"] = bf(t["hc_lowrank"], hc * H)
        dense[f"{base}.input_mix_weight_up.weight"] = bf(hc * H, t["hc_lowrank"])
        if inject:
            dense[f"{base}.block_inject_weight.weight"] = bf(hc, hc * H)

    def attention(sa):
        heads, kvh, hd, ih, ihd = t["num_attention_heads"], t["num_key_value_heads"], t["head_dim"], t["indexer_n_heads"], t["indexer_head_dim"]
        dense[f"{sa}.q_proj.weight"] = bf(heads * hd * 2, H)
        dense[f"{sa}.k_proj.weight"], dense[f"{sa}.v_proj.weight"] = bf(kvh * hd, H), bf(kvh * hd, H)
        dense[f"{sa}.o_proj.weight"] = bf(H, heads * hd)
        dense[f"{sa}.q_norm.weight"], dense[f"{sa}.k_norm.weight"] = bf(hd, scale=0.3), bf(hd, scale=0.3)
        dense[f"{sa}.indexer.index_qk_proj.weight"] = bf((ih + 1) * ihd, H)
        dense[f"{sa}.indexer.q_layernorm.weight"], dense[f"{sa}.indexer.k_layernorm.weight"] = bf(ihd, scale=0.3), bf(ihd, scale=0.3)

    def moe_common(mlp):
        E, S = t["num_experts"], t["shared_expert_intermediate_size"]
        dense[f"{mlp}.gate.weight"], dense[f"{mlp}.shared_expert_gate.weight"] = bf(E, H), bf(1, H)
        dense[f"{mlp}.shared_expert.gate_proj.weight"], dense[f"{mlp}.shared_expert.up_proj.weight"] = bf(S, H), bf(S, H)
        dense[f"{mlp}.shared_expert.down_proj.weight"] = bf(H, S)

    E, I = t["num_experts"], t["moe_intermediate_size"]
    hyper(f"{P}hyper_connection_mixer", False)
    for L, kind in enumerate(t["layer_types"]):
        base = f"{P}layers.{L}"
        hyper(f"{base}.attn_hyper_connection", True)
        hyper(f"{base}.mlp_hyper_connection", True)
        if kind == "linear_attention":
            la = f"{base}.linear_attn"
            nk, nv, kd, vd = t["linear_num_key_heads"], t["linear_num_value_heads"], t["linear_key_head_dim"], t["linear_value_head_dim"]
            conv_dim = 2 * nk * kd + nv * vd
            dense[f"{la}.in_proj_qkv.weight"], dense[f"{la}.in_proj_z.weight"] = bf(conv_dim, H), bf(nv * vd, H)
            dense[f"{la}.in_proj_b.weight"], dense[f"{la}.in_proj_a.weight"] = bf(nv, H), bf(nv, H)
            dense[f"{la}.out_proj.weight"], dense[f"{la}.norm.weight"] = bf(H, nv * vd), (1 + r(vd)).to(torch.bfloat16)
            dense[f"{la}.conv1d.weight"] = bf(conv_dim, 1, t["linear_conv_kernel_dim"], scale=0.3)
            dense[f"{la}.dt_bias"] = torch.ones(nv).to(torch.bfloat16)
            dense[f"{la}.A_log"] = torch.empty(nv).uniform_(0.01, 16, generator=g).log().to(torch.bfloat16)
        else:
            attention(f"{base}.self_attn")
        moe_common(f"{base}.mlp")
        for e in range(E):
            shared = {"weight_scale_2": torch.tensor(0.01 * (e + 1), dtype=torch.float32),
                      "input_scale": torch.tensor(0.002 * (L + 1), dtype=torch.float32)}         # gate == up, as the export has it
            for proj, (out, inn) in (("gate_proj", (I, H)), ("up_proj", (I, H)), ("down_proj", (H, I))):
                b = f"{base}.mlp.experts.{e}.{proj}"
                experts[f"{b}.weight"], experts[f"{b}.weight_scale"] = u8(out, inn // 2), e4(out, inn // 16)
                experts[f"{b}.weight_scale_2"] = shared["weight_scale_2"] if proj != "down_proj" else torch.tensor(0.03, dtype=torch.float32)
                experts[f"{b}.input_scale"] = shared["input_scale"] if proj != "down_proj" else torch.tensor(0.004, dtype=torch.float32)
        if L in F.ple_layers:
            ple = f"{base}.ple"
            D = t["ple_embed_dim"]
            dense[f"{ple}.key_proj.weight"], dense[f"{ple}.value_proj.weight"] = bf(hc * H, D), bf(H, D)
            for norm in ("norm_key", "norm_query", "norm_conv"):
                dense[f"{ple}.{norm}.weight"] = bf(hc * H, scale=0.3)
            dense[f"{ple}.conv1d.weight"] = bf(hc * H, 1, t["ple_conv_kernel_size"], scale=0.3)
            made = NGramHash.splitmix(ngram_size=t["ngram_size"], heads=t["heads_per_ngram"], unigram_vocab=V,
                                      base=t["ngram_vocab_size_base"], table_index=0, seed=t["seed"], eos=t["eos_token_id"])
            dense[f"{ple}.ple_embedding.layer_multipliers"] = made.multipliers.clone()
            dense[f"{ple}.ple_embedding.ngram_heads_offsets"] = made.offsets.clone()
            dense[f"{ple}.ple_embedding.ngram_heads_vocab_sizes"] = made.sizes.clone()
            table = f"{ple}.ple_embedding.ngram_embedding"
            for s in range(t["split_ngram_parts"]):
                fp8[f"{table}.shard_{s}.weight"] = e4(F.ple_rows_per_shard, F.ple_head_dim, lo=-3.0, hi=3.0)
            fp8[f"{table}.weight_scale"] = torch.tensor([0.0002], dtype=torch.bfloat16)
    m = "mtp."
    dense[f"{m}fc_embedding.weight"], dense[f"{m}fc_hidden.weight"] = bf(H, H, scale=H ** -0.5), bf(H, H, scale=H ** -0.5)
    dense[f"{m}pre_fc_norm_embedding.weight"], dense[f"{m}pre_fc_norm_hidden.weight"] = bf(H, scale=0.3), bf(hc * H, scale=0.3)
    hyper(f"{m}hyper_connection_mixer", False)
    hyper(f"{m}layers.0.attn_hyper_connection", True)
    hyper(f"{m}layers.0.mlp_hyper_connection", True)
    attention(f"{m}layers.0.self_attn")
    moe_common(f"{m}layers.0.mlp")
    for e in range(E):
        for proj, (out, inn) in (("gate_proj", (I, H)), ("up_proj", (I, H)), ("down_proj", (H, I))):
            b = f"{m}layers.0.mlp.experts.{e}.{proj}"
            fp8[f"{b}.weight"] = e4(out, inn, lo=-2.0, hi=2.0)
            fp8[f"{b}.weight_scale_inv"] = (torch.rand(out // MTP_BLOCK, inn // MTP_BLOCK, generator=g) * 0.001 + 0.0001).to(torch.bfloat16)
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(nvidia_config()))
    (root / "generation_config.json").write_text(json.dumps({"temperature": 1.0, "top_p": 0.95}))
    (root / "tokenizer.json").write_text("{}")
    files = {"model-00001-of-00003.safetensors": dense, "model-00002-of-00003.safetensors": experts,
             "model-fp8-mtp-ple.safetensors": fp8}
    index, written = {}, {}
    for file, tensors in files.items():
        write_safetensors(root / file, tensors)
        index.update({name: file for name in tensors})
        written.update(tensors)
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))
    return written


@unittest.skipUnless(torch is not None, "requires torch")
class FactsTests(unittest.TestCase):
    def test_the_export_encoding_is_read_off_the_config(self):
        F = facts_of(nvidia_config())
        self.assertEqual((F.mtp_experts, F.mtp_block, F.ngram_divisible), ("fp8_block", MTP_BLOCK, 128))
        self.assertEqual(F.ple_rows_total % 128, 0)
        self.assertEqual(F.ple_rows_per_shard * TINY["split_ngram_parts"], F.ple_rows_total)
        self.assertEqual(F.ple_rows_per_rank, F.ple_rows_per_shard * 2)
        self.assertEqual(F.weight_layout, "st-qwen38-tep4-modelopt-v3")

    def test_the_older_nvfp4_only_config_still_loads_with_bf16_mtp_experts(self):
        F = facts_of(old_config())
        self.assertEqual((F.mtp_experts, F.mtp_block), ("bf16", 0))

    def test_the_mtp_experts_fp8_label_of_either_config_file_is_read(self):
        c = nvidia_config()
        c["quantization_config"]["quantized_layers"]["mtp.layers.0.mlp.experts"] = {"quant_algo": "FP8_BLOCK_SCALES", "group_size": 128}
        self.assertEqual(facts_of(c).mtp_block, 128)

    def test_a_config_that_quantises_more_or_less_is_refused(self):
        for change in ("drop an expert layer", "quantise the head", "another block algo", "an extra module"):
            c = nvidia_config()
            q = c["quantization_config"]
            if change == "drop an expert layer":
                q["quantized_layers"].pop(f"{P}layers.0.mlp.experts")
            elif change == "quantise the head":
                q["ignore"].remove("lm_head")
            elif change == "another block algo":
                q["quantized_layers"]["mtp.layers.0.mlp.experts"] = {"quant_algo": "NVFP4", "group_size": 16}
            else:
                q["quantized_layers"][f"{P}layers.0.self_attn"] = {"quant_algo": "FP8"}
            with self.subTest(change=change), self.assertRaises(ValueError):
                facts_of(c)

    def test_the_served_checkpoint_shard_rows_derive_from_its_config(self):
        """probes/qwen38_config.json is the served checkpoint's config: the derived shard rows are its header's."""
        from probes.engine_qwen38_cells import facts as served
        F = served()
        self.assertEqual((F.ple_rows_per_shard, F.ple_rows_per_rank, F.ple_rows_total), (2_500_012, 80_000_384, 320_001_536))


@unittest.skipUnless(torch is not None, "requires torch")
class PreshardTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name) / "ckpt"
        self.written = nvidia_checkpoint(self.root)
        self.out = Path(self.dir.name) / "st-qwen38-tep4"

    def run_preshard(self, *extra):
        from engine.profiles.qwen38 import preshard
        from contextlib import redirect_stdout
        import io
        with redirect_stdout(io.StringIO()):
            rc = preshard.main(["--ckpt", str(self.root), "--out", str(self.out), "--source-revision", "test", *extra])
        self.assertEqual(rc, 0)

    def test_plan_reads_every_text_tensor_and_no_table_part_is_a_spec(self):
        from engine.profiles.qwen38 import facts, preshard, specs
        F, groups, report, ck = preshard.plan(self.root)
        self.assertEqual(report["unread_text_count"], 0, report["unread_text_tensors"])
        names = [s.name for s in specs.all_specs(F)]
        self.assertEqual(len(names), report["tensors_per_rank"])
        self.assertFalse([n for n in names if ".ple.table." in n])
        self.assertEqual(report["ple"]["shards_per_rank"], TINY["split_ngram_parts"] // facts.TP)
        self.assertEqual(report["mtp_experts"], "fp8_block")
        labels = [label for label, _, _ in groups]
        self.assertIn("layer 0 experts rank 3", labels)
        self.assertIn("mtp experts rank 0", labels)
        # a rank group answers only for its rank; every rank sees each routed spec exactly once
        for rank in range(facts.TP):
            seen = [s.name for _, _, of in groups for s in of(rank)]
            self.assertEqual(len(seen), len(set(seen)))
            self.assertEqual(sorted(seen), sorted(names))

    def test_ranks_tables_and_records_are_written_and_verified(self):
        from engine.base.loader import RankLoader
        from engine.profiles.qwen38 import facts, specs
        from engine.profiles.qwen38.ple_table import PLETable
        self.run_preshard()
        F = facts.load(self.root)
        names = sorted(s.name for s in specs.all_specs(F))
        manifest = json.loads((self.out / "preshard-manifest.json").read_text())
        sums = {name: sha for sha, name in (line.split("  ") for line in (self.out / "SHA256SUMS").read_text().splitlines())}
        self.assertEqual(len(sums), 2 * facts.TP)
        self.assertTrue((self.out / "kernel_shape.json").is_file())
        self.assertTrue((self.out / "config.json").is_file() and (self.out / "tokenizer.json").is_file())
        self.assertFalse((self.out.with_name(self.out.name + ".incomplete")).exists())
        for rank in range(facts.TP):
            with self.subTest(rank=rank):
                loader = RankLoader(self.out / f"rank{rank}of4.safetensors")
                self.assertEqual(loader.metadata["weight_layout"], "st-qwen38-tep4-modelopt-v3")
                self.assertEqual((loader.metadata["rank"], loader.metadata["ple"], loader.metadata["mtp_experts"]), (str(rank), "ssd", "fp8_block"))
                self.assertEqual(sorted(k for k in loader.keys() if not k.startswith("__st_padding__.")), names)
                # the PLE table file: its rank's shards back to back
                path = self.out / facts.ple_file(rank)
                shards = specs.ple_shards(F, 1, rank)
                expect = b"".join(self.written[s].view(torch.uint8).numpy().tobytes() for s in shards)
                self.assertEqual(path.read_bytes(), expect)
                sidecar = json.loads((self.out / facts.ple_sidecar(rank)).read_text())
                self.assertEqual((sidecar["rows"], sidecar["width"], sidecar["shards"]), (F.ple_rows_per_rank, F.ple_head_dim, shards))
                self.assertEqual(sidecar["sha256"], hashlib.sha256(expect).hexdigest())
                self.assertEqual(sums[path.name], sidecar["sha256"])
                self.assertAlmostEqual(sidecar["scale"], float(self.written[f"{P}layers.1.ple.ple_embedding.ngram_embedding.weight_scale"].float()), places=9)
                table = PLETable.open(self.out, rank, F)
                rows = np.array([0, F.ple_rows_per_rank - 1, 5, 5, F.ple_rows_per_shard], dtype=np.int64)
                whole = np.frombuffer(expect, dtype=np.uint8).reshape(F.ple_rows_per_rank, F.ple_head_dim)
                self.assertTrue(np.array_equal(table.gather(rows), whole[rows]))
                table.close()
        self.assertEqual([row["name"] for row in manifest["ple_files"]], [facts.ple_file(r) for r in range(facts.TP)])
        self.assertEqual(manifest["source_revision"], "test")

    def test_the_mtp_experts_are_the_fp8_ones_dequantised_then_nvfp4_encoded_like_the_target(self):
        from engine.base.loader import RankLoader
        from engine.modules.nvfp4_sf import swizzle_sf_batch
        from engine.profiles.qwen38 import facts, specs
        self.run_preshard()
        F = facts.load(self.root)
        I = TINY["moe_intermediate_size"]
        for rank in (0, 3):
            loader = RankLoader(self.out / f"rank{rank}of4.safetensors")
            got = loader.load(["mtp.L0.moe.w13", "mtp.L0.moe.w13_sf", "mtp.L0.moe.w13_alpha", "mtp.L0.moe.w2", "mtp.L0.moe.w2_alpha",
                               "mtp.L0.moe.a13_scale"], device="cpu")
            w13, sf13, g13, w2, g2 = [], [], [], [], []
            for e in range(*F.expert_range(rank)):
                b = f"mtp.layers.0.mlp.experts.{e}."
                parts = [specs.dequant_fp8_block(self.written[b + p + ".weight"], self.written[b + p + ".weight_scale_inv"], MTP_BLOCK)
                         for p in ("gate_proj", "up_proj", "down_proj")]
                p, sc, gs = specs.nvfp4_from_bf16(torch.cat([parts[1], parts[0]], 0))
                w13.append(p); sf13.append(sc); g13.append(gs)
                p, _, gs = specs.nvfp4_from_bf16(parts[2])
                w2.append(p); g2.append(gs)
            self.assertTrue(torch.equal(got["mtp.L0.moe.w13"], torch.stack(w13)))
            self.assertTrue(torch.equal(got["mtp.L0.moe.w13_sf"].view(torch.uint8), swizzle_sf_batch(torch.stack(sf13).view(torch.uint8))))
            self.assertTrue(torch.equal(got["mtp.L0.moe.w13_alpha"], torch.stack(g13)))
            self.assertTrue(torch.equal(got["mtp.L0.moe.w2"], torch.stack(w2)))
            self.assertTrue(torch.equal(got["mtp.L0.moe.w2_alpha"], torch.stack(g2)))
            self.assertTrue(torch.equal(got["mtp.L0.moe.a13_scale"], torch.ones(F.experts_local)))
            self.assertEqual(tuple(got["mtp.L0.moe.w13"].shape), (F.experts_local, 2 * I, TINY["hidden_size"] // 2))

    def test_routed_experts_and_dense_tensors_land_on_their_ranks(self):
        from engine.base.loader import RankLoader
        from engine.profiles.qwen38 import facts
        self.run_preshard()
        F = facts.load(self.root)
        for rank in range(facts.TP):
            loader = RankLoader(self.out / f"rank{rank}of4.safetensors")
            got = loader.load(["L0.moe.w13", "L0.moe.w13_alpha", "embed", "L3.attn.in_proj"], device="cpu")
            lo, hi = F.expert_range(rank)
            want = torch.stack([torch.cat([self.written[f"{P}layers.0.mlp.experts.{e}.up_proj.weight"],
                                           self.written[f"{P}layers.0.mlp.experts.{e}.gate_proj.weight"]], 0) for e in range(lo, hi)])
            self.assertTrue(torch.equal(got["L0.moe.w13"], want))
            self.assertTrue(torch.equal(got["L0.moe.w13_alpha"],
                                        torch.stack([self.written[f"{P}layers.0.mlp.experts.{e}.up_proj.weight_scale_2"] for e in range(lo, hi)])))
            vp = F.vocab_local
            self.assertTrue(torch.equal(got["embed"], self.written[f"{P}embed_tokens.weight"][rank * vp:(rank + 1) * vp]))
            self.assertEqual(tuple(got["L3.attn.in_proj"].shape), (2 * 2 * 32 + 2 * 32 + 4 * 16 + 16, TINY["hidden_size"]))

    def test_a_development_subset_writes_its_layers_and_the_first_table_shards_and_cannot_be_served(self):
        from engine.base.loader import RankLoader
        from engine.profiles.qwen38 import facts, specs
        from engine.profiles.qwen38.ple_table import PLETable
        self.run_preshard("--layers", "0-0", "--ple-shards", "1")
        F = facts.load(self.root)
        loader = RankLoader(self.out / "rank2of4.safetensors")
        keys = sorted(k for k in loader.keys() if not k.startswith("__st_padding__."))
        self.assertEqual(keys, sorted(s.name for s in specs.all_specs(F, [0])))
        self.assertEqual(loader.metadata["layers"], "0-0")
        shards = specs.ple_shards(F, 1, 2)[:1]
        sidecar = json.loads((self.out / facts.ple_sidecar(2)).read_text())
        self.assertEqual((sidecar["shards"], sidecar["rows"], sidecar["dev_shards"]), (shards, F.ple_rows_per_shard, 1))
        self.assertEqual((self.out / facts.ple_file(2)).read_bytes(), self.written[shards[0]].view(torch.uint8).numpy().tobytes())
        with self.assertRaisesRegex(ValueError, "rows"):
            PLETable.open(self.out, 2, F)
        with self.assertRaises(ValueError):                              # the option goes with --layers
            from engine.profiles.qwen38 import preshard
            preshard.main(["--ckpt", str(self.root), "--out", str(self.out) + "-x", "--source-revision", "t", "--ple-shards", "1"])

    def test_a_second_run_into_the_same_directory_is_refused(self):
        from engine.profiles.qwen38 import preshard
        self.run_preshard()
        with self.assertRaisesRegex(ValueError, "new directory"):
            preshard.main(["--ckpt", str(self.root), "--out", str(self.out), "--source-revision", "test"])

    def test_dequant_fp8_block_multiplies_tile_scales(self):
        from engine.profiles.qwen38.specs import dequant_fp8_block
        w = torch.ones(32, 64).to(torch.float8_e4m3fn)
        s = torch.arange(1, 9, dtype=torch.float32).view(2, 4).to(torch.bfloat16)
        got = dequant_fp8_block(w, s, 16)
        self.assertEqual(tuple(got.shape), (32, 64))
        self.assertTrue(torch.equal(got[:16, :16], torch.full((16, 16), 1.0)))
        self.assertTrue(torch.equal(got[16:, 48:], torch.full((16, 16), 8.0)))
        with self.assertRaises(ValueError):
            dequant_fp8_block(w, s, 32)


if __name__ == "__main__":
    unittest.main()
