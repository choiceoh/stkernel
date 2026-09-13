"""engine/profiles/qwen38/weights: the checkpoint reader over a synthetic checkpoint written in the checkpoint's own
layout -- safetensors shards under an index, `model.language_model.` names, bf16 matrices, NVFP4 experts as the
modelopt four-tensor layout, the PLE table as e4m3 shards with one scalar scale. What each encoding must become is
checked against the modules that define it (engine/modules/moe.dequant_nvfp4) or against the bytes written.

The scale is the reason this file exists: a loader that upcast the table's e4m3 bytes without it served rows of the
wrong magnitude and a garbage answer on the real checkpoint (2026-09-13, qualification-notes.md had said so)."""
import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path

torch = None
if importlib.util.find_spec("torch") is not None:
    import torch
    import numpy as np

SAFETENSORS_DTYPE = {torch.bfloat16: "BF16", torch.float32: "F32", torch.uint8: "U8", torch.float8_e4m3fn: "F8_E4M3",
                     torch.int64: "I64"} if torch is not None else {}


def write_safetensors(path: Path, tensors: dict) -> None:
    """A safetensors file by hand: the 8-byte header length, the JSON header, the data."""
    header, blobs, at = {}, [], 0
    for name, t in tensors.items():
        raw = t.contiguous().view(torch.uint8).numpy().tobytes() if t.dtype in (torch.bfloat16, torch.float8_e4m3fn) \
            else t.contiguous().numpy().tobytes()
        header[name] = {"dtype": SAFETENSORS_DTYPE[t.dtype], "shape": list(t.shape), "data_offsets": [at, at + len(raw)]}
        blobs.append(raw)
        at += len(raw)
    encoded = json.dumps(header).encode()
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(encoded)))
        f.write(encoded)
        for raw in blobs:
            f.write(raw)


def synthetic_checkpoint(root: Path, seed: int = 0) -> dict:
    """One GDN layer's worth of the real layout: returns what was written, for the assertions."""
    g = torch.Generator().manual_seed(seed)
    P = "model.language_model."
    H, I, E, rows_per_shard, width = 32, 16, 3, 5, 4
    cfg = {"hidden_size": H, "moe_intermediate_size": I, "num_experts": E, "ple_layer_ids": [1], "eos_token_id": 0,
           "layer_types": ["linear_attention"], "vocab_size": 32}
    (root).mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps({"text_config": cfg}))
    written, files = {}, {}
    # bf16 matrices and a bf16 vector under layer 0
    bf16 = {f"{P}layers.0.linear_attn.in_proj_qkv.weight": (torch.randn(2 * H, H, generator=g) * 0.1).to(torch.bfloat16),
            f"{P}layers.0.linear_attn.A_log": torch.empty(4).uniform_(0.01, 16, generator=g).log().to(torch.bfloat16),
            "lm_head.weight": (torch.randn(32, H, generator=g) * 0.1).to(torch.bfloat16),
            "mtp.pre_fc_norm_embedding.weight": (torch.randn(H, generator=g) * 0.3).to(torch.bfloat16)}   # the MTP head's names are top-level
    files["model-bf16-00001.safetensors"] = bf16
    # NVFP4 experts: packed nibbles, e4m3 block scales, a global scale, an input scale -- the modelopt four tensors
    experts = {}
    for e in range(E):
        for proj, (out, inner) in (("gate_proj", (I, H)), ("up_proj", (I, H)), ("down_proj", (H, I))):
            base = f"{P}layers.0.mlp.experts.{e}.{proj}"
            experts[f"{base}.weight"] = torch.randint(0, 256, (out, inner // 2), generator=g, dtype=torch.int32).to(torch.uint8)
            experts[f"{base}.weight_scale"] = (torch.rand(out, inner // 16, generator=g) * 4 + 0.5).to(torch.float8_e4m3fn)
            experts[f"{base}.weight_scale_2"] = torch.tensor(0.01 * (e + 1), dtype=torch.float32)
            experts[f"{base}.input_scale"] = torch.tensor(0.002, dtype=torch.float32)
    files["layer-00000-experts-0000-0002.safetensors"] = experts
    # the PLE table: three e4m3 shards of `rows_per_shard` rows, and the table's scalar scale in another shard
    table = f"{P}layers.1.ple.ple_embedding.ngram_embedding"
    shards = {f"{table}.shard_{s}.weight": (torch.randn(rows_per_shard, width, generator=g) * 40).to(torch.float8_e4m3fn)
              for s in range(3)}
    files["model-plefp8-00000.safetensors"] = {k: shards[k] for k in list(shards)[:2]}
    files["model-plefp8-00001.safetensors"] = {list(shards)[2]: shards[list(shards)[2]],
                                               f"{table}.weight_scale": torch.tensor([0.0005], dtype=torch.bfloat16)}
    index = {}
    for file, tensors in files.items():
        write_safetensors(root / file, tensors)
        for name in tensors:
            index[name] = file
        written.update(tensors)
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))
    return {"written": written, "cfg": cfg, "table": table, "rows_per_shard": rows_per_shard, "E": E, "H": H, "I": I}


@unittest.skipUnless(torch is not None, "requires torch")
class WeightsTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name) / "ckpt"
        self.meta = synthetic_checkpoint(self.root)

    def test_bf16_tensors_arrive_by_their_composition_name_in_the_requested_dtype(self):
        from engine.profiles.qwen38.weights import Weights
        w = Weights(self.root, dtype="float32")
        got = w("model.layers.0.linear_attn.in_proj_qkv.weight")
        want = self.meta["written"]["model.language_model.layers.0.linear_attn.in_proj_qkv.weight"]
        self.assertEqual(got.dtype, torch.float32)
        self.assertTrue(torch.equal(got, want.float()))
        self.assertTrue(torch.equal(w("model.layers.0.linear_attn.A_log"), self.meta["written"]["model.language_model.layers.0.linear_attn.A_log"].float()))
        self.assertTrue(torch.equal(w("lm_head.weight"), self.meta["written"]["lm_head.weight"].float()))
        self.assertTrue(torch.equal(w("mtp.pre_fc_norm_embedding.weight"),
                                    self.meta["written"]["mtp.pre_fc_norm_embedding.weight"].float()))
        self.assertIs(w("lm_head.weight"), w("lm_head.weight"))                    # kept, not re-read
        with self.assertRaises(KeyError):
            w("model.layers.0.linear_attn.nothing.weight")
        with self.assertRaises(KeyError):                                          # the whole table is never one tensor
            w("model.layers.1.ple.ple_embedding.ngram_embedding.weight")
        self.assertTrue(w("model.layers.0.linear_attn.A_log").is_contiguous())
        w("model.layers.0.linear_attn.A_log")[0] = 1.0                             # a copy off the map: writable

    def test_an_expert_is_its_four_tensors_dequantised_in_gate_up_down_order(self):
        from engine.modules.moe import dequant_nvfp4
        from engine.profiles.qwen38.weights import Weights
        w = Weights(self.root, dtype="float32", expert_cache=2)
        P = "model.language_model."
        for e in range(self.meta["E"]):
            gate_up, down = w.expert(0, e)
            parts = []
            for proj in ("gate_proj", "up_proj", "down_proj"):
                base = f"{P}layers.0.mlp.experts.{e}.{proj}"
                t = self.meta["written"]
                parts.append(dequant_nvfp4(t[f"{base}.weight"], t[f"{base}.weight_scale"], t[f"{base}.weight_scale_2"]))
            self.assertTrue(torch.equal(gate_up, torch.cat(parts[:2])))
            self.assertTrue(torch.equal(down, parts[2]))
            self.assertEqual((tuple(gate_up.shape), tuple(down.shape)), ((2 * self.meta["I"], self.meta["H"]), (self.meta["H"], self.meta["I"])))
        self.assertEqual(len(w._experts), 2)                                       # bounded: the first was evicted
        self.assertIs(w.expert(0, 2)[0], w.expert(0, 2)[0])

    def test_table_rows_are_gathered_across_shards_and_scaled(self):
        from engine.profiles.qwen38.weights import Weights
        w = Weights(self.root, dtype="float32")
        table, per = self.meta["table"], self.meta["rows_per_shard"]
        whole = torch.cat([self.meta["written"][f"{table}.shard_{s}.weight"].float() for s in range(3)])
        scale = self.meta["written"][f"{table}.weight_scale"].float().item()
        rows = torch.tensor([[0, 7, 14], [5, 4, 13]])                              # rows from every shard, twice over
        got = w.table_rows("model.layers.1.ple.ple_embedding.ngram_embedding", rows)
        self.assertEqual(tuple(got.shape), (2, 3, 4))
        torch.testing.assert_close(got, whole[rows] * scale)
        self.assertLess(float(got.abs().max()), 1.0)                               # the scale is what makes them rows of a table
        self.assertAlmostEqual(float(w.table_scale(f"model.language_model.{table[len('model.language_model.'):]}")), scale, places=9)
        self.assertEqual(w.tables[table], [f"{table}.shard_{s}.weight" for s in range(3)])   # shard order by number
        bf = Weights(self.root, dtype="bfloat16")
        self.assertEqual(bf.table_rows("model.layers.1.ple.ple_embedding.ngram_embedding", rows).dtype, torch.bfloat16)

    def test_an_unscaled_table_is_read_as_it_is(self):
        from engine.profiles.qwen38.weights import Weights
        import os
        table = self.meta["table"]
        # rewrite the last shard without the scale entry
        path = self.root / "model-plefp8-00001.safetensors"
        os.remove(path)
        write_safetensors(path, {f"{table}.shard_2.weight": self.meta["written"][f"{table}.shard_2.weight"]})
        index = json.loads((self.root / "model.safetensors.index.json").read_text())
        index["weight_map"].pop(f"{table}.weight_scale")
        (self.root / "model.safetensors.index.json").write_text(json.dumps(index))
        w = Weights(self.root, dtype="float32")
        self.assertIsNone(w.table_scale(table))
        got = w.table_rows("model.layers.1.ple.ple_embedding.ngram_embedding", torch.tensor([[12]]))
        self.assertTrue(torch.equal(got[0, 0], self.meta["written"][f"{table}.shard_2.weight"][2].float()))


if __name__ == "__main__":
    unittest.main()
