"""engine/profiles/qwen38/vision: Qwen3.8's pictures against what the served image computes (tests/fixtures/
qwen38_vision_reference.json, written by probes/qwen38_vision_reference.py inside vllm/vllm-openai:qwen38-flash-next):
the facts and tensor list, the processor's grid and pixel_values bit for bit, the mRoPE positions, the placeholder
expansion, and the tower on a toy configuration against transformers' Qwen3VLVisionModel. Preprocessing tests need
PIL/torchvision (the ST image); the rest run anywhere.

    python3 -m unittest tests.test_engine_qwen38_vision
"""
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.profiles.qwen38 import vision  # noqa: E402
from probes import qwen38_vision_reference as probe  # noqa: E402

REF = json.loads((Path(__file__).parent / "fixtures" / "qwen38_vision_reference.json").read_text())
PICTURES = importlib.util.find_spec("PIL") is not None and importlib.util.find_spec("torchvision") is not None


def facts() -> vision.VisionFacts:
    """The checkpoint's facts: the repo's copy of its config.json and the fixture's preprocessor_config.json."""
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "config.json").write_text((ROOT / "probes" / "qwen38_config.json").read_text())
        (Path(d) / "preprocessor_config.json").write_text(json.dumps(REF["preprocessor_config"]))
        return vision.load(d)


class FactsTests(unittest.TestCase):
    def test_the_checkpoint_s_tower(self):
        V = facts()
        self.assertEqual((V.depth, V.hidden, V.heads, V.inter, V.out_hidden), (27, 1152, 16, 4304, 2560))
        self.assertEqual((V.patch, V.merge, V.temporal, V.grid_side, V.factor, V.patch_dim), (16, 2, 2, 48, 32, 1536))
        self.assertEqual((V.image_token, V.video_token, V.vision_start, V.vision_end), (248056, 248057, 248053, 248054))
        self.assertEqual((V.min_pixels, V.max_pixels), (65536, 16777216))

    def test_the_tensor_list_is_the_checkpoint_s(self):
        S = vision.specs(facts())
        self.assertEqual(len(S), 333)                               # 3 + 27 x 12 + 6: every model.visual.* tensor
        self.assertEqual(len({s.name for s in S}), 333)
        self.assertTrue(all(s.name.startswith(vision.PREFIX) and s.dtype == torch.bfloat16 for s in S))
        params = sum(int(np.prod(s.shape)) for s in S)
        self.assertEqual(params, 1152 * 1536 + 1152 + 2304 * 1152 + 27 * (4 * 1152 + 3 * 1152 * 1152 + 3 * 1152 + 1152 * 1152
                                                                          + 1152 + 2 * 1152 * 4304 + 4304 + 1152)
                         + 2 * 1152 + 4608 * 4608 + 4608 + 2560 * 4608 + 2560)

    def test_a_language_model_only_checkpoint_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            c = json.loads((ROOT / "probes" / "qwen38_config.json").read_text())
            c["language_model_only"] = True
            (Path(d) / "config.json").write_text(json.dumps(c))
            (Path(d) / "preprocessor_config.json").write_text(json.dumps(REF["preprocessor_config"]))
            with self.assertRaises(AssertionError):
                vision.load(d)


class GeometryTests(unittest.TestCase):
    def test_every_case_s_grid_is_the_processor_s(self):
        V = facts()
        for name, ref in REF["processor"]["cases"].items():
            w, h = ref["loaded_size"]
            H, W = vision.smart_resize(h, w, V.factor, V.min_pixels, V.max_pixels)
            with self.subTest(name):
                self.assertEqual([1, H // V.patch, W // V.patch], ref["grid"])
                self.assertEqual(V.tokens(ref["grid"]) * V.merge * V.merge, ref["shape"][0])

    def test_an_extreme_aspect_is_refused(self):
        with self.assertRaises(ValueError):
            vision.smart_resize(10, 2100, 32, 65536, 16777216)


@unittest.skipUnless(PICTURES, "PIL and torchvision (the ST image)")
class ProcessorTests(unittest.TestCase):
    def test_pixel_values_are_the_processor_s_bit_for_bit(self):
        V = facts()
        door = vision.Door(V)
        for name, data in probe.cases().items():
            ref = REF["processor"]["cases"][name]
            with self.subTest(name):
                self.assertEqual(hashlib.sha256(data).hexdigest(), ref["bytes_sha256"], "the picture's encoder moved")
                item = door.prepare("image", data)
                self.assertEqual(list(item["grid"]), ref["grid"])
                self.assertEqual(item["tokens"], ref["shape"][0] // 4)
                pv = vision.pixel_values(V, item["canvas"], item["grid"])
                self.assertEqual(list(pv.shape), ref["shape"])
                self.assertEqual(hashlib.sha256(pv.contiguous().numpy().tobytes()).hexdigest(), ref["sha256"])

    def test_video_is_refused_at_the_door(self):
        with self.assertRaises(ValueError):
            vision.Door(facts()).prepare("video", b"")


class ExpansionTests(unittest.TestCase):
    def item(self, grid, digest="d"):
        return {"kind": "image", "digest": digest, "canvas": np.zeros((1, 3, grid[1] * 16, grid[2] * 16), np.uint8),
                "grid": grid, "tokens": grid[1] * grid[2] // 4}

    def test_each_placeholder_becomes_its_picture_s_run(self):
        V = facts()
        S, I, E = V.vision_start, V.image_token, V.vision_end
        ids = [1, 2, S, I, E, 3, S, I, E, 4]
        out, media = vision.Door(V).expand(ids, [self.item((1, 4, 6), "a"), self.item((1, 2, 2), "b")])
        self.assertEqual(out, [1, 2, S] + [I] * 6 + [E, 3, S, I, E, 4])
        self.assertEqual([m["positions"] for m in media], [list(range(3, 9)), [12]])
        self.assertEqual([m["digest"] for m in media], ["a", "b"])
        self.assertEqual(media[0]["grid"], (1, 4, 6))

    def test_mismatches_are_refused(self):
        V = facts()
        S, I, E = V.vision_start, V.image_token, V.vision_end
        door = vision.Door(V)
        for ids, items in (([S, I, E, S, I, E], [self.item((1, 2, 2))]),           # more placeholders than pictures
                           ([S, I, E], [self.item((1, 2, 2))] * 2),               # more pictures than placeholders
                           ([I], [self.item((1, 2, 2))]),                          # outside the markers
                           ([S, V.video_token, E], [])):                           # a video
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                door.expand(ids, items)


class RopePositionTests(unittest.TestCase):
    def test_positions_and_delta_are_vllm_s(self):
        for name, ref in REF["mrope"].items():
            pos, delta = vision.rope_positions(ref["length"], ref["media"], 2)
            with self.subTest(name):
                self.assertEqual(pos.tolist(), ref["positions"])
                self.assertEqual(delta, ref["delta"])

    def test_text_alone_is_the_ordinary_index(self):
        pos, delta = vision.rope_positions(7, [], 2)
        self.assertEqual(pos.tolist(), [list(range(7))] * 3)
        self.assertEqual(delta, 0)

    def test_a_run_that_disagrees_with_its_grid_is_refused(self):
        with self.assertRaises(ValueError):
            vision.rope_positions(20, [{"positions": list(range(2, 7)), "grid": (1, 4, 6)}], 2)


class PreshardTests(unittest.TestCase):
    def test_the_tower_file_holds_the_checkpoint_s_tensors(self):
        """preshard --vision on a toy checkpoint: vision.safetensors next to the rank files, every tensor's bytes kept."""
        from safetensors.torch import load_file, save_file
        from engine.profiles.qwen38 import preshard
        V = probe.toy_facts()
        w = probe.toy_weights(vision.specs(V))
        with tempfile.TemporaryDirectory() as d:
            ckpt, out = Path(d) / "ckpt", Path(d) / "ranks"
            ckpt.mkdir()
            out.mkdir()
            c = json.loads((ROOT / "probes" / "qwen38_config.json").read_text())
            c["vision_config"] = dict(c["vision_config"], **probe.TOY)
            c["text_config"] = dict(c["text_config"], hidden_size=V.out_hidden)
            (ckpt / "config.json").write_text(json.dumps(c))
            (ckpt / "preprocessor_config.json").write_text(json.dumps(dict(REF["preprocessor_config"], patch_size=V.patch)))
            save_file(dict(w), str(ckpt / "vision-00001.safetensors"))
            weight_map = {k: "vision-00001.safetensors" for k in w} | {"model.language_model.layers.0.x": "absent.safetensors"}
            (ckpt / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
            self.assertEqual(preshard.main(["--ckpt", str(ckpt), "--out", str(out), "--vision"]), 0)
            got = {k: t for k, t in load_file(str(out / vision.FILE)).items() if not k.startswith("__st_padding__")}   # alignment
            self.assertEqual(set(got), set(w))
            self.assertTrue(all(torch.equal(got[k], w[k]) for k in w))


class TowerTests(unittest.TestCase):
    def test_the_toy_tower_is_transformers(self):
        ref = REF["toy_tower"]
        V = probe.toy_facts()
        w = probe.toy_weights(vision.specs(V))
        self.assertEqual(probe.digest(w), ref["weights_sha256"], "the toy weights' generator moved")
        got = vision.Vision(V, w).tower(probe.toy_pixels(), tuple(ref["grid"])).float()
        want = torch.tensor(ref["rows"])
        self.assertEqual(got.shape, want.shape)
        self.assertLess(float((got - want).norm() / want.norm()), 2e-2)      # bf16 weights and activations against fp32
        self.assertGreater(float(torch.nn.functional.cosine_similarity(got, want, dim=-1).min()), 0.999)

    def test_the_ranks_must_agree(self):
        V = probe.toy_facts()
        w = probe.toy_weights(vision.specs(V))

        class Comm:
            world_size = 4

            @staticmethod
            def all_reduce_max(t):
                return t + 1                                        # another rank's sum was larger

        tower = vision.Vision(V, w, comm=Comm())
        canvas = np.zeros((1, 3, 16, 24), np.uint8)
        with self.assertRaises(RuntimeError):
            tower.encode(canvas, (1, 4, 6))


if __name__ == "__main__":
    unittest.main()
