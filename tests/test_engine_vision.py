"""Pictures (45차 §23 A7): the served processor's geometry, frame sampling and patch layout against numbers read
off the production image (tests/fixtures/glm53_vision_reference.json), the placeholder expansion, and the tower's
composition on a toy configuration. Preprocessing tests need PIL/torchvision (the ST image); the rest run anywhere."""
import hashlib
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.profiles.glm53 import vision  # noqa: E402
from engine.profiles.glm53.vision import Door, Vision, VisionFacts, canvas, sample_frame_indices, smart_resize  # noqa: E402

REF = json.loads((Path(__file__).parent / "fixtures" / "glm53_vision_reference.json").read_text())
CKPT = Path("/home/choiceoh/models/glm53-redhat-nvfp4")
MEAN, STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)


def facts(**over) -> VisionFacts:
    base = dict(depth=2, hidden=32, heads=2, inter=48, out_hidden=40, proj_inter=56, patch=14, temporal=2, merge=2, channels=3,
                rms_eps=1e-5, swiglu_limit=10.0, image_token=254, image_start=250, image_end=251, video_token=255, video_start=252,
                video_end=253, mean=MEAN, std=STD, image_min_tokens=16, image_max_tokens=8000, video_min_tokens=16, video_max_tokens=30000,
                video_fps=2.0)
    base.update(over)
    return VisionFacts(**base)


def toy_views(V: VisionFacts, seed=0) -> dict:
    g = torch.Generator().manual_seed(seed)
    out = {}
    for s in vision.specs(V):
        t = torch.randn(s.shape, generator=g) * (0.5 if "norm" in s.name else 0.05)
        if s.name.endswith("norm.weight") or s.name.endswith("norm1.weight") or s.name.endswith("norm2.weight") or s.name.endswith("post_layernorm.weight"):
            t = 1 + t * 0.1
        out[s.name] = t.to(torch.bfloat16)
    return out


def synthetic_rgb(h, w):
    yy, xx = np.mgrid[0:h, 0:w]
    return np.stack([(xx * 255 // (w - 1)), (yy * 255 // (h - 1)), ((xx + yy) % 256)], axis=-1).astype(np.uint8)


class Stamps:
    """A tokenizer for the "N.N seconds" stamps: one token per character (code points below 250)."""
    class Enc:
        def __init__(self, ids):
            self.ids = ids

    def encode(self, text, add_special_tokens=False):
        return self.Enc([ord(c) % 250 for c in text])


class GeometryTests(unittest.TestCase):
    def test_pixel_budgets_and_canvas_sizes_match_the_served_processor(self):
        V = facts()
        self.assertEqual(list(V.pixels("image")), REF["image_budget"])
        self.assertEqual(list(V.pixels("video")), REF["video_budget"])
        lo, hi = V.pixels("image")
        for key, expect in REF["smart_resize_image"].items():
            h, w = (int(x) for x in key.split("x"))
            self.assertEqual(list(smart_resize(2, h, w, 2, 28, 28, lo, hi)), expect, key)
        lo, hi = V.pixels("video")
        for key, expect in REF["smart_resize_video"].items():
            t, h, w = (int(x) for x in key.split("x"))
            self.assertEqual(list(smart_resize(t, h, w, 2, 28, 28, lo, hi)), expect, key)

    def test_frame_sampling_matches_the_served_sampler(self):
        for key, expect in REF["frame_indices"].items():
            total, fps, dur = key.split("/")
            got = sample_frame_indices(int(total), float(fps), float(dur), target_fps=2.0, max_frame_count=2048, temporal=2)
            self.assertEqual(got, expect, key)

    def test_the_largest_shapes_are_what_qualification_encodes(self):
        V = facts()
        n = V.image_max_tokens * 4
        gh = int(n ** 0.5) // 2 * 2
        gw = n // gh // 2 * 2
        self.assertLessEqual(gh * gw, n)
        self.assertGreaterEqual(gh * gw, n * 0.98)


@unittest.skipUnless(all(__import__("importlib").util.find_spec(m) for m in ("PIL", "torchvision")), "PIL + torchvision: the ST image")
class PreprocessingTests(unittest.TestCase):
    """The door's canvas and every rank's patches reproduce the served pixel_values bit for bit."""

    def test_synthetic_image_pixel_values_match_the_served_processor(self):
        import io
        from PIL import Image
        ref = REF["synthetic"]
        h, w = ref["size_hw"]
        buf = io.BytesIO()
        Image.fromarray(synthetic_rgb(h, w), "RGB").save(buf, format="PNG")
        door = Door(facts(), Stamps())
        item = door.prepare_image(buf.getvalue())
        self.assertEqual(list(item["grid"]), ref["grid"])
        self.assertEqual(item["tokens"], ref["grid"][1] * ref["grid"][2] // 4)
        tower = Vision(facts(), toy_views(facts()))
        pv = tower.patches(item["canvas"], item["grid"]).float()          # bf16 on the way to the tower: compare in fp32 before that cast
        raw = tower_patches_fp32(tower, item["canvas"], item["grid"])
        self.assertEqual(list(raw.shape), ref["pixel_values_shape"])
        self.assertEqual(hashlib.sha256(raw.numpy().astype(np.float32).tobytes()).hexdigest(), ref["sha256"])
        self.assertAlmostEqual(float(raw.double().sum()), ref["sum"], places=3)
        self.assertEqual(pv.shape, raw.shape)

    def test_transparency_composites_on_white(self):
        from PIL import Image
        rgba = Image.new("RGBA", (64, 40), (0, 0, 0, 0)); rgba.putpixel((1, 1), (10, 20, 30, 128))
        rgb = vision.to_rgb(rgba)
        self.assertEqual(list(rgb.getpixel((0, 0))), REF["rgba_white"]["corner"])
        self.assertEqual(list(rgb.getpixel((1, 1))), REF["rgba_white"]["px11"])

    def test_synthetic_video_frames_patchify_as_served(self):
        ref = REF["synthetic_video"]
        yy, xx = np.mgrid[0:80, 0:100]
        frames = np.stack([np.roll(np.stack([(xx * 255 // 99), (yy * 255 // 79), np.full((80, 100), i * 30)], -1).astype(np.uint8), i, axis=1)
                           for i in range(8)])                                                  # [8, 80, 100, 3]
        V = facts()
        t = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
        lo, hi = V.pixels("video")
        H, W = smart_resize(8, 80, 100, 2, 28, 28, lo, hi)
        cv = canvas(t, H, W, allow_upscale=8 * 80 * 100 < lo)
        grid = (4, H // 14, W // 14)
        self.assertEqual(list(grid), ref["grid"])
        raw = tower_patches_fp32(Vision(V, toy_views(V)), cv.numpy(), grid)
        self.assertEqual(list(raw.shape), ref["shape"])
        self.assertEqual(hashlib.sha256(raw.numpy().astype(np.float32).tobytes()).hexdigest(), ref["sha256"])

    @unittest.skipUnless(__import__("importlib").util.find_spec("cv2"), "cv2: the ST image")
    def test_the_fixture_clip_yields_frame_pairs_with_stamps(self):
        clip = Path(__file__).parent / "fixtures" / "startup_red_blue.mp4"
        item = Door(facts(), Stamps()).prepare_video(clip.read_bytes())
        t, gh, gw = item["grid"]
        self.assertEqual(item["canvas"].shape[0], 2 * t)
        self.assertEqual(len(item["seconds"]), t)
        self.assertEqual(item["tokens"], t * gh * gw // 4)
        self.assertTrue(all(b >= a for a, b in zip(item["seconds"], item["seconds"][1:])))


def tower_patches_fp32(tower: Vision, canvas_u8, grid) -> torch.Tensor:
    """`Vision.patches` before its bf16 cast: the served pixel_values (float32)."""
    V = tower.V
    x = torch.from_numpy(np.ascontiguousarray(canvas_u8))
    if pad := -x.shape[0] % V.temporal:
        x = torch.cat([x, x[-1:].expand(pad, -1, -1, -1)])
    gt, gh, gw = grid
    x = (x.float() - tower.mean) / tower.std
    x = x.view(gt, V.temporal, V.channels, gh // V.merge, V.merge, V.patch, gw // V.merge, V.merge, V.patch).permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    return x.reshape(gt * gh * gw, V.patch_dim)


class ExpansionTests(unittest.TestCase):
    def test_images_and_videos_expand_into_the_served_placeholders(self):
        V = facts()
        door = Door(V, Stamps())
        image = {"kind": "image", "digest": "a" * 64, "canvas": None, "grid": (1, 4, 4), "tokens": 4}
        video = {"kind": "video", "digest": "b" * 64, "canvas": None, "grid": (2, 2, 2), "tokens": 2, "seconds": [0.0, 1.5]}
        ids = [1, V.image_token, 2, V.video_start, V.video_token, V.video_end, 3]
        out, media = door.expand(ids, [image, video])
        stamp = lambda s: [ord(c) % 250 for c in f"{s:.1f} seconds"]        # noqa: E731
        expect = ([1] + [V.image_token] * 4 + [2, V.video_start]
                  + [V.image_start, V.image_token, V.image_end] + stamp(0.0)
                  + [V.image_start, V.image_token, V.image_end] + stamp(1.5)
                  + [V.video_end, 3])
        self.assertEqual(out, expect)
        self.assertEqual([m["kind"] for m in media], ["image", "video"])
        self.assertEqual(media[0]["positions"], [1, 2, 3, 4])
        first_frame = expect.index(V.image_start) + 1
        self.assertEqual(media[1]["positions"], [first_frame, expect.index(V.image_start, first_frame) + 1])
        self.assertTrue(all(out[p] == V.image_token for m in media for p in m["positions"]))
        with self.assertRaises(ValueError):
            door.expand([V.image_token, V.image_token], [image])          # more placeholders than pictures
        with self.assertRaises(ValueError):
            door.expand([1], [image])                                      # a picture without a placeholder

    def test_limits_are_production_s(self):
        self.assertEqual(Door.limits, {"image": 4, "video": 1})


class TowerTests(unittest.TestCase):
    def test_specs_name_every_tensor_of_the_served_tower(self):
        V = facts(depth=24, hidden=1024, heads=16, inter=4096, out_hidden=4096, proj_inter=10240)
        S = vision.specs(V)
        self.assertEqual(len(S), 347)
        self.assertAlmostEqual(sum(s.numel for s in S) / 1e6, 563.6, places=1)          # 1.13 GB of BF16 per rank
        self.assertTrue(all(s.name.startswith("model.visual.") for s in S))

    def test_frame_groups_attend_within_themselves_and_slices_do_not_change_the_answer(self):
        V = facts()
        tower = Vision(V, toy_views(V))
        gh, gw = 4, 6
        cv = vision.Vision._pattern((4, 3, gh * 14, gw * 14))
        cv[2:] = 255 - cv[2:]                                              # the second pair differs
        whole = tower.encode(cv, (2, gh, gw)).float()
        first = tower.encode(cv[:2], (1, gh, gw)).float()
        second = tower.encode(cv[2:], (1, gh, gw)).float()
        self.assertEqual(whole.shape, (2 * gh * gw // 4, V.out_hidden))
        torch.testing.assert_close(whole[:gh * gw // 4], first, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(whole[gh * gw // 4:], second, atol=2e-2, rtol=2e-2)
        old = vision.SLICE_PATCHES
        try:
            vision.SLICE_PATCHES = gh * gw                                 # one group per slice
            torch.testing.assert_close(tower.encode(cv, (2, gh, gw)).float(), whole, atol=0, rtol=0)
        finally:
            vision.SLICE_PATCHES = old

    def test_rotary_is_two_dimensional_and_in_merge_window_order(self):
        V = facts()
        tower = Vision(V, toy_views(V))
        cos, sin = tower.rotary(4, 6)
        self.assertEqual(cos.shape, (24, V.head_dim // 2))
        self.assertTrue(torch.equal(cos[0], torch.ones_like(cos[0])))       # position (0, 0)
        # merge-window order: the second patch is (0, 1): its row angle is 0, its column angle is 1 * inv_freq
        q = V.head_dim // 4
        self.assertTrue(torch.equal(cos[1, :q], torch.ones(q)))
        self.assertFalse(torch.equal(cos[1, q:], torch.ones(q)))
        # the third patch is (1, 0): row angle 1, column angle 0
        self.assertFalse(torch.equal(cos[2, :q], torch.ones(q)))
        self.assertTrue(torch.equal(cos[2, q:], torch.ones(q)))

    def test_patches_reject_a_canvas_that_disagrees_with_its_grid(self):
        V = facts()
        tower = Vision(V, toy_views(V))
        with self.assertRaises(ValueError):
            tower.patches(np.zeros((2, 3, 28, 28), np.uint8), (1, 2, 4))

    def test_qualification_runs_the_largest_shapes(self):
        V = facts(image_max_tokens=8, video_max_tokens=64)
        paid = Vision(V, toy_views(V)).qualify()
        self.assertEqual(sorted(paid), ["vision/image", "vision/video"])


class StepPatchTests(unittest.TestCase):
    def test_rows_replace_the_embedding_at_their_positions(self):
        from engine.profiles.glm53.net import Step
        ids = torch.tensor([5, 6, 7, 8], dtype=torch.int64)
        rows = torch.ones(2, 3)
        step = Step.prefill(ids, 0, 1, 1, patches=((torch.tensor([1, 2]), rows),))
        self.assertEqual(len(step.patches), 1)
        with self.assertRaises(ValueError):
            Step.prefill(ids, 0, 1, 1, patches=((torch.tensor([1, 4]), rows),))    # outside the step
        with self.assertRaises(ValueError):
            Step.prefill(ids, 0, 1, 1, patches=((torch.tensor([1]), rows),))       # one row per position


@unittest.skipUnless((CKPT / "processor_config.json").exists(), "the GLM-5.3 checkpoint metadata")
class CheckpointFactsTests(unittest.TestCase):
    def test_the_checkpoint_s_facts_are_the_served_ones(self):
        V = vision.load(CKPT)
        self.assertEqual((V.depth, V.hidden, V.heads, V.out_hidden, V.proj_inter, V.head_dim), (24, 1024, 16, 4096, 10240, 64))
        self.assertEqual((V.image_token, V.video_token, V.image_start, V.image_end), (154854, 154855, 154830, 154831))
        self.assertEqual((V.image_max_tokens, V.video_max_tokens), (8000, 30000))
        self.assertEqual(len(vision.specs(V)), 347)


if __name__ == "__main__":
    unittest.main()
