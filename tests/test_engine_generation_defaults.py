"""A request that says nothing samples the way the model ships, not the way a quantiser's re-generated file says.

zai-org/GLM-5.3-Flash's generation_config.json is do_sample, temperature 1.0, top_p 0.95. The quantised
repositories the served meta is cut from regenerate the file from config.json and drop top_p, so the door
used to sample the whole tail at temperature 1 for any client that omitted it. facts.GENERATION holds the
model's defaults; boot.generation_defaults fills what the meta omits and lets the meta win where it speaks.
"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


def _meta(where, **generation):
    (Path(where) / "generation_config.json").write_text(json.dumps(dict(eos_token_id=[1], **generation)))
    return where


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "requires PyTorch")
class GenerationDefaultsTests(unittest.TestCase):
    def test_the_profile_supplies_what_the_quantised_meta_dropped(self):
        from engine.profiles.glm53 import boot, facts
        self.assertEqual(facts.GENERATION, {"temperature": 1.0, "top_p": 0.95})
        with tempfile.TemporaryDirectory() as where:
            self.assertEqual(boot.generation_defaults(_meta(where, temperature=1.0)), {"temperature": 1.0, "top_p": 0.95})
            self.assertEqual(boot.generation_defaults(_meta(where)), {"temperature": 1.0, "top_p": 0.95})

    def test_a_meta_that_speaks_wins(self):
        from engine.profiles.glm53 import boot
        with tempfile.TemporaryDirectory() as where:
            got = boot.generation_defaults(_meta(where, temperature=0.7, top_p=0.9, top_k=40, repetition_penalty=1.1))
        self.assertEqual(got, {"temperature": 0.7, "top_p": 0.9, "top_k": 40, "repetition_penalty": 1.1})

    def test_a_silent_request_gets_the_models_nucleus_and_an_explicit_one_keeps_its_own(self):
        from engine.base.serve import sampling_options
        defaults = {"temperature": 1.0, "top_p": 0.95}
        self.assertEqual(sampling_options({}, defaults), (1.0, {"top_p": 0.95}))
        self.assertEqual(sampling_options({"temperature": 0.3}, defaults), (0.3, {"top_p": 0.95}))
        # a client that asks for the whole tail (top_p 1, as Deneb does) still gets it
        self.assertEqual(sampling_options({"top_p": 1}, defaults), (1.0, {}))
        self.assertEqual(sampling_options({"top_p": 0.8}, defaults), (1.0, {"top_p": 0.8}))


if __name__ == "__main__":
    unittest.main()
