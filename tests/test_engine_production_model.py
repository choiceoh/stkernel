"""Which model production serves (launchers/st_production.py): the selection, the launch environment, the state.

The part that switches a fleet is exercised in tests/test_engine_fleet_ops.py (ProductionModelTests); this is
the file every production path reads, judged on its own: what no choice means, what a bad choice means, and
which keys one model's launch may never inherit from another's.
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "launchers"))

import st_production  # noqa: E402


class ProductionFiles(unittest.TestCase):
    """The selection, the state and the env files in a temp directory: this box's, never the host's."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        (self.dir / "config").mkdir()
        env = {"ST_PRODUCTION_FILE": str(self.dir / "st-production.json"),
               "ST_PRODUCTION_STATE": str(self.dir / "st-production-state.json"),
               "ST_PROFILE_CONFIG_DIR": str(self.dir / "config")}
        patcher = patch.dict(os.environ, env)
        patcher.start()
        self.addCleanup(patcher.stop)


class ProductionSelectionTests(ProductionFiles):
    def test_nothing_chosen_is_glm53(self):
        """Every box that never chose serves exactly what it served before the selection existed."""
        self.assertEqual(st_production.selected(), "glm53")
        self.assertEqual(st_production.DEFAULT, "glm53")

    def test_an_unknown_or_unreadable_choice_is_glm53_not_an_outage(self):
        path = self.dir / "st-production.json"
        for text in ('{"profile": "qwen39"}', "{not json", '["qwen38"]'):
            with self.subTest(text=text):
                path.write_text(text)
                self.assertEqual(st_production.selected(), "glm53")

    def test_a_choice_is_written_whole_and_read_back(self):
        record = st_production.select("qwen38", by="deneb", note="the operator chose it")
        self.assertEqual(st_production.selected(), "qwen38")
        on_disk = json.loads((self.dir / "st-production.json").read_text())
        self.assertEqual({k: on_disk[k] for k in ("profile", "by", "note")},
                         {"profile": "qwen38", "by": "deneb", "note": "the operator chose it"})
        self.assertEqual(on_disk["at"], record["at"])
        self.assertEqual([p.name for p in self.dir.iterdir() if p.name.startswith(".")], [], "no temp file left")

    def test_an_unknown_profile_is_refused_before_anything_is_written(self):
        with self.assertRaises(ValueError):
            st_production.select("qwen39")
        self.assertFalse((self.dir / "st-production.json").exists())
        self.assertEqual(st_production.main(["select", "qwen39"]), 2)

    def test_the_state_carries_what_production_can_serve(self):
        """Deneb reads this file over ssh to list the models it can choose; the box that serves them says which."""
        st_production.publish_state("glm53", "qwen38", "switching", "glm53 -> qwen38")
        view = st_production.show()
        self.assertEqual((view["state"]["serving"], view["state"]["wanted"], view["state"]["phase"]),
                         ("glm53", "qwen38", "switching"))
        self.assertEqual(view["state"]["profiles"]["qwen38"], {"model": "qwen3.8-flash-next", "container": "st-qwen38"})
        self.assertEqual(view["state"]["profiles"]["glm53"], {"model": "glm-5.3-flash", "container": "st-glm53"})
        st_production.publish_state("none", "glm53", "waiting")
        self.assertEqual(st_production.show()["state"]["serving"], "", "nothing serving is empty, never a name")


class LaunchEnvironmentTests(ProductionFiles):
    """systemd hands the supervisor st-glm53.env. What it holds for GLM-5.3 must not boot another model."""

    def setUp(self):
        super().setUp()
        (self.dir / "config" / "st-glm53.env").write_text(
            "# production GLM-5.3\nST_PRODUCTION=1\nRANKS_DIR=/models/glm-ranks\nCKPT='/models/glm meta'\n"
            "PORT=8000\nST_ENGINE_DIR=/home/choiceoh/st-releases/abc\nST_IMAGE=st-engine:prod-abc\n")

    def test_another_model_clears_every_model_key_and_sets_its_own(self):
        base = {"PATH": "/usr/bin", "ST_PRODUCTION": "1", "RANKS_DIR": "/models/glm-ranks", "CKPT": "/models/glm meta",
                "PORT": "8000", "ST_ENGINE_DIR": "/home/choiceoh/st-releases/abc", "ST_IMAGE": "st-engine:prod-abc"}
        env = st_production.launch_env("qwen38", base)
        self.assertEqual(env["RANKS_DIR"], "/home/choiceoh/models/st-qwen38-tep4")
        for key in ("ST_PRODUCTION", "CKPT", "PORT"):
            self.assertNotIn(key, env, f"{key} is GLM-5.3's launch, not Qwen3.8's")
        self.assertEqual(env["PATH"], "/usr/bin")
        # production's tree and image are production's whichever model it serves
        self.assertEqual((env["ST_ENGINE_DIR"], env["ST_IMAGE"]), ("/home/choiceoh/st-releases/abc", "st-engine:prod-abc"))

    def test_the_shell_form_says_the_same(self):
        lines = st_production.env_lines("qwen38").splitlines()
        self.assertIn("unset CKPT", lines)
        self.assertIn("unset ST_PRODUCTION", lines)
        self.assertIn("export RANKS_DIR=/home/choiceoh/models/st-qwen38-tep4", lines)
        self.assertFalse(any("ST_ENGINE_DIR" in line or "ST_IMAGE" in line for line in lines))

    def test_glm53_launches_on_exactly_its_file(self):
        lines = st_production.env_lines("glm53").splitlines()
        self.assertEqual([line for line in lines if line.startswith("unset")], [])
        self.assertIn("export RANKS_DIR=/models/glm-ranks", lines)
        self.assertIn("export CKPT='/models/glm meta'", lines, "quotes stripped once, then shell-quoted")

    def test_a_profile_s_own_file_wins_over_its_defaults(self):
        (self.dir / "config" / "st-qwen38.env").write_text("RANKS_DIR=/models/qwen-other\nST_SPEC_K=3\n")
        env = st_production.launch_env("qwen38", {"ST_SPEC_K": "1"})
        self.assertEqual((env["RANKS_DIR"], env["ST_SPEC_K"]), ("/models/qwen-other", "3"))
        self.assertNotIn("ST_SPEC_K", st_production.launch_env("glm53", {"ST_SPEC_K": "3"}),
                         "and Qwen3.8's knob does not reach GLM-5.3's launch either")


if __name__ == "__main__":
    unittest.main()
