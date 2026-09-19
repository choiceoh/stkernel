"""The benchmark must record the model it actually targets, even beside GLM."""
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


BENCH = Path(__file__).resolve().parents[1] / "bench"
sys.path.insert(0, str(BENCH))
import onepass


class ContainerIdentityTests(unittest.TestCase):
    def test_explicit_qwen_identity_does_not_report_resident_glm(self):
        with patch.dict(os.environ, {"ONEPASS_ST_CONTAINER": "st-qwen38"}), \
             patch("subprocess.run", return_value=SimpleNamespace(stdout="st-glm53\nst-qwen38\n")), \
             patch.object(onepass, "_st_build", return_value={"boot_id": "qwen"}) as inspect:
            self.assertEqual(onepass._served_build("unused"), {"boot_id": "qwen"})
            inspect.assert_called_once_with(["st-glm53", "st-qwen38"], name="st-qwen38")

    def test_missing_explicit_container_never_borrows_glm_identity(self):
        with patch.dict(os.environ, {"ONEPASS_ST_CONTAINER": "st-qwen38"}), \
             patch("subprocess.run", return_value=SimpleNamespace(stdout="st-glm53\n")), \
             patch.object(onepass, "_st_build") as inspect:
            self.assertEqual(onepass._served_build("unused"), {})
            inspect.assert_not_called()

    def test_default_remains_glm(self):
        with patch.dict(os.environ, {}, clear=True), \
             patch("subprocess.run", return_value=SimpleNamespace(stdout="st-glm53\n")), \
             patch.object(onepass, "_st_build", return_value={"boot_id": "glm"}) as inspect:
            self.assertEqual(onepass._served_build("unused"), {"boot_id": "glm"})
            inspect.assert_called_once_with(["st-glm53"], name="st-glm53")


if __name__ == "__main__":
    unittest.main()
