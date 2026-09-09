"""Load the real profile through the launcher library; no launcher execution."""
import os
from pathlib import Path
import shutil
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which("bash")
PROMOTED = ("VLLM_GLM53_TP_SF6_Q0", "VLLM_GLM53_STARTUP_TRIM",
            "VLLM_GLM53_SKIP_UNUSED_GRAPH_PROFILE")
RETAINED_OFF = ("ENABLE_EP", "VLLM_GLM53_EP_PREFILL_LOCAL",
                "VLLM_B12X_EP_WARM_COMPACT", "VLLM_B12X_EP_ZERO_WEIGHT_MICRO",
                "VLLM_B12X_EP_STOCK_TOPK_MICRO", "VLLM_GLM53_B12X_PREFILL_REUSE",
                "VLLM_GLM53_B12X_PREFILL_FC1_N128")
STATIC = "VLLM_GLM53_B12X_STATIC_V2"


def load_profile(overrides=None):
    # A clean environment prevents the invoking shell's experimental flags
    # from contaminating a defaults test. Values are environment data, not code.
    env = {"PATH": os.defpath, "LC_ALL": "C", **(overrides or {})}
    script = r'''set -euo pipefail
source "$1"
ct_load_profile "$2" ENABLE_EP
shift 2
for key in "$@"; do
    printf '%s=%s\n' "$key" "${!key-}"
done
'''
    # Resolve the configured Bash before cleaning PATH; macOS /bin/bash is
    # older than the launcher requires even when a current Bash is installed.
    if BASH is None:
        raise RuntimeError("Bash is required to validate the launcher profile")
    command = [BASH, "--noprofile", "--norc", "-c", script, "profile-default-test",
               str(ROOT / "launchers/lib/common-tp4.sh"), str(ROOT / "profiles/glm53.env"),
               *PROMOTED, *RETAINED_OFF, STATIC]
    result = subprocess.run(command, env=env, check=True, capture_output=True,
                            text=True, timeout=10)
    return dict(line.split("=", 1) for line in result.stdout.splitlines())


class TpQ0ProfileDefaultsTests(unittest.TestCase):
    def test_real_loader_defaults_preserve_tp_backend_and_disable_ep(self):
        actual = load_profile()
        self.assertEqual(actual, {**dict.fromkeys(PROMOTED, "1"),
                                  **dict.fromkeys(RETAINED_OFF, "0"), STATIC: "t,r,sf6"})

    def test_caller_zero_rolls_back_each_promoted_default(self):
        for overrides in ({name: "0"} for name in PROMOTED):
            with self.subTest(overrides=overrides):
                actual = load_profile(overrides)
                self.assertEqual(actual, {**dict.fromkeys(PROMOTED, "1"),
                                          **dict.fromkeys(RETAINED_OFF, "0"),
                                          STATIC: "t,r,sf6", **overrides})
        actual = load_profile(dict.fromkeys(PROMOTED, "0"))
        self.assertEqual(actual, {**dict.fromkeys(PROMOTED + RETAINED_OFF, "0"),
                                  STATIC: "t,r,sf6"})


if __name__ == "__main__":
    unittest.main()
