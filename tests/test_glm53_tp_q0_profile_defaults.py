"""Load the real profile through the launcher library; no launcher execution."""
import os
from pathlib import Path
import shutil
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which("bash")
PROMOTED = ("ENABLE_EP", "VLLM_GLM53_EP_TILED", "VLLM_GLM53_STARTUP_TRIM",
            "VLLM_GLM53_SKIP_UNUSED_GRAPH_PROFILE")
RETAINED_OFF = ("VLLM_GLM53_TP_SF6_Q0", "VLLM_GLM53_EP_PREFILL_LOCAL",
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
ct_load_profile "$2" ENABLE_EP SPEC_K
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
               *PROMOTED, *RETAINED_OFF, STATIC, "SPEC_K", "VLLM_GLM53_PREP_FUSED"]
    result = subprocess.run(command, env=env, check=True, capture_output=True,
                            text=True, timeout=10)
    return dict(line.split("=", 1) for line in result.stdout.splitlines())


class EpTiledProfileDefaultsTests(unittest.TestCase):
    def test_real_loader_selects_ep_sf6_and_preserves_other_defaults(self):
        self.assertEqual(load_profile(), {**dict.fromkeys(PROMOTED, "1"),
            **dict.fromkeys(RETAINED_OFF, "0"), STATIC: "t,r,sf6", "SPEC_K":"5", "VLLM_GLM53_PREP_FUSED":"1"})

    def test_explicit_tp_rollback_and_startup_overrides_survive_loader(self):
        rollback = {"ENABLE_EP":"0", "VLLM_GLM53_EP_TILED":"0", "VLLM_GLM53_TP_SF6_Q0":"1", "SPEC_K":"5"}
        for overrides in (rollback, {**rollback, "VLLM_GLM53_STARTUP_TRIM":"0",
                                      "VLLM_GLM53_SKIP_UNUSED_GRAPH_PROFILE":"0"},
                          {**rollback, "VLLM_GLM53_PREP_FUSED":"0"}):
            with self.subTest(overrides=overrides):
                self.assertEqual(load_profile(overrides), {**dict.fromkeys(PROMOTED,"1"),
                    **dict.fromkeys(RETAINED_OFF,"0"), STATIC:"t,r,sf6", "SPEC_K":"5", "VLLM_GLM53_PREP_FUSED":"1", **overrides})


if __name__ == "__main__":
    unittest.main()
