"""CPU-only literal Docker command fixtures; never process or GPU proof."""
import base64
import hashlib
import json
from pathlib import Path
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import glm53_launch_metadata as m


def command(extra="", *, rank=0, prelude=m._GID_PRELUDE, body=None):
    if body is None:
        body = ("vllm serve /models/private-model --tensor-parallel-size 4 "
                f"--nnodes 4 --node-rank {rank} --max-model-len 1048576 "
                f"--num-gpu-blocks-override 1056 {extra}")
    script = prelude + body + " > /glmlogs/glm53.log 2>&1\n"
    encoded = base64.b64encode(script.encode()).decode()
    return ["-c", f"echo {encoded} | base64 -d > /tmp/serve.sh; bash /tmp/serve.sh"]


class LaunchMetadataTests(unittest.TestCase):
    def test_known_prelude_matches_current_launcher_literal(self):
        source = (ROOT / "launchers/lib/common-tp4.sh").read_text()
        match = re.search(r"CT_GID_PRELUDE=\$\(cat <<'GIDEOF'\n(.*?)\nGIDEOF\n\)", source, re.S)
        self.assertIsNotNone(match)
        self.assertEqual(m._GID_PRELUDE, match[1] + "\n")

    def test_current_head_worker_and_literal_json_are_supported_without_leaking_argv(self):
        extra = "--profiler-config '{\"profiler\":\"torch\",\"torch_profiler_dir\":\"/private/prof\"}'"
        for rank in range(4):
            with self.subTest(rank=rank):
                result = m.launch_parallelism(command(extra + (" --headless" if rank else ""), rank=rank))
                self.assertEqual((result["enabled"], result["tensor_parallel_size"], result["nnodes"], result["node_rank"]),
                                 (False, 4, 4, rank))
                self.assertEqual(set(result), {"schema", "source", "enabled", "tensor_parallel_size", "nnodes", "node_rank",
                                              "command_sha256", "prelude_sha256", "serve_argv_sha256", "serve_argv_without_ep_sha256"})
                self.assertEqual(result["schema"], 1)
                self.assertEqual(result["source"], "container-launch-script")
                self.assertNotIn("private", json.dumps(result))
                self.assertEqual(result["prelude_sha256"], hashlib.sha256(m._GID_PRELUDE.encode()).hexdigest())

    def test_only_exact_ep_argument_is_removed_from_comparison_hash(self):
        baseline = m.launch_parallelism(command())
        candidate = m.launch_parallelism(command("--enable-expert-parallel"))
        self.assertTrue(candidate["enabled"])
        for key in ("command_sha256", "serve_argv_sha256"):
            self.assertNotEqual(baseline[key], candidate[key])
        self.assertEqual(baseline["serve_argv_without_ep_sha256"], candidate["serve_argv_without_ep_sha256"])
        for changed in (command(rank=1), command("--headless"), command(body="vllm serve /models/other --tensor-parallel-size 4 --nnodes 4 --node-rank 0 --max-model-len 262144")):
            self.assertNotEqual(baseline["serve_argv_without_ep_sha256"], m.launch_parallelism(changed)["serve_argv_without_ep_sha256"])

    def test_flag_inside_literal_json_is_not_activation(self):
        result = m.launch_parallelism(command("--profiler-config '{\"note\":\"--enable-expert-parallel\",\"literal\":\"$(not executed); --node-rank 3\"}'"))
        self.assertFalse(result["enabled"])
        self.assertEqual(result["serve_argv_sha256"], result["serve_argv_without_ep_sha256"])

    def test_missing_option_values_cannot_turn_into_ep_activation(self):
        for extra in ("--served-model-name --enable-expert-parallel", "--profiler-config --enable-expert-parallel",
                      "--served-model-name=--enable-expert-parallel", "--unknown-flag --enable-expert-parallel",
                      "--enable_expert_parallel", "--tensor_parallel_size 8"):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                m.launch_parallelism(command(extra))

    def test_duplicate_ambiguous_or_invalid_parallel_options_are_rejected(self):
        for extra in ("--enable-expert-parallel --enable-expert-parallel", "--enable-expert-parallel=false",
                      "--enable-expert-parallel false", "--no-enable-expert-parallel", "--disable-expert-parallel",
                      "--enable-expert", "-ep", "--tensor-parallel-size 4", "--tensor-parallel-size=8",
                      "--tensor-parallel 8", "-tp 8", "--nnodes=4", "--node-rank 0", "-- --enable-expert-parallel"):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                m.launch_parallelism(command(extra))
        for body in ("vllm serve model --nnodes 4 --node-rank 0", "vllm serve model --tensor-parallel-size 0 --nnodes 4 --node-rank 0",
                     "vllm serve model --tensor-parallel-size 4 --nnodes 0 --node-rank 0",
                     "vllm serve model --tensor-parallel-size 4 --nnodes 4 --node-rank 4",
                     "vllm serve model --tensor-parallel-size 4 --nnodes 4 --node-rank -1"):
            with self.subTest(body=body), self.assertRaises(ValueError):
                m.launch_parallelism(command(body=body))

    def test_shell_evaluation_and_multiple_commands_are_rejected(self):
        for extra in ("; echo ignored", "&& true", "| cat", "$(echo --enable-expert-parallel)",
                      '"$(echo --enable-expert-parallel)"', "`echo hidden`", "$EXTRA_FLAGS", "# comment",
                      "\nvllm serve model", " > other.log", "${FLAGS}", "--model *", "--flag 'unterminated"):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                m.launch_parallelism(command(extra))
        for prelude in ("", "echo skipped\n" + m._GID_PRELUDE, m._GID_PRELUDE.replace("seq 0 15", "seq 0 16")):
            with self.subTest(prelude=prelude), self.assertRaises(ValueError):
                m.launch_parallelism(command(prelude=prelude))

    def test_malformed_wrappers_and_encodings_raise_instead_of_defaulting_off(self):
        valid = command()
        for value in (None, [], valid[1], ["-c", valid[1], "extra"], ["-lc", valid[1]],
                      ["-c", valid[1] + "; true"], ["-c", valid[1].replace("base64 -d", "base64 --decode")],
                      ["-c", "echo !!! | base64 -d > /tmp/serve.sh; bash /tmp/serve.sh"],
                      ["-c", "echo a | base64 -d > /tmp/serve.sh; bash /tmp/serve.sh"],
                      ["-c", "echo /w== | base64 -d > /tmp/serve.sh; bash /tmp/serve.sh"]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                m.launch_parallelism(value)


if __name__ == "__main__":
    unittest.main()
