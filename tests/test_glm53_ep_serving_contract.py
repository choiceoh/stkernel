"""Literal CPU fixtures for capacity and matched-arm configuration contracts."""
import base64
import copy
import json
from pathlib import Path
import sys
import unittest
import os
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import glm53_ep_serving_contract as m
import glm53_launch_metadata as launch


DEFAULT_COMPILATION = '{"cudagraph_mode":"FULL_DECODE_ONLY","custom_ops":["all"]}'


def command(*, rank=0, enabled=False, replace=None, extra="", eager=False, compilation=DEFAULT_COMPILATION):
    values = {"host": "127.0.0.1", "port": "18000", "max-model-len": "1048576",
              "max-num-seqs": "4", "max-num-batched-tokens": "8192", "block-size": "2304",
              "num-gpu-blocks-override": "1056", "gpu-memory-utilization": "0.6229"}
    values.update(replace or {})
    body = ("vllm serve /models/private-model --tensor-parallel-size 4 --nnodes 4 "
            f"--node-rank {rank} " + " ".join(f"--{key} {value}" for key, value in values.items()))
    body += " --enable-prefix-caching"
    body += (" --enforce-eager" if eager else
             " --max-cudagraph-capture-size 32 --compilation-config '" + compilation + "'")
    body += " --enable-expert-parallel" if enabled else ""
    body += " --headless" if rank else ""
    script = launch._GID_PRELUDE + body + " " + extra + " > /glmlogs/glm53.log 2>&1\n"
    encoded = base64.b64encode(script.encode()).decode()
    return ["-c", f"echo {encoded} | base64 -d > /tmp/serve.sh; bash /tmp/serve.sh"]


def fixture(enabled=False, *, replace=None, eager=False, compilation=DEFAULT_COMPILATION):
    knobs = {m.EP_LOCAL: str(int(enabled)), m.EP_COMPACT: "1" if enabled else None}
    nodes = {}
    for rank, node in enumerate(m.NODES):
        env = dict(m.COMMON_KNOBS, **{key: value for key, value in knobs.items() if value is not None},
                   PRIVATE_TOKEN="secret-do-not-print", NODE_SPECIFIC=str(rank))
        nodes[node] = dict(cmd=command(rank=rank, enabled=enabled, replace=replace, eager=eager, compilation=compilation),
                           env=[key + "=" + value for key, value in env.items()])
    return nodes, knobs


KV = {"KV_TOKENS": "2000000", "KV_HYBRID_BLOCKS": "187"}


def arm(enabled=False, **kwargs):
    nodes, knobs = fixture(enabled, **kwargs)
    return m.configured_arm(nodes, enabled=enabled, expected_knobs=knobs, kv_witness=KV)


def public_capacity(**kwargs):
    return {node: m.launch_capacity(command(rank=rank, replace={"host": "0.0.0.0", "port": "8000"}, **kwargs), public=True)
            for rank, node in enumerate(m.NODES)}


def compare(arms, *, incoming=None):
    return m.assert_matched_arms(arms, incoming=public_capacity() if incoming is None else incoming)


class EpServingContractTests(unittest.TestCase):
    def test_actual_capacity_and_corrected_gmu_survive_all_three_arms(self):
        arms = [arm(), arm(True), arm()]
        compare(arms)
        controls = arms[0]["controls"]
        self.assertEqual(controls["MAX_LEN"], "1048576")
        self.assertEqual(controls["KV_TOKENS"], "2000000")
        self.assertEqual(controls["GMU"], "0.6229")
        self.assertEqual(controls["CG_UTIL_DELTA"], "0")
        self.assertNotIn("KV_BLOCKS", controls)
        self.assertNotIn("private", json.dumps(arms))
        self.assertNotIn("secret-do-not-print", json.dumps(arms))
        eager = arm(eager=True)
        self.assertEqual(eager["controls"]["EAGER"], "1")
        self.assertNotIn("GRAPH_CAP", eager["controls"])
        self.assertIsNone(eager["nodes"][m.NODES[0]]["capacity"]["compilation_config"])
        self.assertNotIn("COMPILE_CFG", eager["controls"])
        incoming = m.launch_capacity(command(replace={"host": "0.0.0.0", "port": "8000"}), public=True)
        self.assertEqual(m.replay_controls(incoming, KV), controls)

    def test_actual_capacity_change_or_private_endpoint_drift_is_rejected(self):
        for change in ({"max-model-len": "262144"}, {"gpu-memory-utilization": "0.6158"},
                       {"max-num-batched-tokens": "4096"}, {"max-num-seqs": "8"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                compare([arm(), arm(True, replace=change), arm()])
        for change in ({"host": "0.0.0.0"}, {"port": "8000"}, {"num-gpu-blocks-override": "415"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                arm(True, replace=change)

    def test_requested_ep_env_cannot_replace_actual_flag_or_rank_agreement(self):
        nodes, knobs = fixture(True)
        for bad in (command(rank=3), command(rank=2, enabled=True)):
            changed = copy.deepcopy(nodes)
            changed[m.NODES[3]]["cmd"] = bad
            changed[m.NODES[3]]["env"].append("ENABLE_EP=1")
            with self.assertRaisesRegex(ValueError, "EP/TP/rank"):
                m.configured_arm(changed, enabled=True, expected_knobs=knobs, kv_witness=KV)
        with self.assertRaisesRegex(ValueError, "exactly four"):
            m.configured_arm({m.NODES[0]: nodes[m.NODES[0]]}, enabled=True, expected_knobs=knobs, kv_witness=KV)

    def test_only_explicit_two_knobs_may_change(self):
        nodes, knobs = fixture(True)
        for extra in ({"ENABLE_EP": "1"}, {"VLLM_GLM53_PREFILL_SP_FP8": "0"}):
            with self.assertRaisesRegex(ValueError, "allowlisted"):
                m.configured_arm(nodes, enabled=True, expected_knobs={**knobs, **extra}, kv_witness=KV)
        for entry in ("UNRELATED_KNOB=1", "PRIVATE_TOKEN=changed", "VLLM_GLM53_PREFILL_SP_FP8=0"):
            changed = copy.deepcopy(nodes)
            key = entry.split("=", 1)[0]
            changed[m.NODES[2]]["env"] = [v for v in changed[m.NODES[2]]["env"] if not v.startswith(key + "=")] + [entry]
            with self.subTest(entry=entry), self.assertRaises(ValueError):
                candidate = m.configured_arm(changed, enabled=True, expected_knobs=knobs, kv_witness=KV)
                compare([arm(), candidate, arm()])
        missing = {m.EP_LOCAL: "1", m.EP_COMPACT: None}
        with self.assertRaisesRegex(ValueError, "allowlisted"):
            m.configured_arm(nodes, enabled=True, expected_knobs=missing, kv_witness=KV)

    def test_unrelated_serve_options_are_not_erased_with_ep_flag(self):
        nodes, knobs = fixture(True)
        for extra in ("--attention-backend OTHER", "--speculative-config '{\"num_speculative_tokens\":3}'"):
            changed = copy.deepcopy(nodes)
            changed[m.NODES[1]]["cmd"] = command(rank=1, enabled=True, extra=extra)
            candidate = m.configured_arm(changed, enabled=True, expected_knobs=knobs, kv_witness=KV)
            with self.assertRaisesRegex(ValueError, "beyond EP"):
                compare([arm(), candidate, arm()])

    def test_duplicate_and_unknown_commands_and_environment_fail_closed(self):
        for extra in ("--max-model-len=1048576", "--num-gpu-blocks-override 1056", "--enforce-eager",
                      "--no-enable-prefix-caching", "--kv-cache-memory-bytes 1234", "--unknown 1",
                      "--max-cudagraph-capture-size 32", "; echo ignored"):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                m.launch_capacity(command(extra=extra))
        # Current launcher can append two distinct middleware entries.
        m.launch_capacity(command(extra="--middleware first --middleware second"))
        nodes, knobs = fixture()
        for entry in ("PRIVATE_TOKEN=other", "=empty", "missing-equals"):
            changed = copy.deepcopy(nodes)
            changed[m.NODES[0]]["env"].append(entry)
            with self.subTest(entry=entry), self.assertRaises(ValueError):
                m.configured_arm(changed, enabled=False, expected_knobs=knobs, kv_witness=KV)

    def test_missing_nonfinite_or_ambiguous_capacity_does_not_default(self):
        for key, bad in (("max-model-len", "0"), ("max-num-seqs", "-1"), ("max-num-batched-tokens", "8192.0"),
                         ("block-size", "128"), ("gpu-memory-utilization", "NaN"),
                         ("gpu-memory-utilization", "Infinity"), ("gpu-memory-utilization", "1")):
            with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                m.launch_capacity(command(replace={key: bad}))
        decoded = base64.b64decode(launch._WRAPPER.fullmatch(command()[1])[1]).decode()
        for missing in ("--max-model-len 1048576 ", "--num-gpu-blocks-override 1056 ",
                        "--max-cudagraph-capture-size 32 ", "--enable-prefix-caching "):
            raw = decoded.replace(missing, "")
            encoded = base64.b64encode(raw.encode()).decode()
            cmd = ["-c", f"echo {encoded} | base64 -d > /tmp/serve.sh; bash /tmp/serve.sh"]
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                m.launch_capacity(cmd)

    def test_kv_inputs_are_required_and_match_launcher_ceil_with_hybrid_reserve(self):
        configured = m.launch_capacity(command())
        for witness in ({}, {"KV_BLOCKS": "1056"}, {**KV, "KV_BLOCKS": "1056"},
                        {**KV, "KV_TOKENS": "auto"}, {**KV, "KV_HYBRID_BLOCKS": "186"},
                        {**KV, "KV_TOKENS": 2000000}):
            with self.subTest(witness=witness), self.assertRaises(ValueError):
                m.replay_controls(configured, witness)
        for tokens, expected in ((2303, 188), (2304, 188), (2305, 189)):
            configured = m.launch_capacity(command(replace={"num-gpu-blocks-override": str(expected)}))
            controls = m.replay_controls(configured, {**KV, "KV_TOKENS": str(tokens)})
            self.assertEqual(controls["KV_TOKENS"], str(tokens))
        launcher = (ROOT / "launchers/start-glm53-nvfp4-tp4.sh").read_text()
        self.assertIn('int(($KV_TOKENS + 2303) / 2304) + $KV_HYBRID_BLOCKS', launcher)

    def test_incomplete_or_internally_inconsistent_persisted_arms_are_rejected(self):
        for change in ("enabled", "node_rank", "knobs", "digest", "missing", "endpoint"):
            arms = [arm(), arm(True), arm()]
            candidate = arms[1]
            node = candidate["nodes"][m.NODES[0]]
            if change == "enabled":
                node["parallelism"]["enabled"] = False
            elif change == "node_rank":
                node["parallelism"]["node_rank"] = 3
            elif change == "knobs":
                candidate["knobs"][m.EP_LOCAL] = "0"
            elif change == "digest":
                node["parallelism"]["command_sha256"] = ""
            elif change == "missing":
                del node["parallelism"]
            else:
                node["endpoint"] = {"host": "0.0.0.0", "port": "8000"}
            with self.subTest(change=change), self.assertRaises(ValueError):
                compare(arms)

    def test_nondefault_compilation_config_replays_exactly_through_existing_axis(self):
        compilation = '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"],"pass_config":{"fuse_gemm_comms":false}}'
        arms = [arm(enabled, compilation=compilation) for enabled in (False, True, False)]
        compare(arms, incoming=public_capacity(compilation=compilation))
        controls = arms[0]["controls"]
        self.assertEqual(controls["COMPILE_CFG"], compilation)
        self.assertEqual(controls["CUSTOM_OPS_AXIS"], "all")
        self.assertEqual(controls["PIECEWISE"], "0")
        # Invoke only the existing pure shell helper; never source the serving
        # launcher, load a profile, run Docker, or touch the fleet.
        script = '. "$1"; ct_apply_custom_ops_axis "$CUSTOM_OPS_AXIS" 1; printf %s "$COMPILE_CFG"'
        result = subprocess.run(["bash", "-c", script, "axis-fixture", str(ROOT / "launchers/lib/common-tp4.sh")],
                                env=dict(os.environ, **controls), text=True, capture_output=True, check=True)
        self.assertEqual(result.stdout, compilation)

    def test_original_graph_configuration_and_capacity_cannot_drift_in_all_new_arms(self):
        compilation = '{"cudagraph_mode":"FULL_DECODE_ONLY","custom_ops":["all"],"cudagraph_capture_sizes":[1,2,8]}'
        with self.assertRaisesRegex(ValueError, "original public"):
            compare([arm(), arm(True), arm()], incoming=public_capacity(compilation=compilation))
        with self.assertRaisesRegex(ValueError, "original public"):
            compare([arm(enabled, replace={"max-model-len": "262144"}) for enabled in (False, True, False)])
        with self.assertRaisesRegex(ValueError, "exactly four"):
            compare([arm(), arm(True), arm()], incoming={})
        with self.assertRaises(ValueError):
            compare([arm(), arm(True, compilation=compilation), arm()])

    def test_graph_axis_without_byte_preserving_replay_is_rejected(self):
        for compilation in ('{"cudagraph_mode":"FULL_DECODE_ONLY"}',
                            '{"custom_ops":["none"]}', '{"custom_ops": ["all"]}',
                            '{"custom_ops":["all"],"custom_ops":["none"]}', 'not-json'):
            with self.subTest(compilation=compilation), self.assertRaises(ValueError):
                arm(compilation=compilation)


if __name__ == "__main__":
    unittest.main()
