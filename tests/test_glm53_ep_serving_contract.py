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


def fixture(enabled=False, *, replace=None, eager=False, compilation=DEFAULT_COMPILATION, extra=""):
    knobs = {m.EP_LOCAL: str(int(enabled)), m.EP_COMPACT: "1" if enabled else None}
    nodes = {}
    for rank, node in enumerate(m.NODES):
        env = dict(m.COMMON_KNOBS, **{key: value for key, value in knobs.items() if value is not None},
                   PRIVATE_TOKEN="secret-do-not-print", NODE_SPECIFIC=str(rank))
        nodes[node] = dict(cmd=command(rank=rank, enabled=enabled, replace=replace, eager=eager,
                                      compilation=compilation, extra=extra),
                           env=[key + "=" + value for key, value in env.items()],
                           id=format(1 + rank + 100 * enabled, "064x"),
                           started_at="2026-09-08T10:00:00.000000000Z",
                           image="sha256:" + "a" * 64,
                           model={"path": "/models/private-model", "metadata": {"config.json": "b" * 64}},
                           mounts={"/site-packages/private-module.py": "c" * 64},
                           manifest_sha="d" * 64, source_revision="e" * 40,
                           hardware="GPU-" + str(rank), host_config={"Memory": 112 * 2**30})
    return nodes, knobs


KV = {"KV_TOKENS": "2000000", "KV_HYBRID_BLOCKS": "187"}
ENDPOINTS = {node: m.EndpointSubstitution(m.Endpoint("0.0.0.0", 8000),
                                         m.Endpoint("127.0.0.1", 18000))
             for node in m.NODES}


def arm(enabled=False, **kwargs):
    nodes, knobs = fixture(enabled, **kwargs)
    return m.configured_arm(nodes, enabled=enabled, expected_knobs=knobs, kv_witness=KV, endpoint_mapping=ENDPOINTS)


def public_capacity(**kwargs):
    return m.configured_incoming(public_nodes(**kwargs), kv_witness=KV, endpoint_mapping=ENDPOINTS)


def public_nodes(**kwargs):
    nodes, _ = fixture(**kwargs)
    for rank, node in enumerate(m.NODES):
        nodes[node]["cmd"] = command(rank=rank, replace={"host": "0.0.0.0", "port": "8000"}, **kwargs)
        nodes[node]["id"] = format(1000 + rank, "064x")
        nodes[node]["started_at"] = "2026-09-08T09:00:00.000000000Z"
    return nodes


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
                m.configured_arm(changed, enabled=True, expected_knobs=knobs, kv_witness=KV, endpoint_mapping=ENDPOINTS)
        with self.assertRaisesRegex(ValueError, "exactly four"):
            m.configured_arm({m.NODES[0]: nodes[m.NODES[0]]}, enabled=True, expected_knobs=knobs, kv_witness=KV, endpoint_mapping=ENDPOINTS)

    def test_only_explicit_two_knobs_may_change(self):
        nodes, knobs = fixture(True)
        for extra in ({"ENABLE_EP": "1"}, {"VLLM_GLM53_PREFILL_SP_FP8": "0"}):
            with self.assertRaisesRegex(ValueError, "allowlisted"):
                m.configured_arm(nodes, enabled=True, expected_knobs={**knobs, **extra}, kv_witness=KV, endpoint_mapping=ENDPOINTS)
        for entry in ("UNRELATED_KNOB=1", "PRIVATE_TOKEN=changed", "VLLM_GLM53_PREFILL_SP_FP8=0"):
            changed = copy.deepcopy(nodes)
            key = entry.split("=", 1)[0]
            changed[m.NODES[2]]["env"] = [v for v in changed[m.NODES[2]]["env"] if not v.startswith(key + "=")] + [entry]
            with self.subTest(entry=entry), self.assertRaises(ValueError):
                candidate = m.configured_arm(changed, enabled=True, expected_knobs=knobs, kv_witness=KV, endpoint_mapping=ENDPOINTS)
                compare([arm(), candidate, arm()])
        missing = {m.EP_LOCAL: "1", m.EP_COMPACT: None}
        with self.assertRaisesRegex(ValueError, "allowlisted"):
            m.configured_arm(nodes, enabled=True, expected_knobs=missing, kv_witness=KV, endpoint_mapping=ENDPOINTS)

    def test_unrelated_serve_options_are_not_erased_with_ep_flag(self):
        nodes, knobs = fixture(True)
        for extra in ("--attention-backend OTHER", "--speculative-config '{\"num_speculative_tokens\":3}'"):
            changed = copy.deepcopy(nodes)
            changed[m.NODES[1]]["cmd"] = command(rank=1, enabled=True, extra=extra)
            candidate = m.configured_arm(changed, enabled=True, expected_knobs=knobs, kv_witness=KV, endpoint_mapping=ENDPOINTS)
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
                m.configured_arm(changed, enabled=False, expected_knobs=knobs, kv_witness=KV, endpoint_mapping=ENDPOINTS)

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

    def test_persisted_outer_settings_and_controls_remain_bound_to_node_evidence(self):
        original = [arm(), arm(True), arm()]
        # Storage round trips preserve the checksum without retaining raw Env.
        compare(json.loads(json.dumps(original)))
        for change in ("candidate_compact", "both_baselines", "all_controls", "missing_checksum"):
            arms = copy.deepcopy(original)
            if change == "candidate_compact":
                arms[1]["knobs"][m.EP_COMPACT] = "0"
            elif change == "both_baselines":
                for index in (0, 2):
                    arms[index]["knobs"][m.EP_COMPACT] = "0"
            elif change == "all_controls":
                for record in arms:
                    record["controls"]["GMU"] = "0.6158"
            else:
                del arms[1]["envelope_sha256"]
            self.assertEqual([record["nodes"] for record in arms],
                             [record["nodes"] for record in original])
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "arm envelope"):
                compare(arms)

        # A real B2 with compact=0 cannot masquerade as the absent baseline
        # setting by editing only its outer metadata after configuration.
        nodes, knobs = fixture()
        knobs[m.EP_COMPACT] = "0"
        for config in nodes.values():
            config["env"].append(m.EP_COMPACT + "=0")
        b2 = m.configured_arm(nodes, enabled=False, expected_knobs=knobs,
                              kv_witness=KV, endpoint_mapping=ENDPOINTS)
        with self.assertRaisesRegex(ValueError, "restore baseline"):
            compare([arm(), arm(True), b2])
        b2["knobs"][m.EP_COMPACT] = None
        with self.assertRaisesRegex(ValueError, "arm envelope"):
            compare([arm(), arm(True), b2])

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

    def test_original_ep_flag_and_two_settings_are_the_only_allowed_arm_changes(self):
        incoming = public_capacity(enabled=True)
        arms = [arm(), arm(True), arm()]
        compare(arms, incoming=incoming)
        for node in m.NODES:
            original, baseline = incoming[node], arms[0]["nodes"][node]
            self.assertTrue(original["parallelism"]["enabled"])
            self.assertFalse(baseline["parallelism"]["enabled"])
            self.assertEqual(original["original_argv_without_ep_sha256"], baseline["original_argv_without_ep_sha256"])
            self.assertEqual(original["other_env_sha256"], baseline["other_env_sha256"])
            self.assertNotEqual(original["provenance"]["container"], baseline["provenance"]["container"])
        serialized = json.dumps(dict(incoming=incoming, arms=arms))
        self.assertNotIn("secret-do-not-print", serialized)
        self.assertNotIn("/models/private-model", serialized)
        self.assertNotIn("vllm serve", serialized)

    def test_three_identically_changed_launches_cannot_drift_from_original(self):
        for extra in ("--attention-backend OTHER", "--disable-custom-all-reduce",
                      "--speculative-config '{\"num_speculative_tokens\":3}'"):
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, "from original"):
                compare([arm(enabled, extra=extra) for enabled in (False, True, False)])
        # The same mismatch is rejected when it exists only in the original.
        with self.assertRaisesRegex(ValueError, "from original"):
            compare([arm(), arm(True), arm()], incoming=public_capacity(extra="--attention-backend OTHER"))

    def test_original_environment_is_required_and_only_ep_settings_are_excluded(self):
        for entry in ("PRIVATE_TOKEN=changed", "UNRELATED_KNOB=1", "ENABLE_EP=1"):
            nodes = public_nodes()
            key = entry.split("=", 1)[0]
            nodes[m.NODES[2]]["env"] = [value for value in nodes[m.NODES[2]]["env"]
                                        if not value.startswith(key + "=")] + [entry]
            incoming = m.configured_incoming(nodes, kv_witness=KV, endpoint_mapping=ENDPOINTS)
            with self.subTest(entry=entry), self.assertRaisesRegex(ValueError, "environment.*original"):
                compare([arm(), arm(True), arm()], incoming=incoming)
        for missing in ("cmd", "env", "id", "started_at"):
            nodes = public_nodes()
            del nodes[m.NODES[0]][missing]
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                m.configured_incoming(nodes, kv_witness=KV, endpoint_mapping=ENDPOINTS)
        bare_capacity = {node: m.launch_capacity(config["cmd"], public=True)
                         for node, config in public_nodes().items()}
        with self.assertRaisesRegex(ValueError, "configuration identity"):
            compare([arm(), arm(True), arm()], incoming=bare_capacity)

    def test_endpoint_mapping_must_be_explicit_typed_complete_and_exact(self):
        bad_maps = (None, {}, {m.NODES[0]: ENDPOINTS[m.NODES[0]]},
                    {**ENDPOINTS, "unexpected": ENDPOINTS[m.NODES[0]]},
                    {**ENDPOINTS, m.NODES[0]: dict(original="0.0.0.0:8000", replay="127.0.0.1:18000")})
        for mapping in bad_maps:
            with self.subTest(mapping=mapping), self.assertRaisesRegex(ValueError, "typed endpoint"):
                m.configured_incoming(public_nodes(), kv_witness=KV, endpoint_mapping=mapping)
        for host, port in (("127.0.0.1", 18000), ("0.0.0.0", 8001), ("other", 8000)):
            with self.subTest(original=(host, port)), self.assertRaises(ValueError):
                m.EndpointSubstitution(m.Endpoint(host, port), m.Endpoint("127.0.0.1", 18000))
        with self.assertRaises(ValueError):
            m.Endpoint("0.0.0.0", True)
        with self.assertRaises(ValueError):
            m.EndpointSubstitution(m.Endpoint("0.0.0.0", 8000), m.Endpoint("0.0.0.0", 8000))
        for host, port in (("0.0.0.0", "18000"), ("127.0.0.1", "8000")):
            nodes = public_nodes()
            nodes[m.NODES[0]]["cmd"] = command(replace={"host": host, "port": port})
            with self.subTest(actual=(host, port)), self.assertRaises(ValueError):
                m.configured_incoming(nodes, kv_witness=KV, endpoint_mapping=ENDPOINTS)

    def test_endpoint_strings_in_other_values_and_option_spelling_are_not_erased(self):
        original_name = "--served-model-name endpoint-0.0.0.0:8000"
        incoming = public_capacity(extra=original_name)
        compare([arm(enabled, extra=original_name) for enabled in (False, True, False)], incoming=incoming)
        with self.assertRaisesRegex(ValueError, "from original"):
            compare([arm(enabled, extra="--served-model-name endpoint-127.0.0.1:18000")
                     for enabled in (False, True, False)], incoming=incoming)

        def equals_endpoint(cmd):
            raw = base64.b64decode(launch._WRAPPER.fullmatch(cmd[1])[1]).decode()
            raw = raw.replace("--host ", "--host=").replace("--port ", "--port=")
            return ["-c", "echo " + base64.b64encode(raw.encode()).decode()
                    + " | base64 -d > /tmp/serve.sh; bash /tmp/serve.sh"]

        arms = []
        for enabled in (False, True, False):
            nodes, knobs = fixture(enabled)
            for config in nodes.values():
                config["cmd"] = equals_endpoint(config["cmd"])
            arms.append(m.configured_arm(nodes, enabled=enabled, expected_knobs=knobs,
                                         kv_witness=KV, endpoint_mapping=ENDPOINTS))
        # Exact argv permits endpoint value replacement, not option rewriting.
        with self.assertRaisesRegex(ValueError, "from original"):
            compare(arms)
        nodes = public_nodes()
        for config in nodes.values():
            config["cmd"] = equals_endpoint(config["cmd"])
        incoming = m.configured_incoming(nodes, kv_witness=KV, endpoint_mapping=ENDPOINTS)
        compare(arms, incoming=incoming)

    def test_supplied_image_model_source_and_host_provenance_cannot_drift_or_disappear(self):
        changed_values = dict(image="sha256:" + "f" * 64,
                              model={"metadata": {"config.json": "f" * 64}},
                              mounts={"/site-packages/private-module.py": "f" * 64},
                              manifest_sha="f" * 64, source_revision="f" * 40,
                              hardware="GPU-changed", host_config={"Memory": 108 * 2**30})
        for key, value in changed_values.items():
            for remove in (False, True):
                arms = []
                for enabled in (False, True, False):
                    nodes, knobs = fixture(enabled)
                    if remove:
                        del nodes[m.NODES[1]][key]
                    else:
                        nodes[m.NODES[1]][key] = value
                    arms.append(m.configured_arm(nodes, enabled=enabled, expected_knobs=knobs,
                                                 kv_witness=KV, endpoint_mapping=ENDPOINTS))
                with self.subTest(field=key, remove=remove), self.assertRaisesRegex(ValueError, "provenance.*original"):
                    compare(arms)
        # Optional provenance is not synthesized when unavailable. A future
        # live runner must require its full image/model/source inventory.
        original = public_nodes()
        arms = []
        for config in original.values():
            for key in changed_values:
                del config[key]
        for enabled in (False, True, False):
            nodes, knobs = fixture(enabled)
            for config in nodes.values():
                for key in changed_values:
                    del config[key]
            arms.append(m.configured_arm(nodes, enabled=enabled, expected_knobs=knobs,
                                         kv_witness=KV, endpoint_mapping=ENDPOINTS))
        incoming = m.configured_incoming(original, kv_witness=KV, endpoint_mapping=ENDPOINTS)
        compare(arms, incoming=incoming)

    def test_persisted_container_and_original_identity_cannot_be_detached_from_record(self):
        for field in ("id", "started_at"):
            incoming = public_capacity()
            original = incoming[m.NODES[0]]
            original["provenance"]["container"][field] = "changed"
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "configuration identity"):
                compare([arm(), arm(True), arm()], incoming=incoming)
        incoming = public_capacity()
        incoming[m.NODES[1]]["other_env_sha256"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "configuration identity"):
            compare([arm(), arm(True), arm()], incoming=incoming)


if __name__ == "__main__":
    unittest.main()
