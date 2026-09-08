"""CPU-only proof of matched-serving evidence and fail-closed drift checks."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("decode_next_proof", ROOT / "probes/decode_next_runtime_proof.py")
proof = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(proof)
MANIFEST = {"/pkg/ar.cu": "a" * 64, "/pkg/sf6.py": "b" * 64}
COMMON_LOG = """[osar] consumer PDL self-test PASS
[megakernel] AR consumer MHC self-test PASS
[osar] consumer PDL CAPTURED numel=4096
[megakernel] AR consumer MHC CAPTURED T=1 bf16=True vec4=True
"""
CANDIDATE_LOG = """[osar] compact transport self-test cases=21 graph=3 maxerr=0
[oneshot] compact AR CAPTURED numel=4096 ctas=12 tickets=48
[oneshot] inline proxy serving peers=3 inline_bytes=8 wr_reuse=1
[b12x sf6] prepared FC1+FC2; raw prefill scales retained; packed bytes=123456
[b12x sf6] raw fallback: fc2 stage spans 75 byte codes (limit 64); raw prefill scales retained; packed bytes=0
[b12x static v2] lane serving: static2_m6_k4096_n512_t8_r128_tm32f2g2a32wutr16n128k256d256sf6v1 (mac=48, m=6, routed=48, smem=65000 B)
"""
FINALIZED_LOG = "[b12x sf6] packed-only owners finalised: layers=48 raw_bytes_released=268435456; decode and prefill read immutable packed scales\n"


def report(mode="baseline", host="srv1"):
    argv = ["vllm", "serve", "/model", "--gpu-memory-utilization", "0.6329",
            "--num-gpu-blocks-override", "1056", "--chat-template", "/model/template.jinja"]
    return dict(schema=1, host=host, mode=mode, image=proof.IMAGE,
                boot_id=host + "|" + mode, running=True, oom_killed=False,
                config_cmd=["-c", "echo BASE64 | base64 -d > /tmp/serve.sh; bash /tmp/serve.sh"],
                entrypoint=["/bin/bash"], serving_argv=argv, serving_script_sha256="d" * 64,
                gmu="0.6329", kv_blocks="1056", source_sha256=dict(MANIFEST),
                knobs=dict(proof.expected_knobs(mode), VLLM_HOST_IP=host, VLLM_GLM53_SPEC_K="5"),
                mounts=[dict(type="bind", source="/host/pkg", destination="/pkg", rw=False),
                        dict(type="bind", source="/host/model", destination="/model", rw=False)],
                template=dict(path="/model/template.jinja", sha256="c" * 64),
                host_memory_kib=dict(MemTotal=125000000, MemFree=15000000,
                                     MemAvailable=16000000, AnonPages=80000000, Shmem=20, Slab=100),
                markers=proof.parse_markers(COMMON_LOG + (CANDIDATE_LOG if mode == "candidate" else "")))


class RuntimeProofTests(unittest.TestCase):
    def test_actual_small_capture_and_partial_sf6_are_recorded(self):
        candidate = report("candidate")
        self.assertEqual(proof.validate_report(candidate, MANIFEST), [])
        self.assertEqual(candidate["markers"]["compact_capture_numel"], [4096])
        self.assertEqual(candidate["markers"]["sf6_prepared_count"], 1)
        self.assertEqual(candidate["markers"]["sf6_fallback_count"], 1)
        self.assertEqual(candidate["markers"]["sf6_packed_bytes"], 123456)
        self.assertEqual(candidate["markers"]["sf6_fallback_reasons"],
                         ["fc2 stage spans 75 byte codes (limit 64)"])
        for field in ("sf6_packed_only_finalizations", "sf6_packed_only_layers",
                      "sf6_raw_bytes_released"):
            self.assertEqual(candidate["markers"][field], 0)
        self.assertEqual(proof.validate_report(report(), MANIFEST), [])

    def test_new_sf6_logs_record_release_separately_from_preparation(self):
        candidate = report("candidate")
        new_log = COMMON_LOG + CANDIDATE_LOG.replace(" raw prefill scales retained;", "")
        candidate["markers"] = proof.parse_markers(new_log)
        self.assertEqual(proof.validate_report(candidate, MANIFEST), [])
        self.assertEqual(candidate["markers"], report("candidate")["markers"])
        candidate["markers"] = proof.parse_markers(new_log + FINALIZED_LOG)
        self.assertEqual(proof.validate_report(candidate, MANIFEST), [])
        self.assertEqual(candidate["markers"]["sf6_packed_only_finalizations"], 1)
        self.assertEqual(candidate["markers"]["sf6_packed_only_layers"], 48)
        self.assertEqual(candidate["markers"]["sf6_raw_bytes_released"], 268435456)
        mixed = proof.parse_markers(CANDIDATE_LOG + new_log + FINALIZED_LOG + FINALIZED_LOG)
        self.assertEqual(mixed["sf6_prepared_count"], 2)
        self.assertEqual(mixed["sf6_fallback_count"], 2)
        self.assertEqual(mixed["sf6_packed_only_finalizations"], 2)
        self.assertEqual(mixed["sf6_packed_only_layers"], 96)
        self.assertEqual(mixed["sf6_raw_bytes_released"], 536870912)

    def test_release_marker_does_not_replace_required_execution_proof(self):
        for mode in ("baseline", "candidate"):
            value = report(mode)
            value["markers"] = proof.parse_markers(COMMON_LOG + FINALIZED_LOG)
            self.assertTrue(proof.validate_report(value, MANIFEST), mode)
        for malformed in (FINALIZED_LOG.replace("finalised", "pending"),
                          FINALIZED_LOG.replace("layers=48", "layers=-1"),
                          FINALIZED_LOG.replace("268435456;", "268435456.5;")):
            markers = proof.parse_markers(malformed)
            self.assertEqual(markers["sf6_packed_only_finalizations"], 0)
            self.assertEqual(markers["sf6_packed_only_layers"], 0)
            self.assertEqual(markers["sf6_raw_bytes_released"], 0)

    def test_candidate_requires_executed_lanes_not_environment_only(self):
        for field in ("compact_capture_numel", "compact_selftests", "inline_posts",
                      "sf6_m6_serving", "sf6_prepared_count", "sf6_packed_bytes"):
            value = report("candidate")
            value["markers"][field] = [] if isinstance(value["markers"][field], list) else 0
            self.assertTrue(proof.validate_report(value, MANIFEST), field)
        for n in (0, 32769):
            value = report("candidate")
            value["markers"]["compact_capture_numel"] = [n]
            self.assertTrue(proof.validate_report(value, MANIFEST))
        poisoned = COMMON_LOG + CANDIDATE_LOG.replace("maxerr=0", "maxerr=0.01")
        self.assertEqual(proof.parse_markers(poisoned)["compact_selftests"], [])

    def test_baseline_forbids_any_new_mode_proof(self):
        for line in CANDIDATE_LOG.splitlines():
            value = report()
            value["markers"] = proof.parse_markers(COMMON_LOG + line)
            self.assertIn("baseline executed a new candidate lane", proof.validate_report(value, MANIFEST))

    def test_missing_or_changed_identity_fails(self):
        changes = {"image": "sha256:" + "f" * 64, "running": False, "oom_killed": True,
                   "source_sha256": {}, "config_cmd": [], "entrypoint": [],
                   "template": {"path": "/other", "sha256": "c" * 64},
                   "gmu": "0.60", "kv_blocks": "1024", "mounts": [], "host_memory_kib": {}}
        for key, change in changes.items():
            value = report()
            value[key] = change
            self.assertTrue(proof.validate_report(value, MANIFEST), key)
        self.assertTrue(proof.validate_report({}, MANIFEST))
        for invalid in ({}, {"relative": "a" * 64}, {"/a": "oops"}, {"/a b": "a" * 64}):
            with self.assertRaises(ValueError):
                proof.validate_manifest(invalid)

    def test_within_arm_keeps_boot_and_all_configuration(self):
        before = report("candidate")
        after = copy.deepcopy(before)
        after["host_memory_kib"]["MemAvailable"] -= 100
        self.assertEqual(proof.compare_snapshots(before, after), [])
        for key in ("boot_id", "config_cmd", "serving_script_sha256", "mounts", "template", "knobs"):
            changed = copy.deepcopy(after)
            if key == "knobs":
                changed[key]["VLLM_GLM53_SPEC_K"] = "7"
            else:
                changed[key] = None
            self.assertTrue(proof.compare_snapshots(before, changed), key)

    def test_four_rank_pair_allows_only_target_knobs_and_distinct_boots(self):
        base = {host: report("baseline", host) for host in ("srv1", "srv2", "srv3", "srv4")}
        candidate = {host: report("candidate", host) for host in base}
        self.assertEqual(proof.compare_arms(base, candidate), [])
        for key, change in (("boot_id", base["srv3"]["boot_id"]),
                            ("gmu", "0.62"), ("kv_blocks", "1024"),
                            ("source_sha256", {"/pkg/ar.cu": "f" * 64})):
            changed = copy.deepcopy(candidate)
            changed["srv3"][key] = change
            self.assertTrue(proof.compare_arms(base, changed), key)
        changed = copy.deepcopy(candidate)
        changed["srv3"]["knobs"]["VLLM_GLM53_SPEC_K"] = "7"
        self.assertTrue(proof.compare_arms(base, changed))
        changed = copy.deepcopy(candidate)
        changed["srv3"] = copy.deepcopy(changed["srv1"])
        self.assertTrue(proof.compare_arms(base, changed))
        changed = copy.deepcopy(candidate)
        changed["srv3"]["knobs"] = None
        self.assertTrue(proof.compare_arms(base, changed))
        self.assertTrue(proof.compare_arms({"srv1": base["srv1"]}, {"srv1": candidate["srv1"]}))

    def test_serving_argv_and_memory_parsing_are_exact(self):
        text = "export X='multiline\nvalue'\nvllm serve /model --gpu-memory-utilization 0.63 --chat-template '/model/x.jinja' > /glmlogs/glm53.log 2>&1\n"
        argv = proof.serving_argv(text)
        self.assertEqual(argv[-2:], ["--chat-template", "/model/x.jinja"])
        self.assertEqual(proof.cli_option(argv, "--gpu-memory-utilization"), "0.63")
        self.assertEqual(proof.cli_option(["--x=1"], "--x"), "1")
        for script in ("echo vllm serve /model", text + text):
            with self.assertRaises(ValueError):
                proof.serving_argv(script)
        for argv in (["--x"], ["--x", "--other"], ["--x=1", "--x=2"]):
            with self.assertRaises(ValueError):
                proof.cli_option(argv, "--x")
        self.assertEqual(proof.memory_from_text("MemAvailable: 5 kB\nHugePages_Total: 0\n"), {"MemAvailable": 5})
        with self.assertRaises(ValueError):
            proof.memory_from_text("MemAvailable: 5 MB")

    def test_mocked_collection_preserves_actual_docker_and_template_identity(self):
        fixture = report("candidate", "test-host")
        script = " ".join(fixture["serving_argv"]) + " > /glmlogs/glm53.log 2>&1\n"
        inspect = dict(Image=fixture["image"], Id="actual-container-id",
                       State=dict(Running=True, OOMKilled=False, StartedAt="2026-09-09T01:00:00Z"),
                       Config=dict(Cmd=fixture["config_cmd"], Entrypoint=fixture["entrypoint"],
                                   Env=[key + "=" + value for key, value in fixture["knobs"].items()]
                                   + ["UNRELATED_SECRET=not-recorded"]),
                       Mounts=[dict(Type=mount["type"], Source=mount["source"],
                                    Destination=mount["destination"], RW=mount["rw"])
                               for mount in fixture["mounts"]])
        calls = []

        def run(command, **kwargs):
            calls.append(command)
            self.assertIn("timeout", kwargs)
            if command[1] == "ps":
                return "glm53-worker\n"
            if command[1] == "inspect":
                return json.dumps([inspect])
            if command[3] == "cat":
                return script
            self.assertEqual(command[3:5], ["sha256sum", "--"])
            hashes = dict(MANIFEST, **{fixture["template"]["path"]: fixture["template"]["sha256"]})
            return "\n".join(hashes[path] + "  " + path for path in command[5:]) + "\n"

        def read(path, **kwargs):
            if str(path) == "/proc/meminfo":
                return "\n".join(f"{key}: {value} kB" for key, value in fixture["host_memory_kib"].items())
            self.assertEqual(str(path), "/home/choiceoh/glm53-logs/glm53.log")
            return COMMON_LOG + CANDIDATE_LOG

        with patch.object(Path, "read_text", read), patch.object(proof.socket, "gethostname", return_value="test-host"):
            collected = proof.collect_report("candidate", MANIFEST, run=run)
        self.assertEqual(proof.validate_report(collected, MANIFEST), [])
        self.assertEqual(collected["config_cmd"], fixture["config_cmd"])
        self.assertEqual(collected["serving_argv"], fixture["serving_argv"])
        self.assertEqual(collected["template"], fixture["template"])
        self.assertNotIn("UNRELATED_SECRET", collected["knobs"])
        self.assertEqual(len(calls), 4)


if __name__ == "__main__":
    unittest.main()
