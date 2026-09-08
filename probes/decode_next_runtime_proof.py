#!/usr/bin/env python3
"""Read-only per-rank proof for the compact AR, inline RDMA and SF6 arms.

The lever runs this on each node before and after its unchanged onepass.
Collection never imports torch or opens a CUDA context. Validation/comparison
helpers accept plain dictionaries so malformed or drifting proof is CPU-testable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import socket
import subprocess


IMAGE = "sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211"
TARGET_KNOBS = {
    "VLLM_GLM53_AR_COMPACT_CTA",
    "VLLM_GLM53_AR_PROXY_INLINE",
    "VLLM_GLM53_B12X_STATIC_V2",
}
COMMON_KNOBS = {
    "VLLM_GLM53_AR_CONSUMER_PDL": "1",
    "VLLM_GLM53_MK_PDL": "1",
    "VLLM_GLM53_MK_MHC_BF16": "1",
    "VLLM_GLM53_MK_INPUT_CTA": "4",
    "VLLM_GLM53_MK_INPUT_REUSE": "1",
    "VLLM_GLM53_AR_PREFETCH": "0",
}
COMMON_MARKERS = (
    "[osar] consumer PDL self-test PASS",
    "[megakernel] AR consumer MHC self-test PASS",
)
MEMORY_FIELDS = {"MemTotal", "MemFree", "MemAvailable", "AnonPages", "Shmem", "Slab"}
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def expected_knobs(mode):
    if mode not in ("baseline", "candidate"):
        raise ValueError("mode must be baseline or candidate")
    on = mode == "candidate"
    return dict(COMMON_KNOBS, VLLM_GLM53_AR_COMPACT_CTA=str(int(on)),
                VLLM_GLM53_AR_PROXY_INLINE=str(int(on)),
                VLLM_GLM53_B12X_STATIC_V2="t,r,sf6" if on else "t,r")


def validate_manifest(expected):
    if not isinstance(expected, dict) or not expected:
        raise ValueError("nonempty target-to-SHA256 manifest required")
    for path, digest in expected.items():
        if (not isinstance(path, str) or not path.startswith("/")
                or any(c.isspace() for c in path) or ".." in Path(path).parts
                or not isinstance(digest, str) or not SHA256.fullmatch(digest)):
            raise ValueError("invalid manifest target or SHA256")
    return dict(sorted(expected.items()))


def parse_markers(log):
    prepared = re.findall(
        r"\[b12x sf6\] prepared FC1\+FC2; raw prefill scales retained; packed bytes=(\d+)", log)
    fallback = re.findall(
        r"\[b12x sf6\] raw fallback: ([^\n]+?); raw prefill scales retained; packed bytes=0", log)
    return {
        "common": {marker: marker in log for marker in COMMON_MARKERS},
        "ar_capture_numel": [int(n) for n in re.findall(
            r"\[osar\] consumer PDL CAPTURED numel=(\d+)", log)],
        "mhc_capture_tokens": [int(n) for n in re.findall(
            r"\[megakernel\] AR consumer MHC CAPTURED T=(\d+) bf16=True vec4=True", log)],
        "compact_capture_numel": [int(n) for n in re.findall(
            r"\[oneshot\] compact AR CAPTURED numel=(\d+) ctas=12 tickets=48", log)],
        "compact_selftests": [int(n) for n in re.findall(
            r"\[osar\] compact transport self-test cases=(\d+) graph=3 maxerr=0(?:\.0+)?(?:\s|$)", log)],
        "inline_posts": len(re.findall(
            r"\[oneshot\] inline proxy serving peers=3 inline_bytes=8 wr_reuse=1", log)),
        "sf6_m6_serving": len(re.findall(
            r"\[b12x static v2\] lane serving: static2_m6_[^\n]*r16n128k256d256sf6v1[^\n]*\(mac=\d+, m=6,", log)),
        "sf6_prepared_count": len(prepared),
        "sf6_packed_bytes": sum(map(int, prepared)),
        "sf6_fallback_count": len(fallback),
        "sf6_fallback_reasons": fallback,
        "new_marker_present": any(marker in log for marker in (
            "[oneshot] compact AR CAPTURED", "[osar] compact transport self-test",
            "[oneshot] inline proxy serving", "[b12x sf6]", "sf6v1")),
    }


def serving_argv(script):
    commands = []
    for line in script.splitlines():
        if not re.match(r"\s*vllm\s+serve(?:\s|$)", line):
            continue
        tokens = shlex.split(line, comments=True)
        if tokens[:2] == ["vllm", "serve"]:
            # The generated launcher has a single literal vllm command followed
            # by stdout/stderr redirections. Preserve every actual CLI argument.
            stop = next((i for i, token in enumerate(tokens)
                         if token in (">", ">>", "2>", "2>>") or token.startswith("2>")), len(tokens))
            commands.append(tokens[:stop])
    if len(commands) != 1 or len(commands[0]) < 3:
        raise ValueError("expected exactly one generated vllm serve command")
    return commands[0]


def cli_option(argv, name):
    values = []
    for i, token in enumerate(argv):
        if token == name:
            if i + 1 == len(argv) or argv[i + 1].startswith("--"):
                raise ValueError("missing value for " + name)
            values.append(argv[i + 1])
        elif token.startswith(name + "="):
            values.append(token.split("=", 1)[1])
    if len(values) != 1 or not values[0]:
        raise ValueError("expected one " + name)
    return values[0]


def memory_from_text(raw):
    fields = {}
    for line in raw.splitlines():
        parts = line.split()
        if parts and parts[0].rstrip(":") in MEMORY_FIELDS:
            if len(parts) != 3 or parts[2] != "kB":
                raise ValueError("invalid /proc/meminfo units")
            fields[parts[0].rstrip(":")] = int(parts[1])
    return fields


def covering_mount(path, mounts):
    matches = [mount for mount in mounts if path == mount["destination"]
               or path.startswith(mount["destination"].rstrip("/") + "/")]
    return max(matches, key=lambda mount: len(mount["destination"]), default=None)


def validate_report(report, expected):
    """Return concrete proof failures; no device, filesystem or process access."""
    errors = []
    try:
        expected = validate_manifest(expected)
        wanted = expected_knobs(report["mode"])
        if report.get("image") != IMAGE:
            errors.append("immutable image mismatch")
        if report.get("running") is not True or report.get("oom_killed") is not False:
            errors.append("container stopped or OOM state unavailable")
        if not report.get("host") or not report.get("boot_id"):
            errors.append("missing host or boot identity")
        if report.get("source_sha256") != expected:
            errors.append("mounted source manifest mismatch")
        env = report["knobs"]
        if not isinstance(env, dict) or any(env.get(key) != value for key, value in wanted.items()):
            errors.append("actual container mode/common knobs mismatch")
        if not isinstance(report.get("config_cmd"), list) or not report["config_cmd"]:
            errors.append("missing actual Config.Cmd")
        if not isinstance(report.get("entrypoint"), list) or not report["entrypoint"]:
            errors.append("missing actual Config.Entrypoint")
        argv = report["serving_argv"]
        if not isinstance(argv, list) or argv[:2] != ["vllm", "serve"]:
            errors.append("invalid serving argv")
        template = report["template"]
        if (template["path"] != cli_option(argv, "--chat-template")
                or not SHA256.fullmatch(template["sha256"])):
            errors.append("chat template identity mismatch")
        if not SHA256.fullmatch(report["serving_script_sha256"]):
            errors.append("missing serving script identity")
        # Retain both values directly from the actual command, not profile
        # defaults or the caller environment. Across-arm comparison is exact.
        if report["gmu"] != cli_option(argv, "--gpu-memory-utilization"):
            errors.append("GMU not bound to serving argv")
        if report["kv_blocks"] != cli_option(argv, "--num-gpu-blocks-override"):
            errors.append("KV blocks not bound to serving argv")
        if not 0 < float(report["gmu"]) < 1 or int(report["kv_blocks"]) <= 0:
            errors.append("invalid GMU/KV values")
        mounts = report["mounts"]
        for path in [*expected, template["path"], argv[2]]:
            mount = covering_mount(path, mounts)
            if mount is None or mount.get("rw") is not False:
                errors.append("missing read-only serving mount: " + path)
        mem = report["host_memory_kib"]
        if (not all(type(mem.get(key)) is int and mem[key] >= 0 for key in MEMORY_FIELDS)
                or not 0 <= mem["MemAvailable"] <= mem["MemTotal"] or mem["MemTotal"] <= 0):
            errors.append("invalid host memory evidence")
        markers = report["markers"]
        if markers.get("common") != dict.fromkeys(COMMON_MARKERS, True):
            errors.append("missing AR/MHC self-test PASS")
        for field, upper in (("ar_capture_numel", 32768), ("mhc_capture_tokens", 8)):
            values = markers.get(field, [])
            if not values or any(type(n) is not int or not 1 <= n <= upper for n in values):
                errors.append("missing or invalid " + field)
        if report["mode"] == "candidate":
            values = markers.get("compact_capture_numel", [])
            if not values or any(type(n) is not int or not 1 <= n <= 32768 for n in values):
                errors.append("missing or invalid actual compact capture")
            if not markers.get("compact_selftests") or any(n <= 0 for n in markers["compact_selftests"]):
                errors.append("missing exact compact transport self-test")
            for field in ("inline_posts", "sf6_m6_serving", "sf6_prepared_count", "sf6_packed_bytes"):
                if type(markers.get(field)) is not int or markers[field] <= 0:
                    errors.append("missing actual " + field)
            if markers.get("sf6_fallback_count") != len(markers["sf6_fallback_reasons"]):
                errors.append("invalid SF6 fallback accounting")
        elif (markers.get("new_marker_present") is not False or any(markers.get(field)
              for field in ("compact_capture_numel", "compact_selftests", "inline_posts",
                            "sf6_m6_serving", "sf6_prepared_count", "sf6_packed_bytes",
                            "sf6_fallback_count", "sf6_fallback_reasons"))):
            errors.append("baseline executed a new candidate lane")
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        errors.append("malformed runtime proof: " + str(exc))
    return errors


_IMMUTABLE = ("host", "mode", "image", "boot_id", "source_sha256", "knobs",
              "config_cmd", "entrypoint", "serving_argv", "serving_script_sha256",
              "gmu", "kv_blocks", "mounts", "template")


def compare_snapshots(before, after):
    """One host/arm must retain its boot, executable and configuration."""
    if not isinstance(before, dict) or not isinstance(after, dict):
        return ["before/after runtime reports must be objects"]
    errors = []
    for side, value in (("before", before), ("after", after)):
        errors.extend(side + ": " + error for error in validate_report(value, before.get("source_sha256")))
    errors.extend("within-arm drift: " + key for key in _IMMUTABLE
                  if before.get(key) != after.get(key))
    return errors


def compare_arms(baseline_by_host, candidate_by_host):
    """Compare four matching hosts; only the three requested knobs may differ."""
    errors = []
    if (not isinstance(baseline_by_host, dict) or not isinstance(candidate_by_host, dict)
            or len(baseline_by_host) != 4 or set(baseline_by_host) != set(candidate_by_host)
            or any(not isinstance(value, dict) for value in
                   [*baseline_by_host.values(), *candidate_by_host.values()])):
        return ["exactly four matching rank/host reports required"]
    for mode, reports in (("baseline", baseline_by_host), ("candidate", candidate_by_host)):
        if len({report.get("host") for report in reports.values()}) != 4:
            errors.append(mode + ": duplicate or missing host identity")
        if len({report.get("boot_id") for report in reports.values()}) != 4:
            errors.append(mode + ": duplicate or missing rank boot identity")
    reference_sources = next(iter(baseline_by_host.values())).get("source_sha256")
    for host, baseline in baseline_by_host.items():
        candidate = candidate_by_host[host]
        for mode, report in (("baseline", baseline), ("candidate", candidate)):
            if report.get("mode") != mode:
                errors.append(host + ": arm mode mismatch")
            errors.extend(host + " " + mode + ": " + error
                          for error in validate_report(report, reference_sources))
        for key in _IMMUTABLE:
            if key in ("mode", "boot_id", "knobs"):
                continue
            if baseline.get(key) != candidate.get(key):
                errors.append(host + ": across-arm drift: " + key)
        if baseline.get("boot_id") == candidate.get("boot_id"):
            errors.append(host + ": arms reused the same boot")
        baseline_env = baseline.get("knobs") if isinstance(baseline.get("knobs"), dict) else {}
        candidate_env = candidate.get("knobs") if isinstance(candidate.get("knobs"), dict) else {}
        baseline_knobs = {key: value for key, value in baseline_env.items()
                          if key not in TARGET_KNOBS}
        candidate_knobs = {key: value for key, value in candidate_env.items()
                           if key not in TARGET_KNOBS}
        if baseline_knobs != candidate_knobs:
            errors.append(host + ": non-target container knobs changed")
    return errors


def collect_report(mode, expected, *, run=subprocess.check_output):
    expected = validate_manifest(expected)
    names = run(["docker", "ps", "--format", "{{.Names}}"], text=True, timeout=20).splitlines()
    selected = [name for name in names if name in ("glm53", "glm53-worker")]
    if len(selected) != 1:
        raise ValueError("expected exactly one local serving container")
    name = selected[0]
    obj = json.loads(run(["docker", "inspect", name], text=True, timeout=20))[0]
    config, state = obj["Config"], obj["State"]
    env = dict(item.split("=", 1) for item in config["Env"] if "=" in item)
    script = run(["docker", "exec", name, "cat", "/tmp/serve.sh"], text=True, timeout=20)
    argv = serving_argv(script)
    template_path = cli_option(argv, "--chat-template")
    if not template_path.startswith("/") or any(c.isspace() for c in template_path):
        raise ValueError("expected an absolute chat template path")
    wanted_hashes = sorted(set(expected) | {template_path})
    raw_hashes = run(["docker", "exec", name, "sha256sum", "--", *wanted_hashes], text=True, timeout=30)
    hashes = {}
    for line in raw_hashes.splitlines():
        digest, path = line.split(maxsplit=1)
        if path in hashes or path not in wanted_hashes or not SHA256.fullmatch(digest):
            raise ValueError("unexpected or duplicate source hash result")
        hashes[path] = digest
    if set(hashes) != set(wanted_hashes):
        raise ValueError("incomplete mounted source hashes")
    log = Path("/home/choiceoh/glm53-logs/glm53.log").read_text(errors="replace")
    mounts = sorted((dict(type=mount["Type"], source=mount["Source"],
                          destination=mount["Destination"], rw=mount["RW"])
                     for mount in obj["Mounts"]), key=lambda mount: mount["destination"])
    return dict(schema=1, mode=mode, host=socket.gethostname(), container=name,
                image=obj["Image"], boot_id=obj["Id"] + "|" + state["StartedAt"],
                running=state["Running"], oom_killed=state["OOMKilled"],
                config_cmd=config["Cmd"], entrypoint=config["Entrypoint"],
                serving_argv=argv, serving_script_sha256=hashlib.sha256(script.encode()).hexdigest(),
                gmu=cli_option(argv, "--gpu-memory-utilization"),
                kv_blocks=cli_option(argv, "--num-gpu-blocks-override"),
                source_sha256={path: hashes[path] for path in expected},
                knobs={key: value for key, value in sorted(env.items()) if key.startswith("VLLM_")},
                mounts=mounts, template=dict(path=template_path, sha256=hashes[template_path]),
                host_memory_kib=memory_from_text(Path("/proc/meminfo").read_text()),
                log_sha256=hashlib.sha256(log.encode()).hexdigest(), markers=parse_markers(log))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("baseline", "candidate"))
    parser.add_argument("expected", help="JSON manifest mapping mounted target to SHA256")
    args = parser.parse_args(argv)
    try:
        expected = validate_manifest(json.loads(args.expected))
        report = collect_report(args.mode, expected)
        errors = validate_report(report, expected)
        report["errors"] = errors
        report["valid"] = not errors
    except (OSError, KeyError, TypeError, ValueError, subprocess.SubprocessError) as exc:
        report, errors = dict(mode=args.mode, host=socket.gethostname(), valid=False,
                              collection_error=str(exc)), [str(exc)]
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
