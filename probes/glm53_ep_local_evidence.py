"""Read-only binding between EP CPU evidence and the files about to be tested."""
import hashlib
import json
from pathlib import Path
if __package__:
    from .glm53_ep_route_remap_check import compile_cases
else:
    from glm53_ep_route_remap_check import compile_cases

CPU_TEST_MODULES = (
    "test_glm53_ep_prefill_local.py",
    "test_glm53_ep_local_probe.py",
    "test_glm53_probe_lifecycle.py",
    "test_glm53_ep_route_remap.py",
)
CONTRACT_PATHS = tuple("tests/"+name for name in CPU_TEST_MODULES) + (
    "probes/glm53_ep_local_check.py",
    "probes/glm53_ep_local_evidence.py",
    "probes/run_glm53_ep_local_offline.py",
    "probes/glm53_probe_lifecycle.py",
    "probes/glm53_ep_local_compile_check.py",
    "probes/run_glm53_ep_local_cpu_compile.py",
    "probes/glm53_ep_route_remap_check.py",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mounted_sources(root):
    """Files mounted by both the CPU compiler and the isolated GPU runner."""
    rows = {}
    names = set()
    for line in (Path(root)/"build/glm53/manifest.tsv").read_text().splitlines():
        name, target, *_ = line.split("\t")
        if "/flashinfer/" in target or name == "flashinfer_b12x_moe.py":
            if target in rows or name in names or Path(name).name != name:
                raise ValueError("duplicate or invalid MoE manifest entry")
            rows[target] = Path(root)/"build/glm53"/name
            names.add(name)
    required = {"moe_dispatch.py", "moe_dynamic_ep_local.py", "glm53_ep_route_remap.py",
                "flashinfer_b12x_moe.py", "b12x_moe.py"}
    if not required.issubset(names):
        raise ValueError("incomplete MoE compile manifest")
    return rows


def validate_compile_evidence(root, path):
    root = Path(root)
    evidence = json.loads(Path(path).read_text())
    if (evidence.get("arm") != "local" or evidence.get("cuda_initialized") is not False
            or not evidence.get("cache_key")
            or evidence["cache_key"][-1] != "glm53_ep_prefill_local_v1"
            or not evidence.get("artifacts") or not evidence.get("resources")):
        raise ValueError("actual no-device local CuTe compile proof required")
    contracts = evidence.get("contracts", {})
    if (contracts.get("tests_run", 0) <= 0 or contracts.get("failures") != 0
            or contracts.get("errors") != 0 or contracts.get("skips") != 0):
        raise ValueError("passing pinned CPU contracts without skips required")
    remap = evidence.get("remap_compilation", [])
    if ({row.get("label") for row in remap} != {row["label"] for row in compile_cases()}
            or len(remap) != len(compile_cases())
            or not all(row.get("ptx_sha256") and row.get("cubin_sha256") for row in remap)):
        raise ValueError("complete no-device remap compilation proof required")
    for relative in CONTRACT_PATHS:
        if contracts.get("files", {}).get(relative) != digest(root/relative):
            raise ValueError("CPU-tested contract source changed: "+relative)
    mounted = mounted_sources(root)
    if set(evidence.get("mounted_sources", {})) != set(mounted):
        raise ValueError("compiled and mounted MoE file sets differ")
    for target, source in mounted.items():
        if evidence["mounted_sources"][target] != digest(source):
            raise ValueError("compiled overlay source changed: "+source.name)
        # Both maps attest files used by the kernel compiler.
        if target in evidence["sources"] and evidence["sources"][target] != digest(source):
            raise ValueError("kernel and mount source receipts disagree: "+source.name)
    return evidence
