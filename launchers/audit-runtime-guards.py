#!/usr/bin/env python3
"""Audit deployed vLLM sources for token/expert ID lower and upper bounds.

The production image is third-party and the relevant sampler/CUDA sources are
not overlaid by this repository. Missing source is therefore UNKNOWN, never a
PASS. Exit codes: 0=all PASS, 1=at least one FAIL, 2=UNKNOWN/probe error.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Callable


GUMBEL = "vllm/v1/worker/gpu/sample/gumbel.py"
REJECTION = "vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py"
HASH_MOE = "csrc/libtorch_stable/moe/topk_softplus_sqrt_kernels.cu"
MOE_SUM = "csrc/libtorch_stable/moe/moe_align_sum_kernels.cu"
FUSED_SUM = "vllm/model_executor/layers/fused_moe/moe_fused_mul_sum.py"
MOE_UTILS = "vllm/model_executor/layers/fused_moe/utils.py"
REQUIRED_FILES = (GUMBEL, REJECTION, HASH_MOE, MOE_SUM, FUSED_SUM, MOE_UTILS)


@dataclass(frozen=True)
class Result:
    name: str
    status: str
    path: str | None
    detail: str


Checker = Callable[[str], tuple[bool, str]]


def _matches(text: str, pattern: str) -> int:
    return len(re.findall(pattern, text, flags=re.MULTILINE | re.DOTALL))


def _gumbel(text: str) -> tuple[bool, str]:
    pattern = (
        r"tl\.minimum\s*\(\s*block_idx\s*\*\s*BLOCK_SIZE\s*\+\s*idx"
        r"\s*,\s*vocab_size\s*-\s*1\s*\)"
    )
    count = _matches(text, pattern)
    return count >= 1, f"tile argmax upper clamps: {count}/1"


def _rejection(text: str) -> tuple[bool, str]:
    pattern = (
        r"tl\.minimum\s*\(\s*block_idx\s*\*\s*BLOCK_SIZE\s*\+\s*idx"
        r"\s*,\s*vocab_size\s*-\s*1\s*\)"
    )
    count = _matches(text, pattern)
    return count >= 2, f"spec-decode argmax upper clamps: {count}/2"


def _hash_moe(text: str) -> tuple[bool, str]:
    range_count = _matches(
        text, r"token_id\s*>=\s*0\s*&&\s*token_id\s*<\s*hash_table_rows"
    )
    table_rows = bool(
        re.search(
            r"hash_table_rows\s*=\s*tid2eid(?:\.value\(\))?\.size\(0\)",
            text,
        )
    )
    ok = range_count >= 2 and table_rows
    return ok, (
        f"lower+upper token guards: {range_count}/2; "
        f"bound derived from tid2eid rows: {'yes' if table_rows else 'no'}"
    )


def _moe_sum(text: str) -> tuple[bool, str]:
    bounded = bool(
        re.search(
            r"expert_id\s*<\s*0\s*\|\|\s*expert_id\s*>=\s*"
            r"static_cast<int64_t>\(num_experts\)",
            text,
        )
    )
    map_len = bool(
        re.search(
            r"num_experts\s*=\s*static_cast<int32_t>\("
            r"expert_map->numel\(\)\)",
            text,
        )
    )
    return bounded and map_len, (
        f"lower+upper expert guard: {'yes' if bounded else 'no'}; "
        f"bound derived from expert_map length: {'yes' if map_len else 'no'}"
    )


def _fused_sum(text: str) -> tuple[bool, str]:
    bounded = bool(
        re.search(
            r"\(id_val\s*>=\s*0\)\s*&\s*"
            r"\(id_val\s*<\s*num_experts\)",
            text,
        )
    )
    map_len = "expert_map.numel()" in text
    return bounded and map_len, (
        f"lower+upper expert guard: {'yes' if bounded else 'no'}; "
        f"bound derived from expert_map length: {'yes' if map_len else 'no'}"
    )


def _count_experts(text: str) -> tuple[bool, str]:
    bounded = bool(
        re.search(
            r"\(expert_ids\s*>=\s*0\)\s*&\s*"
            r"\(expert_ids\s*<\s*num_global_experts\)",
            text,
        )
    )
    map_len = "expert_map.numel()" in text
    return bounded and map_len, (
        f"global lower+upper expert guard: {'yes' if bounded else 'no'}; "
        f"bound derived from expert_map length: {'yes' if map_len else 'no'}"
    )


def _swiglu(text: str) -> tuple[bool, str]:
    bounded = bool(
        re.search(
            r"\(expert_id\s*>=\s*0\)\s*&\s*"
            r"\(expert_id\s*<\s*num_experts\)",
            text,
        )
    )
    map_len = "expert_map.numel()" in text
    return bounded and map_len, (
        f"lower+upper expert guard: {'yes' if bounded else 'no'}; "
        f"bound derived from expert_map length: {'yes' if map_len else 'no'}"
    )


RULES: tuple[tuple[str, str, Checker], ...] = (
    ("sampler.gumbel", GUMBEL, _gumbel),
    ("sampler.spec_decode", REJECTION, _rejection),
    ("hash_moe.token_table", HASH_MOE, _hash_moe),
    ("expert_map.cuda_sum", MOE_SUM, _moe_sum),
    ("expert_map.triton_sum", FUSED_SUM, _fused_sum),
    ("expert_map.count", MOE_UTILS, _count_experts),
    ("expert_map.swiglu", MOE_UTILS, _swiglu),
)


def _candidate_paths(root: Path, relpath: str):
    yield root / relpath
    if relpath.startswith("vllm/"):
        yield root / relpath.removeprefix("vllm/")


def load_from_root(root: Path) -> dict[str, dict[str, str]]:
    found: dict[str, dict[str, str]] = {}
    for relpath in REQUIRED_FILES:
        for path in _candidate_paths(root, relpath):
            if path.is_file():
                found[relpath] = {
                    "path": str(path.resolve()),
                    "text": path.read_text(encoding="utf-8", errors="replace"),
                }
                break
    return found


def _container_probe() -> str:
    relpaths = repr(REQUIRED_FILES)
    return f"""
import importlib.util
import json
from pathlib import Path

relpaths = {relpaths}
roots = [
    Path("/opt/venv/lib/python3.12/site-packages"),
    Path("/usr/local/lib/python3.12/site-packages"),
    Path("/workspace/vllm"),
    Path("/workspace"),
    Path("/vllm-workspace"),
    Path("/app/vllm"),
    Path("/app"),
    Path("/opt/vllm"),
]
spec = importlib.util.find_spec("vllm")
if spec and spec.origin:
    package = Path(spec.origin).resolve().parent
    roots.extend((package.parent, package))

seen = set()
unique_roots = []
for root in roots:
    key = str(root)
    if key not in seen:
        seen.add(key)
        unique_roots.append(root)

found = {{}}
for relpath in relpaths:
    for root in unique_roots:
        candidates = [root / relpath]
        if relpath.startswith("vllm/"):
            candidates.append(root / relpath.removeprefix("vllm/"))
        path = next((p for p in candidates if p.is_file()), None)
        if path is not None:
            found[relpath] = {{
                "path": str(path.resolve()),
                "text": path.read_text(encoding="utf-8", errors="replace"),
            }}
            break
print(json.dumps(found))
"""


def load_from_container(container: str) -> dict[str, dict[str, str]]:
    errors: list[str] = []
    found: dict[str, dict[str, str]] = {}
    successful_probes = 0
    for python in ("python3", "python", "/opt/venv/bin/python"):
        try:
            proc = subprocess.run(
                ["docker", "exec", container, python, "-c", _container_probe()],
                text=True,
                capture_output=True,
                check=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("docker CLI not found") from exc
        if proc.returncode != 0:
            errors.append(f"{python}: {proc.stderr.strip() or proc.stdout.strip()}")
            continue
        try:
            current = json.loads(proc.stdout)
        except json.JSONDecodeError:
            errors.append(f"{python}: invalid JSON: {proc.stdout[:200]!r}")
            continue
        successful_probes += 1
        for relpath, source in current.items():
            found.setdefault(relpath, source)
        if len(found) == len(REQUIRED_FILES):
            break
    if successful_probes:
        return found
    raise RuntimeError("; ".join(errors))


def evaluate(sources: dict[str, dict[str, str]]) -> list[Result]:
    results: list[Result] = []
    for name, relpath, checker in RULES:
        source = sources.get(relpath)
        if source is None:
            results.append(
                Result(
                    name,
                    "UNKNOWN",
                    None,
                    f"{relpath} is absent; binary presence cannot prove the guard",
                )
            )
            continue
        passed, detail = checker(source["text"])
        results.append(
            Result(
                name,
                "PASS" if passed else "FAIL",
                source["path"],
                detail,
            )
        )
    return results


def _summary(results: list[Result]) -> dict[str, int]:
    return {
        status: sum(result.status == status for result in results)
        for status in ("PASS", "FAIL", "UNKNOWN")
    }


def print_text(results: list[Result], source: str) -> None:
    print(f"runtime ID bounds audit ({source})")
    for result in results:
        location = f" [{result.path}]" if result.path else ""
        print(f"{result.status:7} {result.name}{location}")
        print(f"        {result.detail}")
    summary = _summary(results)
    print(
        "summary: "
        + " ".join(f"{status}={summary[status]}" for status in ("PASS", "FAIL", "UNKNOWN"))
    )


def self_test() -> None:
    sources = {
        GUMBEL: {
            "path": GUMBEL,
            "text": "token_id = tl.minimum(block_idx * BLOCK_SIZE + idx, vocab_size - 1)",
        },
        REJECTION: {
            "path": REJECTION,
            "text": "\n".join(
                [
                    "token_id = tl.minimum(block_idx * BLOCK_SIZE + idx, vocab_size - 1)",
                    "token_id = tl.minimum(block_idx * BLOCK_SIZE + idx, vocab_size - 1)",
                ]
            ),
        },
        HASH_MOE: {
            "path": HASH_MOE,
            "text": "\n".join(
                [
                    "token_id >= 0 && token_id < hash_table_rows",
                    "token_id >= 0 && token_id < hash_table_rows",
                    "hash_table_rows = tid2eid.value().size(0)",
                ]
            ),
        },
        MOE_SUM: {
            "path": MOE_SUM,
            "text": (
                "expert_id < 0 || expert_id >= static_cast<int64_t>(num_experts); "
                "num_experts = static_cast<int32_t>(expert_map->numel())"
            ),
        },
        FUSED_SUM: {
            "path": FUSED_SUM,
            "text": (
                "(id_val >= 0) & (id_val < num_experts); "
                "expert_map.numel()"
            ),
        },
        MOE_UTILS: {
            "path": MOE_UTILS,
            "text": "\n".join(
                [
                    "(expert_ids >= 0) & (expert_ids < num_global_experts)",
                    "(expert_id >= 0) & (expert_id < num_experts)",
                    "expert_map.numel()",
                ]
            ),
        },
    }
    results = evaluate(sources)
    assert all(result.status == "PASS" for result in results), results

    broken = {key: dict(value) for key, value in sources.items()}
    broken[HASH_MOE]["text"] = "token_id >= 0"
    broken_results = evaluate(broken)
    assert next(
        result for result in broken_results if result.name == "hash_moe.token_table"
    ).status == "FAIL"
    print("self-test: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--root", type=Path, help="vLLM repo or site-packages root")
    source.add_argument("--container", default="hy4", help="running container (default: hy4)")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--self-test", action="store_true", help="test the audit rules")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0

    if args.root is not None:
        if not args.root.is_dir():
            print(f"ERROR: audit root does not exist: {args.root}", file=sys.stderr)
            return 2
        source_name = f"root={args.root}"
        sources = load_from_root(args.root)
    else:
        source_name = f"container={args.container}"
        try:
            sources = load_from_container(args.container)
        except RuntimeError as exc:
            if args.json:
                print(json.dumps({"source": source_name, "error": str(exc)}))
            else:
                print(f"ERROR: {exc}", file=sys.stderr)
            return 2

    results = evaluate(sources)
    summary = _summary(results)
    if args.json:
        print(
            json.dumps(
                {
                    "source": source_name,
                    "results": [asdict(result) for result in results],
                    "summary": summary,
                },
                indent=2,
            )
        )
    else:
        print_text(results, source_name)

    if summary["FAIL"]:
        return 1
    if summary["UNKNOWN"]:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
