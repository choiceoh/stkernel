#!/usr/bin/env python3
"""Regenerate the committed AST code graph for engine/base."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "engine" / "base"
OUTPUT = ROOT / "graphify-out"

COMMUNITY_LABELS = {
    0: "Memory Plan",
    1: "Serving Diagnostics",
    2: "Device Memory Arena",
    3: "Fleet Lease & Latency",
    4: "Distributed Communication",
    5: "Prefix Cache & Tenancy",
    6: "Scheduler & KV Blocks",
    7: "Memory Budget Gates",
    8: "Request Cache",
    9: "Sampling Constants",
    10: "Runtime Introspection",
    11: "Composed Model Lifecycle",
    12: "Stateless Randomness",
    13: "Model & Kernel Shapes",
    14: "Snapshot Publication",
    15: "Stage Timing",
    16: "Instrumentation",
    17: "KV Cache Sizing",
    18: "Common Kernel Lanes",
    19: "Stall Watchdog",
    20: "Process Topology",
    21: "Tensor Layout",
    22: "Package Init",
    23: "Free Block State",
    24: "Block Reservation",
    25: "Boundary Cache Release",
    26: "Snapshot Cache Release",
    27: "Phase Measurement",
    28: "Death Notes",
    29: "Distributed Gate Diagnostics",
}


def source_files() -> list[Path]:
    return sorted(
        path
        for path in SOURCE.rglob("*.py")
        if not any(
            part.startswith(".") or part == "__pycache__"
            for part in path.relative_to(ROOT).parts
        )
    )


def source_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(ROOT).as_posix().encode()
        digest.update(relative)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def source_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def graphify_version() -> str:
    try:
        from importlib.metadata import version

        return version("graphifyy")
    except Exception:
        return "unknown"


def stable_questions(graph, communities, labels):
    """Keep sampled betweenness questions reproducible across regenerations."""
    import graphify.analyze as analyze

    original = analyze.nx.betweenness_centrality

    def seeded_betweenness(*args, **kwargs):
        kwargs["seed"] = 0
        return original(*args, **kwargs)

    analyze.nx.betweenness_centrality = seeded_betweenness
    try:
        return analyze.suggest_questions(graph, communities, labels)
    finally:
        analyze.nx.betweenness_centrality = original


def write_readme(node_count: int, edge_count: int, community_count: int, digest: str) -> None:
    revision = source_revision()
    text = f"""# Code graph snapshot

This directory contains a portable code-graph snapshot for `engine/base`.

- Source scope: `engine/base/`
- Source revision: `{revision}`
- Source digest: `{digest}`
- Generated: {date.today().isoformat()}
- Extractor: `graphifyy {graphify_version()}`
- Extraction mode: AST-only (code-only corpus)
- Graph size: {node_count:,} nodes and {edge_count:,} edges across {community_count} communities

Files:

- `graph.html` — interactive browser visualization
- `graph.json` — graph data for programmatic queries
- `GRAPH_REPORT.md` — hubs, communities, connections, and suggested questions
- `manifest.json` — source-file manifest for incremental regeneration
- `source.sha256` — content digest used by the validator

Commands:

```sh
./tools/graphify_engine_base.sh
python3 tools/validate_graphify_engine_base.py
```

The snapshot is intentionally scoped to `engine/base`; regenerate it when the
source digest changes materially.
"""
    (OUTPUT / "README.md").write_text(text)


def generate() -> None:
    os.chdir(ROOT)

    from graphify.analyze import god_nodes, surprising_connections
    from graphify.build import build_from_json
    from graphify.cluster import cluster, score_all
    from graphify.detect import detect, save_manifest
    from graphify.export import to_html, to_json
    from graphify.extract import extract
    from graphify.report import generate as generate_report

    OUTPUT.mkdir(exist_ok=True)
    detected = detect(Path("engine/base"))
    files = [Path(path) for path in detected["files"]["code"]]
    if not files:
        raise SystemExit("No code files found under engine/base")

    extraction = extract(sorted(files), cache_root=Path("."))
    extraction["nodes"] = sorted(extraction["nodes"], key=lambda node: node["id"])
    extraction["edges"] = sorted(
        extraction["edges"],
        key=lambda edge: (
            edge.get("source", ""),
            edge.get("target", ""),
            edge.get("relation", ""),
            edge.get("source_file", ""),
            edge.get("source_location", ""),
        ),
    )
    graph = build_from_json(extraction)
    communities = cluster(graph)
    cohesion = score_all(graph, communities)
    labels = {
        community_id: COMMUNITY_LABELS.get(community_id, f"Community {community_id}")
        for community_id in communities
    }
    questions = stable_questions(graph, communities, labels)
    tokens = {
        "input": extraction.get("input_tokens", 0),
        "output": extraction.get("output_tokens", 0),
    }

    report = generate_report(
        graph,
        communities,
        cohesion,
        labels,
        god_nodes(graph),
        surprising_connections(graph, communities),
        detected,
        tokens,
        "engine/base",
        suggested_questions=questions,
    )
    clean_report = "\n".join(line.rstrip() for line in report.splitlines()) + "\n"
    (OUTPUT / "GRAPH_REPORT.md").write_text(clean_report)
    to_json(graph, communities, str(OUTPUT / "graph.json"))
    to_html(graph, communities, str(OUTPUT / "graph.html"), community_labels=labels)
    save_manifest(detected["files"], manifest_path=str(OUTPUT / "manifest.json"))

    digest = source_digest(source_files())
    (OUTPUT / "source.sha256").write_text(f"{digest}  engine/base\n")
    write_readme(graph.number_of_nodes(), graph.number_of_edges(), len(communities), digest)
    print(
        f"Generated engine/base graph: {graph.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} edges, {len(communities)} communities"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate the committed snapshot instead of regenerating it",
    )
    args = parser.parse_args()
    if args.check:
        subprocess.run(
            ["python3", str(ROOT / "tools" / "validate_graphify_engine_base.py")],
            cwd=ROOT,
            check=True,
        )
    else:
        generate()


if __name__ == "__main__":
    main()
