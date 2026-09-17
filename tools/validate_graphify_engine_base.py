#!/usr/bin/env python3
"""Validate the committed engine/base graph without requiring graphify."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "engine" / "base"
OUTPUT = ROOT / "graphify-out"


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
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def fail(message: str) -> None:
    print(f"graph validation failed: {message}", file=sys.stderr)
    raise SystemExit(1)


def validate_graph_shape(nodes: list[dict], links: list[dict]) -> None:
    node_ids = [node.get("id") for node in nodes]
    if any(not isinstance(node_id, str) or not node_id for node_id in node_ids):
        fail("graph.json contains a node without a string id")
    if len(node_ids) != len(set(node_ids)):
        fail("graph.json contains duplicate node ids")

    known_ids = set(node_ids)
    broken_links = [
        link
        for link in links
        if link.get("source") not in known_ids or link.get("target") not in known_ids
    ]
    if broken_links:
        sample = broken_links[0]
        fail(
            "graph.json contains a link to a missing node "
            f"({sample.get('source')} -> {sample.get('target')})"
        )


def validate_report(report: str, nodes: list[dict], links: list[dict]) -> None:
    if "## God Nodes" not in report or "## Suggested Questions" not in report:
        fail("GRAPH_REPORT.md is missing required sections")

    summary = re.search(
        r"-\s+(\d+) nodes · (\d+) edges · (\d+) communities detected",
        report,
    )
    if not summary:
        fail("GRAPH_REPORT.md is missing its graph summary")

    community_ids = {
        node.get("community") for node in nodes if node.get("community") is not None
    }
    expected = (len(nodes), len(links), len(community_ids))
    reported = tuple(int(value) for value in summary.groups())
    if reported != expected:
        fail(f"GRAPH_REPORT.md summary {reported} disagrees with graph {expected}")


def main() -> None:
    required = [
        "README.md",
        "GRAPH_REPORT.md",
        "graph.html",
        "graph.json",
        "manifest.json",
        "source.sha256",
    ]
    missing = [name for name in required if not (OUTPUT / name).is_file()]
    if missing:
        fail("missing " + ", ".join(missing))

    try:
        graph = json.loads((OUTPUT / "graph.json").read_text())
        manifest = json.loads((OUTPUT / "manifest.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"invalid JSON: {exc}")

    nodes = graph.get("nodes")
    links = graph.get("links")
    if not isinstance(nodes, list) or not nodes:
        fail("graph.json has no nodes")
    if not isinstance(links, list) or not links:
        fail("graph.json has no links")
    if not isinstance(manifest, dict):
        fail("manifest.json is not an object")
    validate_graph_shape(nodes, links)

    current = {path.relative_to(ROOT).as_posix() for path in source_files()}
    recorded = set(manifest)
    if current != recorded:
        missing_sources = sorted(current - recorded)
        removed_sources = sorted(recorded - current)
        detail = []
        if missing_sources:
            detail.append("new=" + ",".join(missing_sources[:3]))
        if removed_sources:
            detail.append("removed=" + ",".join(removed_sources[:3]))
        fail("manifest source set differs (" + "; ".join(detail) + ")")

    recorded_digest = (OUTPUT / "source.sha256").read_text().split()[0]
    actual_digest = source_digest(source_files())
    if recorded_digest != actual_digest:
        fail("source.sha256 does not match engine/base; regenerate the graph")

    graph_sources = {
        node.get("source_file")
        for node in nodes
        if node.get("source_file")
    }
    stale_sources = sorted(graph_sources - current)
    if stale_sources:
        fail("graph.json references missing source files: " + ", ".join(stale_sources[:3]))

    report = (OUTPUT / "GRAPH_REPORT.md").read_text()
    html = (OUTPUT / "graph.html").read_text()
    community_ids = {
        node.get("community") for node in nodes if node.get("community") is not None
    }
    validate_report(report, nodes, links)
    if "<html" not in html.lower():
        fail("graph.html does not look like an HTML document")

    print(
        "graph validation: OK "
        f"({len(nodes)} nodes, {len(links)} edges, "
        f"{len(community_ids)} communities, {len(recorded)} source files)"
    )


if __name__ == "__main__":
    main()
