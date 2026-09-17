# Code graph snapshot

This directory contains a portable code-graph snapshot for `engine/base`.

- Source scope: `engine/base/`
- Source revision: `2dbd7a36`
- Source digest: `f59914908191af7f61c2daa2db3dd501bcbd94046a177d426b3ff03e1fbdef0f`
- Generated: 2026-09-18
- Extractor: `graphifyy 0.4.19`
- Extraction mode: AST-only (code-only corpus)
- Graph size: 1,501 nodes and 4,082 edges across 31 communities

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

The GitHub workflow `.github/workflows/graphify-check.yml` validates the
committed snapshot whenever the source, graph, or graph tooling changes.
