# Code graph snapshot

This directory contains a portable code-graph snapshot for `engine/base`.

- Source scope: `engine/base/`
- Source revision: `1ea3f7fa`
- Source digest: `2e02d6786564d3789b4ad8c008bbce9dd73b20dee495ac277a7ad20fecbcc627`
- Generated: 2026-09-19
- Extractor: `graphifyy 0.4.19`
- Extraction mode: AST-only (code-only corpus)
- Graph size: 1,574 nodes and 4,303 edges across 32 communities

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
