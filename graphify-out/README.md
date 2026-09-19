# Code graph snapshot

This directory contains a portable code-graph snapshot for `engine/base`.

- Source scope: `engine/base/`
- Source revision: `b2d3a24a`
- Source digest: `b14f39fd5e995c8feb7f57cff07225c9649675df3788b3396e13f5fe646ab534`
- Generated: 2026-09-19
- Extractor: `graphifyy 0.4.19`
- Extraction mode: AST-only (code-only corpus)
- Graph size: 1,645 nodes and 4,609 edges across 29 communities

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
