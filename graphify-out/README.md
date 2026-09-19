# Code graph snapshot

This directory contains a portable code-graph snapshot for `engine/base`.

- Source scope: `engine/base/`
- Source revision: `62e5affd`
- Source digest: `9fd4466ac9377c2a4ebfbedb406efbd68687c8cf73781b4e8bbd9afba1092fea`
- Generated: 2026-09-20
- Extractor: `graphifyy 0.4.19`
- Extraction mode: AST-only (code-only corpus)
- Graph size: 1,651 nodes and 4,633 edges across 31 communities

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
