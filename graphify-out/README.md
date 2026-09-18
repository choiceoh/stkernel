# Code graph snapshot

This directory contains a portable code-graph snapshot for `engine/base`.

- Source scope: `engine/base/`
- Source revision: `7be8152d`
- Source digest: `ee3613c8a9ca33ed634ae7c6a6a5c050cd5cb2fc9484ea658f16d5e62abce918`
- Generated: 2026-09-18
- Extractor: `graphifyy 0.4.19`
- Extraction mode: AST-only (code-only corpus)
- Graph size: 1,517 nodes and 4,154 edges across 31 communities

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
