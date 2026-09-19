# Code graph snapshot

This directory contains a portable code-graph snapshot for `engine/base`.

- Source scope: `engine/base/`
- Source revision: `c278bbb`
- Source digest: `60887c4dfa3ba1ba71532256b547c51139a49c807ea3729b95c1d65dafaabd5b`
- Generated: 2026-09-19
- Extractor: `graphifyy 0.4.19`
- Extraction mode: AST-only (code-only corpus)
- Graph size: 1,520 nodes and 4,160 edges across 32 communities

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
