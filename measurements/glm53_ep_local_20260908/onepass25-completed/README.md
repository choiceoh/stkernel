# onepass25 terminal evidence

Frozen source `055914aeb719c1769e05cdb863e43a88b2ee47af`; session `eplocalonepass0909v25`, ticket `1788930114553638`, supervisor `553638` / start tick `43250203`.
Canonical order: B0 → B1 → A → B2 → B3. Started: B0, B1, A, B2, B3. Complete original records: B0, B1, A, B2, B3. Only A enables TP Q0.
Terminal payload rc=0, supervisor rc=0. Bound supervisor and owned holder are absent. Observer is closed. Recovery policy/status is preserved without claiming public service restoration.

`job/onepass.jsonl` and `job/verdicts.jsonl` are original bytes when available. Native per-request `channel_diagnostics` remain unmodified; `channels/` copies these fields for review. Existing combined-text gates and their original verdicts are retained. No output-channel attribution waives a quality failure.

- B0: fixed decode [75.33212651644533, 73.2140140929547, 78.22109335616597] tok/s; pooled 75.53362314061366 tok/s; quality {'ok': 18, 'total': 18}; Korean {'dirty': 0, 'n': 8, 'kinds': {'replacement': 0, 'lone_jamo': 0, 'cjk_mixed': 0, 'control': 0}, 'hits': []}; diagnostics 8/8.
- B1: fixed decode [69.55192448576558, 63.47157274616765, 70.60241497729069] tok/s; pooled 67.7252064352149 tok/s; quality {'ok': 18, 'total': 18}; Korean {'dirty': 0, 'n': 8, 'kinds': {'replacement': 0, 'lone_jamo': 0, 'cjk_mixed': 0, 'control': 0}, 'hits': []}; diagnostics 8/8.
- A: fixed decode [69.21689090375017, 61.292683744519614, 71.70690453713567] tok/s; pooled 67.10184444701716 tok/s; quality {'ok': 18, 'total': 18}; Korean {'dirty': 0, 'n': 8, 'kinds': {'replacement': 0, 'lone_jamo': 0, 'cjk_mixed': 0, 'control': 0}, 'hits': []}; diagnostics 8/8.
- B2: fixed decode [75.5260650007893, 69.15952829063703, 71.18787599464792] tok/s; pooled 71.86124197050981 tok/s; quality {'ok': 18, 'total': 18}; Korean {'dirty': 1, 'n': 8, 'kinds': {'replacement': 0, 'lone_jamo': 0, 'cjk_mixed': 2, 'control': 0}, 'hits': [['fixed2K rep1', {'cjk_mixed': 2}]]}; diagnostics 8/8.
- B3: fixed decode [81.87528081172489, 80.43307532143484, 60.64925656999805] tok/s; pooled 72.93122703744989 tok/s; quality {'ok': 18, 'total': 18}; Korean {'dirty': 0, 'n': 8, 'kinds': {'replacement': 0, 'lone_jamo': 0, 'cjk_mixed': 0, 'control': 0}, 'hits': []}; diagnostics 8/8.

Pooled fixed decode is `sum(completion_tokens - 1) / sum(decode_s)` over all recorded fixed repetitions only when every such entry has valid positive timing. This archive does not decide matched throughput, default promotion, or sanitizer acceptance. Missing/partial arms are not reconstructed as results.

Original CPU24 proof is reused verbatim: revision `82ac3c34173ae63b3dd0a42c49f8421097e96a1a`, result SHA256 `1d96612938f200bd86d078e1d8841cc92df808ba4ab802ca51306fe229b67506`, 165 CPU tests and 30 compiled kernels. The original `cpu24-reuse.json` proves exact mounted/contract source equality for GPU25. No CPU25 compile or relabeled CPU receipt is claimed. CPU used capsule13.0.3 while serving used image13.3.1.

Available strict snapshots verify pinned image/source, TP4/4, EP/local/warm/zero0, Q0 only A, skip1/trim1, and actual image4/video0. Raw Docker Env/Cmd/inspect stay private. Each archived non-inspect snapshot and all passive chunks are hash-verified. Snapshot/canary absence is explicit and confers no proof. Unassigned stream bytes keep that scope.

Only explicitly known or terminal-log-referenced temporary legs are considered. Cleaned files remain missing in `terminal-capture.json` / `legs/availability.json`; no reconstruction is labeled original. Deterministic gzip retains original/stored hashes. This collector performs read-only SSH/file/git operations and writes only this fresh archive.
