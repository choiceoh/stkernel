# onepass27 terminal evidence

Frozen source `ea413ac4c39ba3e6e4009c73587b0d536053b4bf`; session `eplocalonepass0909v27`, ticket `1788932740730747`, supervisor `730747` / start tick `43512869`.
Canonical order: B → A. Started: B, A. Complete original records: B, A. Only A enables TP Q0.
Terminal payload rc=0, supervisor rc=0. Bound supervisor and owned holder are absent. Observer is closed. Recovery policy/status is preserved without claiming public service restoration.

`job/onepass.jsonl` and `job/verdicts.jsonl` are original bytes when available. Native per-request `channel_diagnostics` remain unmodified; `channels/` copies these fields for review. Existing combined-text gates and their original verdicts are retained. No output-channel attribution waives a quality failure.

- B: fixed decode [68.54214500096191, 65.16505354240955, 68.73377076953395] tok/s; pooled 67.43982496349793 tok/s; quality {'ok': 18, 'total': 18}; Korean {'dirty': 0, 'n': 8, 'kinds': {'replacement': 0, 'lone_jamo': 0, 'cjk_mixed': 0, 'control': 0}, 'hits': []}; diagnostics 8/8.
- A: fixed decode [79.66053226240837, 68.42612250167345, 61.86725788534301] tok/s; pooled 69.23416453328194 tok/s; quality {'ok': 18, 'total': 18}; Korean {'dirty': 0, 'n': 8, 'kinds': {'replacement': 0, 'lone_jamo': 0, 'cjk_mixed': 0, 'control': 0}, 'hits': []}; diagnostics 8/8.

Pooled fixed decode is `sum(completion_tokens - 1) / sum(decode_s)` over all recorded fixed repetitions only when every such entry has valid positive timing. This archive does not decide matched throughput, default promotion, or sanitizer acceptance. Missing/partial arms are not reconstructed as results. GPU25 is historical context only: GPU27 uses a newer full source including the required main startup changes, while the original CPU24 kernel/contract bytes remain exact.

Original CPU24 proof is reused verbatim: revision `82ac3c34173ae63b3dd0a42c49f8421097e96a1a`, result SHA256 `1d96612938f200bd86d078e1d8841cc92df808ba4ab802ca51306fe229b67506`, 165 CPU tests and 30 compiled kernels. The original `cpu24-reuse.json` proves exact mounted/contract source equality for GPU27. No CPU27 compile or relabeled CPU receipt is claimed. CPU used capsule13.0.3 while serving used image13.3.1.

Available strict snapshots verify pinned image/source, TP4/4, EP/local/warm/zero0, Q0 only A, skip1/trim1, and actual image4/video0. Raw Docker Env/Cmd/inspect stay private. Each archived non-inspect snapshot and all passive chunks are hash-verified. Snapshot/canary absence is explicit and confers no proof. Unassigned stream bytes keep that scope.

Only explicitly known or terminal-log-referenced temporary legs are considered. Cleaned files remain missing in `terminal-capture.json` / `legs/availability.json`; no reconstruction is labeled original. Deterministic gzip retains original/stored hashes. This collector performs read-only SSH/file/git operations and writes only this fresh archive.
