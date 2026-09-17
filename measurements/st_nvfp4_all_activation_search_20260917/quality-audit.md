# Final decisions, structured output and certificate quality

Keep three different observations separate: reported primary decisions, whether
the response can be parsed, and whether every proof/intermediate field is right.
The original grades in the consumer receipts are unchanged. The diagnostic
encoding maps below never replace a canonical score.

## Why C=2 loses ten strict points in the first pass

Baseline C=2 scores 72/76; candidate C=2 scores 62/76. Both report the correct
primary answers on all twelve cases. The candidate has one malformed response,
so the canonical parsed `result` count is 11/12. Its missing final closing brace
causes five failed checks: format, result, derivation, evidence and counterfactual.
Appending only that brace makes the complete portfolio certificate pass 6/6;
this is a diagnostic, not a regrade or a server-side repair.

| Additional candidate deductions | Points | Observed content |
|---|---:|---|
| Missing JSON closing brace | 5 | Correct primary result and certificate values; invalid response syntax |
| CDE treated as feasible | 2 | Misses the explicit C/E exclusion in the base and changed-budget feasible sets |
| ACE intermediate values | 1 | Cost/staff 18/5 instead of 17/6; no trace proving a mere transcription error |
| Ledger `after_loss` field | 1 | Writes 271 after reservations instead of 308 before reservations; final available inventory/decision are correct |
| Minimal inconsistent core | 1 | Reports U3/U4/U6 while using excluded U1/U2 implicitly |

Both arms also lose four points to logic world/witness bit-string encodings.
Their named assignments and truth classifications are correct; the response
traces support that classification. Thus the full difference is five JSON
points plus five additional certificate points, not ten wrong final decisions.

The candidate's C=1 first pass improves from 52/57 to 54/57. It still has a
portfolio dependency omission at 128K, so the new certificate mistakes are not
exclusive to the C=2 path.

## Limits on a causal explanation

This is one boot per arm with C=2 measured once. The repeated baseline C=1 run
also produces different output hashes for all five requests, despite identical
problem content and temperature zero. Cache salts are metadata, not inserted
prompt text. These observations do not isolate a C=2 kernel defect or establish
a stable concurrency-specific quality drop.

Lower activation SSE does not imply that every downstream projection or answer
improves: the dense tensor fixture itself demonstrates that distinction. Token
choices can change after small numerical changes, but logits/first divergence
were not measured here. That mechanism is a possibility, not a demonstrated
cause of these particular mistakes.

`quality-error-audit.json` retains every failing case's original grade, raw-record
and output hashes, and narrowly scoped response excerpts. Bit-string maps are
checked against both the complete oracle world set and valid witness sets.
Constraint, field-scope and minimal-core errors remain visible even when the
primary result is correct.
