# CPU12 Q0 dual-warp PTX audit

Scoped verdict: **no blocking issue found in emitted PTX and source**. Frozen source `b36dc7f1b19e471fa3280536e244f1711311eaea`; actual CPU receipt is 239/9 PASS. This is not GPU correctness or performance acceptance.

The exact two PTX copies match both the CPU receipt and the independent original archive. Baseline key21/False and candidate key22/True bind the intended constructor; both report REG168 / STACK112 / static SHARED1024 / LOCAL0 and 288 threads.

In `candidate.ptx`, the live batch loop starts at 1167 and returns at 2960. The route-state store (1663) precedes the common CTA barrier (1666); the inactive-token predicate is applied afterward (1667). The input-copy acquire waits remain at 1669/1675. Route-state (1681), first scale (1687), row (1941/2106), scale-row offset (1952), and unequal route scale (2110) loads are after that barrier and inside the batch loop. No cross-batch hoist or reuse was observed. Existing final CTA/fence publication is at 2962–2966.

Warp division by two is emitted at 1143–1151, odd-warp offset32 at 1152–1154, and SF stride64 at 2953. The candidate adds one CTA publication barrier per live batch in place of the single-warp sync; shared allocation declarations do not change. All inactive tail workers and the DMA warp join the common barrier. Payload ownership remains four token rows over eight math warps.

From the first post-Q0 global fence onward, both PTX tails have 20,635 noncomment lines and are identical after consistent register/block-label renaming. Source comparison with parent21531085 changes only the Q0 method and its two host admission methods. SF6 FC1/Q1/FC2/scatter source and native source are byte-identical.

No `.loc` source debug mapping is present: line references above are PTX file lines, correlated through operands and control flow. This does not inspect final SASS or prove race freedom; GPU startup numerics and replay remain required. STACK112/LOCAL0 is not a no-spill claim; actual dynamic/total shared bytes remain unmeasured/null. No additional workload or source mutation was performed.

Exact hashes, resources and line map are in `audit.json`.
