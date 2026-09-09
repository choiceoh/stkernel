# Onepass6 started: identical decode candidate, complete Git history

The onepass5 retry uses a separate checkout with complete Git history. Before submission, the exact candidate HEAD/tree, clean worktree, non-shallow status and current main ancestry were verified. The original failed checkout and log are preserved under [onepass5-failed](../onepass5-failed/README.md); no ancestry or deployment guard was bypassed.

Session `eplocalonepass0909v6`, ticket `17889071393096545`, entered at **2026-09-09 07:38:59 KST**. Frozen source remains `e2a54cff881465c2bb7dbbd3f5ec39ca240c7f74`. B1/A/B2 use the identical candidate controls and canonical workload from [onepass5-queued](../onepass5-queued/README.md), including standard prefill contexts and three fixed 1024-token decode requests. The complete 31-test CPU evidence there applies to this same source/tree; no kernel, test or workload changed for this retry.

The collected status is a startup snapshot, not completed measurement or proof of recovered decode throughput. The owned passive observer starts four log streams only after GO and captures strict A/B1/B2 identities privately when ready. No default promotion, merge or deployment to public serving is claimed.
