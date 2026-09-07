# Serving3 stopped at candidate log collection

B1 and A each completed priming and measured 2K/32K/128K requests. Both
phases in both arms scored retrieval 9/9 and Korean corruption 0/5, with
fresh salts and no traffic/cache issues. B2 did not run; there is no valid
bracket and no measured speedup verdict.

The collector incorrectly read Docker logs even though the launcher
redirects both stdout and stderr to `/glmlogs/glm53.log`. All four archived
Docker logs were empty. The head boot log independently records
`mla prefill32 LAUNCHED T=8192 W=2176`; the failure does not demonstrate
that the candidate fell back. Worker A file logs were overwritten by
recovery before this collector defect was diagnosed, so all-rank proof
cannot be reconstructed for this run.

Measured request TTFT seconds (2K three requests, 32K, 128K):

- B1: 0.884086, 0.878051, 0.871948, 10.465541, 41.165352.
- A: 0.876067, 0.886682, 0.862964, 10.532638, 39.032065.

These incomplete raw timings are preserved, not promoted to a performance
claim. The public default arm was restored with verified four-node
source/image/config identity and health 200 at 21:54:22 KST; completion
reports exit 1 and restored true. No default changed.

The corrected collector reads the file identified by the inspected
container's log bind mount and exact truncating shell redirection. It
rejects an empty/stale log, changed inode or restarted/stopped container,
archives each rank before proof evaluation, and saves the final incomplete
snapshot even when candidate proof is missing. CPU regression tests exercise
the actual embedded collection code with real temporary files.
