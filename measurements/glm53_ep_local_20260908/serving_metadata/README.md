# Configured EP launch metadata

Onepass now records the configured expert-parallel flag and TP/node/rank
topology separately from container environment knobs. Environment and command
lookups use the observed container ID so a replacement under the same name
cannot combine two containers' settings. A strict pure parser accepts the
current launcher wrapper, literal GID prelude and one serve command; unsupported
input records unknown topology. It never executes the command or stores raw
arguments. Only the exact EP flag is omitted from the comparison argv hash.

The eight parser tests and three collector/proof tests passed. The latter
were rerun after binding the environment lookup to the observed container ID;
their original output is in `onepass-tests.log`. `tests.json` records source
hashes and distinguishes the earlier parser test observation from the captured
collector output. The final source also passed 6795 core checks and 38
megakernel regressions (`core-tests.log`); this run includes zero fleet tests.
No kernel source changed and CPU9 was not rerun.

`configured-launch-snapshot.json` was derived from read-only Docker inspections
at 2026-09-08 17:42:46 KST. All four existing public containers parsed as TP4,
four nodes, ranks 0–3, EP disabled, with the same pinned image. This snapshot
tests parser compatibility with real launch inputs. It does not establish
which holder produced them, live process/kernel execution, EP candidate
acceptance, numerical correctness, quality, or performance. The private raw
input remains outside the repository; its hash and the parser hash are in
`snapshot-provenance.json`.

The head proof table now requires the actual EP-local LAUNCHED marker for
`VLLM_GLM53_EP_PREFILL_LOCAL`. The separate SP marker still proves only arming.
All-rank EP-local and MHC execution checks, capacity-preserving B1/A/B2 arms,
and an EP-aware comparator remain required. In particular, generic baseline
classification still reads `knobs` and does not reject EP-only launch changes
using the new `parallelism` metadata. These additions do not make it suitable
for an EP performance verdict.

Verify this directory with `shasum -a 256 -c SHA256SUMS`.
