# GLM prefill follow-up verification

PR #439 adds FP8 unpack/MHC post fusion, independent AG/RS thresholds, and
direct grouped PyNCCL packet exchange. Both new execution paths remain off
by default; the separate thresholds inherit the existing shared boundary.

## First GPU gate: rejected

Fleet `spfused0907` ran 2026-09-07 11:46:31–11:47:10 KST on source
`66747d596f8ce1be786f13383b00128f3ac994b9`, using immutable image
`sha256:a3dd4c0f6cbb053097d65d10cd8ff8f6ae0cb9115cf0ff142e1cafe124c09211`.
The helper SHA was `bed829bc85bc94d93b500538116990e62a30df8e53de181c8e938ae720e9d056`.

Explicit SM121 compilation passed for all five listed kernels, including
`_unpack_sum_mhc_post`. All 28 simulated-rank transport cases passed.
The fused MHC zero-input case passed at 32 rows/rank, but the random case
failed bitwise comparison with the mounted production TileLang MHC post.
The batch stopped there; TP4 direct exchange, threshold sweeps and serving
throughput were **not tested** by this invocation.

Read-only inspection of the production TileLang cache's embedded PTX found
the cause: the compiler rounds `comb[0,j] * residual[0]` first and then uses
`fma(post[j], x, product)`. The initial fusion rounded `post[j] * x` first.
Those mathematically equivalent expressions can differ after BF16 rounding.
The correction follows the compiled order, preserving the preceding BF16
decode rounding as well. A deterministic cancellation fixture uses x=1,
r0=1.5, comb0=1+2^-23, and post=-(1.5+2^-22): stock gives +0, while the
incorrect order gives -2^-24. The GPU probe now checks this boundary.

Raw log: `srv2:/tmp/glm53-prefill-followup.aRyhm1/fleet.log`.
Traffic audit: `srv2:/tmp/glm53-prefill-followup.aRyhm1/traffic.jsonl`.
Reference PTX is embedded in the production cache entry
`tilelang/0.1.12/linux-aarch64/kernels/ca8d6b8e3e7290c732146d1a8a6e7b4f79530eb813f6a3d86cfe3f78a47af006/executable.so`.

## Corrected GPU gate: numerics passed; memory incident

The retry on source `52b23e229231e470236d36ea74bfebc352d512ea` (main
`e1a88d7` included) started under the same fleet session at 12:01:38 KST.
All 28 fused MHC cases passed, including the deterministic rounding boundary,
bit-exact post/pre continuation at 33/532/1728 local rows with and without
RMSNorm, repeated consumers, and nondefault streams. Actual TP4 passed 180
raw-codec cases plus 80 independent-threshold cases (260 total). Both actual
direct exchanges and deferred packet/MHC consumers were exercised.

However, the probe's admission was insufficient: an idle serving process
still held most of the GB10 UMA memory. At 12:01:25, available memory was
9,579 MiB; at 12:01:58 it fell to 6,048 MiB (4.93%). `earlyoom` sent SIGTERM
to serving worker PID 777975. The API exited at 12:02:06. The resulting
microtimings are exploratory offline results, not clean serving evidence.
The arithmetic comparisons remain valid. No serving throughput claim is
derived from this invocation.

Recovery ran through fleet session `spfusedrestore` at 12:06:37–12:11:12,
using the pinned image and production defaults. The launcher observed health
200 before the fleet was handed to the next queued offline experiment.
Probe launchers now check MemAvailable on every participating node before
starting a GPU container, reserving 8 GiB for the probe plus 10% of host RAM
(at least 8 GiB) for existing work. Missing counters and inadequate headroom
fail closed; there is no override. Each subsequent single-GPU phase rechecks.
The observed incident values and threshold edge are covered by CPU tests.

Exploratory observations used to choose the serving candidate:

| Operation / rows | Existing path | Candidate | Decision |
| --- | ---: | ---: | --- |
| Decode + MHC post / 1728 local | 842.11 us | 735.23 us fused | Test fusion in serving |
| All-gather / 2128 global | 1013.62 us BF16 | 795.68 us v3 | Test AG threshold 2048 |
| Reduce-scatter / 2128 global | 1049.87 us BF16 | 1111.54 us v3 ProcessGroup | Retain RS threshold 4096 |
| Reduce-scatter / 4095 global | 1494.93 us ProcessGroup | 39723.73 us direct | Keep direct path off |
| Reduce-scatter / 4096 global | 1281.31 us ProcessGroup | 11458.94 us direct | Size cliff; no promotion |

These are per-arm medians; mirrored per-round ratios and all samples are in
the [raw evidence](GLM53_PREFILL_FOLLOWUP_20260907.json). The direct arm also
has high variance, so ratios of independent medians must not be described as
its paired estimator. No per-kernel percentage is an end-to-end gain.

## Serving gate: failed at 128K; no A/B verdict

Fleet `spfusedserv` ran at 12:15:55–12:25:25 KST on the numerically tested
source `52b23e229231e470236d36ea74bfebc352d512ea`, with `FUSE_MHC=1`,
AG minimum 2048, RS minimum 4096, and direct exchange off. Before boot,
deployment verified all 56 overlays and the manifest on all four nodes.
The actual serving overlay stamp was `284770d1a222`; the initial fleet
deployment message still showed the previous cache stamp `f2cfb08ce330`.
The onepass record uses the actual new stamp and has **3/3 serving proof**.

The exclusive onepass began at 12:23:12 and completed ten requests through
32K with all twelve retrieval checks passing. The 128K request returned no
content or usage because the head worker died. At 12:25:11, earlyoom observed
6,114 MiB available (4.99%) and sent SIGTERM to host worker PID 884391;
the engine reported its death at 12:25:23. This occurred during serving
alone, without the additional probe containers from the earlier incident.
It is a separate long-context headroom failure. The new probe-admission
guard does **not** solve it. The journal establishes host memory pressure
as the immediate cause; it does not isolate the candidate's contribution
without a baseline under the same resource conditions.

The recorded result is **12/15 retrieval, 0/11 Korean corruption**, with
invalid evidence flags `requests remain after the workload` and
`completed requests 10 != own requests 11`. These flags reflect the aborted
request; they are not evidence of external traffic. The 128K field's
40.88 seconds is time until the failed stream ended, **not a valid TTFT**.
The complete arm is excluded from performance conclusions, including its
otherwise completed shorter requests. The baseline arm never ran.

The fleet failure handler released the slot because the queued boot job
`dec3follow0907` took ownership at 12:25:35. A second recovery by this
experiment would interfere with that owner, so none was launched. The
earlier 12:11 recovery remains verified. At **12:36:50 KST**, a read-only
observation of the subsequent `nvs3` defaults boot confirmed **health 200**,
the same immutable image, the existing SP v3/shared-4096 settings, and none
of this PR's new flags in the head container environment. That boot's stamp
is `f2cfb08ce330`, so it is not a baseline for our `284770d1a222` candidate.
This confirms restored availability; it is neither a recovery launched by
this experiment nor acceptance of the next owner's still-active experiment.

**Decision:** keep fusion and direct exchange off, and leave both thresholds
inheriting the current shared default. Direct exchange is rejected for this
candidate due to its size cliffs. Fusion and separate thresholds remain
unpromoted pending a valid matched serving pair with sufficient UMA
headroom. The original 40% throughput target remains unproven.

Raw serving record, journal, failure-log tails, fleet log and surviving-worker
attestation are included under `serving_attempt` in the JSON evidence.

## Scheduled retest preparation

The 13:30 KST read-only check found 143 GiB available disk space on srv1
and no new earlyoom terminations since the 12:25 incident. The fleet was
owned by `dec3step0907`, so no deployment or GPU work was started then.
Main `0b6dc75` has since promoted calibrated NVFP4 scale 16; it is merged
into this branch and will be common to both arms of the new comparison.

The retest uses a dedicated onepass ledger and identical **KV_TOKENS=524288,
MAX_LEN=262144** settings on both sides to leave additional UMA headroom.
The five requested contexts still fit; this is a comparison with reduced
cache capacity, not acceptance at the production 2,000,000-token cache and
1,048,576 maximum length. The test runner restores production capacity after
the bracket if no subsequent boot owns that responsibility.

`ONEPASS_MEMORY_DIR` enables an opt-in guard around the unchanged onepass
workload. It samples MemAvailable on all four nodes before starting its
client and throughout the run. Below 10 GiB, missing memory counters or a
failed SSH observation causes the guard to terminate only its own onepass
client, closing its requests and failing the fleet leg. It never signals
serving workers or changes earlyoom. The guard is sampled and cannot reserve
RAM or guarantee protection from instantaneous spikes. Four model-free
behavioral tests cover malformed counters, refusal before launch, propagation
of the child's exit code and cancellation after a memory drop. They pass,
as do 14 transport dispatch tests, four probe-admission tests, and the existing
6,684 logic checks with 30 megakernel and 32 fleet regressions (the same
torch-dependent host skips remain). Device code is unchanged from the
numerically tested helper. Serving results remain pending.

The retry was admitted as fleet `spfrt0907` at 13:56:11 KST on immutable
source `6f797df28c29e7e4cb606724e2416dbc1c5dcfcc`. All 56 overlay files and
the manifest were verified on four nodes before the first defaults boot at
13:57:10. The planned order is `SPFRT0907B1` / `SPFRT0907A` /
`SPFRT0907B2`, with a public production-capacity restore when required by
fleet ownership. Requests use loopback port 18000 and the exclusive onepass
gate. Run directory: `srv2:/tmp/glm53-prefill-retry.3rfahoko`; its dedicated
`onepass.jsonl` must not be mixed into production-capacity baselines.


The first retry baseline completed at 14:10:44: retrieval **15/15**, Korean
corruption **0/11**, 11 completed requests and no traffic evidence issues.
The 128K TTFT was **41.401 s**. The memory guard recorded 106 observations,
with minimum available GiB of **20.700 / 26.239 / 21.785 / 24.250** on
srv2/srv1/srv3/srv4 and no guard issues. This reduced-capacity workload did
not reproduce the earlier memory termination.

The post-leg attestation then failed because its original implementation
attempted to read the head-only `glm53-cache/.overlay-sha` on workers. This
is an experiment-harness error, not a failed model request. Chain recovery
started before the candidate; no A/B conclusion follows from this run. The
corrected attestation verifies the deployed manifest and all 56 mounted
source hashes against the composed pinned tree on every node, and checks the
cache stamp only on the head. A read-only check on all four recovery
containers passed. It also binds the head container ID/start time to the
onepass record. The recovery observation is not represented as an
attestation of the earlier baseline container.

A fresh bracket was registered at 14:14:26 as `spfrt20907` in
`srv2:/tmp/glm53-prefill-retry2.7gy5_ruh`, with the same immutable source,
image and workload. Its arms are `SPFR20907B1` / `SPFR20907A` /
`SPFR20907B2`; only the external attestation helper changed. It waits for
the previous owner's recovery and does not bypass the queue.


At 14:17:16 the corrected bracket acquired the fleet, then refused **before
deploy**: srv1 had only **18.9 GiB** free disk, below the runner's 32 GiB
floor. No candidate ran. A read-only inventory found **five 44.8 GiB rank
cache artifacts (224 GiB total)**; the entire GLM cache occupied about 264
GiB. The latest three rank-cache writes were 13:33, 13:51 and 14:03, during
the interval when available disk fell from the earlier observation. No data
was deleted. This admission failure is a disk constraint, separate from the
previous srv2 RAM termination. The first baseline remains a completed
workload with incomplete post-leg attestation, not an A/B speedup verdict.

The original recovery had retained the test capacity/private endpoint because
a following boot was queued. The pre-deploy refusal therefore required an
additional official recovery of public port 8000 and production capacity.
That recovery is tracked with the raw results in the [scheduled retry
report](../measurements/glm53_prefill_retry_20260907/README.md).

Public recovery `SPFR20907PUBLIC` finished at **14:25:02 KST**. Verification
confirmed the pinned image and expected defaults on all four nodes, public
port 8000, max length 1,048,576, block override 1,056, and health **200**
from both srv2 and srv1. The repeat automation is paused pending sufficient
disk capacity. The candidate remains unmeasured in this retry.
