The failure remains unresolved and is not waived. Current data does not
identify the failing row or its exact per-row noise-scaled limit. The
concentrated fixture always rewrites all eight cache slots and assigns 6912
rows (54 M128 tiles) to each selected expert, so no padding or shrinking-slot
edge is involved. This reduces suspicion of those specific cases; it does
not prove race freedom.

Historical attempt4 used the same expert/route fixture and comparison logic.
Its changed-input candidate peak maxima were0.03825137; the current overall
maximum is0.04069768. Relative L2 remains below0.02. The inherited candidate
uses BF16 route multiplication and atomic accumulation, whereas compact
accumulates per-pair outputs then performs index_add_. Accumulation grouping
and scheduling are plausible contributors, not an established root cause.
No old timing or near-boundary result can override this FAIL.

After final fleet restoration, the next bounded diagnostic should retain the
original fixture and gate, keep the first FAIL permanently, and capture up to
eight bad row IDs with exact L2/peak/noise/limits and reference norm/maxima,
top8 routes/weights/scales, worst columns and B1/B2/B3/C1 BF16 raw values.
At most two additional same-input candidate results can measure candidate-self
variation without retrying until PASS. If the same error persists beyond self
variation, isolate CPU14 versus CPU16 under the same capsule or compare Q0
packed inputs/scale bytes before changing arithmetic. This diagnostic is not
implemented or submitted by this preparation; no tolerance change is authorized.

collect-after-terminal.py is a prepared read-only terminal collector, not an
executed result. It requires real fleet exit/release/restore proof before
creating gpu5-completed. The current closed capture stays pending-restore.
