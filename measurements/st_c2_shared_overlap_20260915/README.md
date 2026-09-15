# C=2 shared expert beside the routed kernel — rejected (2026-09-15)

**Rejected. Production is unchanged: only up to `spec_k + 1` rows (C=1) overlap the shared expert.**

The candidate let C=2's sixteen verify rows run the shared-expert W4 MLP on the `SharedOverlap` stream, beside the
routed kernel, as C=1's eight rows already do.

- **Exact.** It is bitwise exact against the served serial finalizer: 0 differing bytes in 178 consumer cells, and in
  92.3 M shared-MLP outputs and 5.8 M BF16 activations at sixteen rows.
- **Not faster where C=2 decode lives.** The realistic fixture is the real L3 router on sixteen independent rows
  (U=100). There it measured:
  - warm: +0.25% (run 1), −0.00% (run 2);
  - evicted: +0.10% (run 1), −0.12% (run 2).
- **Why.** The shared MLP alone costs 27 µs warm and 57 µs evicted at sixteen rows, 1.7% / 3.5% of that FFN. The
  overlap recovered 0% of it warm and 3% evicted.
- **U20** (two requests of eight related rows) is neutral: +0.06% / +0.33% warm, −0.30% / −0.14% evicted.
- Only a single request's worth of related rows (U=10) gains consistently: −1.9% / −1.8% evicted. That is not a C=2
  decode step.

This is kernel evidence from one GB10, with no model boot and no transport. It is not an engine speed claim (D17). No
consumer, onepass, acceptance or tok/s run was made; with no component win, none is owed.

## Candidate

Frozen source `b916b9b41451e87adbd759fa369b9a169ae5e4a3`, based on main `1dd3e606` (#975). It includes #974's
`t,r,sf6,batch,q0` recipe, so sixteen rows take the batch M16 tile. The probe revision
`749437460ee82aac665be01edb655783d52af03e` adds only probe code. Both are kept on branch
`ostcode/c2-shared-overlap-candidate`; none of it is merged.

Main moved to `a3712822` (#967 one-shot consumer, #976 drafter cells) while this was written. Neither touches the path
measured here: `net._moe`, `shared_mlp.py`, the batch tile, or the shared expert's 1024×4096 and 4096×512 W4 cells.

- **Gate.** `net.shared_overlap_rows(rows, spec_k, c2=True)` returned `rows <= spec_k + 1 or rows == 2 * (spec_k + 1)`.
  24 and 32 rows kept the chain. `Glm53Net._moe(..., c2_overlap=False)` was the same-build control.
- **Proof.** `SharedOverlap` recorded the widths it joined. `boot.shared_overlap_report` demanded every overlapped
  capture width from `decode_fastpath_rows` (8 and 16), not only one call.
- **CPU.** `tests.test_engine_moe_output`, `tests.test_engine_native_execution` and `tests.test_engine_shared_mlp` ran
  in `st-engine:bracket-9c45086a0622` with CUDA hidden: 15 tests, 6 GPU skips, all passed. They cover the gate over
  1..32 rows with and without the control and finalizer, the proof's per-width demand, and that a failed join records
  no width.

## Method

`probes/engine_moe_shared_overlap.py` (lane `engine_kernel_check --lanes shared_overlap`):

- **Inputs.** The real rank-3 L3 weights of `/home/choiceoh/models/st-glm53-9391-up-gate-full/rank3of4.safetensors`
  (production's rank on srv4, folded scales). Their sha256 values are in each JSON's `identity` record.
  `sh_gate_up` is 1024×4096 and `sh_down` 4096×512, both RTN W4 packs (production packs them with GPTQ from the store;
  kernels and arithmetic are the same).
- **Lane.** `lanes.served()` with the production recipe, the real `Glm53Net._moe` consumer, the production output
  cast/add (`moe_output.combine`) and identity collectives. The router runs outside the timed graphs.
- **Arms.**

  | Arm | Rows | What runs |
  |---|---|---|
  | B | 8, 16 | served (`c2_overlap=False`): at 16 rows the serial finalizer chain after the routed kernel |
  | A | 8, 16 | the candidate: `SharedOverlap` at 8 and 16 rows |
  | F | 16 | diagnostic: the fused shared MLP (`run_smlp2`) inside the finalizer, same stream |
  | S | 8 | C=1's positive control (run 2): no `SharedOverlap` owner, so the serial chain |
  | R, H, C | 8, 16 | components (run 2): routed kernel + cast/add over a zero shared row; fused shared MLP alone; serial chain alone |

  At eight rows B and A are the same code. Their difference is the noise floor.
- **Order.** Every numerical cell ran before any timing.
- **Timing.** Graphs are captured with CUDA events inside, and a 128 MiB flush sits outside the events ("evicted").
  Each entry is the median of 32 replays.
  - Run 1: four B/A(/F)/…/B palindromes.
  - Run 2: six brackets that rotate the palindrome (B/A/S/S/A/B, A/S/B/B/S/A, …). The capture order also alternates
    by fixture.
- **Stream collision** (the 09-14 overlap v2 failure). Main's `_capture` already reuses one warm/capture stream per
  device (v3). The probe also takes that stream before constructing `SharedOverlap` and refuses an alias up front.
  Both runs recorded distinct streams (`streams`: capture/overlap handles 718700464/718700560 and
  1679035056/1679035120), and every overlap cell ran without a parent-stream error.
- **Superseded lane.** The old `moe_pair_overlap` lane compares against ordinary M32 from before #974, so it no longer
  isolates the overlap.

Tickets on the single-GPU lane (srv4 GB10), image `sha256:b45454b5fdc138bfaafa6470cf9cb49bbcd74783ceffb343b0a7b1d0cf87fcd5`
(Torch 2.13.0+cu132, CUDA 13.2):

| Ticket | Source | GO – release (KST) | Brackets |
|---|---|---|---|
| `c2so-gpu1-b916b9b4` | `b916b9b4` | 10:33:09 – 10:34:51 | 4 |
| `c2so-gpu2-74943746` | `74943746` | 10:41:05 – 10:41:47 | 6 |

**No ST serving container ran on srv4 during either ticket.** `st-glm53` had exited shortly after 10:28 on a
parked-tier mismatch between ranks. The lane admits one probe at a time, so nothing else shared the GB10. Peak
allocation was 1.81 GiB.

## Numerics — exact

Both runs found every comparison byte-identical.

| Rows | Eager draws | Shared output bytes differing | BF16 activation bytes differing | Captured replays differing | Distinct BF16 gate / up values |
|---:|---:|---:|---:|---:|---:|
| 8 | 1,408 | 0 of 46,137,344 | 0 of 2,883,584 | 0 of 88 | 8,487 / 8,422 |
| 16 | 1,408 | 0 of 92,274,688 | 0 of 5,767,168 | 0 of 88 | 8,766 / 8,746 |

- **What the table compares.** The fused shared MLP (C=1's form, and the candidate's) against the served serial chain:
  W4 gate/up, Triton SwiGLU, W4 down.
- **Activation.** The epilogue's BF16 activation (through the calibration store) is compared with `swiglu_clamped` of
  the same gate/up rows.
- **Sweep.** 11 input scales from 0 to 256, each with and without a few 64× channels. Values also differed nowhere,
  so there is no sign-of-zero case either.
- **Reading.** At these widths `mk_sigmoid`'s `expf` and Triton's libdevice exp produced the same BF16 activations
  over this coverage. No element needed the rounding-tie allowance that `tests/test_engine_shared_mlp` keeps.

| Rows | Consumer cells | A cells differing | Third-arm cells differing | Replay-to-replay differing | Dispatch (overlap calls / serial linears per two invocations) |
|---:|---:|---:|---:|---:|---|
| 8 | 74 | 0 | S: 0 | 0 | B 2/0, A 2/0, S 0/4 |
| 16 | 104 | 0 | F: 0 | 0 | B 0/4, A 2/0, F 0/0 |

The cells cover:

- unique experts 1–128, with duplicate routes at U<8;
- inputs at 0.05–8, and routes with a zero weight;
- poisoned outputs, reversed replay order, and unchanged-input checks;
- the real router at scales 1e-3–16;
- zero routed weights;
- six real-router draws per request grouping. Unique experts per draw: 16 independent rows U 97–102, two groups of
  eight U 17–21, one group of sixteen U 9–11; at eight rows, independent U 53–62.

## Timing

A vs B is the candidate against served. µs are per FFN call, as the mean of the entry medians.

| Rows | Fixture | U | Cache | Run | B µs | A vs B | F (16) / S (8) vs B |
|---:|---|---:|---|---:|---:|---:|---:|
| 16 | real router, independent rows | 100 | warm | 1 | 1579.8 | +0.25% (+3.9 µs) | F −0.07% |
| 16 | same | 100 | warm | 2 | 1583.2 | −0.00% (−0.0 µs) | F −0.04% |
| 16 | same | 100 | evicted | 1 | 1633.5 | +0.10% (+1.7 µs) | F −0.10% |
| 16 | same | 100 | evicted | 2 | 1640.3 | −0.12% (−1.9 µs) | F −0.01% |
| 16 | real router, two groups of 8 related rows | 20 | warm | 1 | 369.9 | +0.06% | F −0.44% |
| 16 | same | 20 | warm | 2 | 371.8 | +0.33% | F −0.07% |
| 16 | same | 20 | evicted | 1 | 425.3 | −0.30% | F −0.08% |
| 16 | same | 20 | evicted | 2 | 429.0 | −0.14% | F −0.05% |
| 16 | real router, one group of 16 related rows | 10 | warm | 1 | 218.1 | −0.07% | F −0.18% |
| 16 | same | 10 | warm | 2 | 211.1 | +0.20% | F −0.40% |
| 16 | same | 10 | evicted | 1 | 277.7 | −1.87% | F −0.45% |
| 16 | same | 10 | evicted | 2 | 275.2 | −1.78% | F −0.40% |
| 8 | real router, independent rows (identical arms) | 53 | warm | 1 / 2 | 873.0 / 857.4 | −1.74% / +0.18% | S −0.04% (run 2) |
| 8 | same | 53 | evicted | 1 / 2 | 904.2 / 910.2 | −0.07% / +0.82% | S +0.41% (run 2) |
| 8 | real router, one group of 8 (identical arms) | 8 | warm | 1 / 2 | 168.0 / 174.6 | −1.99% / −2.83% | S −2.34% (run 2) |
| 8 | same | 8 | evicted | 1 / 2 | 245.1 / 245.4 | +0.23% / −0.32% | S +0.66% (run 2) |

Per-bracket A vs B at U=100:

- run 1 warm: +0.45, +0.41, +0.04, +0.08%;
- run 1 evicted: −0.12, +0.25, +0.11, +0.18%;
- run 2 warm: −0.04, −0.01, +0.09, −0.03, +0.12, −0.14%;
- run 2 evicted: +1.20, −0.27, −0.80, −0.63, −0.07, −0.13%.

At U=10 evicted all ten brackets fall between −1.48% and −2.36%.

**Identical C=1 arms set the floor.** Warm, they differ by up to 2.8% at small occupancy (U=8, every bracket, both
runs). The offset stayed with B's graph although run 2 rotated bracket positions and captured that fixture in the
opposite order; its cause was not isolated. So small warm differences are not evidence. Evicted, the floor is about
±0.5%, with single outlier brackets up to +4.1% (U=53, run 2).

### Where the shared expert's time goes (run 2, µs)

| Rows | Fixture | U | Cache | R: routed + cast/add | H: fused shared alone | C: serial chain alone | B − R | A − R | Share of min(H, C) the overlap recovered |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|
| 16 | independent | 100 | warm | 1544.9 | 27.3 | 27.1 | 38.3 | 38.2 | 0% |
| 16 | independent | 100 | evicted | 1604.5 | 57.5 | 57.0 | 35.8 | 33.8 | 3% |
| 16 | two groups of 8 | 20 | warm | 334.2 | 27.1 | 27.1 | 37.6 | 38.8 | −5% |
| 16 | two groups of 8 | 20 | evicted | 394.2 | 57.2 | 56.3 | 34.8 | 34.2 | 1% |
| 16 | one group of 16 | 10 | warm | 169.5 | 27.1 | 27.1 | 41.6 | 42.1 | −2% |
| 16 | one group of 16 | 10 | evicted | 240.4 | 57.1 | 56.0 | 34.8 | 29.9 | 9% |
| 8 | independent (B = A = overlap) | 53 | warm | 825.6 | 21.2 | 22.3 | 31.8 | 33.4 | — |
| 8 | independent | 53 | evicted | 884.5 | 49.6 | 48.9 | 25.8 | 33.2 | — |

At eight rows the serial control S costs S − R = 31.4 µs warm and 29.5 µs evicted, against 31.8–33.4 and 25.8–33.2
for the two identical overlap arms.

- **The prize.** At U=100 a perfectly concurrent shared expert could save about 27–57 µs per MoE layer:
  1.1–2.4 ms of a C=2 step over 42 MoE layers.
- **What was realized.** Inside the FFN the shared expert costs the same 34–38 µs whether it follows the routed
  kernel (B) or runs beside it (A). The fused form (F) and the serial chain cost the same inside the finalizer too.
- **C=1's own overlap is also neutral at its realistic occupancy.** Serial S against overlap at U=53 is −0.04% warm
  and +0.41% evicted. So these probes cannot tell two explanations apart:
  - the routed tile leaves no spare GPU capacity at that occupancy;
  - the replayed graph gives the side stream little concurrency.

  Both lead to the same C=2 decision.

## Relation to the 09-14 numbers

`../glm53_c2_moe_20260914` reported M16 plus overlapped shared at −2.17% / −2.47% (U98, warm / evicted). That was
against ordinary M32 with the serial shared chain. The same run put M16 with the serial chain at −0.71% / −1.57%.

- The overlap's share of that was −1.5 / −0.9 points. That is inside the up-to-2.28% variation the same day's
  component tickets reported for unchanged C=1 code (`../glm53_c2_direct_20260914`).
- Measured on the served batch tile against its own serial chain, the overlap's share is 0 ± 0.3%.

## Follow-ups (not done here)

- **C=1 overlap.** Whether C=1's overlap still earns its stream is a separate, C=1 question. The positive control
  above says it is neutral at U≈53 on this build. It was not changed.
- **Old lanes.** `moe_pair_serial` and `moe_pair_overlap` still compare against pre-#974 M32 and no longer describe
  production.

## Files and reproduction

- Raw records: `gpu1-b916b9b4.json` and `gpu2-74943746.json`.
- Launch logs, with preflight and the probe's line records: `gpu1-b916b9b4.log` and `gpu2-74943746.log`.
- Tables: `python3 summarize.py gpu1-b916b9b4.json gpu2-74943746.json` (stdlib only).

To reproduce, from a frozen checkout of the candidate on srv2, through normal single-GPU admission (a new session
name; the queue refuses a reused one with different arguments):

```sh
git -C ~/stkernel fetch -q origin ostcode/c2-shared-overlap-candidate
git -C ~/stkernel worktree add --detach ~/st-worktrees/c2so-74943746 749437460ee82aac665be01edb655783d52af03e
cd ~/st-worktrees/c2so-74943746
ST_IMAGE=sha256:b45454b5fdc138bfaafa6470cf9cb49bbcd74783ceffb343b0a7b1d0cf87fcd5 ST_PROBE_TREE=c2so-74943746 \
bash bench/fleet.sh run --gpu --detach <session> 15 'C2 shared overlap gate' -- \
  bash probes/run_engine_probe.sh probes/engine_kernel_check.py --lanes shared_overlap \
    --ranks /home/choiceoh/models/st-glm53-9391-up-gate-full/rank3of4.safetensors \
    --samples 6 --output /cache/<session>.json
```
