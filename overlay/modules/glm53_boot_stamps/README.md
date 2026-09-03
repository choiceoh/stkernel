# glm53_boot_stamps

Boot phase timing for the serving engine. Additive: two new files, no image
file replaced.

## Why

The 2026-09-02 boot reported one number for the whole middle of the boot --
`init engine (profile, create kv cache, warmup model) took 192.52 s` -- and
the only way to split it was to read the timestamps of whatever else
happened to log inside the window:

| what | measured |
|---|---|
| osar build + connect | 60 s |
| TileLang MHC pair (first) | 5 s |
| megakernel compile + arm | 43 s |
| **no marker at all** | **42 s** |
| torch.compile (2 ranges) | 10 s |
| cudagraph memory profiling | 12 s |
| **no marker at all** | **15 s** |
| cudagraph capture | 2 s |
| **no marker at all (tail)** | **6 s** |

The three compile items are cached since 2026-09-03 (MEASUREMENTS 13차,
14차), so the window should now be ~90 s, of which **63 s has no marker in
it**. That is the largest unattributed boot cost this repo has, and the
switch-fan investigation is the standing reminder of what guessing at an
unmeasured number costs.

## What it does

A `.pth` file imports `deneb_boot_stamps` at interpreter start -- the image
already ships `/usr/lib/python3.12/sitecustomize.py`, which must not be
shadowed, so a `.pth` is the additive way in. An audit hook on `import`
installs timing wrappers as soon as the vLLM worker modules land in
`sys.modules`, around:

- `Worker.load_model`
- `Worker.determine_available_memory` and `GPUModelRunner.profile_run`
- `Worker.initialize_from_config` (KV cache allocation)
- `Worker.compile_or_warm_up_model` and `GPUModelRunner.capture_model`

Each wrapper calls the original, returns its value, and logs
`[boot-stamp] <phase> took X.Xs (at Y.Ys since interpreter start)` to
stderr, which the supervisor already captures into `glm53.log`.

It measures and nothing else. Any failure to install leaves the originals in
place and logs one line. `DENEB_BOOT_STAMPS=0` disables it.

## Reading the output

The absolute offsets are what close the gaps: two consecutive stamps whose
`at` values differ by more than their own durations mean the time went
somewhere neither of them covers, and that is the next thing to name.
