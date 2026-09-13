# Decode boot, packet and component failure handling

This follows the native decode changes in #904 and #906. MoE output remains
on by default. Its CUDA/Triton arithmetic, rounding, graph shapes and normal
packet launch order are unchanged; `source.json` records that boundary.

The `expert-eval0913-MAIN906` boot at `91280525b148` failed on all four ranks
in `DrafterDecodeGraphs.observations`, before serving opened. The direct
traceback is `observe_committed -> _observe -> write_draft_kv`:
`AttributeError: 'int' object has no attribute 'numel'`. `boot-failure.log`
preserves rank 0's shape/configuration and traceback. The complete rank logs
remain under `/home/choiceoh/glm53-logs/st-bracket-dumps/` in
`expert-eval0913-MAIN906-hold-91280525b148/` on srv2. The boot's rank-0 image
was `sha256:1fdddb4c04fe77eaabba7076d983226c8ff15c4f541661ff04f83d20d357ea28`.

`observe_committed` had passed `positions.numel()` as a KV-write count when
decode calibration was active. The collector accepted that integer for a
mask comparison; the native writer required an int64 device tensor. That
call originated in #862, before #904/#906; the serving calibration default
now exercises it. The native writer now receives an int64 device scalar
created with a capture-safe constant fill, matching the independently
prepared #909 fix already undergoing a fleet boot. The collector keeps its
explicit committed-row mask. Masked asynchronous observations still pass
their original device count. Precision, calibration policy and fused MoE
remain enabled as before.

The old packet wrappers could enqueue a native exchange and then raise while
leaving the ring reusable. Ordinary exchange, direct producers and MoE output
now share one publication owner: it claims the ring before the first native
call and poisons it on any later failure, including descriptor construction.
A packet consumer cannot recursively consume its descriptor or return on a
different CUDA stream. Invalid input checks still happen before publication.

The decode bundle formerly killed only the immediate Python child on timeout.
Each component now owns a process group; timeout and INT/TERM cancellation
kill that group's descendants and reap its leader. A cancelled bundle exits
without starting another component. Start failures and timeouts retain their
own result and allow independent later checks, while cleanup errors abort the
bundle. The result scope no longer says serving defaults are disabled.

Validation is host-only: 71 packet/component CPU tests pass, with five
GPU-only skips (76 total), plus 46 drafter/calibration CPU tests with ten
GPU-only skips (56 total), in the pinned ST image
`sha256:09d9ba96a4c7e1113f91100b892a94c1ab859dae8e46db3e7b02dfa2564f93bc`.

- `before.log` reproduces four failing packet cases in the original code:
  both exchange APIs remain reusable after a native error, recursive consume
  succeeds, and a consumer's stream change goes unnoticed.
- `cpu-tests.log` includes those regressions, descriptor failure, four-rank
  C=1..4 output/state comparisons, and existing communication contracts.
- `draft-before.log` reproduces the real boot exception at all eight K=7
  observation lengths, 1..8. `draft-cpu-tests.log` includes the passing fix
  and related drafter, precision, calibration and ring-write contracts.
  The regression replaces only the Triton launch: it retains the real
  `observe_committed`, dense observer and `write_draft_kv` host checks, then
  checks the entire ring and exact calibration Gram/count. Device-counted
  accepted prefixes 0/1/7/8 retain their original tensor and untouched cells.
- The process tests use real Python children and grandchildren, including a
  grandchild ignoring TERM. Timeout/cancellation stop that group, leave an
  unrelated process alive, restore signal handlers and preserve exit codes.
- Native OneShot build inputs and the build function are unchanged, so prior
  compilation evidence is reused. The failed boot's image differs from the
  pinned CPU image above; this is host-contract evidence, not a successful
  fleet reboot or a GPU/engine speed claim.

The admitted component source and ticket remain frozen:
`st-decode-output-bundle0913v2`, ticket `17893109571846042`, source `12c9ce9a7`.
It validates the same kernel arithmetic. These new host failure checks run on
CPU; the existing reservation is not restarted to repeat that kernel gate.
