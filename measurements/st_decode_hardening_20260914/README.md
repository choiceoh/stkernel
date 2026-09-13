# Decode packet and component failure handling

This follows the native decode changes in #904 and #906. MoE output remains
on by default. CUDA/Triton arithmetic, rounding, graph shapes and normal
launch order are unchanged; `source.json` records that boundary.

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

Validation is host-only: 71 focused CPU tests pass, with five GPU-only skips
(76 total), in the pinned ST image
`sha256:09d9ba96a4c7e1113f91100b892a94c1ab859dae8e46db3e7b02dfa2564f93bc`.

- `before.log` reproduces four failing packet cases in the original code:
  both exchange APIs remain reusable after a native error, recursive consume
  succeeds, and a consumer's stream change goes unnoticed.
- `cpu-tests.log` includes those regressions, descriptor failure, four-rank
  C=1..4 output/state comparisons, and existing communication contracts.
- The process tests use real Python children and grandchildren, including a
  grandchild ignoring TERM. Timeout/cancellation stop that group, leave an
  unrelated process alive, restore signal handlers and preserve exit codes.
- Native OneShot build inputs and the build function are unchanged, so prior
  compilation evidence is reused. No GPU or engine speed claim is made.

The admitted component source and ticket remain frozen:
`st-decode-output-bundle0913v2`, ticket `17893109571846042`, source `12c9ce9a7`.
It validates the same kernel arithmetic. These new host failure checks run on
CPU; the existing reservation is not restarted to repeat that kernel gate.
