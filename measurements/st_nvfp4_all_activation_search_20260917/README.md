# NVFP4 activation scale search extension — 2026-09-17

The user requested the same three-candidate search at all three identified
serving gaps: routed MoE FC1, dynamic-prefill FC2 and the first three dense MLPs.
The `as1` recipe selects that extension. `ss1` retains its original meaning:
search only the routed static FC2 input. They cannot be combined. `as2` retains
the optional five-candidate reference. No experts, weights, global scales,
activation/clamp semantics or accumulation arithmetic change.

## Coverage

- Static v4/v5 FC1: ordinary pack, vector loads, C=1 token cache and C=2
  register fanout, including the per-expert unequal-scale branch.
- Dynamic prefill: FC1 and FC2 in the generic and gated bodies; Q0, raw-scale
  Q0, short/long SF6 and FP8 packet-input route producers. Packet transport is
  unchanged; search scores the existing BF16-rounded inputs to FP4 packing.
- Dense MLP: FC1 and FC2 in the micro and stock-static kernels. The admitted
  dense geometry is E=1, H=4096, I=3072/rank. Long-prefill W4A16 still uses BF16
  activations, so it has no FP4 activation scales to search.
- The bound routed/dense NVFP4 geometry gates the extension. Unrelated shapes
  and formats retain their quantizers. Both in-memory and persistent compile
  keys include the option. Inherited-source pins and provenance were updated.

## Validation so far

GPU-free checks used the existing serving image
`sha256:e9e80b94d41277b171cef5785483989acd1c91d450d0d1068269e99f60bc75bd`,
PyTorch 2.13.0+cu132 and SM121a native compilation.

- 74 focused CPU tests passed (policy, quantizer oracle, profile defaults,
  source/dependency contracts, route producers, packets, SF6 staging and cache).
- 30 kernels compiled. Ten existing-ss1/extended-as1 pairs cover routed
  8/16/32 rows, dense micro/static, generic dynamic, raw/SF6 Q0, long prefill
  and packet prefill. Every enabled binary differs from its ss1 control;
  register, stack, shared and local resource counts are unchanged in all ten.
- C=1/C=2 routed kernels use 96 registers and zero stack/local memory.
  Dense/static and dynamic kernels retain their existing nonzero stack usage;
  this is not a claim that every kernel is spill-free.
- The disabled quantizer is byte-identical to the original quantizer.

Receipts: `cpu.log`, `compile.jsonl`, `compile-summary.json`. The compile source
snapshot precedes a helper docstring update and the probe's expanded fingerprint;
no compiled arithmetic changed after that check.

## Consumer gate

Pending. Compare existing `ss1` with `as1` using the same implementation and
harness, production shape, C=1/C=2 fixed 1024-token throughput, natural-EOS
acceptance and clarified ko-reasoning-v3 quality, including 32K and 128K C=1.
Quality must be preserved and C=1 output throughput must remain at least 95%
of the matched baseline. Compile success is not a throughput or quality verdict.
