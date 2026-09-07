# FP4 byte address diagnosis and correction

The M1/U8 GPU graph checks passed, but M2/U8 failed with max absolute error
3.5 versus limit 0.1875. A device-free inspection of the actual layouts found:

- A2 has byte swizzle `S<2,4,3>` and an outer layout in FP4 nibble units.
- The original candidate applied the swizzle to nibble offsets, then divides by two.
- The consumer and existing M32 store use swizzling of byte offsets instead.
- Row 0 agrees. For row 1 / column 0 the candidate stores byte 72; the
  existing consumer mapping requires byte 64. Row 4 gives 256 instead of 288.

The compile-time uniqueness check was insufficient: a permutation can be
bijective and still disagree with the consumer. The correction is to
convert the outer offset to bytes before applying the byte swizzle, and
compare every write address against the actual consumer mapping.

The original speed-only queue was cancelled when the user requested fixing the
numerical error in the speed candidate too. The corrected store converts the
outer nibble offset to bytes, then applies `byte ^ ((byte >> 3) & 0x30)`.
Compilation now checks every byte against the consumer map, including rows 1
and 4 that exposed the bug. All three tile changes remain enabled together.

Evidence: `cpu/address-inspection.log`, `cpu/address-inspection.py` and
`gpu-numerics/bundle.log`. The inspection is CPU-only and is not proof that a
future correction passes GPU numerical/graph checks.
