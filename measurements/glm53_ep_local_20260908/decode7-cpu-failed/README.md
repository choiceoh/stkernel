# Decode repair CPU7: artifact filename collision

Source `54eef0fba65459d938e7a9d81365f7f648d6e2c6` reached both expected
CuTe cache keys (candidate M32 and control M64). The two compile dumps used
the same generated basename, so the second replaced the first. The required
two-artifact check failed before Triton lowering or CPU contracts. This run
is not a complete CPU PASS and contains no GPU execution or throughput.

The preserved artifact is the final dump only. The follow-up changes the
probe to move each pass's fresh output into its own directory immediately;
it does not change either serving kernel or the workload.
