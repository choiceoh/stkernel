# Decode repair CPU8: stale test fixture after SF6 merge

Source `739a7be83c7649203dae6d20157940e39c6f245f` compiled both distinct
CuTe micro artifacts and all 24 fused-preparation Triton variants. Its CPU
suite reached 43 tests and failed on 16 short-decode test cases/subcases:
the test fixture lacked the `_sf6_weight_views=None` field initialized by
the real wrapper on main. The serving implementation has that field.

This run remains FAIL. The follow-up adds the actual EP initialization
state to the test fixture and reruns the complete check. All original
artifacts, including per-pass keys, are retained in the evidence tarball.
No GPU execution or throughput was measured here.
