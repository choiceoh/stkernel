# qwen38_oneshot_wiring

Hooks `CudaCommunicator.all_reduce` in the `qwen38-fi618:local` image so small
decode tensors try `tp_oneshot_ar`'s host-register RDMA one-shot AllReduce first
(27 µs against NCCL's 67 µs on this fabric; GLM-5.3 serves with it). Whole-file
OVERRIDE, image-specific by construction — generated from the image's stock
`cuda_communicator.py` by `tools/qwen38_oneshot_wiring_gen.py` (stock SHA pinned,
one anchored edit: stock `all_reduce` → `_all_reduce_impl`, wrapper on top; the
wrapper text is `dsv4_oneshot_wiring`'s). The kernel and shim are `tp_oneshot_ar`'s
(`requires`).

Armed by `ONESHOT_AR=1` in the profile (launcher: `VLLM_DSV4_ONESHOT_AR`). Off by
default until bracketed on this model: a TEP=4 decode step carries 48 × 2 small
all-reduces, so the upper bound is ~4 ms of a 56 ms step. Pre-commit failures
vote back to NCCL; post-commit faults are fatal on purpose (see the shim).

```
S=<image>/usr/local/lib/python3.12/dist-packages/vllm/distributed/device_communicators/cuda_communicator.py
python3 tools/qwen38_oneshot_wiring_gen.py --stock $S --check overlay/modules/qwen38_oneshot_wiring/cuda_communicator.py
```
