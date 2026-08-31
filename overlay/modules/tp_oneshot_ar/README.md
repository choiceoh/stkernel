# tp_oneshot_ar

Host-register RDMA one-shot AllReduce for small decode tensors (27us against
NCCL's 67us on this fabric). Both files are new, so this half is portable;
the CUDACommunicator.all_reduce hook is per-image and lives in a
`*_oneshot_wiring` module.

Armed by `VLLM_DSV4_ONESHOT_AR`. Every failure path falls back to NCCL.

One kernel launch per collective (`k_oneshot`, #89's 5->3 followed by 3->1):
ring guard and peer wait spin on globals inside the kernel, and the publish
step is taken by the last block via a never-reset monotonic completion counter
(`done_ctr % ARGRID`), which is what makes it cudagraph-replay-safe.
