# dsv4_oneshot_wiring

Hooks `CUDACommunicator.all_reduce` so small decode tensors try the
one-shot path first. Whole-file replacement, so the contract is
image-specific: `aidendle94/sparkrun-vllm-ds4-gb10:production-hybrid-1.6`.

The kernel and shim are in `tp_oneshot_ar`.
