# glm53_oneshot_wiring

Hooks `CUDACommunicator.all_reduce` so small decode tensors try the
one-shot path first. Whole-file replacement, so the contract is
image-specific: `glm53:v13-b12x`.

The kernel and shim are in `tp_oneshot_ar`.
