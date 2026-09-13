"""Model-free kernels: their arguments are the shape.

Nothing in this package reads a model constant, a kernel shape (engine/base/kernel_shape) or the
environment, and nothing here imports a model-bound kernel package or a profile. Any profile may
use these; the engine's default lanes (engine/base/lanes.py) bind them, so a profile inherits them
instead of wiring each one. The shape wizard lists them as the `universal` lane.

    sampler           sort-free top-k/top-p over a mixed batch, one launch     (engine/base/sampler)
    block_verify      speculative block acceptance, one launch                 (engine/base/sampler)
    vocab_candidates  unique int64 candidate keys and partial-max selection    (engine/modules/vocab)
    decode_commit     accepted tokens, EOS, limit and context advance          (engine/base/lanes: commit)
    norm_rope         RMSNorm, add+RMSNorm and RMSNorm+RoPE with a warmed table (engine/base/lanes)
    swiglu            silu(gate) * up over a fused projection                  (engine/base/lanes)
    native_cache      content-addressed native builds                          (every native lane's build)

A kernel that needs a model constant does not belong here: it lives with its lane and reads the
bound kernel shape. tests/test_engine_kernel_common.py holds the boundary.
"""
