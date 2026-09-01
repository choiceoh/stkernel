# dspark_drafter

DSpark drafter: model, speculator and its utils. One unit -- the speculator
calls into the model's `compute_logits` and Markov head.

The FP8 copy helper separates UE8M0 requantization from DeepGEMM layout
packing and validates the post-requant scale bits first. Invalid scales raise
on the host instead of reaching the packer's device-side trap; positive zero
remains valid for all-zero/padded weight blocks.

After DFlash block tables are initialized, the speculator also JIT-warms the
active rejection-sampler dtype signatures and every DSpark input-preparation
`BLOCK_SIZE` bucket (8 through 256) with anchor sampling enabled. The warmup
is best-effort and defaults on; `VLLM_DSV4_SPEC_WARMUP=0` is the rollback.
