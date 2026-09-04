# glm53_dflash_early_fc

The DFlash drafter's `fc` (aux hidden states [tokens, 5 x 4096] -> [tokens,
4096]; 168 MB bf16, 301 us on the MK W4 lane) needs only the target's aux
hidden states, which exist when the target forward returns -- but the stock
speculator computes it inside `propose()`, after the head GEMM, the logits
AllGather and the rejection sampler. That window is not DRAM-bound, so the fc
streams its weights there for free (ceiling ~0.3 ms/step, MEASUREMENTS 27차
census: fc is the only tail GEMM whose inputs are ready that early).

- producer: a wrapper on `GPUModelRunner.execute_model` (installed from the
  GLM model module import, like `glm53_prep_fused`); after the forward it
  cats the aux states and runs the drafter's `fc` on a side stream into a
  persistent buffer, recording an event
- consumer: `DFlash2Qwen3ForCausalLM.combine_hidden_states` (the drafter
  overlay) takes the pending buffer for this step and token count after
  waiting on the event, else runs the stock computation

The consumer runs before `precompute_and_store_context_kv` and the drafter
graph, so the fc's MK launch never overlaps another megakernel launch (the
lane's ticket barrier is one-launch-at-a-time). Numerics identical: same
kernel, same inputs. Knob `VLLM_GLM53_DFLASH_EARLY_FC=1`, default 0; any
producer failure disables it for the boot and logs. Speed only: bracket on
C=1 step/s stacked on the EXP-10 arm (the fc must be on the lane to matter).
