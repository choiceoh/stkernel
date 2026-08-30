# glm53_drop_audit

Instrument, not an optimization. Inert unless `VLLM_DENEB_DROP_AUDIT=1`.

The Korean corruption is the canonical tokenization minus exactly one
single-byte glue token. `" 붙여"` encodes as `[Ġë¶][Ļ][ìĹ¬]` — there is no
whole-syllable token — so the model emitting `Ġë¶` is correct and only the next
step is in question. Either it emitted `Ļ` and something removed it, or its
argmax there was `여`. Nothing measured separates those: the logprobs we can read
belong to the steps that *did* produce output, and a step whose token vanished
leaves no logprob to read. Three config A/Bs (speculation off, async scheduling
inverted, fp8 KV off) each eliminated a suspect without reaching one.

## `discard_request_mask` — the only silent drop without speculation

`get_output()` calls `.clear()` on a request's tokens for the whole step when
this mask is set. The mask exists for requests still being prefilled, whose
sampled token is meaningless, and is computed as
`optimistic_seq_lens < request.num_tokens` — a runner-side counter that runs
ahead against a scheduler-side delivered count.

With `max_gen_len == 1` this is the **only** path by which a generated token
fails to reach the client, and the forward has already run, so the KV keeps what
the output loses. That is the shape of the corruption exactly.

Discarding during prefill is correct and stays silent. The audit fires only when
a request past its prompt length is discarded.

## `parse_output` contiguity — the assumption nobody checks

The rejection sampler writes accepted tokens from position 0 and leaves
`PLACEHOLDER_TOKEN_ID` after them, so valid entries are a prefix. Three places
depend on that and none verifies it:

- `_update_states_after_model_execute` counts non-placeholders and calls the
  result the accepted count, with the assumption written in a comment
- the eagle padded-prepare kernel sums the same mask
- `parse_output` filters with a boolean mask, which **closes** a hole rather
  than reporting one

If the prefix ever breaks, all three are wrong and the symptom is a missing
token. This says so instead.

## Reading it

A warning during steady-state decode names the culprit. Silence across a run
that reproduces the corruption eliminates the whole family and moves the
remaining weight onto the model itself (b12x MoE, the NVFP4 weights, CUDA
graphs).

Base contract from `glm53:v13-b12x`. Both files are whole-file replacements, so
the pins have to be refreshed on an image bump.
