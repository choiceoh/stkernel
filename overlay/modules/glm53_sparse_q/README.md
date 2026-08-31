# glm53_sparse_q

Probabilistic draft sampling without the dense-vocab tax. The rejection ratio
test reads the draft distribution at exactly ONE index per position (the
sampled token); the fork's dense pipeline nevertheless materializes
[tokens, vocab] probs, caches [reqs, k, vocab] (17-69 MB persistent), stages
row copies in the runner, and sweeps q over the full vocab in the recovery
kernel. #102's measurement: +12.5 pct tokens/step bought, ~6 pct step/s paid.

This takeover gathers q(token) at the sampling site and carries [num_tokens]
floats through the pipeline. Acceptance is bit-identical (the ratio test is
exact either way); only the post-rejection replacement token changes law,
from the exact (p-q)+ residual to a target sample -- the law this stack
served before probabilistic mode. VLLM_SPEC_GATHER_Q=0 restores the dense
exact-residual pipeline (rollback).

Gates: pos-1 acceptance unchanged vs the dense-probabilistic arm (+3.4 pct
points must survive), 9/9 retrieval, 0/16 Korean. Expect step/s to recover
most of the ~6 pct; if it does not, the residual cost is the fp64 Gumbel
pairing, a different lever.

## Candidate-restricted sampling — `VLLM_SPEC_CAND_SAMPLE` (default off)

The probabilistic path still paid for a full-vocab draft law every step:
softmax over [T, 154880], top-(Q_POOL) over the probs, and a float(leak)
host sync. The drafter's support is the candidate head's top-k (16) --
`compute_candidates` already reduces to it without any full gather. With
this knob the draft token is sampled from softmax over those k logits and
q IS that k-way categorical: the (p-q)+ recovery stays exact BY
CONSTRUCTION (q is zero outside the support -- no leak check, no sync, no
full softmax/topk). The law change vs the dense path is only the
renormalization over the support, which the old leak check bounded at
<=1e-3 of tail mass. RNG, greedy-row and K-row temperature discipline are
mirrored 1:1 in `_candidate_probs_and_sample` (tested bitwise-of-formula in
tests/test_logic.py, torch-gated). Rollback: the env alone.
