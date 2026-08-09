# Runtime ID bounds audit

Date: 2026-08-09  
Scope: `aidendle94/sparkrun-vllm-ds4-gb10:production-hybrid-1.6` as deployed by this repository

## Verdict

**Repository-level result: UNKNOWN — not PASS.**

The seven files in `overlay/manifest.tsv` do not replace the sampler or MoE
files where untrusted token/expert IDs become table indices. Those files remain
inside the third-party base image, whose build source and commit are not
recorded in this repository. A repository-only review therefore cannot prove
that both bounds are present.

Run the executable audit against the actual container:

```bash
python3 launchers/audit-runtime-guards.py --container hy4
```

For an extracted or checked-out vLLM tree:

```bash
python3 launchers/audit-runtime-guards.py --root /path/to/vllm
```

Exit codes are `0=all PASS`, `1=guard missing in source (FAIL)`, and
`2=source unavailable or probe error (UNKNOWN)`. `--json` emits a
machine-readable report. Missing C++ source in a binary-only image is
intentionally UNKNOWN; library symbols or a version string do not prove the
indexing guard.

## Audited invariants

| Layer | Required protection | Upstream reference |
|---|---|---|
| Gumbel sampler | tile-local argmax is clamped to `vocab_size - 1`; its lower bound is structural because Triton program IDs and reduction indices are non-negative | [vLLM #50843](https://github.com/vllm-project/vllm/pull/50843) |
| Spec-decode sampler | both tile-local argmax sites are clamped to `vocab_size - 1` | [vLLM #50843](https://github.com/vllm-project/vllm/pull/50843) |
| DeepSeek-V4 hash-MoE | `0 <= token_id < tid2eid.size(0)` before every `tid2eid` gather | [vLLM #50844](https://github.com/vllm-project/vllm/pull/50844) |
| CUDA/Triton expert-map consumers | `0 <= expert_id < expert_map.numel()` before every `expert_map` gather | [vLLM #50845](https://github.com/vllm-project/vllm/pull/50845) |

The lower and upper checks are both required at each data-dependent table
lookup. An upper-only sampler clamp is not the downstream defense: the hash-MoE
and expert-map consumers must reject negative and oversized IDs independently.

## Decision rule

- **PASS**: all seven semantic checks are present in the deployed source.
- **FAIL**: a relevant source file is present but one or more required checks
  are absent. Do not promote the image; rebuild or backport the referenced fix.
- **UNKNOWN**: relevant source is absent. Obtain a source/commit attestation for
  the image or rebuild it from a verified revision before treating it as safe.
