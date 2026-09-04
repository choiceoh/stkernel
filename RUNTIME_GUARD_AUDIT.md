# Runtime ID bounds audit

Date: 2026-08-09  
Scope: `aidendle94/sparkrun-vllm-ds4-gb10:production-hybrid-1.6` as deployed by this repository

## Verdict

**Pinned production image: FAIL — `PASS=0 FAIL=7 UNKNOWN=0`.**

The eleven files this repository overlaid at the time do not replace the
sampler or MoE files where token/expert IDs become table indices. (Layout note,
2026-09: overlays are modules now and the manifest is composed per profile into
`build/<profile>/manifest.tsv` -- dsv4 23 rows. The verdict below is about the
image, so growing the overlay set does not change it; re-running the audit is
what would.) We reconstructed the exact
sources from the registry layers of the pinned image and ran this repository's
audit over them. All seven semantic checks failed: prod1.6 does not contain the
requested upper/lower protections.

Attested production provenance:

- image manifest: `sha256:bc99a7e29498be6c5e99714bb33ce90935ada715e76317f396d62a092582629c`
- image/config ID: `sha256:b763d81b57f7611378a514fa0faf859c3b0d0ec1010f8c5115bea11a60d49ec3`
- vLLM commit label: `fcc614141e5e9ab18cb304c476f7feed2a9552e3`
- `/opt/vllm` source layer: `sha256:6c7580da3d4d2642fece4f03d879efcbf1f352aa6020f9bb6469583cac8558d3`
- `/opt/venv` installed-tree layer: `sha256:0a8189023383d3aa06bd9fca015fa175ff87b168f467c9644860a3392a61a322`
- final Gumbel overwrite layer: `sha256:a0582d70753436b2e398077dc15822a0816298e274dd3268aceb0bc26e4d4a43`

Exact audited final-source hashes:

| File | SHA-256 |
|---|---|
| `gumbel.py` | `8b467430511d890ed452eb7a81dce772880a816bade4e9d10bd20c2d4ba31753` |
| `rejection_sampler_utils.py` | `58d1aa702766602cd295ed1039aaa1ac7396a0cdedf9200bb37fbed328afbf3e` |
| `topk_softplus_sqrt_kernels.cu` | `3feae7580f1f2e2e4a4cc81e0fadd415cc138250537a57c8eeac23f86deaa603` |
| `moe_align_sum_kernels.cu` | `54f099fb7fc4423efc3519ce4ca3ddc0a87bb6b8c09dfc8f051c84b59750bc9a` |
| `moe_fused_mul_sum.py` | `30fbb37b84b1cf6a8962d674abd9c3ab4d2be56d028fac82f55833c88d9729e8` |
| `fused_moe/utils.py` | `7707cb5fdc4945c061d0972fb7386cb13c0c164701e15bffa946a8daec25b9e8` |

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
machine-readable report. Missing C++ source in a binary-only runtime view is
intentionally UNKNOWN; library symbols or a version string do not prove the
indexing guard. That limitation does not apply to the verdict above because the
pinned registry layers supplied all six exact sources (`UNKNOWN=0`).

## Audited invariants

| Layer | Required protection | prod1.6 | Upstream reference |
|---|---|---:|---|
| Gumbel sampler | tile-local argmax is clamped to `vocab_size - 1`; its lower bound is structural because Triton program IDs and reduction indices are non-negative | FAIL (0/1) | [vLLM #50843](https://github.com/vllm-project/vllm/pull/50843) |
| Spec-decode sampler | both tile-local argmax sites are clamped to `vocab_size - 1` | FAIL (0/2) | [vLLM #50843](https://github.com/vllm-project/vllm/pull/50843) |
| DeepSeek-V4 hash-MoE | `0 <= token_id < tid2eid.size(0)` before every `tid2eid` gather | FAIL (0/2; no row bound) | [vLLM #50844](https://github.com/vllm-project/vllm/pull/50844) |
| CUDA/Triton expert-map consumers | `0 <= expert_id < expert_map.numel()` before every `expert_map` gather | FAIL (4/4 consumers) | [vLLM #50845](https://github.com/vllm-project/vllm/pull/50845) |

The lower and upper checks are both required at each data-dependent table
lookup. An upper-only sampler clamp is not the downstream defense: the hash-MoE
and expert-map consumers must reject negative and oversized IDs independently.

The opt-in DSpark top-k overlay has a separate fail-closed boundary. The launcher
accepts only `0` (disabled) or `1..129280`; the model overlay repeats the
check against the draft model's actual vocabulary, requires target/draft
vocabulary equality, and verifies a replicated Markov W2 of shape
`[vocab_size, markov_rank]` before global top-k token IDs index it. It does
not call `index.max().item()` in the captured hot path: indices are produced
by `topk` on that same validated vocabulary tensor, while static dimensions
provide the lower/upper contract without a device synchronization.

## Decision rule

- **PASS**: all seven semantic checks are present in the deployed source.
- **FAIL**: a relevant source file is present but one or more required checks
  are absent. Do not promote the image; rebuild or backport the referenced fix.
- **UNKNOWN**: relevant source is absent. Obtain a source/commit attestation for
  the image or rebuild it from a verified revision before treating it as safe.
