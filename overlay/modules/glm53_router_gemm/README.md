# glm53_router_gemm — the cuBLAS router tier GB10 was excluded from (vLLM #54048)

`gate_linear.py` replaces `vllm/model_executor/layers/fused_moe/router/gate_linear.py`
(image build `0.1.dev20051+g487ecf187`, base sha in `manifest.tsv`).

## Why a module of its own

The subject belongs to `moe_gate_sm121`, but that module is shared with the
`dsv4` profile and the two images ship **different** `gate_linear.py`
(glm53 `6fca8c7f...`, `production-hybrid-1.6` `b9cfa27c...`). One preimage row
cannot serve both, and putting the glm53 hash in the shared module would abort
the dsv4 deploy on the first file it verifies.

## The defect

The image gates the fused bf16 x bf16 -> fp32 router GEMM on
`allow_specialized_router_gemm`:

```python
is_hopper   = current_platform.is_device_capability((9, 0))
is_blackwell = current_platform.is_device_capability_family(100)   # :60
can_use_specialized_kernels = is_cuda() and (is_hopper or is_blackwell) and not bias
```

GB10 is compute-capability **family 120**, so both arms are False and the tier
is skipped — even though that tier is nothing but `torch.mm`'s `out_dtype`
epilogue (plain cuBLAS, no NVVM or CuteDSL codegen) and runs anywhere. The
router therefore falls to bf16 `F.linear` plus a standalone bf16 -> fp32 copy,
which bf16-rounds the logits on the way.

## The change and why it is off by default

`allow_cublas_router_gemm` gets its own family-120 arm, at both the constructor
and the `set_out_dtype` recompute. It is armed by
`VLLM_GLM53_ROUTER_CUBLAS_F120=1` only.

Default 0 because of a local interaction: `DenebGateLinear` in
`moe_gate_sm121` refuses to arm unless every accelerated tier above the
`F.linear` one is off, so turning this on **disarms** the measured M <= 32
fused gate (1.71 -> 1.18 ms/step, C=1 +3.5 % on DSv4). The two want composing —
cuBLAS for prefill's M > 32, the Triton split-K for decode — and the bracket
that decides how is the reason this knob exists.

Numerics: enabling it removes a bf16 round trip, so prefill router logits land
strictly closer to the fp32 reference but are bit-different from baseline.
Same reason `moe_gate_sm121` carries a kill switch.
