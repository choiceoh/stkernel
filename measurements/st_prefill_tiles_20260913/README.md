# Tile-ready KDA input projection on GB10 TP4

Implemented, default off (`prefill_project_tiles=0`). No serving speedup is
claimed. This is the second experiment after the direct MHC receiver.

Prefill's local token shard is exchanged in at most four balanced tiles.
Two explicitly owned slots hold packed payload, receive bytes and BF16
values. A CUDA event releases each slot only after its input projection and
rank-major output copy. The next tile's communication can run while the
previous tile's KDA input GEMM runs on the normal compute stream.

The FP8/BF16 decision uses the original total row count (4096 threshold),
never the smaller tile count. FP8 transport blocks remain per 2048 values;
each 4096-column row keeps the same scales and BF16 decode boundary. Input
projection uses the existing weights and per-token, 128-column quantization.
All projected rows are restored to original rank/token order before the
unchanged full-sequence FP32 KDA recurrence. DSA and FFN paths are unchanged.
The chunk-order and layer-major entry points both support this path.

The latest main increases prefill chunks to 32,256 tokens. The tile count is
bounded at four across that change; balanced tails remain in the FP8 prefill
GEMM path instead of entering the <=32-row W4 decode path.

Validation:

- The 30-test CPU ST-image suite passed before integrating the latest main:
  true LocalTP rank gather/sum, exact output/auxiliary/cache-state comparison,
  two-slot ownership, execution plans, defaults and boot paths.
- The registered `probes/engine_prefill_tiles_check.py` tests real CUDA stream
  readiness/reuse, delayed synthetic peers, the precision threshold and exact
  FP8 projection against the ordinary full gather. This is not NIC proof.
- Real TP4 and full onepass C=1/C=4 quality, tok/s, TTFT and per-item latency
  remain pending. Extra NCCL launches can offset overlap; keep default off.

Stream synchronization follows [PyTorch's CUDA collective contract](https://docs.pytorch.org/docs/stable/distributed):
`Work.wait()` orders the active CUDA stream; an explicit event orders the
consumer on the other stream. No partially completed collective is read.
