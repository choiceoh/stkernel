# Source attribution

This package includes ST's forks of kernels from vLLM, Flash Linear Attention,
and FlashInfer. The per-file source paths and pre-port SHA256 hashes are in
`SOURCES.json`. Original copyright and SPDX headers are retained where present.

- vLLM contributors: KDA integration, causal convolution, kpool, and mHC
  sources, under Apache License 2.0 (see this package's `LICENSE`). The
  Qwen3.8 QSA kernels in `qsa.py` are ported from vLLM's
  `models/qwen3_8_flash_next/nvidia/ops/qsa.py` under the same license.
  The QSA tile-union prefill kernels in `qsa_tile_union.py` are ported
  from vLLM pull request #55430 (`vllm/models/qwen4_exp/nvidia/ops/
  qsa_tile_union.py`, jschmied/vllm commit
  `c5d7eba35823043331b295fa2359e4bd6a85cfd2`), contributed to the vLLM
  project under the same license.
- FlashInfer contributors: b12x API, dispatch, and CuTe DSL kernels, under
  Apache License 2.0. ST maintains its modified copies in `b12x/` and uses
  the installed FlashInfer package for shared utilities and compilation.
- B12X authors: Copyright (c) 2025 by the b12x authors, Apache License 2.0.
  Immediate-offset shared subword loads and packed FP8 widening are adapted
  from `local-inference-lab/b12x`, commit
  `12b4eb2574416c524eef0da273e2c063d35347d3`, `b12x/_lib/intrinsics.py`
  and `b12x/attention/_shared/mla/decode_math.py`. ST's versions in
  `b12x/moe_static_common.py` and `mla/glm53_megakernel.cu` retain volatile
  ring reads and qualify direct BF16 widening for the SM121 CUDA compiler.
- Flash Linear Attention: Copyright (c) 2023-2025, Songlin Yang, Yu Zhang.
  The KDA files carry the vLLM integration's Apache headers and the original
  MIT attribution. The original MIT permission notice follows.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

DeepGEMM, CUTLASS DSL, Triton, TileLang, CUDA and PyTorch remain separate
library/toolchain dependencies. DeepGEMM's compiled library and headers are
preserved together when building the ST image; they are not committed here.
