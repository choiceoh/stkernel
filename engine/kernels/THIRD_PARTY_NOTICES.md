# Source attribution

This package includes ST's forks of kernels from vLLM, Flash Linear Attention,
and FlashInfer. The per-file source paths and pre-port SHA256 hashes are in
`SOURCES.json`. Original copyright and SPDX headers are retained where present.

- vLLM contributors: KDA integration, causal convolution, kpool, and mHC
  sources, under Apache License 2.0 (see this package's `LICENSE`).
- FlashInfer contributors: b12x API, dispatch, and CuTe DSL kernels, under
  Apache License 2.0. ST maintains its modified copies in `b12x/` and uses
  the installed FlashInfer package for shared utilities and compilation.
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
