#!/usr/bin/env python3
"""Execute extracted CUDA control flow as C++ without CUDA or torch.

The operand labels and synchronization counters below replace device operations,
but the publication branches come from the shipped
.cu file. These tests catch indexing and missing-barrier regressions; GPU
racecheck and stock-op numerical comparisons remain necessary before deployment.
"""
from __future__ import annotations

import pathlib
import shutil
import subprocess
import tempfile
import unittest


REPO = pathlib.Path(__file__).resolve().parents[1]
CUDA_SOURCE = REPO / "overlay/modules/glm53_megakernel/glm53_megakernel.cu"


def _block(source: str, marker: str) -> str:
    """Return one source construct through its balanced closing brace."""
    start = source.index(marker)
    opening = source.index("{", start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def _statement(source: str, marker: str) -> str:
    start = source.index(marker)
    return source[start:source.index(";", start) + 1]


def harness_source(source: str) -> str:
    publication = source[source.index("if (pend >= 0)"):
                         source.index("      xv = nxv;")]
    probe = _block(source, "std::vector<int64_t> mk_probe_state()")
    restore = _block(source, "void mk_restore_probe_state(")
    return r'''
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

void require(bool ok, const char* what) {
  if (!ok) throw std::runtime_error(what);
}
#define TORCH_CHECK(ok, ...) require((ok), "probe-state validation")
constexpr int KBLK_MAX = 32, MK_GRID_CAP = 96;
int g_probe_ksr2;
''' + probe + "\n" + restore + r'''


struct { int x = 0; } threadIdx;
int syncs = 0, fences = 0;
unsigned g_mk_mhc_tok_arrive[32];
void __syncthreads() { ++syncs; }
void __threadfence() { ++fences; }
unsigned atomicAdd(unsigned* p, unsigned n) { unsigned old = *p; *p += n; return old; }

void mhc_test() {
  // Simulate every publication branch. Each token's shared-tile read must
  // be followed by a block barrier before the next iteration can write it.
  // The batch optimization must still publish each completed token once.
  for (int groups = 1; groups <= 9; ++groups) {
    for (int tokens = 1; tokens <= 32; ++tokens) {
      std::fill(std::begin(g_mk_mhc_tok_arrive), std::end(g_mk_mhc_tok_arrive), 0u);
      struct { int num_tokens; } a{tokens};
      for (int g = 0; g < std::min(groups, tokens); ++g) {
        int pend = -1;
        for (int t = g; t < tokens; t += groups) {
          syncs = fences = 0;
''' + publication + r'''
          require(syncs > 0, "MHC shared tile reused without a block barrier");
          require(fences == (pend >= 0 ? 0 : 1), "MHC publication fence batching changed");
        }
        require(pend == -1, "MHC left a completed token unpublished");
      }
      for (int t = 0; t < tokens; ++t)
        require(g_mk_mhc_tok_arrive[t] == 1u, "MHC token published more or less than once");
    }
  }
}

void probe_state_test() {
  const int64_t imax = std::numeric_limits<int>::max();
  const int64_t imin = std::numeric_limits<int>::min();
  // one knob (ksr2, the forced split) since 34차 §8: the persistent lane's
  // split, the lane switch, the local-quant, bg-control and tail knobs are
  // gone with their lanes
  const std::vector<std::vector<int64_t>> snapshots = {
    {-1}, {0}, {5}, {imax}, {imin}, {-2}
  };
  for (const auto& snapshot : snapshots) {
    mk_restore_probe_state(snapshot);
    require(mk_probe_state() == snapshot, "probe snapshot did not round trip");
  }
  const auto before = mk_probe_state();
  auto reject = [&](const std::vector<int64_t>& bad) {
    bool rejected = false;
    try { mk_restore_probe_state(bad); } catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "invalid probe snapshot was accepted");
    require(mk_probe_state() == before, "invalid snapshot partially changed probe state");
  };
  reject({});
  reject({0,0});
  const int64_t low[] = {imin-1};
  const int64_t high[] = {imax+1};
  for (int i = 0; i < 1; ++i) {
    auto bad = before; bad[i] = low[i]; reject(bad);
    bad[i] = high[i]; reject(bad);
  }
}

int main(int argc, char** argv) {
  try {
    require(argc == 2, "select a regression case");
    const std::string test = argv[1];
    if (test == "mhc") mhc_test();
    else if (test == "probe") probe_state_test();
    else throw std::runtime_error("unknown regression case");
    std::cout << test << ": PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
'''


class MegakernelCudaRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("clang++") or shutil.which("g++") or shutil.which("c++")
        if compiler is None:
            raise RuntimeError("a C++ compiler is required for the extracted CUDA checks")
        cls.tmp = tempfile.TemporaryDirectory(prefix="mk-cuda-regressions-")
        cls.addClassCleanup(cls.tmp.cleanup)
        cpp = pathlib.Path(cls.tmp.name) / "regressions.cpp"
        cls.exe = pathlib.Path(cls.tmp.name) / "regressions"
        cpp.write_text(harness_source(CUDA_SOURCE.read_text()))
        subprocess.run([compiler, "-std=c++17", "-O2", "-Wno-unknown-pragmas",
                        str(cpp), "-o", str(cls.exe)], check=True,
                       capture_output=True, text=True)

    def _run(self, case):
        result = subprocess.run([str(self.exe), case], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_mhc_shared_tile_reuse(self):
        self._run("mhc")

    def test_probe_state_restoration_is_exact_and_atomic(self):
        self._run("probe")


if __name__ == "__main__":
    unittest.main()
