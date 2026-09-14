"""The PDL consumer's reach, read from the engine's own kernel source: which grid it launches, what its stash and
block ownership cover, how its publication tickets interleave with the other launches, and what `reduce` sends it."""
import ast
import importlib.util
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from engine.kernels.cells import ONESHOT_CONSUMER_MAX_ELEMENTS, ONESHOT_MAX_ELEMENTS

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'engine/kernels/oneshot/dsv4_oneshot_ar.cu'
HEADER = ROOT/'engine/kernels/oneshot/dsv4_oneshot_transport.h'


class ConsumerLaunchTests(unittest.TestCase):
    def test_engine_build_launches_the_ordinary_grid_for_the_consumer(self):
        # The engine never defines OSAR_COMPACT_CTA, so the consumer is k_oneshot_impl<true> on the fixed
        # 48 x 256 grid: one ticket per CTA and the VECITER stash, not the compact 12-CTA/two-vector form.
        tree = ast.parse((ROOT/'engine/kernels/oneshot/__init__.py').read_text())
        build = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'build')
        self.assertNotIn('COMPACT', ast.unparse(build))
        source = SOURCE.read_text()
        self.assertIn('#ifndef OSAR_COMPACT_CTA\n#define OSAR_COMPACT_CTA 0\n#endif', source)
        impl = source[source.index('static at::Tensor py_oneshot_impl('):source.index('static at::Tensor py_oneshot(')]
        consumer = impl[impl.index('if (consumer_pdl) {'):impl.index('} else {')]
        common, compact = consumer.split('#if OSAR_COMPACT_CTA', 1)
        self.assertIn('cfg.gridDim = dim3(ARGRID);', common)
        self.assertIn('cfg.blockDim = dim3(ARTHREADS);', common)
        disabled = compact.split('#else', 1)[1].split('#endif', 1)[0]
        self.assertIn('cudaLaunchKernelEx(&cfg, k_oneshot_consumer, g_ctrl, src,', disabled)
        entry = source[source.index('__global__ void k_oneshot_consumer('):]
        self.assertIn('k_oneshot_impl<true>(c, src, dst, n, nbytes, h, rank);', entry[:entry.index('\n}')])
        kernel = source[source.index('__device__ __forceinline__ void k_oneshot_impl('):source.index('// Distinct entry points')]
        self.assertIn('uint4 mine[COMPACT ? OSAR_COMPACT_VECITER : VECITER];', kernel)
        self.assertIn('last = atomicAdd((unsigned long long *)&c->done_ctr, 1ULL) %\n               ARGRID == ARGRID - 1;', kernel)

    @unittest.skipUnless(shutil.which('g++'), 'requires a host C++ compiler')
    def test_stash_and_ownership_cover_every_consumer_size_through_maxel(self):
        source = SOURCE.read_text()
        defines = {name: re.search(rf'^#define {name} (.+)$', source, re.M).group(1)
                   for name in ('ARGRID', 'ARTHREADS', 'VECITER')}
        start = source.index('template <bool VECTOR_EXACT>')
        owns = source[start:source.index('\n}', start) + 2].replace('__host__ __device__ ', '')
        rules = HEADER.read_text()
        rules = rules[:rules.index('// Include verbs.h')]
        program = '#include <cstdio>\n#include <cstdint>\n#include <random>\n' + rules + f"""
#define MAXEL {ONESHOT_MAX_ELEMENTS}
#define ARGRID {defines['ARGRID']}
#define ARTHREADS {defines['ARTHREADS']}
#define VECITER {defines['VECITER']}
constexpr int CONSUMER = {ONESHOT_CONSUMER_MAX_ELEMENTS};
""" + owns + r"""
// One (block, thread) of k_oneshot_impl: the copy/reduce vector loop (its trip count indexes the stash) and the
// scalar tail, with the kernel's own loop bounds and strides.
struct Visit { int trips; bool vector, tail; };
Visit visit(int b, int t, int n) {
  const int nv = n >> 3;
  Visit out{0, false, false};
  for (int v = b * ARTHREADS + t, k = 0; v < nv; v += ARGRID * ARTHREADS, k++) {
    out.trips = k + 1;
    out.vector = true;
  }
  for (int i = (nv << 3) + b * ARTHREADS + t; i < n; i += ARGRID * ARTHREADS) out.tail = true;
  return out;
}
int check(int n) {
  int trips = 0;
  for (int b = 0; b < ARGRID; ++b) {
    bool touched = false;
    for (int t = 0; t < ARTHREADS; ++t) {
      const Visit v = visit(b, t, n);
      if (v.trips > trips) trips = v.trips;
      touched |= v.vector || v.tail;
    }
    const bool consumer = osar_block_owns<true>(b, ARTHREADS, n);
    const bool ordinary = osar_block_owns<false>(b, ARTHREADS, n);
    // A block that copies or reduces anything must guard, fence and wait; the consumer retires exactly the rest.
    if (consumer != (touched || b == 0) || (touched && !ordinary)) {
      std::fprintf(stderr, "ownership n=%d block=%d consumer=%d ordinary=%d touched=%d\n", n, b, consumer, ordinary, touched);
      return -1;
    }
  }
  return trips;
}
int main() {
  if (CONSUMER <= 0 || CONSUMER > MAXEL || VECITER * ARGRID * ARTHREADS * 8 < MAXEL) return 2;
  int sizes = 0, worst = 0, owners_at_bound = 0;
  // Every 8-element size through the consumer bound and a stride past it, then row multiples, the grid-stride
  // edges and odd tails through MAXEL.
  for (int n = 0; n <= CONSUMER + 12288 * 8; n += 8) {
    const int trips = check(n);
    if (trips < 0 || trips > VECITER) return 3;
    worst = trips > worst ? trips : worst;
    ++sizes;
  }
  for (int n = 1; n <= MAXEL; n += n < CONSUMER ? 4093 : 65531) {
    for (int d = -9; d <= 9; ++d) {
      const int m = n + d;
      if (m < 0 || m > MAXEL) continue;
      const int trips = check(m);
      if (trips < 0 || trips > VECITER) return 4;
      ++sizes;
    }
  }
  for (int rows = 1; rows * 4096 <= MAXEL; ++rows) {
    const int trips = check(rows * 4096);
    if (trips < 0 || trips > VECITER) return 5;
    ++sizes;
  }
  if (check(MAXEL) != VECITER) return 6;
  for (int b = 0; b < ARGRID; ++b) owners_at_bound += osar_block_owns<true>(b, ARTHREADS, CONSUMER);
  // Tickets: consumer and ordinary launches add one per CTA and publish on done_ctr % ARGRID; packets add one per
  // CTA on the wrap-safe rule; MAX and gather add ARGRID from one CTA. Whichever CTA adds the last ticket publishes.
  std::mt19937 rng(965);
  uint64_t counter = 0;
  for (uint64_t seq = 1; seq <= 20000; ++seq) {
    const unsigned kind = rng() % 4;
    const unsigned grid = kind == 3 ? 1 : ARGRID, weight = kind == 3 ? ARGRID : 1;
    unsigned publishers = 0;
    for (unsigned done = 0; done < grid; ++done) {
      const uint64_t old = counter;
      counter += weight;
      const bool last = kind < 2 ? old % ARGRID == ARGRID - 1 : osar_publication_last(old, weight, seq);
      if (last && done != grid - 1) return 7;
      publishers += last;
    }
    if (publishers != 1 || counter != seq * ARGRID) return 8;
  }
  std::printf("%d sizes through MAXEL %d; stash %d trips of VECITER %d (worst %d at <= %d elements); "
              "%d owning CTAs at the consumer bound; 20000 mixed publications PASS\n",
              sizes, MAXEL, check(MAXEL), VECITER, worst, CONSUMER + 12288 * 8, owners_at_bound);
  return 0;
}
"""
        with tempfile.TemporaryDirectory() as temp:
            path, binary = Path(temp)/'consumer.cpp', Path(temp)/'consumer'
            path.write_text(program)
            subprocess.run(['g++', '-O2', '-std=c++17', str(path), '-o', str(binary)],
                           check=True, capture_output=True, text=True, timeout=120)
            result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=300)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('PASS', result.stdout)
            self.assertIn(f'owning CTAs at the consumer bound', result.stdout)
            owners = int(re.search(r'(\d+) owning CTAs at the consumer bound', result.stdout).group(1))
            self.assertEqual(owners, (ONESHOT_CONSUMER_MAX_ELEMENTS // 8 + 255) // 256)


@unittest.skipUnless(importlib.util.find_spec('torch') is not None, 'requires PyTorch')
class ConsumerDispatchTests(unittest.TestCase):
    def test_reduce_sends_c1_and_c2_sums_to_the_consumer_and_larger_sums_to_the_ordinary_kernel(self):
        import torch
        from engine.kernels import oneshot
        self.assertEqual(oneshot.CONSUMER_MAX_ELEMENTS, 16 * 4096)
        transport = object.__new__(oneshot.OneShot)
        transport.closed, transport.pending, transport.packet_failed = False, None, False
        transport.ext = SimpleNamespace(healthy=lambda: True, oneshot_ar=Mock(return_value='ordinary'),
                                        oneshot_ar_consumer=Mock(return_value='consumer'))
        for elements, expected in ((8, 'consumer'), (8 * 4096, 'consumer'), (16 * 4096, 'consumer'),
                                   (16 * 4096 + 8, 'ordinary'), (17 * 4096, 'ordinary'), (64 * 4096, 'ordinary')):
            value = SimpleNamespace(is_cuda=True, dtype=torch.bfloat16, is_contiguous=lambda: True,
                                    numel=lambda n=elements: n, data_ptr=lambda: 1 << 20)
            self.assertEqual(transport.reduce(value), expected, elements)


if __name__ == '__main__':
    unittest.main()
