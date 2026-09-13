"""Adversarial rank agreement using the transport's actual FP32 sum helper."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which('g++'), 'requires a host C++ compiler')
class RankOrderedSumTests(unittest.TestCase):
    def test_cancellation_and_bf16_inputs_agree_on_all_ranks(self):
        header = (ROOT/'engine/kernels/oneshot/dsv4_oneshot_transport.h').read_text()
        rules = header[:header.index('// Include verbs.h')]
        program = r'''
#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
''' + rules + r'''
float fold(const std::array<float,4>& x, int rank) {
  std::array<float,3> peers{};
  int slot=0;
  for (int i=0; i<4; ++i) if (i!=rank) peers[slot++]=x[i];
  return osar_sum_rank_order(x[rank],peers[0],peers[1],peers[2],rank);
}
int main() {
  const std::array<std::array<float,4>,4> inputs{{
    {{16777216.f,-16777216.f,1.f,1.f}},
    {{256.f,-256.f,0x1p-16f,0x1p-16f}},
    {{1.f,2.f,3.f,4.f}}, {{-1.f,1.f,0x1p-24f,-0x1p-24f}}
  }};
  const std::array<float,4> expected{{2.f,0x1p-15f,10.f,0.f}};
  for (int test=0; test<4; ++test) for (int rank=0; rank<4; ++rank)
    assert(fold(inputs[test],rank)==expected[test]);
  std::mt19937 rng(5518);
  for (int trial=0; trial<65536; ++trial) {
    std::array<float,4> x{};
    for (float& value:x) {
      uint32_t bf16=(rng()%2u)<<31 | (100u+rng()%51u)<<23 | (rng()%128u)<<16;
      std::memcpy(&value,&bf16,sizeof(value));
    }
    volatile float reference=x[0]+x[1];
    reference=reference+x[2]; reference=reference+x[3];
    const float wanted=reference;
    for (int rank=0; rank<4; ++rank) {
      const float actual=fold(x,rank);
      assert(std::memcmp(&actual,&wanted,sizeof(float))==0);
    }
  }
  std::cout << "4 cancellation fixtures and 65536 BF16 vectors agree on all ranks PASS\n";
}
'''
        with tempfile.TemporaryDirectory() as temp:
            source, binary = Path(temp)/'check.cpp', Path(temp)/'check'
            source.write_text(program)
            subprocess.run(['g++', '-O2', '-std=c++17', str(source), '-o', str(binary)],
                           check=True, capture_output=True, timeout=60)
            result = subprocess.run([str(binary)], check=True, capture_output=True, text=True, timeout=30)
            self.assertIn('65536 BF16 vectors agree on all ranks PASS', result.stdout)


if __name__ == '__main__':
    unittest.main()
