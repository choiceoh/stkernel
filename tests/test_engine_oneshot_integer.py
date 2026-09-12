"""CPU oracle for signed-key reduction and mixed one-shot publication tickets."""
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which('g++'), 'requires a host C++ compiler')
class IntegerTransportTests(unittest.TestCase):
    def test_signed_packets_odd_tails_and_publication_weight(self):
        source = (ROOT/'engine/kernels/oneshot/dsv4_oneshot_ar.cu').read_text()
        header = (ROOT/'engine/kernels/oneshot/dsv4_oneshot_transport.h').read_text()
        start = source.index('__host__ __device__ constexpr int64_t osar_max_int64')
        maximum = source[start:source.index('\n}', start)+2].replace('__host__ __device__', '')
        weight = re.search(r'constexpr unsigned weight = ([^;]+);', source).group(1)
        # Header's device-independent publication rules, without the verbs wrapper.
        rules = header[:header.index('// Include verbs.h')]
        program = '''#include <array>
#include <algorithm>
#include <cassert>
#include <cstring>
#include <limits>
#include <random>
#include <iostream>
''' + rules + maximum + '''
constexpr unsigned ARGRID = 48;
template<bool MAX_INT64, bool COMPACT> constexpr unsigned tickets() { return ''' + weight + '''; }
int main() {
  std::mt19937_64 rng(913);
  for (int n=1; n<=64; ++n) for (int trial=0; trial<128; ++trial) {
    std::array<std::array<int64_t,64>,4> packets{};
    for (auto &rank: packets) for (int i=0; i<n; ++i) {
      uint64_t bits=rng(); std::memcpy(&rank[i], &bits, 8);
    }
    packets[0][0]=std::numeric_limits<int64_t>::min();
    packets[3][n-1]=std::numeric_limits<int64_t>::max();
    // Exactly two keys per uint4 transfer, followed by an optional odd tail.
    std::array<int64_t,64> out{};
    for (int v=0; v<n/2; ++v) for (int p=0; p<2; ++p) {
      int i=2*v+p;
      out[i]=osar_max_int64(packets[0][i],packets[1][i],packets[2][i],packets[3][i]);
    }
    if (n%2) out[n-1]=osar_max_int64(packets[0][n-1],packets[1][n-1],packets[2][n-1],packets[3][n-1]);
    for (int i=0; i<n; ++i)
      assert(out[i]==std::max({packets[0][i],packets[1][i],packets[2][i],packets[3][i]}));
  }
  uint64_t counter=0;
  for (uint64_t seq=1; seq<4096; ++seq) {
    int grid=seq%3==0 ? 1 : (seq%3==1 ? 12 : 48);
    unsigned weight=grid==1 ? tickets<true,false>() : (grid==12 ? tickets<false,true>() : tickets<false,false>());
    assert(grid*weight==48);
    for (int block=0; block<grid; ++block) {
      bool last=osar_publication_last(counter,weight,seq);
      assert(last==(block==grid-1));
      counter+=weight;
    }
    assert(counter==seq*48);
  }
  for (uint64_t seq: {uint64_t(0),uint64_t(1),std::numeric_limits<uint64_t>::max()})
    assert(osar_publication_last((seq-1)*48,tickets<true,false>(),seq));
  std::cout << "8192 signed packet cases; 4095 mixed publications; MAX wrap boundaries PASS\\n";
}
'''
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'check.cpp'
            path.write_text(program)
            binary = Path(temp)/'check'
            subprocess.run(['g++', '-O2', '-std=c++17', str(path), '-o', str(binary)],
                           check=True, capture_output=True, timeout=60)
            result = subprocess.run([str(binary)], check=True, capture_output=True, text=True, timeout=30)
            self.assertIn('PASS', result.stdout)


if __name__ == '__main__':
    unittest.main()
