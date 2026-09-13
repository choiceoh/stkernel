"""Device-free watchdog: rapid polls and idle gaps use proxy publication time."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


@unittest.skipUnless(shutil.which('g++'), 'requires a host C++ compiler')
class ProxyHealthTests(unittest.TestCase):
    def test_capture_bursts_real_stalls_thread_exit_and_restart(self):
        source = (Path(__file__).resolve().parents[1] /
                  'engine/kernels/oneshot/dsv4_oneshot_transport.h').read_text()
        rules = source[:source.index('// Include verbs.h')]
        program = '#include <cassert>\n' + rules + r'''
int main() {
  OsarProxyHealth h;
  constexpr uint64_t second = 1000000000ULL;
  assert(!h.check(false, 0, 0));
  assert(!h.check(true, 0, 10));
  // Repeated reads within one proxy publication interval remain healthy.
  for (uint64_t i = 0; i < 10000; ++i) assert(h.check(true, 100, 100+i));
  assert(h.check(true, 100, 100+OsarProxyHealth::stale_ns-1));
  assert(!h.check(true, 100, 100+OsarProxyHealth::stale_ns));
  // A beat unseen by the caller is already stale on the next request. The
  // observer-based implementation incorrectly renewed grace here (PR #830).
  assert(!h.check(true, second, 11*second));
  OsarProxyHealth first_query;
  assert(!first_query.check(true, second, 11*second));
  // Genuine proxy progress recovers health; exit wins over a fresh timestamp.
  assert(h.check(true, 11*second, 11*second));
  assert(!h.check(false, 11*second, 11*second));
  // Restart grace is anchored to connect, even when first queried much later.
  assert(h.check(true, 20*second, 20*second+1));
  assert(!h.check(true, 20*second, 23*second));
  assert(!h.check(true, 24*second, 23*second));
  assert(h.check(true, UINT64_MAX-10, UINT64_MAX));
  assert(!h.check(true, UINT64_MAX-10, 10));
}
'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'health.cpp'
            path.write_text(program)
            binary = path.with_suffix('')
            subprocess.run(['g++', '-std=c++17', '-O2', str(path), '-o', str(binary)],
                           check=True, capture_output=True, timeout=30)
            subprocess.run([str(binary)], check=True, timeout=10)


if __name__ == '__main__':
    unittest.main()
