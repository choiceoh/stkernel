"""Device-free watchdog regression: fast capture polls are not a dead proxy."""
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
  assert(!h.check(false, 0, 0));
  assert(h.check(true, 0, 10));
  // The old comparison fails at the second query if its nonzero beat repeats.
  assert(h.check(true, 123, 100));
  for (uint64_t i = 0; i < 10000; ++i) assert(h.check(true, 123, 100+i));
  assert(h.check(true, 123, 100+OsarProxyHealth::stale_ns-1));
  assert(!h.check(true, 123, 100+OsarProxyHealth::stale_ns));
  assert(h.check(true, 124, 100+OsarProxyHealth::stale_ns));
  assert(!h.check(false, 125, 101+OsarProxyHealth::stale_ns));
  h = {};
  assert(h.check(true, 0, 200+OsarProxyHealth::stale_ns));
  assert(!h.check(true, 0, 200+2*OsarProxyHealth::stale_ns));
  assert(h.check(true, UINT64_MAX, 201+2*OsarProxyHealth::stale_ns));
  assert(h.check(true, 0, 202+2*OsarProxyHealth::stale_ns));
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
