"""CPU oracle for one-shot rail placement and the proxy's shared-header writes."""
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'engine/kernels/oneshot/dsv4_oneshot_ar.cu'
HEADER = ROOT/'engine/kernels/oneshot/dsv4_oneshot_transport.h'


class RailPlacementTests(unittest.TestCase):
    def test_python_rule_puts_one_peer_of_every_rank_on_the_second_rail(self):
        from engine.kernels.oneshot import pair_rail
        for rails in (1, 2):
            for a in range(4):
                peers = [b for b in range(4) if b != a]
                placed = [pair_rail(a, b, rails) for b in peers]
                self.assertEqual(placed, [pair_rail(b, a, rails) for b in peers], (rails, a))
                self.assertEqual(sum(placed), 1 if rails == 2 else 0, (rails, a, placed))
        self.assertEqual([(a, b) for a in range(4) for b in range(a + 1, 4) if pair_rail(a, b, 2)],
                         [(0, 1), (2, 3)])

    @unittest.skipUnless(shutil.which('g++'), 'requires a host C++ compiler')
    def test_header_rule_matches_the_python_rule(self):
        from engine.kernels.oneshot import pair_rail
        header = HEADER.read_text()
        rules = header[:header.index('// Include verbs.h')]
        program = '#include <cstdio>\n' + rules + r'''
int main() {
  for (int rails = 1; rails <= 2; ++rails)
    for (int a = 0; a < 4; ++a)
      for (int b = 0; b < 4; ++b)
        if (a != b) std::printf("%d %d %d %d\n", rails, a, b, osar_pair_rail(a, b, rails));
}
'''
        with tempfile.TemporaryDirectory() as d:
            source, binary = Path(d)/'rails.cc', Path(d)/'rails'
            source.write_text(program)
            subprocess.run(['g++', '-std=c++17', '-O1', str(source), '-o', str(binary)], check=True)
            lines = subprocess.run([str(binary)], check=True, capture_output=True, text=True).stdout.split('\n')
        rows = [tuple(map(int, line.split())) for line in lines if line]
        self.assertEqual(len(rows), 24)
        for rails, a, b, rail in rows:
            self.assertEqual(rail, pair_rail(a, b, rails), (rails, a, b))

    def test_every_queue_pair_resource_follows_its_peers_rail(self):
        source = SOURCE.read_text()
        # One device per rail; no single-device global survives the split.
        for name in ('g_cq', 'g_mr', 'g_pd', 'g_ctx', 'g_sgid'):
            self.assertRegex(source, rf'static [^;]*\b{name}\[OSAR_RAILS\]', name)
            self.assertNotRegex(source, rf'\b{name}\b(?!\[)', name)
        self.assertIn('osar_pair_rail(rank, g_peers[s], OSAR_RAILS)', source)
        self.assertIn('ibv_create_qp(g_pd[rail], &qia)', source)
        self.assertIn('g_mr[g_peer_rail[p]]->lkey', source)
        self.assertIn('osar_pair_rail(peer, g_rank, OSAR_RAILS) == g_peer_rail[s]', source)
        self.assertRegex(source, r'for \(int rail = 0; rail < OSAR_RAILS; \+\+rail\) \{\s+struct ibv_wc wc\[16\];')

    def test_the_proxy_keeps_its_poll_count_out_of_the_registered_header(self):
        source = SOURCE.read_text()
        proxy = source[source.index('static void *proxy_fn(void *)'):source.index('// ---------------- setup')]
        self.assertNotIn('g_ctrl->proxy_beat', proxy)
        self.assertEqual(len(re.findall(r'g_ctrl->\w+(?:\[[^\]]*\])*\s*=[^=]', proxy)), 2, proxy)   # flag_src, ack_seq
        self.assertIn('if (periodic) {', proxy)


if __name__ == '__main__':
    unittest.main()
