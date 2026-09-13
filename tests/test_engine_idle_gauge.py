"""st:idle_seconds -- how long since a request last arrived or was answered (the quiet gate's shortcut)."""
import re
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import test_engine_serve as T                                          # noqa: E402


def gauge(text):
    m = re.search(r"^st:idle_seconds(?:\{[^}]*\})? +(\d+)", text, re.M)
    return None if m is None else int(m.group(1))


class IdleGaugeTests(unittest.TestCase):
    def test_it_counts_from_boot_and_restarts_at_every_request(self):
        s = T.server(rows=2)
        s._last_request_at -= 500                                      # as if it booted eight minutes ago
        self.assertGreaterEqual(gauge(s.metrics()), 500)
        request, _ = s.submit([3], 1, 0)
        self.assertLessEqual(gauge(s.metrics()), 1, "a request arrived: the count restarts")
        for _ in range(50):
            if not s.once() and not s._waiting:
                break
        s.take_result(request)
        self.assertLessEqual(gauge(s.metrics()), 1, "and again when it was answered")
        self.assertIn("vllm:request_success_total", s.metrics())


if __name__ == "__main__":
    unittest.main()
