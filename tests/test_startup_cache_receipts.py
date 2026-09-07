"""Exercise the real boot receipt gate without restarting a fleet."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


class BootReceiptTests(unittest.TestCase):
    def check_receipts(self, stage, fast, fast_hits, legacy_hits, suffix="", mode="pack-io"):
        script = (Path(__file__).resolve().parents[1] / "bench/startup_cache_boots.sh").read_text()
        gate = script.split('"$stage" "$MODE" <<\'PY\'\n', 1)[1].split('\nPY\n', 1)[0]
        with tempfile.TemporaryDirectory() as root:
            for node in (1, 2, 3, 4):
                Path(root, f"TEST-srv{node}.log").write_text(
                    "[rank-cache] hit rank=0\n" + 2 * (
                        "[fp8-cache] enabled=True hit=1 miss=0 errors=0\n"
                        f"[mk-pack-io] model fast={fast} fast_hits={fast_hits} legacy_hits={legacy_hits}\n"
                    ) + suffix)
            with patch.object(sys, "argv", ["gate", root, "TEST", stage, mode]), \
                    contextlib.redirect_stdout(io.StringIO()):
                exec(compile(gate, "startup_cache_boots receipt gate", "exec"), {})

    def test_prime_accepts_new_packs_but_timed_arms_require_hits(self):
        self.check_receipts("PRIME", 0, 0, 0)
        self.check_receipts("BASE1", 0, 0, 253)
        self.check_receipts("FAST1", 1, 253, 0)
        for stage, fast in (("BASE1", 0), ("FAST1", 1)):
            with self.subTest(stage=stage), self.assertRaisesRegex(AssertionError, "wrong pack IO path"):
                self.check_receipts(stage, fast, 0, 0)

    def test_prime_still_rejects_wrong_path_and_restore_errors(self):
        for fast, hits, suffix in ((1, 0, ""), (0, 1, ""),
                                  (0, 0, "pack cache example unreadable\n"),
                                  (0, 0, "MK W4 pack build FAILED\n")):
            with self.subTest(fast=fast, hits=hits, suffix=suffix), self.assertRaises(AssertionError):
                self.check_receipts("PRIME", fast, hits, 0, suffix)

    def test_rank_default_requires_an_enabled_nonempty_restore_receipt(self):
        self.check_receipts("DEFAULT", 0, 0, 0,
                            "[rank-cache-io] prefetch=1 chunks=3 bytes=129\n", "rank-default")
        for suffix in ("", "[rank-cache-io] prefetch=0 chunks=3 bytes=129\n",
                       "[rank-cache-io] prefetch=1 chunks=0 bytes=0\n"):
            with self.subTest(suffix=suffix), self.assertRaisesRegex(AssertionError, "wrong rank prefetch path"):
                self.check_receipts("DEFAULT", 0, 0, 0, suffix, "rank-default")


if __name__ == "__main__":
    unittest.main()
