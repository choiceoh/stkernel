"""What the next tenant of the fleet must not inherit.

A restart keeps its parked conversations -- that is D16, and the clients come back to find their turn where they
left it. A handover is not a restart: the previous holder's clients are gone, and its prefix tier is warm with
boundaries the new holder never computed. Correctness survives either way (a boundary is salted by tenant), but a
run that inherits a warm tier is not the run anybody thinks they are timing.
"""
from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from engine.base import tenancy   # noqa: E402


class TenancyTests(unittest.TestCase):
    def dir(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return pathlib.Path(tmp.name)

    def state(self, where):
        (where / "conversations").mkdir(parents=True, exist_ok=True)
        (where / "conversations" / "7.json").write_text("{}")
        (where / "prefix").mkdir(parents=True, exist_ok=True)
        (where / "prefix" / "blocks.bin").write_bytes(b"warm")

    def test_an_unclaimed_directory_is_simply_claimed(self):
        where = self.dir()
        self.state(where)
        self.assertIsNone(tenancy.claim(where, "me@srv4/1"))
        self.assertTrue((where / "conversations" / "7.json").exists(), "nobody's is not somebody else's")
        self.assertEqual(tenancy.held_by(where), "me@srv4/1")

    def test_the_same_owner_keeps_everything(self):
        """A restart. D16 says those conversations survive it."""
        where = self.dir()
        tenancy.claim(where, "me@srv4/1")
        self.state(where)
        self.assertIsNone(tenancy.claim(where, "me@srv4/1"))
        self.assertTrue((where / "conversations" / "7.json").exists())
        self.assertTrue((where / "prefix" / "blocks.bin").exists())

    def test_a_handover_empties_it_and_says_who_it_was(self):
        where = self.dir()
        tenancy.claim(where, "them@srv1/9")
        self.state(where)
        left = tenancy.claim(where, "me@srv4/1")
        self.assertEqual(left, "them@srv1/9", "the boot has to be able to say what it cleared and whose")
        self.assertFalse((where / "conversations").exists())
        self.assertFalse((where / "prefix").exists())
        self.assertEqual(tenancy.held_by(where), "me@srv4/1")

    def test_the_marker_survives_the_clearing_it_caused(self):
        where = self.dir()
        tenancy.claim(where, "them@srv1/9")
        tenancy.claim(where, "me@srv4/1")
        self.assertEqual(tenancy.held_by(where), "me@srv4/1")
        self.assertIsNone(tenancy.claim(where, "me@srv4/1"), "and the next restart is not a handover")

    def test_it_clears_directories_and_files_alike(self):
        where = self.dir()
        tenancy.claim(where, "them@srv1/9")
        (where / "loose.bin").write_bytes(b"x")
        (where / "deep" / "deeper").mkdir(parents=True)
        removed = []
        tenancy.claim(where, "me@srv4/1", clear=lambda p: removed.append(p.name))
        self.assertEqual(sorted(removed), ["deep"], "directories go through `clear`")
        self.assertFalse((where / "loose.bin").exists(), "and plain files are unlinked")

    def test_a_plain_string_path_works_too(self):
        """The signature says `str | Path` and the engine passes a Path; a caller that passes a string must not
        get a str with no `.iterdir`."""
        where = self.dir() / "rank0"
        tenancy.claim(str(where), "them@srv1/9")
        (where / "prefix").mkdir()
        self.assertEqual(tenancy.claim(str(where), "me@srv4/1"), "them@srv1/9")
        self.assertFalse((where / "prefix").exists())

    def test_a_missing_directory_is_created_rather_than_an_error(self):
        where = self.dir() / "rank0"
        self.assertIsNone(tenancy.claim(where, "me@srv4/1"))
        self.assertTrue(where.is_dir())


if __name__ == "__main__":
    unittest.main()
