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
import unittest.mock

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

    def test_a_handover_renames_and_deletes_on_a_thread_not_in_the_ranks_way(self):
        """Deleting here took the fleet down. Every rank runs this, each has a different amount to delete,
        the ones that finish first enter the next collective and wait, and the slow ones are still in
        rmtree -- so NCCL's watchdog timed the collective out and all four aborted (2026-09-12, first boot
        after a real serving run handed over: 34 parked conversations, 3.38 GiB of prefix tier). A rename
        is one inode operation and costs every node the same, which is what the ranks need of each other."""
        where = self.dir()
        tenancy.claim(where, "them@srv1/9")
        (where / "loose.bin").write_bytes(b"x")
        (where / "deep" / "deeper").mkdir(parents=True)
        removed = []
        tenancy.claim(where, "me@srv4/1", clear=lambda p: removed.append(p.name), background=False)
        # the old state is out of the tenant's way the instant claim returns: what is left is the marker and
        # the discard the stub `clear` did not actually delete
        left = sorted(c.name for c in where.iterdir())
        self.assertNotIn("loose.bin", left); self.assertNotIn("deep", left)
        self.assertEqual([n for n in left if not n.startswith(tenancy.DISCARD)], [tenancy.MARKER])
        self.assertEqual(len(removed), 1, "one discard directory, not one call per child")
        self.assertTrue(removed[0].startswith(tenancy.DISCARD))

    def test_the_rename_happens_before_the_delete_does(self):
        """The point is the ordering: if `clear` can still see the old state, the rank is still paying for it."""
        where = self.dir()
        tenancy.claim(where, "them@srv1/9")
        self.state(where)
        seen = []
        tenancy.claim(where, "me@srv4/1", background=False,
                      clear=lambda p: seen.append(sorted(c.name for c in where.iterdir())))
        self.assertEqual(len(seen), 1)
        self.assertNotIn("conversations", seen[0], "the old state was already out of the way")
        self.assertNotIn("prefix", seen[0])

    def test_a_discard_left_by_a_boot_that_died_is_swept_by_the_next_claim(self):
        where = self.dir()
        tenancy.claim(where, "me@srv4/1")
        orphan = where / f"{tenancy.DISCARD}.deadbeef"
        (orphan / "conversations").mkdir(parents=True)
        removed = []
        tenancy.claim(where, "me@srv4/1", clear=lambda p: removed.append(p.name), background=False)
        self.assertEqual(removed, [orphan.name], "nothing accumulates across boots")

    def test_a_filesystem_that_will_not_rename_still_gets_a_clean_handover(self):
        """A slow handover is better than a dirty one."""
        where = self.dir()
        tenancy.claim(where, "them@srv1/9")
        self.state(where)
        removed = []
        with unittest.mock.patch.object(pathlib.Path, "rename", side_effect=OSError("no")):
            tenancy.claim(where, "me@srv4/1", clear=lambda p: removed.append(p.name), background=False)
        self.assertEqual(sorted(removed), ["conversations", "prefix"], "deleted in place instead")

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
