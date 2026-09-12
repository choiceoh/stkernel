#!/usr/bin/env python3
"""Every document says whether it is still supposed to be true.

The repository's markdown is three different things wearing one shape: a live reference that must
track the code, a study that was true on its date, and a record that must not be edited at all.
From a directory listing they are indistinguishable, and that is where rot hides -- on 2026-09-12
a snapshot size of "77 MiB" sat in three comparison documents long after it was 45, because nothing
in them claimed to stay true and so nobody re-read them (45차, PR #755).

So line two says which kind it is, and this refuses a document that does not.
"""
from __future__ import annotations

import pathlib
import subprocess
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
KINDS = ("살아 있는 참조", "그날의 조사", "그대로 두는 기록")
EXEMPT = ("MEASUREMENTS.md",                    # the ledger is the thing the others are dated against
          "engine/kernels/THIRD_PARTY_NOTICES.md")
NAVIGATED = ("engine/", "bench/", "docs/", "profiles/")
"""What a reader navigates BY: the repository root and these four trees.

Not overlay/modules/*/README.md or probes/*.md -- those sit next to the code they describe and go
stale with it, where a reader is already looking. The rot this catches is the other kind: a
document one directory away from anything that would contradict it.
"""


def tracked_markdown():
    out = subprocess.run(["git", "ls-files", "*.md"], cwd=ROOT, capture_output=True, text=True, check=True)
    for name in out.stdout.split():
        if name in EXEMPT:
            continue
        if "/" in name and not name.startswith(NAVIGATED):
            continue
        if name.startswith(("measurements/", ".github/", "docs/agent-rules/")):
            continue
        yield name


class DocumentStatusTests(unittest.TestCase):
    def test_every_document_says_which_kind_it_is(self):
        missing = []
        for name in tracked_markdown():
            head = (ROOT / name).read_text(errors="replace").splitlines()[:8]
            if not any(line.startswith("> ") and any(k in line for k in KINDS) for line in head):
                missing.append(name)
        self.assertEqual(missing, [], "these carry no status banner; add one under the title")

    def test_a_live_reference_says_that_being_wrong_is_a_bug(self):
        """The three are not decoration: they say what to DO when a fact moves. A live reference gets
        corrected, a study gets left alone and superseded in the ledger, a record is never edited."""
        live = [n for n in tracked_markdown()
                if any(KINDS[0] in l for l in (ROOT / n).read_text(errors="replace").splitlines()[:8])]
        self.assertIn("engine/README.md", live)
        self.assertIn("engine/CHARTER.md", live)
        for name in live:
            head = "\n".join((ROOT / name).read_text(errors="replace").splitlines()[:8])
            self.assertIn("버그", head, name)

    def test_the_kinds_are_explained_where_a_reader_starts(self):
        self.assertIn("살아 있는 참조", (ROOT / "README.md").read_text())
        self.assertIn("그대로 두는 기록", (ROOT / "README.md").read_text())


if __name__ == "__main__":
    unittest.main()
