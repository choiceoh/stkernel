"""A ledger entry has to be referable (MEASUREMENTS.md 규칙 9).

Three sessions merged into main within minutes of each other on 2026-09-12 and each picked its
section number by reading the end of the file. §30, §31 and §32 were each written twice, and from
that point "§31 참조" no longer names anything. The rule is now that an entry is named by its PR,
which GitHub issues and no session has to coordinate.

This is the rule as a check rather than a sentence: every `###` entry must be reachable by a name
that belongs to it alone.
"""
import re
import unittest
from pathlib import Path

LEDGER = Path(__file__).resolve().parents[1] / "MEASUREMENTS.md"
HEADING = re.compile(r"^#{2,3} (?:(?P<round>\S+차) )?(?:§(?P<num>\d+) )?— ?(?P<title>.*)$", re.M)
PR = re.compile(r"PR #(\d+)")


def entries():
    """(round, section number or None, the PR it names, the whole heading) for every ledger entry."""
    out = []
    for line in LEDGER.read_text().splitlines():
        if not line.startswith(("## ", "### ")) or " — " not in line:
            continue
        m = HEADING.match(line)
        if m is None:
            continue
        pr = PR.search(line)
        out.append((m.group("round"), m.group("num"), pr.group(1) if pr else None, line))
    return out


class LedgerNumberingTests(unittest.TestCase):
    def test_no_two_entries_answer_to_the_same_name(self):
        """A number used twice in the same round is only allowed if each says which PR it is."""
        seen = {}
        for rnd, num, pr, line in entries():
            if num is None:
                continue
            key = (rnd, num)
            if key in seen:
                first_pr, first_line = seen[key]
                self.assertIsNotNone(pr, f"§{num} is used twice and this one names no PR:\n  {line}")
                self.assertIsNotNone(first_pr, f"§{num} is used twice and this one names no PR:\n  {first_line}")
                self.assertNotEqual(pr, first_pr, f"two entries claim §{num} and PR #{pr}:\n  {first_line}\n  {line}")
            else:
                seen[key] = (pr, line)

    def test_the_rule_is_written_where_an_entry_is_written(self):
        rules = LEDGER.read_text()
        rules = rules[rules.index("## 이 원장을 쓰는 규칙"):]
        rules = rules[:rules.index("\n## ", 10)]
        self.assertIn("PR 번호", rules, "rule 9 is what this test is the enforcement of")
        self.assertIn("9. ", rules)

    def test_the_entries_that_shared_a_number_each_say_which_pr(self):
        """The concrete damage of 2026-09-12: three numbers, six entries. Each must be readable."""
        shared = {}
        for rnd, num, pr, line in entries():
            if rnd == "45차" and num in {"30", "31", "32"}:
                shared.setdefault(num, []).append((pr, line))
        for num, rows in shared.items():
            self.assertEqual(len(rows), 2, f"§{num} was written twice; that is the case this pins")
            self.assertEqual(len({pr for pr, _ in rows}), 2,
                             f"§{num}'s two entries do not name two different PRs: {[l for _, l in rows]}")
            for pr, line in rows:
                self.assertIsNotNone(pr, f"§{num} is ambiguous without a PR:\n  {line}")


if __name__ == "__main__":
    unittest.main()
