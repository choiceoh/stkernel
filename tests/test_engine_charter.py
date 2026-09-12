#!/usr/bin/env python3
"""The charter comes before the code, so the decisions it carries are pinned like code.

Only D17 is here, and only because it is the one that binds work this repository has already
merged without it: on 2026-09-12 eight performance PRs landed on the step and prefill paths and
not one carried a fleet measurement. When main was finally booted that evening it decoded at
12.95 step/s against the old stack\'s 19.87-21.83 on the same workload -- and nobody can say
which of the eight, because none of them was measured.
"""
from __future__ import annotations

import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


class CharterD17Tests(unittest.TestCase):
    def setUp(self):
        self.charter = (ROOT / "engine/CHARTER.md").read_text()

    def test_the_charter_binds_speed_claims_to_a_fleet_measurement(self):
        self.assertIn("### D17.", self.charter)
        for fragment in ("bench/onepass.py", "start-st-glm53.sh", "engine_shape"):
            self.assertIn(fragment, self.charter, fragment)

    def test_it_asks_for_two_runs_and_says_why(self):
        """One run cannot tell a compile tail from a regression -- the same day, 2K cold went
        97 -> 132 tok/s and 32K cold 701 -> 1,639 between the first and second run of one boot."""
        d17 = self.charter[self.charter.index("### D17."):]
        d17 = d17[:d17.index("## 2. ")]
        self.assertIn("두 판", d17)
        self.assertIn("1,639", d17, "the number that shows why one run is not enough")

    def test_the_pull_request_template_asks_for_it_where_a_session_will_see_it(self):
        """A rule nobody reads at the moment of opening a PR is a rule that does not run."""
        template = (ROOT / ".github/pull_request_template.md").read_text()
        self.assertIn("D17", template)
        self.assertIn("bench/onepass.py", template)
        self.assertIn("engine_shape", template)


if __name__ == "__main__":
    unittest.main()
