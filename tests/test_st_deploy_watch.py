"""The deploy watcher's three refusals (45차 §34 뒤).

Everything here is the part that DECIDES. The part that acts -- cutting a release, stopping the
supervisor, launching four nodes -- is not exercised: it can only be judged by doing it, and doing
it takes production down. What is judged is every way this says no, because a deploy watcher is
worth exactly what its refusals are worth.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "launchers"))

import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "st_deploy_watch", Path(__file__).resolve().parents[1] / "launchers/st-deploy-watch.py")
watch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watch)


class WantedTests(unittest.TestCase):
    def test_the_deployed_sha_is_not_a_candidate(self):
        self.assertIsNone(watch.wanted("abc123", {"deployed": "abc123"}))

    def test_a_moved_main_is(self):
        self.assertIn("abc123", watch.wanted("abc123", {"deployed": "def456"}))

    def test_a_first_deploy_is_a_candidate(self):
        self.assertIsNotNone(watch.wanted("abc123", {}))

    def test_a_rejected_sha_is_not_retried_until_main_moves_past_it(self):
        """A merge the gate refused must not become a restart loop."""
        held = {"deployed": "old", "rejected": "bad"}
        self.assertIsNone(watch.wanted("bad", held))
        self.assertIsNotNone(watch.wanted("newer", held), "and the next merge is looked at again")

    def test_nothing_to_do_without_a_head(self):
        self.assertIsNone(watch.wanted("", {"deployed": "abc"}))


class BusyTests(unittest.TestCase):
    def metrics(self, running=0, waiting=0, handing=0, drop=()):
        rows = {"vllm:num_requests_running": running, "vllm:num_requests_waiting": waiting,
                "st:handing_over": handing}
        return "\n".join(f"{k} {v}" for k, v in rows.items() if k not in drop) + "\n"

    def test_an_idle_engine_reads_zero(self):
        self.assertEqual(watch.busy(self.metrics()), 0)

    def test_a_request_in_the_step_or_in_the_queue_counts(self):
        self.assertEqual(watch.busy(self.metrics(running=2)), 2)
        self.assertEqual(watch.busy(self.metrics(waiting=3)), 3)
        self.assertEqual(watch.busy(self.metrics(running=1, waiting=1)), 2)

    def test_a_handover_in_progress_is_never_quiet(self):
        """The fleet is being given to another session: whatever the counters say, this is not a gap."""
        self.assertGreaterEqual(watch.busy(self.metrics(handing=1)), 1)

    def test_an_engine_that_cannot_be_asked_is_not_known_to_be_quiet(self):
        """None is not zero. A door that answers something unrecognisable must not read as a gap."""
        self.assertIsNone(watch.busy(""))
        self.assertIsNone(watch.busy(self.metrics(drop=("vllm:num_requests_waiting",))))
        self.assertIsNone(watch.busy("vllm:num_requests_running not-a-number\n"))

    def test_labels_and_floats_are_read(self):
        body = ('vllm:num_requests_running{model="glm-5.3"} 0.0\n'
                'vllm:num_requests_waiting{model="glm-5.3"} 0.0\n'
                "st:handing_over 0.0\n")
        self.assertEqual(watch.busy(body), 0)

    def test_a_prefix_of_the_name_is_not_the_name(self):
        body = ("vllm:num_requests_running_total 9\nvllm:num_requests_running 0\n"
                "vllm:num_requests_waiting 0\nst:handing_over 0\n")
        self.assertEqual(watch.busy(body), 0)


class GateTests(unittest.TestCase):
    """A dozen engine test files fail on any tree today, so the gate is a difference, not a bar."""

    def test_the_same_failures_are_not_a_regression(self):
        both = {"test_engine_kda_ring": "FAILED (errors=7)", "test_engine_serve": "FAILED (errors=12)"}
        self.assertEqual(watch.regressed(both, dict(both)), [])

    def test_a_newly_failing_file_is(self):
        before = {"test_engine_serve": "FAILED (errors=12)"}
        after = {**before, "test_engine_sampling": "FAILED (failures=1)"}
        self.assertEqual(watch.regressed(before, after), ["test_engine_sampling"])

    def test_the_same_file_failing_worse_is(self):
        self.assertEqual(watch.regressed({"test_engine_serve": "FAILED (errors=12)"},
                                         {"test_engine_serve": "FAILED (errors=13)"}),
                         ["test_engine_serve"])

    def test_a_file_that_stopped_failing_is_not_held_against_the_candidate(self):
        self.assertEqual(watch.regressed({"test_engine_serve": "FAILED (errors=12)"}, {}), [])

    def test_a_file_with_no_verdict_at_all_is_a_regression(self):
        """A run that crashed before unittest could say anything is not a pass."""
        self.assertEqual(watch.regressed({}, {"test_engine_glm53": "NO VERDICT"}), ["test_engine_glm53"])


class GateIsNotOptionalTests(unittest.TestCase):
    def test_a_first_deploy_with_nothing_to_compare_against_is_refused_not_waved_through(self):
        """Skipping here would make the only ungated deploy the first one, which nobody is watching."""
        source = (Path(__file__).resolve().parents[1] / "launchers/st-deploy-watch.py").read_text()
        body = source[source.index("    elif a.gate:"):]
        body = body[:body.index("\n    log(", 10)]
        self.assertIn("REFUSED", body)
        self.assertIn("return 1", body)
        self.assertIn("--seed", body, "and it says how to give the gate something to compare with")


class StateTests(unittest.TestCase):
    def test_an_unreadable_state_file_reads_as_nothing_deployed(self):
        self.assertEqual(watch.state_of(Path("/nonexistent/deploy-state.json")), {})

    def test_the_units_point_at_the_rsynced_tree_not_a_worktree(self):
        """systemd runs what is on the head node, which is what the launcher rsynced there."""
        here = Path(__file__).resolve().parents[1] / "launchers"
        unit = (here / "st-deploy-watch.service").read_text()
        self.assertIn("/home/choiceoh/st-engine/launchers/st-deploy-watch.py", unit)
        self.assertIn("--once", unit, "the loop belongs to the timer, not to a service that never exits")
        self.assertIn("After=st-glm53.service", unit, "it restarts that service: it must not race its start")
        timer = (here / "st-deploy-watch.timer").read_text()
        self.assertNotIn("Persistent=true", timer, "a missed cycle sees the same main; there is nothing to catch up")


if __name__ == "__main__":
    unittest.main()
