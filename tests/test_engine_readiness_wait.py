import unittest

from probes.wait_engine_ready import wait_ready


class ReadinessTests(unittest.TestCase):
    def run_clock(self, ready_at, timeout=1800, alive=True):
        now = [0.0]
        def sleep(seconds):
            now[0] += seconds
        wait_ready(lambda: now[0] >= ready_at, lambda: alive,
                   timeout=timeout, interval=5, clock=lambda: now[0], sleep=sleep)
        return now[0]

    def test_cold_boot_longer_than_seven_minutes_stays_alive(self):
        self.assertEqual(self.run_clock(700), 700)

    def test_ready_at_the_deadline_is_accepted(self):
        self.assertEqual(self.run_clock(10, timeout=10), 10)

    def test_final_partial_interval_is_checked(self):
        self.assertEqual(self.run_clock(7, timeout=7), 7)

    def test_a_failed_container_does_not_wait_for_the_deadline(self):
        with self.assertRaisesRegex(RuntimeError, 'candidate stopped'):
            self.run_clock(100, alive=False)

    def test_a_live_but_unready_container_times_out(self):
        with self.assertRaises(TimeoutError):
            self.run_clock(100, timeout=10)
