"""Host-side admission checks for a second CUDA process on GB10 UMA."""
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / "probes/glm53_probe_memory.py"
spec = importlib.util.spec_from_file_location("probe_memory", path)
memory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(memory)


class ProbeMemoryTests(unittest.TestCase):
    def test_idle_serving_incident_is_refused(self):
        result = memory.assess("MemTotal: 125507584 kB\nMemAvailable: 9808896 kB\n")
        self.assertFalse(result["passed"])
        self.assertGreater(result["required_kib"], 16 * 1024**2)

    def test_offline_headroom_is_admitted(self):
        self.assertTrue(memory.assess(
            "MemTotal: 125507584 kB\nMemAvailable: 99572736 kB\n")["passed"])

    def test_boundary_keeps_probe_and_system_reserves(self):
        total = 128 * 1024**2
        required = 8 * 1024**2 + (total + 9) // 10
        for available, expected in ((required - 1, False), (required, True)):
            self.assertEqual(memory.assess(
                f"MemTotal: {total} kB\nMemAvailable: {available} kB\n")["passed"], expected)

    def test_missing_or_invalid_counters_fail_closed(self):
        for text in ("", "MemTotal: 100 kB", "MemTotal: 100 kB\nMemAvailable: -1 kB",
                     "MemTotal: 100 kB\nMemAvailable: 101 kB", "MemTotal: 100 MB\nMemAvailable: 10 kB"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                memory.assess(text)


if __name__ == "__main__":
    unittest.main()
