"""Does Runner.parked stay at PARKED_RECORDS_KEPT when conversations park through the server?

From the repo root (stk-test):  python3 measurements/parked_record_bound_20260919/probe_growth.py
"""
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "tests")
import test_engine_serve as T  # noqa: E402


def quiet(s, steps=400):
    for _ in range(steps):
        ran = s.once()
        if not ran and not s._waiting and not s._retiring and not s._resuming and not s._restoring:
            return
    raise AssertionError("not quiet")


s = T.server(rows=2, blocks=64, keep_idle=True, tiered=True)
for i in range(40):
    request, _ = s.submit([3 + i % 200, 4, 5], 2, 0)
    quiet(s)
    s.take_result(request)
print(f"parked on the tier: {len(s.runner.parked_keys())}; records held in Runner.parked: {len(s.runner.parked)} "
      f"(PARKED_RECORDS_KEPT = {s.runner.PARKED_RECORDS_KEPT}); digests: {len(s.runner.digests)}")
