"""The engine's tests are a package so `python3 -m unittest tests.test_engine_x` works from the repository root.

Several of them borrow a fixture from a sibling by its bare name (`from test_engine_tier import MemoryTier`).
That import needs this directory on `sys.path`, and until this file existed the only thing that put it there was
`PYTHONPATH=tests`, which is written down nowhere -- so two files' verdicts (test_engine_prefix, and
test_engine_fleet_lease through test_engine_serve) depended on an environment variable a reader could not know
about. Putting it here makes the verdict the same however the suite is started.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
