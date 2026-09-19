"""engine/profiles/qwen38/fleet: what rank 0's recorders hold is written when the process ends (close_on_exit).

The MTP input tap wrote a shard at 4,096 rows or on its 30 s timer, the draft ledger flushed only when a record came,
and the draft-query tap drained every 30 s a slack behind its counter -- and the launcher's `stop` was `docker rm -f`, a
SIGKILL, while the container's python is its PID 1, where an unhandled SIGTERM is not even delivered. The 2026-09-19
tuning window stopped the fleet right after its last data request, so each data boot could lose its tail (PR #1275, the
window's record, section 7). Held here on the CPU:

- `close` writes what a recorder holds -- fewer rows than a shard, a shard still with the writer, the ledger's buffer,
  the query tap's rows behind its slack -- and returns once it is on disk;
- a process sent SIGTERM writes them and exits 143, its main thread idle or stuck in C, where no Python handler runs
  (without close_on_exit, the same process dies with nothing written);
- the tap's writer outlives a shard it cannot write, and `close` still returns;
- the boot installs it before the rendezvous and closes the device's recorder last.

The launcher's side (rank 0 by `docker stop` before any `rm -f`) is tests/test_engine_fleet_ops.py's.
"""
import importlib.util
import json
import os
import select
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

torch = None
if importlib.util.find_spec("torch") is not None:
    import torch

ROOT = Path(__file__).resolve().parents[1]

# A process with rank 0's two host-side recorders: it records 100 positions (fewer than a shard) and three ledger rows,
# says "ready", and waits -- in Python ("idle"), or with its main thread in C for good ("stuck": a mutex another thread
# holds, which a signal does not return from), or idle with no close_on_exit at all ("unhandled").
CHILD = r'''
import ctypes, sys, threading
from pathlib import Path
import torch
from engine.profiles.qwen38.fleet import DraftLedger, MTPInputTap, close_on_exit
out, how = Path(sys.argv[1]), sys.argv[2]
tap = MTPInputTap(out / "mtp-inputs", every_s=3600)
ledger = DraftLedger(out / "draft-ledger")
if how != "unhandled":
    close_on_exit([tap.close, ledger.close])
tap(5, 0, list(range(2000, 2100)), torch.arange(800, dtype=torch.float32).view(100, 8).to(torch.bfloat16), False)
for i in range(3):
    ledger({"seq": 5, "matched": i})
if how == "stuck":
    libc = ctypes.CDLL(None)
    mutex = ctypes.create_string_buffer(256)
    libc.pthread_mutex_init(mutex, None)
    held = threading.Event()
    def hold():
        libc.pthread_mutex_lock(mutex)
        held.set()
        threading.Event().wait()
    threading.Thread(target=hold, daemon=True).start()
    held.wait()
    print("ready", flush=True)
    libc.pthread_mutex_lock(mutex)
else:
    print("ready", flush=True)
    threading.Event().wait()
'''


def streams(n: int, width: int = 8):
    return torch.arange(n * width, dtype=torch.float32).view(n, width).to(torch.bfloat16)


@unittest.skipUnless(torch is not None, "requires torch")
class CloseTests(unittest.TestCase):
    def test_close_writes_rows_that_never_filled_a_shard(self):
        import numpy as np
        from engine.profiles.qwen38.fleet import MTPInputTap
        from engine.profiles.qwen38.mtp_tune import shards
        with tempfile.TemporaryDirectory() as d:
            tap = MTPInputTap(d, every_s=3600)                      # 4,096 rows a shard; a timer that never fires here
            fed = streams(100)
            tap(3, 0, list(range(1000, 1060)), fed[:60], False)     # a prompt
            tap(3, 60, list(range(1060, 1100)), fed[60:], True)     # verify steps' kept positions
            self.assertEqual(shards([d]), [])                       # 100 rows: nothing written before the end
            said = tap.close()
            files = shards([d])                                     # on disk the moment close returns
            self.assertEqual(len(files), 1)
            with np.load(files[0]) as shard:
                self.assertEqual(shard["meta"].tolist(), [[3, p, 1000 + p, int(p >= 60)] for p in range(100)])
                self.assertTrue(torch.equal(torch.from_numpy(shard["streams"]).view(torch.bfloat16), fed))
            self.assertEqual(said, "mtp inputs: 100 rows held at close; this boot 1 shards written, 0 still waiting, "
                                   "0 failed")
            tap(3, 100, [1100], fed[:1], True)                      # after close: not recorded
            self.assertTrue(tap.close().startswith("mtp inputs: 0 rows held at close; this boot 1 shards written"))
            self.assertEqual(len(shards([d])), 1)

    def test_close_waits_for_a_shard_still_with_the_writer(self):
        """A shard handed over just before the end is renamed into place before close returns, not after."""
        import numpy as np
        from engine.profiles.qwen38.fleet import MTPInputTap
        from engine.profiles.qwen38.mtp_tune import shards
        real = np.savez

        def slow(file, **arrays):
            time.sleep(0.5)
            real(file, **arrays)

        with tempfile.TemporaryDirectory() as d, patch("numpy.savez", slow):
            tap = MTPInputTap(d, rows=64, every_s=3600)
            tap(1, 0, list(range(80)), streams(80), False)          # past 64 rows: a shard, to the writer
            tap(1, 80, list(range(80, 90)), streams(10), True)      # ten held
            said = tap.close()
            self.assertEqual([np.load(f)["meta"].shape[0] for f in shards([d])], [80, 10])
            self.assertIn("this boot 2 shards written, 0 still waiting, 0 failed", said)
            self.assertEqual([f.name for f in Path(d).iterdir() if f.name.endswith(".part")], [])

    def test_the_writer_outlives_a_shard_it_cannot_write(self):
        """A full disk used to end the writer thread: every later shard queued in memory and `close` waited it out."""
        import numpy as np
        from engine.profiles.qwen38.fleet import MTPInputTap
        from engine.profiles.qwen38.mtp_tune import shards
        real, calls = np.savez, []

        def full_once(file, **arrays):
            calls.append(1)
            if len(calls) == 1:
                raise OSError(28, "No space left on device")
            real(file, **arrays)

        with tempfile.TemporaryDirectory() as d, patch("numpy.savez", full_once):
            tap = MTPInputTap(d, rows=2, every_s=3600)
            tap(1, 0, [5, 6], streams(2), False)                    # the first shard meets the full disk
            tap(1, 2, [7], streams(1), True)
            self.assertEqual(tap.close(), "mtp inputs: 1 rows held at close; this boot 1 shards written, "
                                          "0 still waiting, 1 failed")
            self.assertEqual([np.load(f)["meta"].tolist() for f in shards([d])], [[[1, 2, 7, 1]]])
            self.assertEqual([f.name for f in Path(d).iterdir() if f.name.endswith(".part")], [])

    def test_close_writes_what_the_ledger_buffers(self):
        from engine.profiles.qwen38.fleet import DraftLedger
        with tempfile.TemporaryDirectory() as d:
            ledger = DraftLedger(d, every=64)
            for i in range(3):
                ledger({"seq": 1, "matched": i})
            self.assertEqual(ledger.close(), f"draft ledger: 3 records in {ledger.path.name}")
            self.assertEqual([json.loads(l)["matched"] for l in ledger.path.read_text().splitlines()], [0, 1, 2])
            ledger({"seq": 1, "matched": 3})                        # the file stays open for a record still coming
            ledger.close()
            self.assertEqual(len(ledger.path.read_text().splitlines()), 4)
            ledger.file.close()

    def test_close_drains_the_query_tap_past_its_slack(self):
        """RowTap.drain keeps `slack` rows behind the counter unless `final`: the 30 s drains never took the last."""
        import numpy as np
        from engine.profiles.qwen38.fleet import DraftQueries

        class Ring:                                                 # RowTap's drain, on the host
            slack = 64

            def __init__(self):
                self.count, self.drained = torch.zeros(1, dtype=torch.int64), 0

            def drain(self, *, final=False):
                count = int(self.count)
                end = count if final else max(self.drained, count - self.slack)
                rows, ids = torch.ones(end - self.drained, 4, dtype=torch.bfloat16), torch.arange(self.drained, end)
                self.drained = end
                return rows, ids, count

        with tempfile.TemporaryDirectory() as d:
            ring = Ring()
            ring.count += 5                                         # the boot's warmup and capture rows
            queries = DraftQueries(ring, d, every_s=3600)
            ring.count += 10                                        # ten queries, fewer than the slack
            self.assertEqual(queries.close(), "draft queries: 10 rows in the last drain")
            files = sorted(Path(d).glob("draft-queries-*.npz"))
            self.assertEqual([np.load(f)["ids"].tolist() for f in files], [list(range(5, 15))])

    def test_the_boot_installs_it_before_the_rendezvous_and_closes_the_device_last(self):
        fleet = (ROOT / "engine/profiles/qwen38/fleet.py").read_text()
        main = fleet[fleet.index("def main("):]
        self.assertLess(main.index("close_on_exit(closers)"), main.index("comm = Comm.init()"))   # a stop mid-boot too
        order = [main.index(line) for line in ("closers.append(model.drafter.inputs_tap.close)",
                                               "closers.append(model.drafter.ledger.close)",
                                               "closers.append(DraftQueries(net.draft_tap, ")]
        self.assertEqual(order, sorted(order))


@unittest.skipUnless(torch is not None, "requires torch")
@unittest.skipUnless(hasattr(signal, "SIGTERM") and os.name == "posix", "POSIX signals")
class SigtermTests(unittest.TestCase):
    """The real signal, in a process of its own: a handler here would be the test runner's."""

    def stopped(self, how: str):
        """(exit status, stdout, the shards' meta rows, the ledger's lines) of CHILD sent SIGTERM once it is ready."""
        import numpy as np
        with tempfile.TemporaryDirectory() as d:
            child = subprocess.Popen([sys.executable, "-c", CHILD, d, how], cwd=ROOT, text=True,
                                     env=dict(os.environ, PYTHONPATH=str(ROOT)),
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                if not select.select([child.stdout], [], [], 120)[0]:
                    self.fail("the child never said it was ready")
                ready = child.stdout.readline()
                if ready != "ready\n":
                    self.fail(f"the child said {ready!r}: {child.stderr.read()}")
                child.send_signal(signal.SIGTERM)
                out, err = child.communicate(timeout=60)
            finally:
                if child.poll() is None:
                    child.kill()
                    child.communicate()
            meta = [row for f in sorted(Path(d, "mtp-inputs").glob("mtp-inputs-*.npz"))
                    for row in np.load(f)["meta"].tolist()]
            lines = [json.loads(l) for f in Path(d, "draft-ledger").glob("*.jsonl") for l in f.read_text().splitlines()]
            return child.returncode, out + err, meta, lines

    def check_written(self, how: str):
        status, out, meta, lines = self.stopped(how)
        self.assertEqual(status, 128 + signal.SIGTERM, out)
        self.assertEqual(meta, [[5, p, 2000 + p, 0] for p in range(100)])            # exactly the rows it held
        self.assertEqual([r["matched"] for r in lines], [0, 1, 2])
        self.assertIn("  SIGTERM: mtp inputs: 100 rows held at close; this boot 1 shards written, 0 still waiting", out)
        self.assertIn("  SIGTERM: draft ledger: 3 records in draft-ledger-", out)

    def test_sigterm_writes_what_is_held_and_exits_143(self):
        self.check_written("idle")

    def test_a_main_thread_stuck_in_c_does_not_keep_it_unwritten(self):
        """The watch thread reads the wakeup fd: nothing waits for the main thread to run Python again."""
        self.check_written("stuck")

    def test_without_it_sigterm_loses_the_tail(self):
        status, _, meta, _ = self.stopped("unhandled")
        self.assertEqual(status, -signal.SIGTERM)
        self.assertEqual(meta, [])


if __name__ == "__main__":
    unittest.main()
