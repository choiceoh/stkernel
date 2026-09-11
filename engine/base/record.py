"""The last N steps, kept always; written out only when the process dies (base).

D12 chose "dump on death" over always-on tracing (the profiler killed the
engine in the 40th campaign) and over nothing (one death fingerprint cost
three crashes and the logs were overwritten by the next boot). The choice is
cheap to state and expensive to honour, because of what killed us:

  every one of the six deaths was earlyoom's SIGTERM.

So the dump path must work INSIDE a SIGTERM handler, which constrains it to
things that need no allocation and no Python teardown:

  the ring is one bytearray, sized at boot
  the file is opened at boot and its fd kept
  the handler does os.write from memoryview slices, then re-raises the signal

SIGKILL cannot be caught by anyone; nothing here pretends otherwise. earlyoom
sends SIGTERM first and that is the window.

Records survive boots: the file name carries a boot id and nothing here ever
deletes one. Deleting is a person's job.
"""
from __future__ import annotations

import os
import signal
import struct
import time
from pathlib import Path

MAGIC = b"STKR"                # file header: magic, version, record_bytes, capacity
HEADER = struct.Struct("<4sIII")


class Ring:
    """Fixed-size records in a preallocated buffer. Oldest is overwritten."""

    def __init__(self, capacity: int, record_bytes: int):
        if capacity <= 0 or record_bytes <= 0:
            raise ValueError("a ring needs capacity and a record size")
        self.capacity, self.record_bytes = capacity, record_bytes
        self.buf = bytearray(capacity * record_bytes)
        self.view = memoryview(self.buf)
        self.count = 0                      # total pushed, ever

    def push(self, record: bytes) -> None:
        if len(record) > self.record_bytes:
            raise ValueError(f"record of {len(record)} bytes exceeds {self.record_bytes}")
        slot = self.count % self.capacity
        base = slot * self.record_bytes
        self.view[base:base + len(record)] = record
        if len(record) < self.record_bytes:
            self.view[base + len(record):base + self.record_bytes] = bytes(self.record_bytes - len(record))
        self.count += 1

    def ordered(self) -> "list[bytes]":
        """Oldest to newest. For readers, not for the handler."""
        n = min(self.count, self.capacity)
        start = (self.count - n) % self.capacity
        return [bytes(self.view[((start + i) % self.capacity) * self.record_bytes:
                                ((start + i) % self.capacity + 1) * self.record_bytes])
                for i in range(n)]


class DeathDump:
    """Opens the file now; writes the ring when a fatal signal arrives."""

    def __init__(self, directory: "str | Path", ring: Ring, boot_id: "str | None" = None,
                 signals=(signal.SIGTERM,)):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.boot_id = boot_id or f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
        self.path = directory / f"steps-{self.boot_id}.ring"
        self.ring = ring
        # opened at boot; the handler must not open anything
        self.fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        self.header = HEADER.pack(MAGIC, 1, ring.record_bytes, ring.capacity)
        self.written = False
        self._previous = {}
        for sig in signals:
            self._previous[sig] = signal.signal(sig, self._on_signal)

    def write_now(self) -> int:
        """The handler's body. No allocation beyond the write syscalls."""
        if self.written:
            return 0
        os.lseek(self.fd, 0, os.SEEK_SET)
        total = os.write(self.fd, self.header)
        # count as a trailer-free prefix: readers reconstruct order from it
        total += os.write(self.fd, struct.pack("<Q", self.ring.count))
        total += os.write(self.fd, self.ring.view)
        os.fsync(self.fd)
        self.written = True
        return total

    def _on_signal(self, signum, frame):
        self.write_now()
        previous = self._previous.get(signum)
        signal.signal(signum, previous if callable(previous) else signal.SIG_DFL)
        os.kill(os.getpid(), signum)          # die the way we were asked to

    def close(self) -> None:
        os.close(self.fd)


def read(path: "str | Path") -> "list[bytes]":
    """Records oldest to newest, from a dumped file."""
    raw = Path(path).read_bytes()
    magic, version, record_bytes, capacity = HEADER.unpack_from(raw, 0)
    if magic != MAGIC:
        raise ValueError(f"{path}: not a step ring")
    (count,) = struct.unpack_from("<Q", raw, HEADER.size)
    body = raw[HEADER.size + 8:]
    n = min(count, capacity)
    start = (count - n) % capacity
    return [bytes(body[((start + i) % capacity) * record_bytes:((start + i) % capacity + 1) * record_bytes])
            for i in range(n)]


def _selfcheck() -> None:
    import tempfile
    ring = Ring(capacity=4, record_bytes=8)
    for i in range(6):
        ring.push(f"step{i:03d}".encode())
    assert [r.rstrip(b"\0") for r in ring.ordered()] == [b"step002", b"step003", b"step004", b"step005"]
    with tempfile.TemporaryDirectory() as d:
        dump = DeathDump(d, ring, boot_id="selfcheck", signals=())
        n = dump.write_now()
        assert n == HEADER.size + 8 + 4 * 8
        back = read(dump.path)
        assert [r.rstrip(b"\0") for r in back] == [b"step002", b"step003", b"step004", b"step005"]
        assert dump.write_now() == 0                       # idempotent: one dump per death
        dump.close()
        # the real path: a child installs the handler, gets SIGTERM, the file has the ring
        import subprocess, sys, textwrap
        code = textwrap.dedent(f"""
            import os, signal, sys, time
            sys.path.insert(0, {str(Path.cwd())!r})
            from engine.base.record import Ring, DeathDump
            r = Ring(3, 8)
            for i in range(5): r.push(f"s{{i}}".encode())
            d = DeathDump({d!r}, r, boot_id="child")
            print("armed", flush=True)
            while True: time.sleep(0.05)
        """)
        p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
        assert p.stdout.readline().strip() == "armed"
        p.send_signal(signal.SIGTERM); rc = p.wait(timeout=10)
        assert rc == -signal.SIGTERM, f"child should die of SIGTERM, got {rc}"
        got = [r.rstrip(b"\0") for r in read(Path(d) / "steps-child.ring")]
        assert got == [b"s2", b"s3", b"s4"], got
    print("  record: ring order, dump/read round trip, SIGTERM handler wrote then died OK")


if __name__ == "__main__":
    _selfcheck()
