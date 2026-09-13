"""A step that never ends, bounded (45차, 2026-09-12 22:31).

The fleet's worst failure is not a rank that dies but four that live: on 2026-09-12 the
ranks entered one all_gather with two shapes (seven rows against six) and sat in it for
sixteen minutes with the GPUs at 0% and requests queued behind them -- and nothing in
the process was going to end that. The NCCL watchdog's own bound was two hours away
(TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200), SIGTERM cannot reach a Python handler while
the main thread is inside a wedged CUDA call, and the step ring that would have said
where it stopped is written by a death handler that never ran.

So a thread beside the loop keeps time for it. The loop says what it is doing
(`enter("step 2299")`) and when it is done (`leave()`); after `note_s` seconds the
watch says so in the log, once, the way the one-shot kernel's stall word does; after
`trap_s` it writes a note file naming the phase and how long it waited, writes the step
ring itself (the DeathDump's `write_now`, which is why it exists as a method) and kills
the process with SIGKILL -- the one signal a stuck main thread cannot ignore. The
container exits, the launcher and the bracket see a dead rank instead of a door that
answers nothing, and the note in the dump directory outlives the container.

Nothing here allocates on the device or takes a lock the loop holds. Thresholds:
ST_STEP_STALL_NOTE_S (60) and ST_STEP_STALL_TRAP_S (300); 0 disables either.
"""
from __future__ import annotations

import json
import os
import signal
import threading
import time
from pathlib import Path

NOTE_S = float(os.environ.get("ST_STEP_STALL_NOTE_S", "60"))
TRAP_S = float(os.environ.get("ST_STEP_STALL_TRAP_S", "300"))


class StepWatch:
    """Keeps time for one loop; `enter`/`leave` are the loop's, `check` is the thread's (or a test's)."""

    def __init__(self, rank: int, *, note_s: float = NOTE_S, trap_s: float = TRAP_S, notes_dir=None,
                 dump=None, clock=time.monotonic, kill=None, say=print, period_s: float = 1.0):
        for name, value in (("note_s", note_s), ("trap_s", trap_s), ("period_s", period_s)):
            if type(value) not in (int, float) or value < 0 or value != value:
                raise ValueError(f"{name} must be a nonnegative number")
        if period_s <= 0:
            raise ValueError("period_s must be positive")
        self.rank = int(rank)
        self.note_s, self.trap_s, self.period_s = float(note_s), float(trap_s), float(period_s)
        self.notes_dir = Path(notes_dir) if notes_dir else None
        self.dump, self.clock, self.say = dump, clock, say
        self.kill = kill if kill is not None else self._kill
        self._lock = threading.Lock()
        self._began = None
        self._what = None
        self._noted = False
        self.notes = 0                              # stall notes said so far
        self.trapped = None                         # the note written at the trap, if it fired
        self._thread = None
        self._stop = threading.Event()

    # -- the loop's side -------------------------------------------------------------
    def enter(self, what: str) -> None:
        """The loop is about to do `what`; the clock starts (or restarts) now."""
        with self._lock:
            self._began, self._what, self._noted = self.clock(), str(what), False

    def leave(self) -> None:
        with self._lock:
            self._began = None
            self._what = None

    @property
    def current(self) -> "str | None":
        """What the loop is doing now, for a death note."""
        with self._lock:
            return self._what

    @property
    def armed(self) -> bool:
        return self.note_s > 0 or self.trap_s > 0

    def start(self) -> None:
        """The thread; idempotent, a no-op when both thresholds are 0."""
        if not self.armed or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="step-watch", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    # -- the thread's side -----------------------------------------------------------
    def check(self, now: "float | None" = None) -> "str | None":
        """'note' or 'trap' when a threshold was crossed just now, else None."""
        now = self.clock() if now is None else now
        with self._lock:
            began, what, noted = self._began, self._what, self._noted
        if began is None:
            return None
        waited = now - began
        if self.trap_s > 0 and waited >= self.trap_s:
            self._trap(what, waited)
            return "trap"
        if self.note_s > 0 and waited >= self.note_s and not noted:
            with self._lock:
                self._noted = True
            self.notes += 1
            self.say(f"[serve] STALL rank={self.rank} {what} has not returned after {waited:.0f}s"
                     + (f"; a trap follows at {self.trap_s:.0f}s" if self.trap_s > 0 else ""), flush=True)
            return "note"
        return None

    def _run(self) -> None:
        while not self._stop.wait(self.period_s):
            try:
                if self.check() == "trap":
                    return
            except Exception as exc:                # noqa: BLE001 -- the watch must outlive its own mistakes
                self.say(f"[serve] step watch: {type(exc).__name__}: {exc}", flush=True)

    def _trap(self, what: str, waited: float) -> None:
        note = dict(rank=self.rank, what=what, waited_s=round(waited, 1), trap_s=self.trap_s,
                    t=time.strftime("%F %T"), pid=os.getpid(), notes=self.notes,
                    why="the step loop has not returned; the ring was written by the watch and the process SIGKILLed")
        self.trapped = note
        if self.notes_dir is not None:
            try:
                self.notes_dir.mkdir(parents=True, exist_ok=True)
                path = self.notes_dir / f"stall-rank{self.rank}-{time.strftime('%Y%m%d-%H%M%S')}.json"
                path.write_text(json.dumps(note, ensure_ascii=False, indent=1))
                note["path"] = str(path)
            except OSError as exc:
                note["path_error"] = f"{type(exc).__name__}: {exc}"
        self.say(f"[serve] STALL TRAP rank={self.rank} {what} has not returned after {waited:.0f}s: "
                 f"writing the ring and killing this rank (note: {note.get('path', 'not written')})", flush=True)
        if self.dump is not None:
            try:
                self.dump()
            except Exception as exc:                # noqa: BLE001 -- the kill must follow regardless
                self.say(f"[serve] step watch: ring not written: {type(exc).__name__}: {exc}", flush=True)
        self.kill()

    @staticmethod
    def _kill() -> None:
        os.kill(os.getpid(), signal.SIGKILL)
