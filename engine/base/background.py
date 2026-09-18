"""Host work a boot runs where it is already waiting (base): a thread started early, joined where its result is needed.

Every profile's fleet boot has the same two kinds of dead time -- a rendezvous where the fast ranks wait for the slow
one, and a device phase where python holds nothing -- and the same host work that fits in them: importing the kernel
packages, reading the tokenizer and the chat template. GLM-5.3's boot measured it first (engine/profiles/glm53/boot.py
`fleet`, `Prelude`); this is the piece of it that knows nothing about a model.
"""
from __future__ import annotations

import threading
import time


class Background:
    """Host work started where the boot is already waiting, and joined where its result is needed.

    The boot has two kinds of dead time -- a rendezvous where the fast ranks wait for the slow one, and
    a device phase where python holds nothing. Both are free seconds for work that touches no CUDA and
    reads nothing the engine has built. The join is always its own recorder row, so a job that fails to
    hide says so in seconds instead of disappearing into the phase it was supposed to hide under.
    """

    def __init__(self, work, name: str):
        self.work, self.result, self.error, self.seconds = work, None, None, 0.0
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)

    def start(self) -> "Background":
        self._thread.start()
        return self

    def _run(self) -> None:
        start = time.perf_counter()
        try:
            self.result = self.work()
        except BaseException as exc:            # noqa: BLE001 -- re-raised on the main thread, in its phase
            self.error = exc
        finally:
            self.seconds = time.perf_counter() - start

    def take(self):
        self._thread.join()
        if self.error is not None:
            raise self.error
        return self.result


__all__ = ["Background"]
