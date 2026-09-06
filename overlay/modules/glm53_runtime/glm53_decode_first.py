# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3 decode-first prefill admission, v2 (39차, operator item 2).

Stock vLLM V1 fills each step's token budget (MAX_BATCHED=8192 here) with
prefill chunks regardless of who else is running: a decoder that shares a
step with a 100K prompt waits for the whole chunk (~2.8 s at ~3000 tok/s)
before its next token, because a mixed step is as long as its prefill chunk
-- and the step leaves the uniform, graph-captured decode path as well. FCFS
admission has a second head-of-line problem: a 2K prompt that arrives behind
a 128K prompt is admitted only after the long one's whole prefill (~45 s).

This AsyncScheduler subclass evaluates three policies per step from the
running and waiting queues (stock scheduling otherwise):

  mixed  >= MIN_DECODERS decoders and prefill work pending. Two shapes:
         MODE=alternate (v3, default): the steps never mix. DECODE_STEPS pure
         decode steps (graph-captured, megakernel, prep-fused, full draft
         acceptance) are followed by ONE pure prefill step in which the
         decoders sit out (the stock per-request decode eligibility gate)
         and the prefill chunk, capped at MIXED_CHUNK tokens, runs with the
         SP / KDA / NVFP4 pure-prefill lanes engaged. Measured on 39차 DF1:
         a mixed step on this stack costs ~0.4 s of fixed overhead (eager
         path, megakernel and SP off), so chunking inside mixed steps bought
         ITL at 2.7x the prefill time; alternating pure steps keeps both
         sides on their fast paths.
         MODE=mixed (v2): every prefill request's chunk is capped at
         MIXED_CHUNK tokens for the step (the stock
         ``long_prefill_token_threshold`` cap, applied for the step only);
         with PREFILL_EVERY=N only every N-th mixed step carries prefill.
  fair   no mixed mode and >= 2 prefills pending (running chunks plus the
         waiting requests that fit under max_num_seqs): the chunk is the step
         budget divided by that count, so the short prompt is admitted on the
         same step as the long one's chunk. The batch stays pure prefill, so
         the SP / KDA / NVFP4 prefill lanes still engage.
  floor  a running prefill whose progress since admission has fallen below
         PREFILL_FLOOR tokens/s (after FLOOR_GRACE_S) gets its deficit added
         to the chunk, up to the step budget, and is never deferred: pacing
         under a saturated decode load cannot starve a prompt.

Pure prefill of a single prompt (no decoders, nothing waiting) is untouched.
Knobs (env, read once at init; the launcher forwards the profile's
VLLM_GLM53_* keys):

  VLLM_GLM53_SCHED_MODE           alternate (default) | mixed
  VLLM_GLM53_SCHED_DECODE_STEPS   alternate: pure decode steps per prefill step (6)
  VLLM_GLM53_SCHED_MIXED_CHUNK    prefill tokens per request per prefill step
                                  (1152 = half of the 2304-token block)
  VLLM_GLM53_SCHED_PREFILL_EVERY  mixed: 1 = prefill on every mixed step (1)
  VLLM_GLM53_SCHED_MIN_DECODERS   decoders needed for mixed mode (1)
  VLLM_GLM53_SCHED_FAIR           1 = share the budget among pending prefills (1)
  VLLM_GLM53_SCHED_PREFILL_FLOOR  guaranteed prefill tokens/s per prompt (1000; 0 = off)
  VLLM_GLM53_SCHED_FLOOR_GRACE_S  seconds after admission before the floor
                                  is enforced (2.0)

Armed by the launcher's DECODE_FIRST=1
(``--scheduler-cls vllm.v1.core.sched.glm53_decode_first.Glm53DecodeFirstScheduler``).
Log anchor at init: ``[decode-first] scheduler armed (...)``; a periodic
``[decode-first] steps ...`` line counts what it did.
"""

import os
import time

from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)

_REPORT_EVERY = 2000


def _int_env(name: str, default: int, lo: int) -> int:
    raw = os.environ.get(name, "")
    try:
        value = int(raw) if raw.strip() else default
    except ValueError:
        logger.warning("[decode-first] %s=%r is not an integer; using %d", name, raw, default)
        value = default
    return max(lo, value)


def _float_env(name: str, default: float, lo: float) -> float:
    raw = os.environ.get(name, "")
    try:
        value = float(raw) if raw.strip() else default
    except ValueError:
        logger.warning("[decode-first] %s=%r is not a number; using %s", name, raw, default)
        value = default
    return max(lo, value)


class Glm53DecodeFirstScheduler(AsyncScheduler):
    """Cap, share and pace prefill chunks around the decoders."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        mode = os.environ.get("VLLM_GLM53_SCHED_MODE", "alternate").strip().lower() or "alternate"
        if mode not in ("alternate", "mixed"):
            logger.warning("[decode-first] VLLM_GLM53_SCHED_MODE=%r unknown; using alternate", mode)
            mode = "alternate"
        self.mode = mode
        self.alternate = mode == "alternate"
        self.decode_steps = _int_env("VLLM_GLM53_SCHED_DECODE_STEPS", 6, 1)
        self.mixed_chunk = _int_env("VLLM_GLM53_SCHED_MIXED_CHUNK", 1152, 1)
        self.prefill_every = _int_env("VLLM_GLM53_SCHED_PREFILL_EVERY", 1, 1)
        self.min_decoders = _int_env("VLLM_GLM53_SCHED_MIN_DECODERS", 1, 1)
        self.fair = _int_env("VLLM_GLM53_SCHED_FAIR", 1, 0) > 0
        self.prefill_floor = _int_env("VLLM_GLM53_SCHED_PREFILL_FLOOR", 1000, 0)
        self.floor_grace_s = _float_env("VLLM_GLM53_SCHED_FLOOR_GRACE_S", 2.0, 0.0)
        # request_id -> (admission time, num_computed_tokens at admission)
        self._prefill_starts: dict[str, tuple[float, int]] = {}
        self._mixed_steps = 0
        self._capped_steps = 0
        self._deferred_steps = 0
        self._fair_steps = 0
        self._boost_steps = 0
        self._max_boost = 0
        self._cycle = 0
        logger.warning(
            "[decode-first] scheduler armed (v3 %s: decode_steps=%d, chunk=%d, prefill_every=%d, "
            "min_decoders=%d, fair=%s, prefill_floor=%d tok/s after %.1fs, "
            "max_num_batched_tokens=%d, max_num_seqs=%d, stock threshold=%d)",
            self.mode,
            self.decode_steps,
            self.mixed_chunk,
            self.prefill_every,
            self.min_decoders,
            self.fair,
            self.prefill_floor,
            self.floor_grace_s,
            self.max_num_scheduled_tokens,
            self.max_num_running_reqs,
            self.scheduler_config.long_prefill_token_threshold,
        )

    def _plan(self, now: float) -> tuple[str, int, int, list] | None:
        """Return (mode, chunk cap, floor boost, decoders) or None for a stock step."""
        decoders = []
        prefills = []
        for request in self.running:
            if request.is_prefill_chunk:
                prefills.append(request)
            else:
                decoders.append(request)
        num_decoders = len(decoders)
        admissible = min(len(self.waiting), max(0, self.max_num_running_reqs - len(self.running)))
        num_prefills = len(prefills) + admissible

        boost = 0
        starts = self._prefill_starts
        if self.prefill_floor > 0 and prefills:
            live = set()
            for request in prefills:
                rid = request.request_id
                live.add(rid)
                rec = starts.get(rid)
                if rec is None:
                    starts[rid] = (now, request.num_computed_tokens)
                    continue
                age = now - rec[0]
                if age <= self.floor_grace_s:
                    continue
                deficit = int(self.prefill_floor * age) - (request.num_computed_tokens - rec[1])
                if deficit > boost:
                    boost = deficit
            if len(starts) != len(live):
                for rid in [rid for rid in starts if rid not in live]:
                    del starts[rid]
        elif starts:
            starts.clear()

        if num_prefills > 0 and num_decoders >= self.min_decoders:
            mode, threshold = "mixed", self.mixed_chunk
        elif self.fair and num_prefills >= 2:
            mode, threshold = "fair", max(self.mixed_chunk, self.max_num_scheduled_tokens // num_prefills)
        else:
            return None
        if boost > 0:
            threshold = min(self.max_num_scheduled_tokens, threshold + boost)
        return mode, threshold, boost, decoders

    def _decode_only_step(self) -> SchedulerOutput:
        # Decoders only. The stock deferral path skips in-progress prefill
        # chunks and new admissions; its capacity flag would cancel it
        # whenever the waiting queue is non-empty, which is exactly the case
        # we pace, so it is cleared for this step.
        self._deferred_steps += 1
        self.prefill_capacity_bound = False
        return super().schedule(True)

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        plan = self._plan(time.monotonic())
        if plan is None:
            return super().schedule(throttle_prefills)
        mode, threshold, boost, decoders = plan
        if mode == "mixed":
            self._mixed_steps += 1
            if self.alternate:
                self._cycle += 1
                if boost <= 0 and self._cycle % (self.decode_steps + 1):
                    return self._decode_only_step()
                # The prefill step: every decoder sits this one out through
                # the stock per-request eligibility gate (schedule() has
                # already advanced current_step when it compares), so the
                # batch is pure prefill and the prefill lanes engage. The
                # gate expires by itself on the next step.
                gate = self.current_step + 2
                for request in decoders:
                    request.next_decode_eligible_step = gate
                self._cycle = 0
            elif boost <= 0 and self.prefill_every > 1 and self._mixed_steps % self.prefill_every:
                return self._decode_only_step()
            self._capped_steps += 1
        else:
            self._fair_steps += 1
        if boost > 0:
            self._boost_steps += 1
            if boost > self._max_boost:
                self._max_boost = boost
        managed = self._mixed_steps + self._fair_steps
        if managed % _REPORT_EVERY == 0:
            logger.info(
                "[decode-first] %s steps mixed=%d (prefill=%d decode-only=%d) fair=%d boosted=%d max_boost=%d",
                self.mode,
                self._mixed_steps,
                self._capped_steps,
                self._deferred_steps,
                self._fair_steps,
                self._boost_steps,
                self._max_boost,
            )

        config = self.scheduler_config
        saved = config.long_prefill_token_threshold
        if saved <= 0 or saved > threshold:
            config.long_prefill_token_threshold = threshold
        try:
            return super().schedule(throttle_prefills)
        finally:
            config.long_prefill_token_threshold = saved
