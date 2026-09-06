# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3 decode-first prefill admission (39차, operator item 2).

Stock vLLM V1 fills each step's token budget (MAX_BATCHED=8192 here) with
prefill chunks regardless of who else is running: a decoder that shares a
step with a 100K prompt waits for the whole chunk (~2.8 s at ~3000 tok/s)
before its next token, because a mixed step is as long as its prefill chunk
-- and the step leaves the uniform, graph-captured decode path as well.

This AsyncScheduler subclass keeps decoders interactive while prefill work
is pending:

  * with >= MIN_DECODERS active decoders, every request's prefill chunk is
    capped at MIXED_CHUNK tokens for that step (the stock
    ``long_prefill_token_threshold`` cap, applied for the step only), and
  * optionally only every PREFILL_EVERY-th such step carries prefill at all
    (the stock DP prefill-throttle deferral, reused single-DP); the steps in
    between are pure decode steps.

Pure prefill (no decoders) is untouched: the full 8192-token chunk, so the
SP / KDA / NVFP4 pure-prefill lanes still engage. Knobs (env, read once at
init; the launcher forwards the profile's VLLM_GLM53_* keys):

  VLLM_GLM53_SCHED_MIXED_CHUNK    prefill tokens per request per mixed step (512)
  VLLM_GLM53_SCHED_PREFILL_EVERY  1 = prefill on every mixed step (1)
  VLLM_GLM53_SCHED_MIN_DECODERS   decoders needed to engage (1)

Armed by the launcher's DECODE_FIRST=1
(``--scheduler-cls vllm.v1.core.sched.glm53_decode_first.Glm53DecodeFirstScheduler``).
Log anchor at init: ``[decode-first] scheduler armed (...)``; a periodic
``[decode-first] mixed steps=...`` line counts what it did.
"""

import os

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


class Glm53DecodeFirstScheduler(AsyncScheduler):
    """Cap and pace prefill chunks whenever decoders share the step."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mixed_chunk = _int_env("VLLM_GLM53_SCHED_MIXED_CHUNK", 512, 1)
        self.prefill_every = _int_env("VLLM_GLM53_SCHED_PREFILL_EVERY", 1, 1)
        self.min_decoders = _int_env("VLLM_GLM53_SCHED_MIN_DECODERS", 1, 1)
        self._mixed_steps = 0
        self._capped_steps = 0
        self._deferred_steps = 0
        logger.warning(
            "[decode-first] scheduler armed (mixed_chunk=%d, prefill_every=%d, "
            "min_decoders=%d, max_num_batched_tokens=%d, stock threshold=%d)",
            self.mixed_chunk,
            self.prefill_every,
            self.min_decoders,
            self.max_num_scheduled_tokens,
            self.scheduler_config.long_prefill_token_threshold,
        )

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        num_decoders = 0
        prefill_pending = bool(self.waiting)
        for request in self.running:
            if request.is_prefill_chunk:
                prefill_pending = True
            else:
                num_decoders += 1
        if num_decoders < self.min_decoders or not prefill_pending:
            return super().schedule(throttle_prefills)

        self._mixed_steps += 1
        if self._mixed_steps % _REPORT_EVERY == 0:
            logger.info(
                "[decode-first] mixed steps=%d capped=%d deferred=%d (decoders now=%d)",
                self._mixed_steps,
                self._capped_steps,
                self._deferred_steps,
                num_decoders,
            )
        if self.prefill_every > 1 and self._mixed_steps % self.prefill_every:
            # Off-cadence step: decoders only. The stock deferral path skips
            # in-progress prefill chunks and new admissions; the capacity flag
            # would cancel it whenever the waiting queue is non-empty, which
            # is exactly the case we pace, so it is cleared for this step.
            self._deferred_steps += 1
            self.prefill_capacity_bound = False
            return super().schedule(True)

        config = self.scheduler_config
        saved = config.long_prefill_token_threshold
        if saved <= 0 or saved > self.mixed_chunk:
            config.long_prefill_token_threshold = self.mixed_chunk
        self._capped_steps += 1
        try:
            return super().schedule(throttle_prefills)
        finally:
            config.long_prefill_token_threshold = saved
