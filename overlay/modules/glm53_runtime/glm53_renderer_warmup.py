# SPDX-License-Identifier: Apache-2.0
"""Overlap GLM's CPU MM warmup with spawned engine initialization.

Both processors still run the stock warmup and cache cleanup. The normal
warmup call joins the work before HTTP readiness; failed processors retry
there. Chat template warmup remains in its original location with the actual
ChatParams. Nothing runs concurrently with incoming requests.
"""
import logging
import os
import time

logger = logging.getLogger(__name__)


def start_renderer_warmup(renderer):
    if os.environ.get("VLLM_GLM53_EARLY_MM_WARMUP", "0") != "1":
        return False
    try:
        config = renderer.config
        mm = config.model_config.multimodal_config
        if (config.model_config.hf_config.model_type != "glm5_next"
                or os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") != "spawn"
                or config.parallel_config._api_process_count != 1
                or mm is None or mm.mm_ipc_gpu_memory_gb != 0
                or mm.use_gpu_video_backend() or mm.mm_device_do_normalize
                or renderer.mm_processor is None):
            logger.info("[early-mm-warmup] skipped unsupported renderer configuration")
            return False
        if getattr(renderer, "_glm53_early_mm_future", None) is not None:
            return True
        from vllm.utils.torch_utils import set_default_torch_num_threads

        original = renderer._warmup_mm_processor
        normal_warmup = renderer.warmup
        normal_shutdown = renderer.shutdown
        targets = [(renderer.mm_processor, "Multi-modal", renderer.clear_mm_cache)]
        if renderer._readonly_mm_processor is not None:
            readonly = renderer._readonly_mm_processor
            targets.append((readonly, "Readonly multi-modal",
                            lambda: renderer._clear_processor_cache(readonly)))
        warmed = []

        def run():
            started = time.perf_counter()
            # Matches BaseRenderer.warmup's thread guard. This changes no
            # environment variables; spawn children keep their normal thread
            # configuration. Reuse the renderer's serial MM executor.
            with set_default_torch_num_threads(1):
                for processor, label, clear in targets:
                    try:
                        try:
                            original(processor, log_prefix=label)
                        finally:
                            clear()
                    except Exception:
                        logger.warning("[early-mm-warmup] %s failed; normal warmup will retry",
                                       label, exc_info=True)
                    else:
                        warmed.append(processor)
            logger.info("[early-mm-warmup] completed processors=%d/%d elapsed_s=%.3f",
                        len(warmed), len(targets), time.perf_counter() - started)

        future = renderer._mm_executor.submit(run)

        def consume(processor, *, log_prefix):
            started = time.perf_counter()
            try:
                future.result()
            except Exception:
                logger.warning("[early-mm-warmup] background setup failed; using normal warmup",
                               exc_info=True)
            for i, done in enumerate(warmed):
                if processor is done:
                    warmed.pop(i)
                    logger.info("[early-mm-warmup] reused %s join_s=%.3f",
                                log_prefix, time.perf_counter() - started)
                    return None
            return original(processor, log_prefix=log_prefix)

        def joined_warmup(*args, **kwargs):
            # Join before BaseRenderer enters its own global Torch thread
            # guard or touches the chat tokenizer. Nested guards in two
            # threads could otherwise restore the wrong process setting.
            try:
                future.result()
            except Exception:
                logger.warning("[early-mm-warmup] early task failed; retrying in normal warmup",
                               exc_info=True)
            return normal_warmup(*args, **kwargs)

        def joined_shutdown(*args, **kwargs):
            # BaseRenderer closes the processor cache before a nonblocking
            # executor shutdown. Finish using it before that close on errors
            # or an explicit shutdown during engine initialization.
            try:
                future.result()
            except Exception:
                pass
            return normal_shutdown(*args, **kwargs)

        renderer.warmup = joined_warmup
        renderer.shutdown = joined_shutdown
        renderer._glm53_early_mm_future = future
        renderer._warmup_mm_processor = consume
        logger.info("[early-mm-warmup] submitted processors=%d before engine startup", len(targets))
        return True
    except Exception:
        logger.warning("[early-mm-warmup] setup unavailable; using normal warmup", exc_info=True)
        return False
