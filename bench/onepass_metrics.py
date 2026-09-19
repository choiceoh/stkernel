"""Onepass's ST counter sampler, independent of the retired overlay benches.

The public vllm:* metric names remain part of ST's HTTP metric contract.
Acceptance and step-window arithmetic are preserved from the former helpers.
"""
import os
from pathlib import Path
import re
import subprocess
import threading
import time
import urllib.request

from window_metrics import metric_sum, traffic_state


API_PORT = int(os.environ.get("GLM53_API_PORT", "8000"))
if not 1024 <= API_PORT <= 65535:
    raise ValueError("GLM53_API_PORT must be 1024..65535")
URL = f"http://127.0.0.1:{API_PORT}/v1/chat/completions"
METRICS = f"http://127.0.0.1:{API_PORT}/metrics"


def _git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10,
                              cwd=Path(__file__).resolve().parent).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _parse_spec_metrics(text):
    out = {}
    for line in text.splitlines():
        match = re.fullmatch(r"([A-Za-z0-9_:]*spec_decode[A-Za-z0-9_:]*)(?:\{[^}]*\})?\s+([0-9.eE+-]+)\s*", line)
        if match:
            name, value = match.groups()
            out[name] = out.get(name, 0.0) + float(value)
    return out


def _spec_delta(before, after):
    delta = {k: after.get(k, 0.0) - before.get(k, 0.0) for k in after}
    accepted = sum(v for k, v in delta.items() if "accept" in k)
    drafted = sum(v for k, v in delta.items() if "draft" in k and "accept" not in k)
    exact_accepted = sum(v for k, v in delta.items() if k.endswith("num_accepted_tokens_total"))
    exact_drafted = sum(v for k, v in delta.items() if k.endswith("num_draft_tokens_total"))
    return (accepted / drafted if drafted > 0 else None,
            exact_accepted / exact_drafted if exact_drafted > 0 else None)


def spec_k_eff(before, after):
    delta = {k: after.get(k, 0.0) - before.get(k, 0.0) for k in after}
    tokens = sum(v for k, v in delta.items() if k.endswith("num_draft_tokens_total"))
    rounds = sum(v for k, v in delta.items() if k.endswith("num_drafts_total"))
    return tokens / rounds if rounds > 0 and tokens > 0 else None


class _StepWindows:
    def __init__(self, endpoint, period=2.0):
        self.endpoint, self.period = endpoint, period
        self.samples, self.traffic_samples = [], []
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._run, daemon=True)

    def _steps(self):
        try:
            text = urllib.request.urlopen(self.endpoint.METRICS, timeout=5).read().decode()
        except Exception:
            return None
        self.traffic_samples.append(traffic_state(text))
        total = metric_sum(text, "vllm:iteration_tokens_total_count")
        if total is not None:
            return total
        parts = [metric_sum(text, name) for name in ("st:steps_prefill_total", "st:steps_decode_total")]
        return sum(parts) if all(value is not None for value in parts) else None

    def _run(self):
        while not self._stop.is_set():
            steps = self._steps()
            if steps is not None:
                self.samples.append((time.monotonic(), steps))
            self._stop.wait(self.period)

    def __enter__(self):
        self._th.start()
        return self

    def __exit__(self, *args):
        self._stop.set()
        self._th.join(timeout=self.period + 5)
