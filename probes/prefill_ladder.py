#!/usr/bin/env python3
"""Where does prefill time go? A length ladder that separates the candidates.

Prefill on this fleet measures ~2,000 tok/s and the step budget only accounts
for a sixth of it -- weight reads ~13%, MoE FLOPs ~1.7%, the compact path's
gather ~1.0%, its host syncs ~0.4%. The rest has been inferred, not measured,
and the inference has been wrong before.

One sweep separates the candidates without a boot:

  tok/s flat across lengths        -> throughput bound (weights, compile, fixed
                                      per-forward cost); attention is not it
  tok/s falls as length grows      -> attention is superlinear here
  first sample slow, rest fast     -> JIT compile, not steady-state cost
  tok/s rises then plateaus        -> a fixed per-forward cost being amortised;
                                      the plateau is the real rate

Each length is sent COLD first (a fresh prefix, so prefix caching cannot hide
the work) and then repeated; the gap between the two is the compile/one-off
tax, which is the channel #164's load-time warmup has to move.

  BENCH_MODEL=glm-5.3-flash python3 probes/prefill_ladder.py
  BENCH_MODEL=... python3 probes/prefill_ladder.py 1024 4096 16384
"""

import json
import os
import sys
import time
import urllib.request

URL = os.environ.get("BENCH_URL", "http://localhost:8000/v1/chat/completions")
def _resolve_model(default: str) -> str:
    """The served name, asked of the server, not assumed.

    The literal default is the dsv4 lane's name. Pointed at the glm53 server
    it 404s, which has silently voided prefill and decode runs in this lane
    more than once -- the harness raised, the boot script's grep found no
    SUMMARY line, and the section read as "measured nothing" rather than
    "never ran". Ask; fall back to the literal only if the server cannot say.
    """
    named = os.environ.get("BENCH_MODEL")
    if named:
        return named
    try:
        import urllib.request as _u
        base = URL.split("/v1/", 1)[0]
        with _u.urlopen(f"{base}/v1/models", timeout=5) as r:
            data = json.loads(r.read().decode())["data"]
        if len(data) == 1:
            return data[0]["id"]
        for entry in data:
            if entry["id"] == default:
                return default
        if data:
            return data[0]["id"]
    except Exception:
        pass
    return default


MODEL = _resolve_model("deepseek-v4-flash")
REPEATS = int(os.environ.get("PREFILL_REPEATS", "2"))
DEFAULT_LENGTHS = (512, 1024, 2048, 4096, 8192, 16384)

# ~4 chars per token for this tokenizer on Korean prose; the exact ratio does
# not matter because the server reports the true prompt_tokens back.
_FILLER = "인공지능 기술의 발전과 그 사회적 영향에 대하여 자세히 서술한다. "


def _prompt(target_tokens: int, seed: int) -> str:
    """A prompt of roughly the requested length, unique per seed.

    The seed goes in FIRST so no two samples share a prefix -- prefix caching
    would otherwise turn a cold sample into a warm one and hide the very cost
    this probe is looking for.
    """
    head = f"[샘플 {seed}] "
    body = _FILLER * max(1, (target_tokens * 4) // len(_FILLER))
    return head + body


def _one(target_tokens: int, seed: int) -> tuple[int, float]:
    """Send one prompt, return (prompt_tokens, seconds to first/only token)."""
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": _prompt(target_tokens, seed)}],
        "max_tokens": 1,
        "temperature": 0,
        "chat_template_kwargs": {"thinking": False},
    }).encode()
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"}
    )
    start = time.time()
    with urllib.request.urlopen(req, timeout=1800) as resp:
        out = json.load(resp)
    return int(out["usage"]["prompt_tokens"]), time.time() - start


def main(lengths) -> int:
    print(f"model={MODEL}  repeats={REPEATS}")
    print(f"{'요청':>7} {'실제tok':>8} {'cold tok/s':>11} {'warm tok/s':>11} "
          f"{'cold TTFT':>10} {'warm TTFT':>10} {'cold세금':>9}")
    rows = []
    for i, want in enumerate(lengths):
        seed = int(time.time() * 1000) % 100000 + i * 7919
        samples = []
        for r in range(max(2, REPEATS)):
            # Every repeat gets DIFFERENT content at the SAME length. Reusing
            # the prompt would score a prefix-cache hit and the "warm" column
            # would measure that instead of the thing being asked about --
            # whether the shape's kernel was already compiled.
            tok, secs = _one(want, seed + r * 1000)
            samples.append((tok, secs))
        tok = samples[0][0]
        cold = samples[0][1]
        warm = min(s[1] for s in samples[1:])
        rows.append((want, tok, cold, warm))
        tax = 100.0 * (cold - warm) / cold if cold > 0 else 0.0
        print(f"{want:>7} {tok:>8} {tok/cold:>11.1f} {tok/warm:>11.1f} "
              f"{cold:>9.2f}s {warm:>9.2f}s {tax:>8.1f}%")

    if len(rows) >= 2:
        print()
        base_tok, base_warm = rows[0][1], rows[0][3]
        base_rate = base_tok / base_warm
        top_tok, top_warm = rows[-1][1], rows[-1][3]
        top_rate = top_tok / top_warm
        drift = 100.0 * (top_rate - base_rate) / base_rate
        print(f"  warm 처리량: {base_rate:.0f} -> {top_rate:.0f} tok/s "
              f"({drift:+.1f}% over {base_tok} -> {top_tok} tokens)")
        if drift < -20:
            print("  -> 길이에 따라 나빠진다: 어텐션이 초선형. "
                  "sparse-MLA / KDA 쪽을 본다")
        elif drift > 20:
            print("  -> 길이에 따라 좋아진다: 포워드당 고정비가 상각되는 것. "
                  "짧은 프리필의 고정비를 본다")
        else:
            print("  -> 평탄: 처리량 바운드. 어텐션은 범인이 아니고, "
                  "가중치 읽기와 컴파일이 남는다")
        worst = max(rows, key=lambda r: (r[2] - r[3]) / max(r[2], 1e-9))
        tax = 100.0 * (worst[2] - worst[3]) / worst[2]
        print(f"  cold 세금 최대: {worst[1]} tok 에서 {tax:.1f}% "
              f"({worst[2]:.2f}s -> {worst[3]:.2f}s)")
        if tax > 15:
            print("  -> 첫 노출이 크게 비싸다: JIT 컴파일이 실물. "
                  "VLLM_B12X_EP_WARM_COMPACT=1 로 A/B 하라")
        else:
            print("  -> 첫 노출 세금이 작다: 컴파일은 정상상태 비용이 아니다")
    return 0


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:]] or list(DEFAULT_LENGTHS)
    raise SystemExit(main(args))
