#!/usr/bin/env python3
"""Feed a calibration boot (VLLM_GLM53_MK_CALIB=1) the traffic whose input
Hessians the GPTQ packer wants: diverse prompts -- Korean technical and
everyday text, English, code, a long-form summary -- with a short
completion each, so the token budget is mostly PREFILL rows (eager, never
a graph replay) of the layers' real inputs.

Every rank accumulates H = sum x x^T per served linear until
VLLM_GLM53_MK_CALIB_TOKENS rows, then dumps <MK_CALIB_DIR>/rank<r>/<name>.pt
and logs "[megakernel] calib: dumped N Hessians". Run this once after the
boot is healthy, then watch the head's log for that line on every rank.

    python3 bench/mk-calib-run.py [--rounds 4] [--max-tokens 64]

4 rounds of the 36 prompts are ~40K tokens, past the 32768 default budget.
"""
import argparse
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_common import resolve_model  # noqa: E402

URL = os.environ.get("BENCH_URL", "http://127.0.0.1:8000/v1/chat/completions")

KO_TECH = [
    "조력 발전의 원리를 한국어로 자세히 설명해줘.",
    "한국의 전력 계통에서 주파수를 유지하는 방법을 설명해줘.",
    "변압기의 냉각 방식(유입 자냉식, 유입 풍냉식, 송유 수냉식)의 차이를 한국어로 설명해줘.",
    "태양광 인버터의 MPPT 제어를 한국어로 설명해줘.",
    "리튬이온 배터리의 BMS가 셀 밸런싱을 하는 이유와 방법을 한국어로 설명해줘.",
    "HVDC 송전이 교류 송전보다 유리한 경우를 한국어로 정리해줘.",
    "반도체 공정에서 EUV 노광의 원리와 한계를 한국어로 설명해줘.",
    "트랜스포머 모델의 어텐션 메커니즘을 한국어로 설명해줘.",
    "TCP 혼잡 제어 알고리즘 세 가지를 한국어로 비교해줘.",
    "쿠버네티스에서 파드가 재시작되는 원인을 한국어로 정리해줘.",
]
KO_DAILY = [
    "서울에서 부산까지 기차로 가는 방법을 한국어로 안내해줘.",
    "한국어 맞춤법에서 띄어쓰기가 어려운 사례를 다섯 개 들어줘.",
    "김치를 담그는 과정을 순서대로 한국어로 설명해줘.",
    "한국의 사계절 기후 특징을 한국어로 서술해줘.",
    "직장 동료에게 보내는 정중한 회의 연기 요청 이메일을 한국어로 써줘.",
    "초등학생에게 분수의 덧셈을 한국어로 설명해줘.",
    "제주도 2박 3일 여행 일정을 한국어로 짜줘.",
    "한국 전통 음식 다섯 가지와 그 유래를 한국어로 소개해줘.",
    "면접에서 자기소개를 1분 안에 하는 요령을 한국어로 알려줘.",
    "아파트 층간소음 문제를 이웃과 원만하게 해결하는 방법을 한국어로 조언해줘.",
]
EN = [
    "Explain how a PID controller works and how to tune it.",
    "Summarize the causes of the 2008 financial crisis in five paragraphs.",
    "What is the difference between TCP and UDP? Give examples of each.",
    "Describe the process of photosynthesis in detail.",
    "Write a short essay on the ethics of autonomous vehicles.",
    "How does public-key cryptography work? Explain RSA briefly.",
    "Explain the CAP theorem with practical examples.",
    "Describe how a modern GPU schedules work across streaming multiprocessors.",
]
CODE = [
    "Write a Python function that merges two sorted lists without using sorted().",
    "Implement an LRU cache in C++ with O(1) get and put.",
    "Write a SQL query that finds the second highest salary per department.",
    "Explain this Rust borrow checker error and fix it: `cannot borrow `v` as mutable because it is also borrowed as immutable`.",
    "Write a bash script that rotates log files older than 7 days.",
    "Write a CUDA kernel that computes a row-wise softmax for a [M, N] fp32 matrix.",
]
LONG = [
    "다음 글을 한국어로 요약해줘: " + ("전력 계통의 주파수는 발전량과 수요의 균형이 무너질 때 변한다. 발전이 수요보다 많으면 주파수가 올라가고, 적으면 내려간다. 계통 운영자는 예비력을 확보하고, 자동 발전 제어와 조속기 응답으로 초 단위의 변동을 흡수한다. 재생에너지 비중이 높아지면 관성이 줄어 주파수 변동이 커지므로 에너지 저장 장치와 합성 관성이 중요해진다. " * 6),
    "Summarize the following text in three bullet points: " + ("Large language models are trained on vast corpora of text using next-token prediction. Their capabilities emerge with scale, but so do their costs: training requires thousands of accelerators for months, and serving requires careful attention to memory bandwidth, batching, and speculative decoding. Quantization reduces the bytes each weight occupies, trading a small amount of accuracy for large gains in throughput. " * 6),
]


def ask(model, prompt, max_tokens):
    body = json.dumps({"model": model, "max_tokens": max_tokens, "temperature": 0,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    u = d.get("usage", {})
    return int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=64)
    args = ap.parse_args()
    model = resolve_model("glm-5.3-flash", URL)
    prompts = KO_TECH + KO_DAILY + EN + CODE + LONG
    tot_p = tot_c = 0
    t0 = time.time()
    for r in range(args.rounds):
        for i, p in enumerate(prompts):
            # a different suffix per round changes the prefill rows (no
            # prefix cache hit repeats the same activations)
            q = p if r == 0 else f"{p} (답변 {r + 1}번째 버전, 이전과 다르게)"
            pt, ct = ask(model, q, args.max_tokens)
            tot_p += pt; tot_c += ct
        print(f"round {r + 1}: {tot_p} prompt tokens, {tot_c} completion tokens so far "
              f"({time.time() - t0:.0f}s)")
    print(f"done: {len(prompts) * args.rounds} requests, {tot_p + tot_c} tokens; "
          "look for '[megakernel] calib: dumped' on every rank")
    return 0


if __name__ == "__main__":
    sys.exit(main())
