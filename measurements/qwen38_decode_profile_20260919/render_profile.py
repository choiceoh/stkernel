"""Render a Qwen3.8 decode-step profile (qwen_profile.py's JSON) beside GLM-5.3's 09-14 table.

    python render_profile.py <profile.json> [<glm decode-profile.json>]
"""
import json
import re
import sys

FAMILIES = (
    ("TP collective (one-shot/NCCL)", r"oneshot|osar|publish_packets|k_publish|_consumer|nccl|allreduce|all_reduce|allgather|reduce_scatter"),
    ("memcpy / memset", r"memcpy|memset"),
    ("MoE (b12x)", r"moe|micro|static|dynamic|b12x|kernel_cutlass|pair_sum"),
    ("skinny GEMV (router, mixer down/up)", r"skinny|_gemv|rows_dot|split_sum|_down_gates|_up_mean"),
    ("vocab head (FP8 rows / deep_gemm)", r"fp8_rows|deep_gemm|sm120_fp8"),
    ("mixer Triton (norm streams, leave, mean)", r"_norm_streams|_leave|_mix_mean|_gate_store|gated_residual"),
    ("gates (mixer or GDN: one name)", r"^_gates$"),
    ("GDN / KDA / conv", r"gdn|kda|recurrent|chunk_|_ring|gated_delta|conv"),
    ("QSA", r"qsa|expand_|select_blocks|norm_rope|_select"),
    ("dense W4/FP8 (megakernel)", r"mk_|w4a|dense|input_pack|quant"),
    ("cuBLAS/cutlass GEMM", r"gemm|nvjet|cutlass|cublas|matmul"),
    ("drafter glue (argmax, draft)", r"draft|argmax|vocab|candidates|partials|_finish|row_tap"),
    ("norm / rope", r"norm|rope|rms"),
    ("torch elementwise / copy", r"elementwise|vectorized|unrolled|reduce_kernel|fill|copy|catarray|index|gather|scatter|where"),
)


def family(name: str) -> str:
    low = name.lower()
    for label, pattern in FAMILIES:
        if re.search(pattern, low):
            return label
    return "other"


def main(path, glm=None):
    r = json.load(open(path, encoding="utf-8"))
    print(f"# {r['label']} ({r['model']})\n")
    print("| run | steps | device us/step (profiler) | step ms profiled | step ms plain | kernel share of the plain step | tokens/step | acceptance |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    good = []
    for run in r["runs"]:
        dev = run.get("device_us_per_step")
        plain = run["unprofiled"].get("step_ms")
        prof = run["profiled"].get("step_ms")
        share = f"{100 * dev / 1000 / plain:.0f}%" if dev and plain else "—"
        print(f"| C={run['concurrency']} #{run['repeat']} | {run.get('steps')} | {dev} | {prof and round(prof, 2)} | "
              f"{plain and round(plain, 2)} | {share} | {run['unprofiled'].get('tokens_a_step') and round(run['unprofiled']['tokens_a_step'], 2)} | "
              f"{run['unprofiled'].get('acceptance') and round(run['unprofiled']['acceptance'], 3)} |")
        if run["kernels"]:
            good.append(run)
    for run in good:
        fams, calls = {}, {}
        for k in run["kernels"]:
            f = family(k["kernel"])
            fams[f] = fams.get(f, 0.0) + k["us_per_step"]
            calls[f] = calls.get(f, 0) + k["calls"] / max(1, run["steps"])
        total = sum(fams.values())
        print(f"\n## C={run['concurrency']} #{run['repeat']}: {round(total)} us of kernels a step "
              f"(top {len(run['kernels'])} kernels), launches a step {round(sum(calls.values()))}\n")
        print("| family | us/step | share | launches/step |")
        print("|---|---:|---:|---:|")
        for f, us in sorted(fams.items(), key=lambda kv: -kv[1]):
            print(f"| {f} | {us:.0f} | {100 * us / total:.1f}% | {calls[f]:.0f} |")
        print("\n| kernel (top 30) | family | us/step | calls/step |")
        print("|---|---|---:|---:|")
        for k in run["kernels"][:30]:
            print(f"| `{k['kernel'][:90]}` | {family(k['kernel'])} | {k['us_per_step']} | {k['calls'] / max(1, run['steps']):.1f} |")


if __name__ == "__main__":
    main(*sys.argv[1:])
