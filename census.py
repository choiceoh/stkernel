#!/usr/bin/env python3
"""스텝 재분해: 커널 인구조사. 스텝 수는 추정하지 않고 센다.

CUPTI 의 절대 시간은 못 믿는다 -- 앞선 세션에서 GPU 바쁜 시간이 136ms 로
부풀려졌다(실측 스텝 47.6ms). 개수는 프로파일링 오버헤드와 무관하므로 개수를
정본으로 쓰고, 시간은 참고로만 병기한다.

분모: 스텝당 정확히 1회 도는 커널(리젝션 샘플러). 앞선 인구조사가 스텝 수를
추정해서 틀렸다.
"""
import gzip, json, re, sys
from collections import defaultdict

path = sys.argv[1]
with gzip.open(path, "rt") as f:
    doc = json.load(f)
evs = [e for e in doc.get("traceEvents", [])
       if e.get("ph") == "X" and "kernel" in (e.get("cat") or "")]
if not evs:
    print("!! kernel 이벤트 없음 — 캡처 실패"); sys.exit(1)

cnt, dur = defaultdict(int), defaultdict(float)
for e in evs:
    n = e.get("name", "?")
    cnt[n] += 1; dur[n] += e.get("dur", 0.0)

# 스텝 분모 후보 (스텝당 1회)
STEP_CANDS = ["_get_num_sampled_and_rejected", "rejection", "sampled_and_rejected"]
steps = None; used = None
for pat in STEP_CANDS:
    hits = [(k, v) for k, v in cnt.items() if pat in k]
    if hits:
        k, v = max(hits, key=lambda x: x[1]); steps = v; used = k; break
print(f"# 트레이스 {path}")
print(f"# 커널 이벤트 {len(evs):,} · 고유 {len(cnt):,}종")
if steps:
    print(f"# 스텝 분모 = {used!r} x {steps}  (스텝당 1회 가정)")
else:
    print("# !! 스텝 분모 커널을 못 찾음 — 아래는 총계이고 스텝당이 아니다")
    steps = 1

# 그룹핑 규칙 (우리 소유 / 남의 것)
GROUPS = [
    ("우리 · osar AR",      r"k_oneshot|k_guard|k_copy_in|k_signal|k_wait|k_reduce"),
    ("우리 · MoE 게이트",    r"_deneb_gate"),
    ("MoE b12x",            r"b12x|moe|topk|Moe"),
    ("mhc (MHC 압축)",       r"mhc"),
    ("KDA/FLA 청크",         r"kda|chunk|recurrent_gated|causal_conv1d|wy_|solve_tril|l2norm|cumsum"),
    ("어텐션 (MLA)",         r"Attention|attn|mla|MLA|flash"),
    ("cutlass/cublas GEMM", r"cutlass|cublas|gemm|Kernel2|splitK"),
    ("elementwise 글루",     r"elementwise|vectorized|unrolled"),
    ("복사/산포/수집",        r"copy|memcpy|scatter|gather|index_"),
    ("NCCL",                r"nccl"),
    ("샘플러/스펙",          r"sampl|reject|dflash|spec|argmax|topp|topk_"),
    ("정규화/양자화",         r"norm|quant|fp8|silu|act_and_mul|rope"),
]
def group(n):
    for g, pat in GROUPS:
        if re.search(pat, n, re.I): return g
    return "기타"

gc, gd = defaultdict(float), defaultdict(float)
for n, c in cnt.items():
    g = group(n); gc[g] += c; gd[g] += dur[n]
tot = sum(gc.values())
print()
print(f"{'그룹':<22} {'커널/스텝':>10} {'비중':>7}   {'us/스텝(참고)':>14}")
print("-" * 60)
for g in sorted(gc, key=lambda x: -gc[x]):
    print(f"{g:<22} {gc[g]/steps:10.1f} {gc[g]/tot*100:6.1f}%   {gd[g]/steps:13.1f}")
print("-" * 60)
print(f"{'합계':<22} {tot/steps:10.1f} {100.0:6.1f}%   {sum(gd.values())/steps:13.1f}")
print()
print("# 상위 개별 커널 (개수순)")
for n, c in sorted(cnt.items(), key=lambda x: -x[1])[:25]:
    print(f"  {c/steps:8.1f}/step  {dur[n]/steps:8.1f}us  {n[:88]}")
print()
print("# ⚠ us 값은 CUPTI 왜곡이 있다(앞선 세션: GPU 바쁜시간 136ms vs 실측 47.6ms).")
print("#   개수는 오버헤드와 무관하므로 개수를 정본으로 쓸 것.")
