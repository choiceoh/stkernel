#!/usr/bin/env python3
"""스텝 재분해: 커널 인구조사. 스텝 수는 추정하지 않고 센다.

CUPTI 의 절대 시간은 못 믿는다 -- 앞선 세션에서 GPU 바쁜 시간이 136ms 로
부풀려졌다(실측 스텝 47.6ms). 개수는 프로파일링 오버헤드와 무관하므로 개수를
정본으로 쓰고 시간은 참고로만 병기한다.

분모: 스텝당 정확히 1회 도는 커널(리젝션 샘플러). 앞선 인구조사가 스텝 수를
추정해서 틀렸다.

인접성 모드 (--after REGEX): 특정 커널(예: k_oneshot) 직후 같은 스트림에서
몇 번째 커널이 도는지 센다. "AR 뒤에 무엇이 붙는다"류 주장을 소스 추정이
아니라 트레이스로 확정할 때 쓴다 -- 그룹 합계는 소스를 읽기 전까지 의문점이다.
"""
import gzip, json, os, re, sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools"))
from trace_common import owner  # 소유 판정은 트레이스 도구와 한 곳에서


def iter_events(path):
    """traceEvents 를 한 개씩 흘려보낸다.

    json.load 는 40 MB 트레이스에서 수 GB 를 잡는다 -- 부팅 중인 노드에서는
    그 자체가 earlyoom 감이라(원장 26차) 스트리밍으로 읽는다. 커널 이름 안에
    `{lambda()#3}` 같은 중괄호가 있으므로 문자열 상태를 추적해서 자른다.
    """
    with gzip.open(path, "rt") as f:
        head = ""
        while '"traceEvents"' not in head:              # 배열 앞의 메타는 건너뛴다
            chunk = f.read(1 << 16)
            if not chunk:
                return
            head = head[-32:] + chunk
        tail = head[head.index('"traceEvents"'):]
        buf, depth, in_str, esc, started = [], 0, False, False, False
        while True:
            for c in tail:
                if in_str:
                    if esc:            esc = False
                    elif c == "\\":    esc = True
                    elif c == '"':     in_str = False
                elif c == '"':         in_str = True
                elif c == "{":
                    depth += 1
                    if depth == 1:
                        buf, started = [], True
                elif c == "}":
                    depth -= 1
                    if depth == 0 and started:
                        buf.append(c)
                        yield json.loads("".join(buf))
                        buf, started = [], False
                        continue
                if started:
                    buf.append(c)
            tail = f.read(1 << 20)
            if not tail:
                return


# ---- args: [trace] [--after REGEX] [--depth N] --------------------------
_path = None
after = None
depth = 2
_args = sys.argv[1:]
_i = 0
while _i < len(_args):
    _a = _args[_i]
    if _a == "--after" and _i + 1 < len(_args):
        after = _args[_i + 1]; _i += 2
    elif _a == "--depth" and _i + 1 < len(_args):
        depth = int(_args[_i + 1]); _i += 2
    elif _path is None:
        _path = _a; _i += 1
    else:
        print(f"?? 인수 무시: {_a}"); _i += 1
if not _path:
    print("usage: census.py <trace.json.gz> [--after REGEX] [--depth N]")
    sys.exit(1)
path = _path

cnt, dur = defaultdict(int), defaultdict(float)
n_ev = 0
evs = [] if after else None   # 인접성 모드만 (pid,tid,ts,name) 을 모은다
for e in iter_events(path):
    if e.get("ph") != "X" or "kernel" not in (e.get("cat") or ""):
        continue
    n_ev += 1
    n = e.get("name", "?")
    cnt[n] += 1; dur[n] += e.get("dur", 0.0)
    if evs is not None:
        evs.append((e.get("pid"), e.get("tid"), e.get("ts", 0), n))
if not cnt:
    print("!! kernel 이벤트 없음 — 캡처 실패"); sys.exit(1)

# 스텝 분모 후보 (스텝당 1회)
STEP_CANDS = ["_get_num_sampled_and_rejected", "rejection", "sampled_and_rejected"]
steps = None; used = None
for pat in STEP_CANDS:
    hits = [(k, v) for k, v in cnt.items() if pat in k]
    if hits:
        k, v = max(hits, key=lambda x: x[1]); steps = v; used = k; break
print(f"# 트레이스 {path}")
print(f"# 커널 이벤트 {n_ev:,} · 고유 {len(cnt):,}종")
if steps:
    print(f"# 스텝 분모 = {used!r} x {steps}  (스텝당 1회 가정)")
else:
    print("# !! 스텝 분모 커널을 못 찾음 — 아래는 총계이고 스텝당이 아니다")
    steps = 1

# ---- 인접성 모드: 매칭 커널 직후 같은 (pid,tid) 스트림에서 도는 커널 ----
if after:
    pat = re.compile(after)
    # 토치 프로파일러 커널 이벤트는 pid=디바이스, tid=스트림이므로 (pid,tid) 가
    # 스트림 식별자다. 다른 스트림의 커널은 '직후'가 아니다(profile-step.py 와
    # 같은 규율).
    streams = defaultdict(list)
    for pid, tid, ts, n in evs:
        streams[(pid, tid)].append((ts, n))
    for s in streams.values():
        s.sort(key=lambda t: t[0])
    succ = [defaultdict(int) for _ in range(depth + 1)]
    anchors = 0
    for s in streams.values():
        for idx, (_ts, n) in enumerate(s):
            if not pat.search(n):
                continue
            anchors += 1
            for d in range(1, depth + 1):
                if idx + d < len(s):
                    succ[d][s[idx + d][1]] += 1
    print(f"\n# 인접성: /{after}/ 매칭 {anchors}회 ({anchors/steps:.1f}/스텝)")
    if anchors == 0:
        print("!! 매칭 0 — 정규식 확인"); sys.exit(0)
    for d in range(1, depth + 1):
        tot = sum(succ[d].values())
        print(f"\n## +{d}번째 커널 (총 {tot}, {tot/anchors*100:.0f}%가 매칭에 뒤따름)")
        for n, c in sorted(succ[d].items(), key=lambda x: -x[1])[:15]:
            print(f"  {c/anchors*100:5.1f}%  {c/steps:7.1f}/step  {n[:96]}")
    print("\n# ⚠ 개수는 정본, us 는 왜곡 — census 본문 주석과 동일 규율.")
    sys.exit(0)

# 그룹핑 규칙 (우리 소유 / 남의 것)
GROUPS = [
    ("우리 · 메가커널 세그먼트", r"mk_gemm2_kernel|mk_mhc_kernel|mk_mla_kernel"),
    # ^ 앵커 필수: 앵커 없는 `k_reduce` 는 deep_gemm 의 split_k_reduce 42발/스텝을
    # 우리 AR 로 셌다(08-31 인구조사의 "우리 소유 186" 이 그래서 42 만큼 부풀었다).
    ("우리 · osar AR",      r"^k_oneshot|^k_guard|^k_copy_in|^k_signal|^k_wait|^k_reduce"),
    ("우리 · MoE 게이트",    r"_deneb_gate"),
    ("우리 · 준비/인덱서",    r"_glm53_prep_fused|_gate_splitk"),
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
# 소유권 축: 이 리포에서 컴파일되는 커널 vs 이미지의 커널
oc, od = defaultdict(float), defaultdict(float)
for n, c in cnt.items():
    o = owner(n); oc[o] += c; od[o] += dur[n]
print()
print(f"{'소유':<22} {'커널/스텝':>10} {'비중':>7}   {'us/스텝(참고)':>14}")
print("-" * 60)
for o, label in (("ours", "우리 커널 (리포)"), ("image", "이미지 커널")):
    print(f"{label:<22} {oc[o]/steps:10.1f} {oc[o]/tot*100:6.1f}%   {od[o]/steps:13.1f}")

print()
print("# 상위 개별 커널 (개수순)")
for n, c in sorted(cnt.items(), key=lambda x: -x[1])[:25]:
    print(f"  {c/steps:8.1f}/step  {dur[n]/steps:8.1f}us  {n[:88]}")
print()
print("# ⚠ us 값은 CUPTI 왜곡이 있다(앞선 세션: GPU 바쁜시간 136ms vs 실측 47.6ms).")
print("#   개수는 오버헤드와 무관하므로 개수를 정본으로 쓸 것.")
