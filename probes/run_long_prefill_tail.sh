#!/usr/bin/env bash
# 40차 후속: reproduce the long-prefill engine death WITH the decode, and
# explain the 2.5x tail that APC-on showed on repeated 128K prefills.
#
# Both deaths happened inside onepass's 128K stage, which generates after the
# long prefill; the max_tokens=1 probe stopped before the decode and did not
# reproduce them (6/6 on both sides of the APC A/B). So this one generates.
# The probe records num_preemptions_total, gpu_cache_usage_perc and the
# prefix-cache counters per request, so a run that does NOT die still says
# whether the tail is eviction-driven preemption.
#
# One arm, production defaults, no leg (the probe IS the measurement, and an
# --after failure does not abort chain.sh the way a dead onepass leg does), and
# no deploy -- it runs on whatever build is already deployed.
set -uo pipefail
REPO=${REPO:-/home/choiceoh/stkernel}
LOGD=${LOGD:-/home/choiceoh/glm53-logs}
S=${FLEET_SESSION:-tail}
CTX=${CTX:-128000}
N=${N:-8}
MAXTOK=${MAXTOK:-300}
OUT=$LOGD/tail-$S
mkdir -p "$OUT"
cd "$REPO" || exit 1
FLEET_SESSION=$S bash bench/chain.sh \
  TAILREPRO="" \
  --legs TAILREPRO none \
  --after TAILREPRO "python3 $LOGD/prefill_long_repeat.py --ctx $CTX --n $N --max-tokens $MAXTOK --json $OUT/tail.json"
rc=$?
# The arm's own head-log copy is taken BEFORE the --after step, so a death
# during the probe lands in a log the next boot overwrites. Keep our own.
docker logs glm53 > "$OUT/head.log" 2>&1 || echo "docker logs glm53 failed"
echo "== [tail] done $(date +%T); evidence in $OUT"
[ -s "$OUT/tail.json" ] && python3 -c "
import json
d = json.load(open('$OUT/tail.json'))
print(f\"  survived {d['survived']}/{d['n']}\" + ('' if not d['died'] else f\" -- died on #{d['died']['at']}: {d['died']['detail'][:120]}\"))
for i, r in enumerate(d['runs'], 1):
    c = r.get('counters', {})
    print(f\"  {i:>2}: {r['wall']:6.1f}s  {r['tok']/r['wall']:,.0f} tok/s  \" +
          ' '.join(f'{k}={v:g}' for k, v in sorted(c.items()) if v))"
exit $rc
