#!/usr/bin/env bash
# 40차: does the long-prefill engine death follow prefix caching?
#
# Two deaths carry the same fingerprint, both on arms with NO knobs set: a
# 6,912-token chunk (3 x the 2304 block, the APC align-mode chunk) at a
# block-aligned deep prefix, six KV blocks queued for zeroing in two runs of
# three, KV at 11-13%, worker rank 0 gone with no Python traceback. With
# PREFIX_CACHE=0 the mamba cache leaves "align" mode and the chunk stops being
# clipped to a block multiple, so the death should stop following it.
#
# Two arms, no leg on either -- the repeat probe IS the measurement, and it must
# be free to kill the engine without aborting the chain (an --after failure does
# not stop chain.sh, a dead onepass leg does). The defaults arm runs LAST so the
# serving left behind is production's own configuration.
#
# No deploy: this runs on whatever build is already deployed, so it needs no
# fresh baseline and costs one boot per arm and nothing else.
#
#   scp probes/run_apc_long_prefill_ab.sh srv2:glm53-logs/apc-ab-<S>.sh
#   scp probes/prefill_long_repeat.py     srv2:glm53-logs/
#   ssh srv2 'FLEET_SESSION=<S> bash ~/glm53-logs/fleet.sh run --gpu <S> 40 "..." \
#             -- bash ~/glm53-logs/apc-ab-<S>.sh'
set -uo pipefail
REPO=${REPO:-/home/choiceoh/stkernel}
LOGD=${LOGD:-/home/choiceoh/glm53-logs}
S=${FLEET_SESSION:-apc}
CTX=${CTX:-128000}
N=${N:-6}
OUT=$LOGD/apc6912-$S
mkdir -p "$OUT"
cd "$REPO" || exit 1
probe() { echo "python3 $LOGD/prefill_long_repeat.py --ctx $CTX --n $N --json $OUT/$1.json"; }
FLEET_SESSION=$S bash bench/chain.sh \
  APCOFF="PREFIX_CACHE=0" \
  APCON="" \
  --legs APCOFF none --legs APCON none \
  --after APCOFF "$(probe off)" \
  --after APCON "$(probe on)"
rc=$?
echo "== [apc-ab] done $(date +%T); evidence in $OUT"
for f in "$OUT"/off.json "$OUT"/on.json; do
  [ -s "$f" ] && python3 -c "
import json,sys
d=json.load(open('$f'))
print(f\"  {'$f'.rsplit('/',1)[-1]:>9}: survived {d['survived']}/{d['n']}\" +
      ('' if not d['died'] else f\" -- died on #{d['died']['at']}: {d['died']['detail'][:90]}\"))"
done
exit $rc
