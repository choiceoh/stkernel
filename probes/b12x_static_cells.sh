#!/usr/bin/env bash
# One probe PROCESS per spec cell (39차, idea 10): a CUDA illegal instruction in
# one cell poisons the context for every cell after it in the same process --
# PR #374's `t,h` took `t,h,s`, `u,h`, `z`, `z,h`, `z,s` AND the closing `u`
# numerics down with it (18:47), so six cells were never judged. Each cell here
# is its own run_mk_probe.sh container: ~20 s of overhead per cell, and the
# stock column is re-measured per cell (a free drift check).
#
#   bash probes/b12x_static_cells.sh "u|t|t,h|z" [probe args...]
#   OUT=/path/cells.out bash probes/b12x_static_cells.sh "u|z,h" --us 8,40,64
#
# The probe's own --isolate (#397) isolates the same way one level down: one
# child PROCESS per spec inside ONE container, plus a summary table (U=40 us,
# GB/s, gain over that child's own stock, numerics counts, status). Cheaper
# (no container or JIT per cell) but it cannot outlive a wedged container, so
# use this script when a cell may hang, --isolate when they only fault.
set -uo pipefail
SPECS=${1:?usage: b12x_static_cells.sh "spec|spec|..." [probe args...]}
shift
REPO=$(cd "$(dirname "$0")/.." && pwd)
OUT=${OUT:-$REPO/b12x_cells.out}
: > "$OUT"
IFS='|' read -r -a cells <<< "$SPECS"
pass=0; fail=0
for c in "${cells[@]}"; do
  echo "== cell [$c] $(date +%T) ==" | tee -a "$OUT"
  if timeout -k 20 "${CELL_TIMEOUT_S:-600}" bash "$REPO/probes/run_mk_probe.sh" probes/b12x_static_probe.py --configs "$c" "$@" >> "$OUT" 2>&1; then
    pass=$((pass+1))
  else
    fail=$((fail+1)); echo "== cell [$c] FAILED rc=$? ==" | tee -a "$OUT"
    # a killed docker client leaves its container running: remove any probe
    # container before the next cell (the gap runner's lesson)
    for ct in $(docker ps --format '{{.Names}}' | grep -v '^glm53'); do
      docker inspect --format '{{.Config.Image}}' "$ct" 2>/dev/null | grep -q 'glm53:v13-b12x' && docker rm -f "$ct" >/dev/null 2>&1
    done
  fi
done
echo "== cells: $pass ok, $fail failed -- $OUT =="
grep -hE "^v2\[|^stock U=|numerics .*-> (PASS|FAIL)|ERROR|illegal" "$OUT" | cut -c1-140
[ "$fail" = 0 ]
