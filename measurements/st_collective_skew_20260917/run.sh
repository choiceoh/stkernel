#!/bin/bash
# Reproduce this record. `record.py` takes the recording on the live door (it RESERVES it, and
# releases it in a finally); the decomposition is 09-14's own tool, unchanged.
set -euo pipefail
OUT=${1:-/tmp/skew-traces}
python3 "$(dirname "$0")/record.py" "skew$(date +%s)" "$OUT"
cd "$OUT"
for f in rank*-decode-*.trace.json.gz; do            # 09-14's tool wants rank{N}-decode{S}.json.gz
  cp "$f" "$(echo "$f" | sed -E 's/rank([0-9])-decode-([0-9]+)\.trace\.json\.gz/rank\1-decode\2.json.gz/')"
done
python3 "$(dirname "$0")/../st_oneshot_rails_20260914/decompose_trace.py" "$OUT" 2 3 4 5
python3 "$(dirname "$0")/per_rank_work.py" "$OUT" 2 3 4 5
