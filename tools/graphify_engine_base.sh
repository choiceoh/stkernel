#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
graphify_bin="${GRAPHIFY_BIN:-$(command -v graphify || true)}"
python_bin="${GRAPHIFY_PYTHON:-}"

if [[ -z "$python_bin" && -n "$graphify_bin" ]]; then
    shebang="$(sed -n '1s/^#!//p' "$graphify_bin")"
    if [[ "$shebang" == /usr/bin/env\ * ]]; then
        python_bin="${shebang##* }"
    else
        python_bin="$shebang"
    fi
fi

python_bin="${python_bin:-python3}"
if ! "$python_bin" -c 'import graphify' >/dev/null 2>&1; then
    echo "graphify is not importable by $python_bin" >&2
    echo "Install graphifyy or set GRAPHIFY_PYTHON to its interpreter." >&2
    exit 1
fi

export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
exec "$python_bin" "$repo_root/tools/graphify_engine_base.py" "$@"
