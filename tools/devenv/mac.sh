#!/bin/zsh
# The Mac's side of tools/devenv (versions.env is the source): ~/.venvs/stkernel -- Python 3.12, torch's macOS wheel (CPU
# and MPS) and PY_PACKAGES -- first on PATH in interactive zsh, graphify at the committed extractor's version where it is
# missing, then the repo's doctor and a report of the tools Homebrew and uv keep. There is no macOS triton: the Triton
# tests run in the stk-test container (python:3.12 with torch's CPU wheel and triton, the files copied in), which this
# script only reports on. Idempotent; `zsh tools/devenv/mac.sh`.
set -euo pipefail
DIR=${0:A:h}
ROOT=${DIR:h:h}
. "$DIR/versions.env"
VENV=$HOME/.venvs/stkernel
UVBIN=$(command -v uv || echo "$HOME/.local/bin/uv")

if [ ! -x "$VENV/bin/python" ]; then
  mkdir -p "$HOME/.venvs"
  "$UVBIN" venv -q --seed --python 3.12 --python-preference only-managed "$VENV"
  echo "venv: $("$VENV/bin/python" --version) at $VENV"
fi
"$UVBIN" pip install -q --python "$VENV/bin/python" "torch==$TORCH_VERSION" ${=PY_PACKAGES}

MARK='# stkernel dev env (tools/devenv/mac.sh)'
if ! grep -qF "$MARK" "$HOME/.zshrc" 2>/dev/null; then
  touch "$HOME/.zshrc"
  cp -p "$HOME/.zshrc" "$HOME/.zshrc.pre-devenv"
  printf '\n%s: the repo'"'"'s Python 3.12 venv first -- python3, pip, ruff, hf resolve here\n%s\n' "$MARK" \
    'export PATH="$HOME/.venvs/stkernel/bin:$PATH"' >> "$HOME/.zshrc"
  echo "zshrc: ~/.venvs/stkernel/bin first on PATH (the previous file kept as ~/.zshrc.pre-devenv)"
fi
path=("$VENV/bin" $path)

if ! command -v graphify >/dev/null; then
  "$UVBIN" tool install -q "graphifyy==$GRAPHIFY_VERSION"
  echo "graphify: graphifyy $GRAPHIFY_VERSION"
fi

cd "$ROOT"
echo "doctor:"
python3 tools/dev_doctor.py --strict | sed 's/^/  /'
echo "tools:"
for t in python3 uv gh git docker node npm codex claude graphify mergiraf wt ruff codespell py-spy hf; do
  if command -v $t >/dev/null; then
    printf '  %-9s %s\n' $t "$($t --version 2>/dev/null | head -1)"
  else
    printf '  %-9s -- missing\n' $t
  fi
done
echo "  stk-test  $(docker ps --filter name=stk-test --format '{{.Image}} {{.Status}}' 2>/dev/null || true)"
