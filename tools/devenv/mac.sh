#!/bin/zsh
# The Mac's side of tools/devenv (versions.env is the source): ~/.venvs/stkernel -- Python 3.12, torch's macOS wheel (CPU
# and MPS) and PY_PACKAGES -- first on PATH in interactive zsh, graphify at the committed extractor's version (a venv of
# its own, linked into that one's bin), then the repo's doctor and a report of the tools Homebrew and uv keep. There is
# no macOS triton: the Triton tests run in the stk-test container (python:3.12 with torch's CPU wheel and triton, the
# files copied in), which this script only reports on. Idempotent; `zsh tools/devenv/mac.sh`.
set -euo pipefail
DIR=${0:A:h}
ROOT=${DIR:h:h}
. "$DIR/versions.env"
VENV=$HOME/.venvs/stkernel
UVBIN=$(command -v uv || echo "$HOME/.local/bin/uv")

if [ -x "$VENV/bin/python" ]; then              # a venv made with another Python is set aside (never deleted), then remade
  have=$("$VENV/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
  if [ "$have" != "$PYTHON_VERSION" ]; then
    aside="$VENV.pre-devenv-$have"
    [ -e "$aside" ] && aside="$aside.$(date +%Y%m%d-%H%M%S)"
    mv "$VENV" "$aside"
    echo "venv: Python $have, not $PYTHON_VERSION -- set aside as $aside"
  fi
fi
if [ ! -x "$VENV/bin/python" ]; then
  mkdir -p "$HOME/.venvs"
  "$UVBIN" venv -q --seed --python "$PYTHON_VERSION" --python-preference only-managed "$VENV"
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

# graphify at GRAPHIFY_VERSION, the extractor graphify-out/ is made with, the way node.sh keeps it on the nodes: a
# venv of its own (its dependencies stay out of the repo's), reinstalled when the pin moves, linked into the repo
# venv's bin -- first on PATH. A graphify installed some other way (pipx, uv tool) is left where it is; the link comes
# first in these shells.
GVENV=$HOME/.venvs/stkernel-graphify
graphify_version() { "$GVENV/bin/python" -c 'import importlib.metadata as m; print(m.version("graphifyy"))' 2>/dev/null || true; }
had=$(graphify_version)
if [ "$had" != "$GRAPHIFY_VERSION" ]; then
  [ -x "$GVENV/bin/python" ] || "$UVBIN" venv -q --python "$PYTHON_VERSION" --python-preference only-managed "$GVENV"
  "$UVBIN" pip install -q --python "$GVENV/bin/python" "graphifyy==$GRAPHIFY_VERSION"
  echo "graphify: graphifyy $GRAPHIFY_VERSION in $GVENV (was ${had:-not there})"
fi
ln -sfn "$GVENV/bin/graphify" "$VENV/bin/graphify"

cd "$ROOT"
echo "doctor:"
python3 tools/dev_doctor.py --strict | sed 's/^/  /'
echo "tools:"
for t in python3 uv gh git docker node npm codex claude graphify mergiraf wt ruff codespell py-spy hf; do
  if ! command -v $t >/dev/null; then
    printf '  %-9s -- missing\n' $t
  elif [ $t = graphify ]; then                    # it has no --version
    printf '  %-9s graphifyy %s (%s)\n' $t "$(graphify_version)" "$(command -v graphify)"
  else
    printf '  %-9s %s\n' $t "$($t --version 2>/dev/null | head -1)"
  fi
done
echo "  stk-test  $(docker ps --filter name=stk-test --format '{{.Image}} {{.Status}}' 2>/dev/null || true)"
