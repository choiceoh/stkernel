#!/bin/bash
# One node's development environment, to tools/devenv/versions.env -- the GB10 nodes srv1..srv4 (aarch64) and the RTX 5050
# PC ost-97x (x86_64, WSL2). The repo is the source: sync.sh applies this to every node and srv4's devenv-sync timer runs
# main's copy every morning. Idempotent: whatever is missing or at another version than the manifest is installed -- a
# file it replaces is moved to ~/.local/bin/.pre-devenv/, never deleted -- and a git setting or a checkout the node
# already has is left as it is. Run on the node (`bash tools/devenv/node.sh [--verify]`) or piped after versions.env
# through ssh (sync.sh).
#
#   ~/.bashrc        ~/.local/bin first on PATH, for non-interactive ssh too
#   ~/.local/bin     uv uvx gh mergiraf wt git-wt from their releases; node's commands (~/node-sdk); the agent CLIs
#                    (~/.npm-global)
#   python3 (3.12)   its user site: torch (CUDA 13.0) and triton, PY_PACKAGES; graphifyy as a uv tool
#   git              the operator's global settings where unset; gh answers github.com's credentials
#   ~/stkernel       cloned where missing; a clean main brought to origin/main, anything else left alone
#
# The verdict is tools/dev_doctor.py --strict with the GPU hidden -- production or a training job may hold it, and the
# doctor's probe would open a context beside them. --verify also runs a few CPU test files.
set -euo pipefail
if [ -z "${DEVENV_MANIFEST:-}" ]; then
  . "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/versions.env"
fi
VERIFY=0
[ "${1:-}" = --verify ] && VERIFY=1
log() { echo "[$(hostname -s) $(date +%H:%M:%S)] $*"; }
BIN=$HOME/.local/bin
mkdir -p "$BIN"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# -- PATH ---------------------------------------------------------------------------------------------------------------
if ! grep -q '^# Codex non-interactive SSH PATH' "$HOME/.bashrc" 2>/dev/null; then
  touch "$HOME/.bashrc"
  cp -p "$HOME/.bashrc" "$HOME/.bashrc.pre-devenv"
  { echo '# Codex non-interactive SSH PATH'
    echo 'case ":$PATH:" in *":$HOME/.local/bin:"*) ;; *) PATH="$HOME/.local/bin:$PATH";; esac'
    cat "$HOME/.bashrc.pre-devenv"; } > "$HOME/.bashrc.devenv"
  mv "$HOME/.bashrc.devenv" "$HOME/.bashrc"
  log "bashrc: ~/.local/bin first for non-interactive shells (the previous file kept as ~/.bashrc.pre-devenv)"
fi
export PATH="$BIN:$PATH"
hash -r

# -- single binaries ----------------------------------------------------------------------------------------------------
case "$(uname -m)" in                             # the release names: Rust target, Go arch, node's arch
  aarch64) TRIPLE=aarch64-unknown-linux GOARCH=arm64 NODEARCH=arm64 ;;
  x86_64) TRIPLE=x86_64-unknown-linux GOARCH=amd64 NODEARCH=x64 ;;
  *) echo "tools/devenv: no releases named for $(uname -m)" >&2; exit 1 ;;
esac
version_of() {    # the first x.y.z a command prints for --version, or nothing
  "$@" --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || true
}
keep() {          # a file (not a link) at a path about to be replaced goes aside with its version, never away
  local path=$1
  if [ -e "$path" ] && [ ! -L "$path" ]; then
    mkdir -p "$BIN/.pre-devenv"
    mv "$path" "$BIN/.pre-devenv/$(basename "$path")-$(version_of "$path")"
    log "kept the previous $(basename "$path") in ~/.local/bin/.pre-devenv/"
  fi
}
fetch() {         # url -> a fresh directory holding the archive's contents
  local url=$1 out
  out=$(mktemp -d -p "$TMP")
  curl -fsSL --retry 3 "$url" -o "$out/archive"
  case "$url" in
    *.tar.xz) tar -xJf "$out/archive" -C "$out" ;;
    *) tar -xzf "$out/archive" -C "$out" ;;
  esac
  rm -f "$out/archive"
  echo "$out"
}
binary() {        # name version url [more names]: the archive's `name` (and the others) into ~/.local/bin at `version`
  local name=$1 want=$2 url=$3 dir b whole=1
  shift 3
  for b in "$@"; do [ -x "$BIN/$b" ] || whole=0; done
  [ "$(version_of "$BIN/$name")" = "$want" ] && [ "$whole" = 1 ] && return 0
  dir=$(fetch "$url")
  for b in "$name" "$@"; do
    keep "$BIN/$b"
    install -m 0755 "$(find "$dir" -type f -name "$b" | head -1)" "$BIN/$b"
  done
  log "$name: $want"
}
binary uv "$UV_VERSION" "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-$TRIPLE-gnu.tar.gz" uvx
binary gh "$GH_VERSION" "https://github.com/cli/cli/releases/download/v$GH_VERSION/gh_${GH_VERSION}_linux_$GOARCH.tar.gz"
binary mergiraf "$MERGIRAF_VERSION" \
  "https://codeberg.org/mergiraf/mergiraf/releases/download/v$MERGIRAF_VERSION/mergiraf_$TRIPLE-gnu.tar.gz"
binary wt "$WORKTRUNK_VERSION" \
  "https://github.com/max-sixty/worktrunk/releases/download/v$WORKTRUNK_VERSION/worktrunk-$TRIPLE-musl.tar.xz" git-wt

# -- node and the agent CLIs --------------------------------------------------------------------------------------------
NODE_DIR=$HOME/node-sdk/node-v$NODE_VERSION-linux-$NODEARCH
if [ "$(version_of "$BIN/node")" != "$NODE_VERSION" ]; then
  if [ ! -x "$NODE_DIR/bin/node" ]; then
    mkdir -p "$HOME/node-sdk"
    mv "$(fetch "https://nodejs.org/dist/v$NODE_VERSION/node-v$NODE_VERSION-linux-$NODEARCH.tar.xz")/node-v$NODE_VERSION-linux-$NODEARCH" \
      "$NODE_DIR"
  fi
  for b in node npm npx corepack; do
    keep "$BIN/$b"
    ln -sfn "$NODE_DIR/bin/$b" "$BIN/$b"
  done
  hash -r
  log "node: v$NODE_VERSION ($NODE_DIR)"
fi
NPM_GLOBAL=$HOME/.npm-global
grep -q '^prefix=' "$HOME/.npmrc" 2>/dev/null || echo "prefix=$NPM_GLOBAL" >> "$HOME/.npmrc"
for pkg in $AGENT_NPM; do
  cmd=${pkg##*/}
  cmd=${cmd%-code}                                  # @openai/codex -> codex, @anthropic-ai/claude-code -> claude
  command -v "$cmd" >/dev/null && continue
  if [ ! -x "$NPM_GLOBAL/bin/$cmd" ]; then          # installed there but off PATH: only the link is missing
    npm install -g --prefix "$NPM_GLOBAL" --no-fund --no-audit --loglevel=error "$pkg" >/dev/null
  fi
  keep "$BIN/$cmd"
  ln -sfn "$NPM_GLOBAL/bin/$cmd" "$BIN/$cmd"
  log "$cmd: $(version_of "$BIN/$cmd") ($pkg)"
done
hash -r

# -- Python -------------------------------------------------------------------------------------------------------------
PIP=(python3 -m pip install --user --break-system-packages --disable-pip-version-check --no-warn-script-location -q)
if ! python3 -c "import sys, torch, triton; sys.exit(torch.__version__.split('+')[0] != '$TORCH_VERSION' or triton.__version__ != '$TRITON_VERSION')" 2>/dev/null; then
  "${PIP[@]}" --index-url "$TORCH_INDEX" --extra-index-url https://pypi.org/simple "torch==$TORCH_VERSION" "triton==$TRITON_VERSION"
  log "python: torch $TORCH_VERSION ($TORCH_INDEX), triton $TRITON_VERSION"
fi
# shellcheck disable=SC2086
"${PIP[@]}" $PY_PACKAGES
if ! uv tool list 2>/dev/null | grep -qx "graphifyy v$GRAPHIFY_VERSION"; then
  uv tool install -q --force "graphifyy==$GRAPHIFY_VERSION"
  log "graphify: graphifyy $GRAPHIFY_VERSION"
fi

# -- git ----------------------------------------------------------------------------------------------------------------
gset() { git config --global --get "$1" >/dev/null 2>&1 || { git config --global "$1" "$2"; log "git: $1"; }; }
gset user.name choiceoh
gset user.email choiceoh@topsolar.kr
gset pull.ff only
gset fetch.prune true
gset rerere.enabled true
gset diff.algorithm histogram
gset merge.conflictstyle zdiff3
gset push.autosetupremote true
gset branch.sort -committerdate
gset checkout.defaultremote origin
gset worktree.guessremote true
gset core.pager cat
gset alias.graph "log --oneline --graph --all -20"
gset http.postbuffer 524288000
gset merge.mergiraf.name mergiraf
gset merge.mergiraf.driver "$BIN/mergiraf merge --git %O %A %B -s %S -x %X -y %Y -p %P -l %L"
for host in https://github.com https://gist.github.com; do
  if ! git config --global --get-all "credential.$host.helper" 2>/dev/null | grep -q 'gh auth git-credential'; then
    git config --global --add "credential.$host.helper" ''
    git config --global --add "credential.$host.helper" "!$BIN/gh auth git-credential"
    log "git: credential.$host.helper -> gh"
  fi
done

# -- the checkout -------------------------------------------------------------------------------------------------------
REPO=$HOME/stkernel
if [ -d "$REPO/.git" ]; then
  git -C "$REPO" fetch -q origin
  # untracked files (a worktree directory, say) do not stop a fast-forward; git refuses one it would overwrite
  if [ "$(git -C "$REPO" branch --show-current)" = main ] && [ -z "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]; then
    git -C "$REPO" merge -q --ff-only origin/main 2>/dev/null || log "stkernel: main could not fast-forward -- left as it is"
  else
    log "stkernel: fetched; not main or changed files, so the checkout is left as it is"
  fi
else
  git clone -q https://github.com/choiceoh/stkernel.git "$REPO"
  log "stkernel: cloned"
fi

# -- the verdict --------------------------------------------------------------------------------------------------------
cd "$REPO"
log "doctor ($(git log -1 --format='%h %cs'), $(git branch --show-current)):"
if [ -f tools/dev_doctor.py ]; then
  CUDA_VISIBLE_DEVICES= python3 tools/dev_doctor.py --strict | sed 's/^/  /'
else
  echo "  this checkout predates tools/dev_doctor.py (left as it is above)"
fi
if [ "$VERIFY" = 1 ]; then
  modules=()
  for m in test_docs_status test_engine_devenv test_engine_qwen38_draft_ahead test_engine_gated_residual_rows; do
    [ -f "tests/$m.py" ] && modules+=("tests.$m")         # the checkout's own: a file main does not have yet is skipped
  done
  log "tests:"
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 TRITON_INTERPRET=1 python3 -m unittest -q "${modules[@]}" 2>&1 | tail -3 \
    | sed 's/^/  /'
fi
log "tools: uv $(version_of uv) · gh $(version_of gh) · node $(version_of node) · mergiraf $(version_of mergiraf)" \
  "· wt $(version_of wt) · codex $(version_of codex) · claude $(version_of claude)" \
  "· $(uv tool list 2>/dev/null | grep '^graphifyy') · $(python3 -c 'import torch, triton; print("torch", torch.__version__, "· triton", triton.__version__)')"
