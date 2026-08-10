#!/bin/bash
# Docker Hub tag/digest watch for the pinned third-party base image.
#
# Why (MEASUREMENTS.md "소프트웨어 개선 경로 — 종결 선언"): the remaining gain
# sources include "저자 신규 이미지 발행 감시". A new image generation is also
# the delivery vehicle for #47808-class adaptive verification and for wiring
# the DSpark confidence head (present in the checkpoint, dropped by the
# current loader; see arXiv:2607.05147 — confidence-scheduled verification is
# the paper's production win at concurrency).
#
# Detects BOTH new tags and digest changes on existing tags (the author
# re-pushing production-hybrid-1.6 matters: the launcher pins the image ID
# and would fail preflight on drift).
#
# Exit codes: 0 = no change, 10 = changes detected (report printed), 1 = error.
# State: $STATE_FILE (default ~/.local/state/stkernel/image-tags.tsv).
# Alert: appends to $FLAG_FILE and POSTs $WEBHOOK_URL (JSON) when set.
#
# Install on srv2 as a user timer (script + units live in this repo):
#   cp ~/stkernel/launchers/watch-image-tags.sh ~/watch-image-tags.sh
#   cp ~/stkernel/launchers/dsv4-image-watch.{service,timer} ~/.config/systemd/user/
#   systemctl --user daemon-reload && systemctl --user enable --now dsv4-image-watch.timer
set -euo pipefail

REPO_SLUG="${REPO_SLUG:-aidendle94/sparkrun-vllm-ds4-gb10}" \
STATE_FILE="${STATE_FILE:-$HOME/.local/state/stkernel/image-tags.tsv}" \
FLAG_FILE="${FLAG_FILE:-$HOME/hybrid-stack/NEW-IMAGE-TAGS.txt}" \
WEBHOOK_URL="${WEBHOOK_URL:-}" \
python3 - <<'PYEOF'
import json
import os
import sys
import time
import urllib.request

slug = os.environ["REPO_SLUG"]
state_file = os.environ["STATE_FILE"]
flag_file = os.environ["FLAG_FILE"]
webhook = os.environ.get("WEBHOOK_URL", "")

# --- fetch every tag page ---------------------------------------------------
url = f"https://hub.docker.com/v2/repositories/{slug}/tags/?page_size=100"
current: dict[str, tuple[str, str]] = {}
try:
    while url:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.load(resp)
        for t in data.get("results", []):
            name = t.get("name") or ""
            if name:
                current[name] = (
                    t.get("digest") or "",
                    (t.get("last_updated") or "")[:19],
                )
        url = data.get("next")
except Exception as exc:  # noqa: BLE001 - single retry then hard fail
    print(f"WARN: fetch failed ({exc}); retrying once in 10s", file=sys.stderr)
    time.sleep(10)
    try:
        current = {}
        url = f"https://hub.docker.com/v2/repositories/{slug}/tags/?page_size=100"
        while url:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = json.load(resp)
            for t in data.get("results", []):
                name = t.get("name") or ""
                if name:
                    current[name] = (
                        t.get("digest") or "",
                        (t.get("last_updated") or "")[:19],
                    )
            url = data.get("next")
    except Exception as exc2:  # noqa: BLE001
        sys.exit(f"ERROR: Docker Hub fetch failed for {slug}: {exc2}")
if not current:
    sys.exit(f"ERROR: empty tag list for {slug}")


def dump_state(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as handle:
        for name in sorted(current):
            digest, updated = current[name]
            handle.write(f"{name}\t{digest}\t{updated}\n")
    os.replace(tmp, path)


# --- first run just seeds the state -----------------------------------------
if not os.path.exists(state_file):
    dump_state(state_file)
    print(f"initialized state with {len(current)} tags ({state_file})")
    sys.exit(0)

old: dict[str, tuple[str, str]] = {}
with open(state_file, encoding="utf-8") as handle:
    for line in handle:
        parts = line.rstrip("\n").split("\t")
        if parts and parts[0]:
            old[parts[0]] = (
                parts[1] if len(parts) > 1 else "",
                parts[2] if len(parts) > 2 else "",
            )

changes = []
for name in sorted(current):
    digest, updated = current[name]
    if name not in old:
        changes.append(f"NEW TAG   {name}  {digest}  {updated}")
    elif old[name][0] != digest:
        changes.append(f"REPUSHED  {name}  {old[name][0]} -> {digest}  {updated}")
for name in sorted(set(old) - set(current)):
    changes.append(f"REMOVED   {name}")

dump_state(state_file)

if not changes:
    print(f"no change ({slug})")
    sys.exit(0)

stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
report = f"[{stamp}] {slug} image feed changed:\n" + "\n".join(changes)
print(report)
try:
    os.makedirs(os.path.dirname(flag_file), exist_ok=True)
    with open(flag_file, "a", encoding="utf-8") as handle:
        handle.write(report + "\n\n")
except OSError as exc:
    print(f"WARN: could not append flag file {flag_file}: {exc}", file=sys.stderr)
if webhook:
    body = json.dumps({"text": "stkernel image watch:\n" + "\n".join(changes)})
    req = urllib.request.Request(
        webhook,
        data=body.encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(req, timeout=15).read()
    except Exception as exc:  # noqa: BLE001
        print(f"WARN: webhook POST failed: {exc}", file=sys.stderr)
sys.exit(10)
PYEOF
